from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tranpy.datasets import (
    rotor_angle_separation,
    stability_labels,
    transient_stability_index,
    unwrapped_rotor_angles,
)
from tranpy.simulation import MockEngine, generate_events
from tranpy.simulation.engine import FaultScenario
from tranpy.simulation.powerfactory import PowerFactoryEngine
from tranpy.simulation.protocol import PROTOCOLS, NormalOperationSampler, StressTestSampler
from tranpy_core.data import SignalDataset
from tranpy_core.errors import ConfigurationError


def test_protocol_names() -> None:
    assert set(PROTOCOLS) == {"baseline", "stress_test", "normal_operation"}


def test_stress_test_sampler() -> None:
    sampler = StressTestSampler(n_lines=6, n_loads=3)
    scenarios = sampler.sample(200, seed=1, first_id=10)
    assert [s.scenario_id for s in scenarios] == list(range(10, 210))
    assert {s.fault_start_s for s in scenarios} == {1.0}
    assert {s.fault_position for s in scenarios} == {0.0, 1.0}
    assert all(0.05 <= s.clearing_duration_s <= 0.40 for s in scenarios)
    assert all(0.8 <= s.operating_point <= 1.2 for s in scenarios)
    assert all(s.load_factors is not None and len(s.load_factors) == 3 for s in scenarios)
    assert all(s.trip_line is None for s in scenarios)
    assert sampler.simulation_time_s == pytest.approx(6.4)
    assert sampler.sample(5, seed=1) == sampler.sample(5, seed=1)

    along = StressTestSampler(
        n_lines=3, n_loads=1, fault_position="along", second_trip=True,
        line_neighbours=((1,), (0, 2), ()),
    )  # fmt: skip
    for s in along.sample(50, seed=2):
        assert 0.0 <= s.fault_position <= 1.0
        assert s.trip_line == (None if s.fault_line == 2 else s.trip_line)
        if s.trip_line is not None:
            assert s.trip_line in along.line_neighbours[s.fault_line]  # type: ignore[index]


def test_stress_test_sampler_validation() -> None:
    with pytest.raises(ConfigurationError) as info:
        StressTestSampler(
            n_lines=0, n_loads=1, loading=(1.2, 0.8), load_noise=-1, second_trip=True,
            fault_position="middle",  # type: ignore[arg-type]
        )  # fmt: skip
    text = str(info.value)
    for part in ("at least one line", "loading", "load_noise", "fault_position", "second_trip"):
        assert part in text


def test_normal_operation_sampler() -> None:
    profiles = np.array([[1.0, 2.0], [0.5, 1.0]])
    sampler = NormalOperationSampler(n_lines=3, load_profiles=profiles, line_weights=[0, 1, 0])
    scenarios = sampler.sample(400, seed=0)
    assert sampler.n_loads == 2
    assert {s.fault_line for s in scenarios} == {1}
    assert {s.load_factors for s in scenarios} == {(1.0, 2.0), (0.5, 1.0)}
    single = np.mean([s.fault_type == "single_phase" for s in scenarios])
    assert 0.7 < single < 0.9
    durations = np.array([s.clearing_duration_s for s in scenarios])
    assert np.all((durations <= 0.15) | (durations >= 0.30))
    assert sampler.simulation_time_s == pytest.approx(6.5)
    with pytest.raises(ConfigurationError, match="line_weights"):
        NormalOperationSampler(n_lines=3, load_profiles=profiles, line_weights=[1, 1])
    with pytest.raises(ConfigurationError, match="load_profiles"):
        NormalOperationSampler(n_lines=3, load_profiles=-profiles, backup_share=2.0)


def _wrapped(events: SignalDataset) -> SignalDataset:
    cols = events.channel_index("*|c:firot*")
    values = events.values.copy()
    values[:, cols] = (values[:, cols] + 180.0) % 360.0 - 180.0
    return events.replace(values=values)


def test_unwrapping_restores_angles(mock_events_full: SignalDataset) -> None:
    raw = unwrapped_rotor_angles(mock_events_full)
    unwrapped = unwrapped_rotor_angles(_wrapped(mock_events_full))
    # equal up to a whole number of turns per event and generator
    shift = unwrapped - raw
    np.testing.assert_allclose(shift, 360.0 * np.round(shift / 360.0), atol=1e-6)
    np.testing.assert_allclose(
        rotor_angle_separation(_wrapped(mock_events_full)),
        rotor_angle_separation(mock_events_full),
        atol=1e-6,
    )


def test_label_criteria(mock_events_full: SignalDataset) -> None:
    for criterion in ("out_of_step", "angle_360", "coi_180"):
        labels = stability_labels(mock_events_full, criterion=criterion)  # type: ignore[arg-type]
        assert labels.shape == (mock_events_full.n_samples,)
    with pytest.raises(ConfigurationError, match="unknown label criterion"):
        stability_labels(mock_events_full, criterion="other")  # type: ignore[arg-type]
    np.testing.assert_allclose(transient_stability_index([0.0, 360.0]), [100.0, 0.0])


def test_generate_events_annotates_stress_test_scenarios() -> None:
    engine = MockEngine("ieee_9bus", simulation_time_s=3.0)
    sampler = StressTestSampler(n_lines=engine.n_lines, n_loads=engine.n_loads, post_clearing_s=1.6)
    events = generate_events(
        engine, sampler.sample(20, seed=4), criterion="angle_360", protocol="stress_test"
    )
    assert events.metadata["protocol"] == "stress_test"
    assert events.metadata["label_criterion"] == "angle_360"
    for key in ("tsi", "angle_separation_deg", "unstable_coi_180", "mean_load_factor"):
        assert key in events.annotations
    np.testing.assert_array_equal(
        events.labels, (np.asarray(events.annotations["angle_separation_deg"]) > 360).astype(int)
    )


def test_mock_single_phase_faults_are_milder() -> None:
    engine = MockEngine("ieee_9bus", simulation_time_s=3.0)
    three = FaultScenario(scenario_id=0, fault_line=0, fault_start_s=1.0, clearing_duration_s=0.3)
    single = replace(three, fault_type="single_phase")
    a, b = engine.simulate([three, single])
    va, vb = (r.values[:, 0] for r in (a, b))
    assert vb.min() > va.min()


def test_powerfactory_rejects_new_protocol_fields() -> None:
    engine = object.__new__(PowerFactoryEngine)
    scenario = FaultScenario(
        scenario_id=3, fault_line=0, fault_start_s=1.0, clearing_duration_s=0.1,
        fault_type="single_phase", fault_position=1.0,
    )  # fmt: skip
    with pytest.raises(ConfigurationError, match="fault_type, fault_position"):
        engine._simulate_one(scenario)

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tranpy.datasets import FAULT_START
from tranpy.simulation import FaultScenario, ScenarioSampler, generate_events
from tranpy_core.errors import ConfigurationError

pytestmark = [pytest.mark.requires_andes, pytest.mark.slow]
pytest.importorskip("andes")


@pytest.fixture(scope="module")
def engine():  # type: ignore[no-untyped-def]
    from tranpy.simulation.andes import AndesEngine

    return AndesEngine("new_england", simulation_time_s=4.0, step_s=0.01)


def test_network_description(engine) -> None:  # type: ignore[no-untyped-def]
    assert engine.n_lines == 34  # transmission lines only, as in the PowerFactory model
    assert engine.n_loads == 19
    assert engine.nominal_frequency_hz == 60.0
    neighbours = engine.line_neighbours
    assert len(neighbours) == engine.n_lines
    for i, adjacent in enumerate(neighbours):
        assert i not in adjacent
        assert all(i in neighbours[j] for j in adjacent)  # symmetric


def test_generator_islands_are_not_simulated(engine) -> None:  # type: ignore[no-untyped-def]
    pairs = [(i, j) for i in range(engine.n_lines) for j in range(engine.n_lines) if i != j]
    base = FaultScenario(scenario_id=0, fault_line=0, fault_start_s=0.5, clearing_duration_s=0.1)
    splits = [replace(base, fault_line=i, trip_line=j) for i, j in pairs]
    splits = [s for s in splits if engine.islands(s)]
    with_generator = next(s for s in splits if engine._separation(s)[2])
    loads_only = next(s for s in splits if not engine._separation(s)[2])
    assert not engine.simulate([with_generator])[0].completed
    assert engine.simulate([loads_only])[0].completed  # cut-off loads are switched off


def test_fault_duration_decides_stability(engine) -> None:  # type: ignore[no-untyped-def]
    line = engine._lines.index("Line_21")  # bus 16 - bus 17, inside the meshed network
    short = FaultScenario(
        scenario_id=1, fault_line=line, fault_start_s=0.5, clearing_duration_s=0.05
    )
    long = replace(short, scenario_id=2, clearing_duration_s=0.8)
    events = generate_events(engine, [short, long])
    assert events.n_samples >= 1
    by_id = dict(zip(events.sample_ids.tolist(), events.labels.tolist(), strict=True))  # type: ignore[union-attr]
    assert by_id == {1: 0, 2: 1}
    assert events.channel_names[0] == "Bus 01|m:u in p.u."
    assert "G 10|s:outofstep" in events.channel_names
    assert np.all(
        np.asarray(events.annotations["simulation_end_s"])
        > np.asarray(events.annotations[FAULT_START])
    )


def test_validation_errors(engine) -> None:  # type: ignore[no-untyped-def]
    from tranpy.simulation.andes import AndesEngine

    with pytest.raises(ConfigurationError, match="outside the network"):
        engine.simulate(
            [
                FaultScenario(
                    scenario_id=0, fault_line=99, fault_start_s=0.5, clearing_duration_s=0.1
                )
            ]
        )
    with pytest.raises(ConfigurationError, match="no built-in ANDES case"):
        AndesEngine("ieee_9bus")
    with pytest.raises(ConfigurationError):
        AndesEngine("new_england", step_s=0)


def test_sampler_draws_topological_neighbours(engine) -> None:  # type: ignore[no-untyped-def]
    sampler = ScenarioSampler(
        n_lines=engine.n_lines,
        n_loads=engine.n_loads,
        trip_adjacent_line=True,
        line_neighbours=engine.line_neighbours,
    )
    for scenario in sampler.sample(50, seed=0):
        if scenario.trip_line is not None:
            assert scenario.trip_line in engine.line_neighbours[scenario.fault_line]

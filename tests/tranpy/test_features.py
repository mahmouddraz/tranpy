from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks

from tranpy.datasets import FAULT_CLEAR, FAULT_START
from tranpy.features import TrajectoryFeatures, window_end_times
from tranpy_core.data import SignalDataset
from tranpy_core.errors import ConfigurationError, DataError


def _naive_rotor(sample, gen: str, end: float) -> dict[str, float]:  # type: ignore[no-untyped-def]
    """Straightforward per-event reference of the rotor features (the old algorithm)."""
    keep = sample.time <= end
    t = sample.time[keep]
    a = sample.channel(f"{gen}|c:firot in deg")[keep]
    pre = t < sample.annotations[FAULT_START]
    a0 = a[pre][-1] if pre.any() else a[0]
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    rate = np.diff(a) / dt
    accel = np.diff(rate) / dt[:-1]
    post = a[~pre]
    peaks, _ = find_peaks(np.abs(post - a0), distance=3)
    return {
        "delta_max": float(np.max(np.abs(a - a0))),
        "delta_final": float(a[-1] - a0),
        "delta_rate_max": float(np.max(np.abs(rate))),
        "delta_rate_mean": float(np.mean(np.abs(rate))),
        "delta_accel_max": float(np.max(np.abs(accel))),
        "first_swing_amp": float(np.abs(post[peaks[0]] - a0))
        if peaks.size
        else float(np.max(np.abs(post - a0))),
    }


@pytest.mark.parametrize("reference", ["fault_start", "fault_clear", "absolute"])
def test_rotor_features_match_naive_reference(mock_events: SignalDataset, reference: str) -> None:
    window = 0.5 if reference != "absolute" else 1.5
    table = TrajectoryFeatures(window_s=window, reference=reference).transform(mock_events)  # type: ignore[arg-type]
    ends = window_end_times(mock_events, window, reference)  # type: ignore[arg-type]
    for i in (0, 7, 19):
        ref = _naive_rotor(mock_events[i], "G 02", ends[i])
        for key, value in ref.items():
            assert table.features[i, table.feature_names.index(f"G02_{key}")] == pytest.approx(
                value, rel=1e-12, abs=1e-12
            )


def test_feature_set_and_names(mock_events: SignalDataset) -> None:
    table = TrajectoryFeatures().transform(mock_events)
    names = set(table.feature_names)
    assert list(table.feature_names) == sorted(table.feature_names)
    expected_names = (
        "G01_delta_max", "G03_speed_integral", "G02_freq_dev_max", "G01_power_swing_amp",
        "G01_reactive_peak", "sys_voltage_nadir", "sys_voltage_recovery_time", "sys_rocof_max",
        "sys_angle_spread_max", "sys_total_power_swing", "sys_coi_angle_max_dev",
    )  # fmt: skip
    for expected in expected_names:
        assert expected in names
    assert np.isfinite(table.features).all()
    np.testing.assert_array_equal(table.labels, mock_events.labels)
    rec = table.features[:, table.feature_names.index("sys_voltage_recovery_time")]
    assert np.all(rec >= 0)  # measured from fault inception, not absolute time


def test_feature_groups_can_be_switched_off(mock_events: SignalDataset) -> None:
    only_voltage = TrajectoryFeatures(
        include_rotor=False, include_frequency=False, include_power=False, include_coherency=False
    ).transform(mock_events)
    assert all(n.startswith("sys_voltage") for n in only_voltage.feature_names)
    nothing = TrajectoryFeatures(
        include_rotor=False,
        include_voltage=False,
        include_frequency=False,
        include_power=False,
        include_coherency=False,
    ).transform(mock_events)
    assert nothing.features.shape == (mock_events.n_samples, 0)
    no_channels = TrajectoryFeatures().transform(
        mock_events.select_channels("Line*")
        if mock_events.channel_index("Line*").size
        else mock_events.select_channels("G 01|s:xspeed in p.u.")
    )
    assert "sys_total_power_swing" in no_channels.feature_names


def test_window_is_causal(mock_events: SignalDataset) -> None:
    """Changing data after the window end must not change any feature (no look-ahead)."""
    extractor = TrajectoryFeatures(window_s=0.3, reference="fault_start")
    before = extractor.transform(mock_events).features
    ends = window_end_times(mock_events, 0.3, "fault_start")
    future = mock_events.time > ends[mock_events.sample_index]
    values = mock_events.values.copy()
    values[future] = (
        np.random.default_rng(0).normal(size=(int(future.sum()), values.shape[1])) * 1e3
    )
    tampered = mock_events.replace(values=values)
    np.testing.assert_array_equal(extractor.transform(tampered).features, before)


def test_shift_invariance_of_deviation_features(mock_events: SignalDataset) -> None:
    """Adding a constant to all rotor angles leaves deviation-type features unchanged."""
    cols = mock_events.channel_index("*|c:firot in deg")
    values = mock_events.values.copy()
    values[:, cols] += 37.0
    a = TrajectoryFeatures().transform(mock_events)
    b = TrajectoryFeatures().transform(mock_events.replace(values=values))
    for name in (
        "G01_delta_max",
        "G02_delta_rate_mean",
        "G03_first_swing_amp",
        "sys_angle_spread_max",
        "sys_angle_std_max",
    ):
        j = a.feature_names.index(name)
        np.testing.assert_allclose(b.features[:, j], a.features[:, j], rtol=1e-9, atol=1e-9)


def test_settings_validation_and_missing_annotations(mock_events: SignalDataset) -> None:
    with pytest.raises(ConfigurationError) as info:
        TrajectoryFeatures(window_s=-1.0)
    assert len(info.value.problems) == 2
    with pytest.raises(ConfigurationError, match="unknown reference"):
        TrajectoryFeatures(reference="fault_peak")  # type: ignore[arg-type]
    no_clear = mock_events.replace(annotations={FAULT_START: mock_events.annotations[FAULT_START]})
    with pytest.raises(DataError, match="no 'fault_clear_s' annotation"):
        TrajectoryFeatures(window_s=0.2, reference="fault_clear").transform(no_clear)
    nan_clear = mock_events.with_annotations(
        **{FAULT_CLEAR: np.full(mock_events.n_samples, np.nan)}
    )
    with pytest.raises(DataError, match="have no detected"):
        TrajectoryFeatures(window_s=0.2, reference="fault_clear").transform(nan_clear)
    unannotated = mock_events.replace(annotations={})
    assert TrajectoryFeatures().transform(unannotated).n_samples == mock_events.n_samples


def test_trajectory_sequence_shape_and_causality(mock_events: SignalDataset) -> None:
    from tranpy.features import TrajectorySequence

    rep = TrajectorySequence(window_s=0.2, reference="fault_start", step_s=0.05, history_s=0.1)
    assert rep.n_steps == 7
    table = rep.transform(mock_events)
    n_channels = table.metadata["n_channels"]
    assert table.features.shape == (mock_events.n_samples, 7 * n_channels)
    assert table.feature_names[0].startswith("t000|")
    assert table.feature_names[n_channels].startswith("t001|")
    ends = window_end_times(mock_events, 0.2, "fault_start")
    future = mock_events.time > ends[mock_events.sample_index]
    values = mock_events.values.copy()
    values[future] = 1e6
    tampered = rep.transform(mock_events.replace(values=values))
    np.testing.assert_array_equal(tampered.features, table.features)
    with pytest.raises(ConfigurationError) as info:
        TrajectorySequence(window_s=0, reference="later")  # type: ignore[arg-type]
    assert len(info.value.problems) == 2

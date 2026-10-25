from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tranpy_core.data import SignalDataset
from tranpy_core.errors import ConfigurationError, DataError
from tranpy_core.features import WindowStatistics, crop_to_window
from tranpy_core.features import segments as seg


@st.composite
def segmented(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    lengths = draw(st.lists(st.integers(1, 7), min_size=1, max_size=6))
    values = draw(arrays(np.float64, (sum(lengths), 2), elements=st.floats(-1e3, 1e3)))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    parts = [values[offsets[i] : offsets[i + 1]] for i in range(len(lengths))]
    return values, offsets, parts


@settings(max_examples=60, deadline=None)
@given(segmented())
def test_reductions_match_per_segment_numpy(
    case: tuple[np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    values, offsets, parts = case
    np.testing.assert_array_equal(seg.segment_max(values, offsets), [p.max(0) for p in parts])
    np.testing.assert_array_equal(seg.segment_min(values, offsets), [p.min(0) for p in parts])
    np.testing.assert_allclose(
        seg.segment_sum(values, offsets), [p.sum(0) for p in parts], atol=1e-9
    )
    np.testing.assert_allclose(
        seg.segment_mean(values, offsets), [p.mean(0) for p in parts], atol=1e-9
    )
    np.testing.assert_array_equal(seg.segment_first(values, offsets), [p[0] for p in parts])
    np.testing.assert_array_equal(seg.segment_last(values, offsets), [p[-1] for p in parts])
    mask = values[:, 0] > 0
    np.testing.assert_array_equal(
        seg.segment_any(mask, offsets), [(p[:, 0] > 0).any() for p in parts]
    )
    expected_first = [int(np.argmax(p[:, 0] > 0)) if (p[:, 0] > 0).any() else -1 for p in parts]
    assert seg.first_true_index(mask, offsets).tolist() == expected_first
    expected_last = [
        p[np.flatnonzero(p[:, 0] > 0)[-1]] if (p[:, 0] > 0).any() else p[0] for p in parts
    ]
    np.testing.assert_array_equal(seg.segment_last_where(values, offsets, mask), expected_last)
    rows, new_off = seg.compact(mask, offsets)
    assert np.array_equal(values[rows], values[mask])
    assert np.diff(new_off).tolist() == [int((p[:, 0] > 0).sum()) for p in parts]


@settings(max_examples=40, deadline=None)
@given(segmented())
def test_segment_diff(case: tuple[np.ndarray, np.ndarray, list[np.ndarray]]) -> None:
    values, offsets, parts = case
    diffs, off = seg.segment_diff(values, offsets)
    for i, p in enumerate(parts):
        np.testing.assert_array_equal(diffs[off[i] : off[i + 1]], np.diff(p, axis=0))


def test_segment_checks() -> None:
    with pytest.raises(ValueError, match="offsets end"):
        seg.segment_max(np.ones((3, 1)), np.array([0, 2]))
    with pytest.raises(ValueError, match="at least one row"):
        seg.segment_max(np.ones((3, 1)), np.array([0, 0, 3]))
    with pytest.raises(ValueError, match="default"):
        seg.segment_last_where(np.ones(2), np.array([0, 2]), np.ones(2, bool), default="last")
    rows, off = seg.compact(np.zeros(0, bool), np.array([0]))
    assert rows.size == 0 and off.tolist() == [0]


def _ramp_dataset() -> SignalDataset:
    times = [np.arange(0.0, 1.01, 0.1), np.arange(0.0, 2.01, 0.1)]
    values = [np.column_stack([t, -t]) for t in times]
    return SignalDataset.from_samples(
        times,
        values,
        channel_names=("A|x", "B|y"),
        labels=[0, 1],
        label_names=("a", "b"),
        name="ramp",
    )


def test_crop_to_window_per_sample_bounds() -> None:
    ds = _ramp_dataset()
    cropped = crop_to_window(ds, end=[0.5, 1.0], start=0.2)
    assert np.allclose(cropped[0].time, [0.2, 0.3, 0.4, 0.5])
    assert cropped[1].time.max() == pytest.approx(1.0)
    with pytest.raises(DataError, match="fewer than"):
        crop_to_window(ds, end=0.05)


def test_window_statistics() -> None:
    ds = _ramp_dataset()
    table = WindowStatistics(
        channels=("A|*",),
        statistics=("mean", "std", "min", "max", "first", "last", "range"),
        end=1.0,
    ).transform(ds)
    assert table.feature_names[0] == "A|x__mean"
    np.testing.assert_allclose(table.features[:, 0], [0.5, 0.5])
    np.testing.assert_allclose(table.features[:, 1], np.std(np.arange(0, 1.01, 0.1)))
    np.testing.assert_allclose(table.features[:, -1], [1.0, 1.0])
    full = WindowStatistics().transform(ds)
    assert full.n_features == 2 * 5
    np.testing.assert_allclose(full.features[1, full.feature_names.index("B|y__min")], -2.0)
    with pytest.raises(ConfigurationError, match="unknown statistics"):
        WindowStatistics(statistics=("median",))  # type: ignore[arg-type]

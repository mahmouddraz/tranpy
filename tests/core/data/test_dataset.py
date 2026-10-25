from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tranpy_core.data import SignalDataset, TabularDataset
from tranpy_core.errors import DataError


def _signals(lengths: list[int], n_channels: int = 2, seed: int = 0) -> SignalDataset:
    rng = np.random.default_rng(seed)
    times = [np.arange(n) * 0.1 for n in lengths]
    values = [rng.normal(size=(n, n_channels)) for n in lengths]
    return SignalDataset.from_samples(
        times,
        values,
        channel_names=tuple(f"E{i}|v" for i in range(n_channels)),
        labels=np.arange(len(lengths)) % 2,
        label_names=("a", "b"),
        name="s",
        annotations={"t0": np.arange(len(lengths), dtype=float)},
    )


def test_tabular_is_read_only_and_copied(toy: TabularDataset) -> None:
    src = np.ones((3, 2))
    ds = TabularDataset(
        features=src, labels=[0, 1, 0], feature_names=("a", "b"), label_names=("x", "y"), name="t"
    )
    src[0, 0] = 5
    assert ds.features[0, 0] == 1
    with pytest.raises(ValueError, match="read-only"):
        ds.features[0, 0] = 2
    assert ds.n_samples == len(ds) == 3
    assert ds.class_counts().tolist() == [2, 1]
    assert "t" in repr(ds)


def test_tabular_validation_reports_all_problems() -> None:
    with pytest.raises(DataError) as info:
        TabularDataset(
            features=np.ones((3, 2)),
            labels=[0, 1],
            feature_names=("a", "a", "c"),
            label_names=("x",),
            name="bad",
            sample_ids=[1, 2],
        )
    msg = str(info.value)
    for part in ("2 labels", "sample_ids", "feature names", "labels must lie"):
        assert part in msg
    with pytest.raises(DataError, match="2-dimensional"):
        TabularDataset(
            features=np.ones(3), labels=[0], feature_names=(), label_names=("x",), name="b"
        )


def test_tabular_subset_select_frame_duplicates(toy: TabularDataset) -> None:
    sub = toy.subset([5, 2])
    assert sub.sample_ids is not None and sub.sample_ids.tolist() == [5, 2]
    assert np.array_equal(sub.features, toy.features[[5, 2]])
    sel = toy.select_features(["f2", "f0"])
    assert sel.feature_names == ("f2", "f0")
    assert np.array_equal(sel.features[:, 1], toy.features[:, 0])
    with pytest.raises(DataError, match="unknown features"):
        toy.select_features(["nope"])
    frame = toy.to_frame()
    assert list(frame.columns)[-1] == "label" and len(frame) == toy.n_samples
    dup = TabularDataset(
        features=[[1, 2], [3, 4], [1, 2]],
        labels=[0, 1, 0],
        feature_names=("a", "b"),
        label_names=("x", "y"),
        name="d",
    )
    groups = dup.duplicate_groups()
    assert groups[0] == groups[2] != groups[1]


def test_signal_dataset_access_and_subset() -> None:
    ds = _signals([3, 5, 4])
    assert ds.n_samples == 3 and ds.n_channels == 2
    assert ds.lengths.tolist() == [3, 5, 4]
    assert ds.sample_index.tolist() == [0] * 3 + [1] * 5 + [2] * 4
    s = ds[1]
    assert s.time.shape == (5,) and s.values.shape == (5, 2)
    assert s.annotations["t0"] == 1.0
    assert np.array_equal(s.channel("E1|v"), s.values[:, 1])
    with pytest.raises(DataError):
        s.channel("missing")
    assert ds[-1].sample_id == 2
    with pytest.raises(IndexError):
        ds[3]
    sub = ds.subset([2, 0])
    assert sub.lengths.tolist() == [4, 3]
    assert np.array_equal(sub[0].values, ds[2].values)
    assert np.array_equal(sub[1].values, ds[0].values)
    assert sub.annotations["t0"].tolist() == [2.0, 0.0]
    assert "SignalDataset" in repr(ds)


@given(st.lists(st.integers(2, 6), min_size=1, max_size=6), st.data())
def test_signal_subset_matches_per_sample_slices(lengths: list[int], data: st.DataObject) -> None:
    ds = _signals(lengths)
    idx = data.draw(st.lists(st.integers(0, len(lengths) - 1), min_size=1, max_size=8))
    sub = ds.subset(idx)
    for k, i in enumerate(idx):
        assert np.array_equal(sub[k].time, ds[i].time)
        assert np.array_equal(sub[k].values, ds[i].values)


def test_channel_selection() -> None:
    ds = _signals([3, 3], n_channels=3)
    assert ds.channel_index(["E0|*", "E2|v"]).tolist() == [0, 2]
    assert ds.select_channels("E1|*").channel_names == ("E1|v",)
    assert ds.drop_channels("E1|*").channel_names == ("E0|v", "E2|v")
    with pytest.raises(DataError, match="no channel"):
        ds.select_channels("X*")
    with_ann = ds.with_annotations(extra=[1.0, 2.0])
    assert set(with_ann.annotations) == {"t0", "extra"}


def test_signal_validation() -> None:
    with pytest.raises(DataError) as info:
        SignalDataset(
            time=np.arange(4.0),
            values=np.ones((3, 2)),
            offsets=[0, 1, 4],
            channel_names=("a",),
            labels=[0, 5],
            label_names=("x", "y"),
            name="bad",
            annotations={"k": [1.0]},
            sample_ids=[1],
        )
    msg = str(info.value)
    for part in (
        "value rows",
        "channel names",
        "at least two",
        "sample_ids",
        "annotation 'k'",
        "labels must lie",
    ):
        assert part in msg
    with pytest.raises(DataError, match="offsets"):
        SignalDataset(
            time=np.arange(4.0),
            values=np.ones((4, 1)),
            offsets=[0, 3],
            channel_names=("a",),
            labels=[0],
            label_names=("x",),
            name="b",
        )
    with pytest.raises(DataError, match="time vectors"):
        SignalDataset.from_samples(
            [np.arange(3)], [], channel_names=("a",), labels=[0], label_names=("x",), name="b"
        )

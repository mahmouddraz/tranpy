from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.conftest import write_powerfactory_csvs

import tranpy
from tranpy.datasets import (
    FAULT_CLEAR,
    FAULT_START,
    data_registry,
    detect_fault_timing,
    list_datasets,
    load_events,
    read_event_csvs,
    stability_labels,
)
from tranpy.systems import IEEE_9BUS, NEW_ENGLAND, get_system
from tranpy.vocabulary import LABEL_NAMES, ChannelName, ElementKind, Stability, element_kind
from tranpy_core.data import SignalDataset, save_signals
from tranpy_core.errors import ConfigurationError, DataError


@pytest.mark.parametrize(
    ("loader", "shape", "counts"),
    [
        (tranpy.load_new_england, (5000, 77), [2650, 2350]),
        (tranpy.load_ieee_9bus, (5000, 17), [3599, 1401]),
    ],
)
def test_bundled_snapshots(loader, shape: tuple[int, int], counts: list[int]) -> None:  # type: ignore[no-untyped-def]
    ds = loader()
    assert ds.features.shape == shape
    assert ds.class_counts().tolist() == counts
    assert ds.label_names == LABEL_NAMES == ("stable", "unstable")
    assert ds.feature_names[0] in ("Bus 1 voltage (p.u.)", "Bus 01 voltage (p.u.)")
    assert ds.feature_names[-1].endswith("(deg, bus unknown)")
    assert ds.metadata["legacy_feature_names"][0] == "F_0"
    assert ds.metadata["source_file"].endswith(".pickle")
    assert np.isfinite(ds.features).all()


def test_snapshot_checksum_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = {
        **data_registry(),
        "snapshots": {
            "ieee_9bus": {**data_registry()["snapshots"]["ieee_9bus"], "sha256": "0" * 64}
        },
    }
    monkeypatch.setattr("tranpy.datasets.data_registry", lambda: registry)
    from tranpy_core.errors import ChecksumError

    with pytest.raises(ChecksumError):
        tranpy.load_snapshots("ieee_9bus")
    with pytest.raises(DataError, match="no snapshot dataset"):
        tranpy.load_snapshots("new_england")


def test_list_datasets() -> None:
    table = list_datasets()
    assert set(table["kind"]) == {"snapshots", "events"}
    assert (table[table["kind"] == "snapshots"]["location"] == "bundled").all()
    assert table[table["kind"] == "events"]["location"].str.startswith("https://").all()


def test_systems_and_vocabulary() -> None:
    assert get_system("NewEngland") is NEW_ENGLAND
    assert get_system(IEEE_9BUS) is IEEE_9BUS
    with pytest.raises(ConfigurationError):
        get_system("ieee_14bus")
    assert Stability.UNSTABLE == 1
    names = ("Bus 01", "G 01", "G1", "Line 4-5", "Load A", "T2", "Results")
    assert [element_kind(n) for n in names] == [
        ElementKind.BUS, ElementKind.GENERATOR, ElementKind.GENERATOR, ElementKind.LINE,
        ElementKind.LOAD, ElementKind.TRANSFORMER, ElementKind.OTHER,
    ]  # fmt: skip
    ch = ChannelName.parse("G 01|c:firot in deg")
    assert ch.kind is ElementKind.GENERATOR and str(ch) == "G 01|c:firot in deg"
    with pytest.raises(DataError):
        ChannelName.parse("no separator")


def test_read_event_csvs_round_trip(mock_events_full: SignalDataset, tmp_path: Path) -> None:
    subset = mock_events_full.subset(np.arange(6))
    csv_dir = write_powerfactory_csvs(subset, tmp_path / "csv")
    ds = read_event_csvs(csv_dir, "ieee_9bus")
    assert ds.channel_names == subset.channel_names
    np.testing.assert_allclose(ds.values, subset.values, rtol=1e-12)
    assert np.array_equal(ds.labels, subset.labels)
    assert ds.sample_ids is not None and ds.sample_ids.tolist() == sorted(
        subset.sample_ids.tolist()
    )  # type: ignore[union-attr]
    picked = read_event_csvs(csv_dir, "ieee_9bus", event_ids=[subset.sample_ids[2]])  # type: ignore[index]
    assert picked.n_samples == 1
    with pytest.raises(DataError, match="events not found"):
        read_event_csvs(csv_dir, "ieee_9bus", event_ids=[999])
    with pytest.raises(DataError, match="no results_event"):
        read_event_csvs(tmp_path, "ieee_9bus")
    # load_events accepts the CSV folder and drops the label channels by default
    events = load_events("ieee_9bus", source=csv_dir)
    assert not events.channel_index("*|s:outofstep").size
    assert (
        load_events("ieee_9bus", source=csv_dir, include_label_channels=True)
        .channel_index("*|s:outofstep")
        .size
    )
    assert load_events("ieee_9bus", source=csv_dir, channels=["Bus*"]).n_channels == 27


def test_read_event_csvs_rejects_inconsistent_files(
    mock_events_full: SignalDataset, tmp_path: Path
) -> None:
    subset = mock_events_full.subset([0, 1])
    csv_dir = write_powerfactory_csvs(subset, tmp_path / "csv")
    other = mock_events_full.subset([2]).select_channels("Bus*")
    write_powerfactory_csvs(other, csv_dir)
    with pytest.raises(DataError, match="different columns"):
        read_event_csvs(csv_dir, "ieee_9bus")
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame(
        [[1.0, 2.0]], columns=pd.MultiIndex.from_arrays([["Bus 1", "Bus 1"], ["m:u in p.u.", "x"]])
    ).to_csv(bad / "results_event_0.csv", index=False)
    with pytest.raises(DataError, match="first column must be the simulation time"):
        read_event_csvs(bad, "ieee_9bus")


def test_load_events_from_npz_and_registry(
    mock_events_full: SignalDataset, tmp_path: Path, isolated_cache: Path
) -> None:
    path = save_signals(mock_events_full, tmp_path / "e.npz")
    assert load_events("ieee_9bus", source=path).n_samples == mock_events_full.n_samples
    with pytest.raises(DataError, match="not 'new_england'"):
        load_events("new_england", source=path)
    with pytest.raises(DataError, match="open_event_shards"):
        load_events("ieee_9bus")


def test_labels_from_out_of_step_flags_and_angle_fallback(mock_events_full: SignalDataset) -> None:
    assert np.array_equal(stability_labels(mock_events_full), mock_events_full.labels)
    assert 0 < mock_events_full.labels.sum() < mock_events_full.n_samples
    no_flags = mock_events_full.drop_channels("*|s:outofstep")
    fallback = stability_labels(no_flags, angle_spread_limit_deg=360.0)
    # the COI-relative angle spread exceeds 360 deg only for events that lost synchronism
    assert np.all(fallback <= mock_events_full.labels)
    with pytest.raises(DataError, match="cannot label"):
        stability_labels(mock_events_full.select_channels("Bus*"))


def test_detect_fault_timing_on_simulated_events(mock_events: SignalDataset) -> None:
    start, clear = detect_fault_timing(mock_events)
    true_start = np.asarray(mock_events.annotations[FAULT_START])
    assert np.all(np.abs(start - true_start) <= 0.011)
    assert np.all(clear > start)
    no_bus = mock_events.select_channels("G*")
    s2, c2 = detect_fault_timing(no_bus)
    assert np.isnan(s2).all() and np.isnan(c2).all()
    assert FAULT_CLEAR in mock_events.annotations


def test_detect_fault_timing_branches() -> None:
    t = np.round(np.arange(0, 1.0, 0.1), 3)

    def ds(v: np.ndarray) -> SignalDataset:
        return SignalDataset.from_samples(
            [t],
            [v[:, None]],
            channel_names=("Bus 1|m:u in p.u.",),
            labels=[0],
            label_names=LABEL_NAMES,
            name="x",
        )

    dip_then_recover = np.array([1, 1, 0.5, 0.5, 1, 0.5, 1, 1, 1, 1.0])
    s, c = detect_fault_timing(ds(dip_then_recover))
    assert (s[0], c[0]) == (0.2, 0.4)
    single_block = np.array([1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1.0])
    assert detect_fault_timing(ds(single_block))[1][0] == 0.5
    collapse = np.array([1, 1, 0.5, 0.3, 0.2, 0.25, 0.5, 0.6, 0.7, 0.75])
    assert detect_fault_timing(ds(collapse))[1][0] == pytest.approx(0.4)
    flat_collapse = np.array([1, 1, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    assert detect_fault_timing(ds(flat_collapse))[1][0] == pytest.approx(0.3)
    no_dip = np.ones(10)
    assert np.isnan(detect_fault_timing(ds(no_dip))[0][0])
    end_gap = np.array([1, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 0.5])
    assert detect_fault_timing(ds(end_gap))[1][0] == pytest.approx(0.2)

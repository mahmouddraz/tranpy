from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from tests.conftest import write_powerfactory_csvs

from tranpy.cli import main
from tranpy.datasets import (
    data_registry,
    event_csv_ids,
    open_event_shards,
    write_event_shards,
)
from tranpy.early_prediction import (
    evaluate_window_tables,
    extract_window_tables,
    run_early_prediction,
)
from tranpy_core.data import ShardManifest, SignalDataset, sha256_file
from tranpy_core.errors import ConfigurationError, DataError


@pytest.fixture
def shard_dir(mock_events_full: SignalDataset, tmp_path: Path) -> Path:
    csv = write_powerfactory_csvs(mock_events_full, tmp_path / "csv")
    write_event_shards(csv, "ieee_9bus", tmp_path / "shards", shard_size=15)
    return tmp_path / "shards"


def test_write_and_open_event_shards(mock_events_full: SignalDataset, shard_dir: Path) -> None:
    manifest = open_event_shards("ieee_9bus", source=shard_dir)
    assert manifest.n_samples == mock_events_full.n_samples
    assert [s.n_samples for s in manifest.shards] == [15, 15, 10]
    assert manifest.metadata["system"] == "ieee_9bus"
    assert open_event_shards("NineBusSystem", source=shard_dir / "manifest.json").n_samples == 40
    with pytest.raises(DataError, match="not 'new_england'"):
        open_event_shards("new_england", source=shard_dir)
    assert len(event_csv_ids(shard_dir.parent / "csv")) == 40
    # resumable: nothing left to convert
    again = write_event_shards(shard_dir.parent / "csv", "ieee_9bus", shard_dir, shard_size=15)
    assert len(again.shards) == 3


def test_open_registered_shards(
    shard_dir: Path, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = shard_dir / "manifest.json"
    remote = ShardManifest.load(manifest_path).to_dict() | {"base_url": shard_dir.as_uri()}
    published = shard_dir.parent / "published.json"
    published.write_text(json.dumps(remote))
    reg = data_registry()
    entry = {
        **reg["events"]["ieee_9bus"],
        "manifest_url": published.as_uri(),
        "manifest_sha256": sha256_file(published),
    }
    monkeypatch.setattr(
        "tranpy.datasets.data_registry",
        lambda: {**reg, "events": {**reg["events"], "ieee_9bus": entry}},
    )
    manifest = open_event_shards("ieee_9bus")
    assert "local_dir" not in manifest.metadata
    assert manifest.shard_url(manifest.shards[0]).startswith("file://")
    unregistered = {**reg, "events": {**reg["events"], "ieee_9bus": {"n_samples": 0}}}
    monkeypatch.setattr("tranpy.datasets.data_registry", lambda: unregistered)
    with pytest.raises(DataError, match="no sharded event dataset"):
        open_event_shards("ieee_9bus")


def test_streamed_extraction_equals_in_memory(
    mock_events_full: SignalDataset, shard_dir: Path, tmp_path: Path
) -> None:
    manifest = open_event_shards("ieee_9bus", source=shard_dir)
    kwargs = {
        "reference": "fault_start",
        "windows_s": (0.1, 0.3),
        "sequences": True,
        "sequence_settings": {"step_s": 0.05},
    }
    streamed = extract_window_tables(manifest, **kwargs)
    # same data in memory: read from the CSVs, so fault timing is detected in both cases
    from tranpy.datasets import read_event_csvs

    events = read_event_csvs(shard_dir.parent / "csv", "ieee_9bus")
    in_memory = extract_window_tables(events, **kwargs)
    assert (
        set(streamed)
        == set(in_memory)
        == {(0.1, "features"), (0.3, "features"), (0.1, "sequence"), (0.3, "sequence")}
    )
    for key in streamed:
        np.testing.assert_allclose(streamed[key].features, in_memory[key].features, rtol=1e-12)
        assert not any("outofstep" in n for n in streamed[key].feature_names)
    cache = tmp_path / "tables"
    first = extract_window_tables(manifest, cache_directory=cache, **kwargs)
    (shard_dir / manifest.shards[0].filename).unlink()  # cached tables need no raw data
    second = extract_window_tables(manifest, cache_directory=cache, **kwargs)
    for key in first:
        np.testing.assert_array_equal(first[key].features, second[key].features)
    memory_cached = extract_window_tables(events, cache_directory=cache, **kwargs)
    assert set(memory_cached) == set(first)


def test_evaluate_window_tables(mock_events: SignalDataset) -> None:
    tables = extract_window_tables(
        mock_events, reference="fault_start", windows_s=(0.2,), sequences=True
    )
    result = evaluate_window_tables(tables, {"lr": LogisticRegression(max_iter=500)}, n_splits=2)
    assert result["input"].tolist() == ["features"]
    with pytest.raises(ConfigurationError):
        evaluate_window_tables(tables)
    broken = dict(tables)
    broken[(0.2, "sequence")] = tables[(0.2, "sequence")].subset(np.arange(5))
    with pytest.raises(ConfigurationError, match="different events"):
        evaluate_window_tables(
            broken, {"lr": LogisticRegression()}, sequence_models={"s": LogisticRegression()}
        )
    with pytest.raises(ConfigurationError):
        run_early_prediction(mock_events, reference="fault_start")


def test_cli_build_shards(
    mock_events_full: SignalDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = write_powerfactory_csvs(mock_events_full.subset(np.arange(6)), tmp_path / "csv")
    assert (
        main(["build-shards", str(csv), "ieee_9bus", str(tmp_path / "out"), "--shard-size", "4"])
        == 0
    )
    assert "2 shards, 6 events" in capsys.readouterr().out

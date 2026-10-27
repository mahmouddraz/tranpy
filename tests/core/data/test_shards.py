from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tranpy_core.data import (
    ShardManifest,
    ShardWriter,
    SignalDataset,
    TabularDataset,
    concat_tabular,
    iter_shards,
    load_signals,
    map_shards,
    save_signals,
)
from tranpy_core.errors import ChecksumError, ConfigurationError, DataError
from tranpy_core.features import WindowStatistics


def _decimal_signals(n: int = 7, seed: int = 0) -> SignalDataset:
    rng = np.random.default_rng(seed)
    times = [np.round(np.arange(0, 0.01 * (20 + i), 0.01), 6) for i in range(n)]
    values = [np.round(rng.normal(size=(t.size, 3)) * 100, 6) for t in times]
    return SignalDataset.from_samples(
        times,
        values,
        channel_names=("a|x", "b|y", "c|z"),
        labels=np.arange(n) % 2,
        label_names=("neg", "pos"),
        name="dec",
        sample_ids=np.arange(100, 100 + n),
        annotations={"t0": np.arange(n, dtype=float)},
    )


def test_compact_encoding_is_bit_exact_and_smaller(tmp_path: Path) -> None:
    data = _decimal_signals()
    compact = save_signals(data, tmp_path / "c.npz")
    plain = save_signals(data, tmp_path / "p.npz", compact=False)
    assert compact.stat().st_size < plain.stat().st_size
    for path in (compact, plain):
        back = load_signals(path)
        assert np.array_equal(back.values, data.values)
        assert back.values.dtype == np.float64


def test_non_decimal_values_fall_back_to_float64(
    tmp_path: Path, mock_events: SignalDataset
) -> None:
    path = save_signals(mock_events, tmp_path / "m.npz")
    import zipfile

    with zipfile.ZipFile(path) as zf:
        header = json.loads(str(np.load(zf.open("__header__.npy")).reshape(-1)[0]))
    assert header["values_encoding"]["kind"] == "float64"
    assert np.array_equal(load_signals(path).values, mock_events.values)
    nan_data = mock_events.replace(
        values=np.where(mock_events.values > 1e9, np.nan, mock_events.values)
    )
    assert np.array_equal(
        load_signals(save_signals(nan_data, tmp_path / "n.npz")).values,
        nan_data.values,
        equal_nan=True,
    )


def test_unknown_encoding_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tranpy_core.data import storage

    original = storage._write

    def write_bad(path, header, arrays):  # type: ignore[no-untyped-def]
        header["values_encoding"] = {"kind": "zstd"}
        return original(path, header, arrays)

    monkeypatch.setattr(storage, "_write", write_bad)
    path = save_signals(_decimal_signals(), tmp_path / "bad.npz")
    with pytest.raises(DataError, match="unknown values encoding"):
        load_signals(path)


def test_writer_shards_resume_and_manifest(tmp_path: Path) -> None:
    data = _decimal_signals(7)
    out = tmp_path / "shards"
    with ShardWriter(out, name="dec", shard_size=3, metadata={"system": "x"}) as writer:
        writer.add(data.subset([0, 1]))
        writer.add(data.subset([2, 3, 4]))
    manifest = ShardManifest.load(out / "manifest.json")
    assert [s.n_samples for s in manifest.shards] == [3, 2]
    assert manifest.shards[0].sample_ids == (100, 102)
    assert manifest.n_samples == 5
    resumed = ShardWriter(out, name="dec", shard_size=3)
    assert resumed.done_sample_ids() == {100, 101, 102, 103, 104}
    resumed.add(data.subset([5, 6]))
    manifest = resumed.close()
    assert manifest.n_samples == 7
    assert dict(manifest.label_counts()) == {"neg": 4, "pos": 3}
    assert "local_dir" not in json.loads((out / "manifest.json").read_text())["metadata"]
    streamed = list(iter_shards(manifest))
    ids = np.concatenate([s.sample_ids for s in streamed])  # type: ignore[misc]
    assert ids.tolist() == list(range(100, 107))
    assert np.array_equal(np.concatenate([s.values for s in streamed]), data.values)


def test_writer_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        ShardWriter(tmp_path, name="x", shard_size=0)
    writer = ShardWriter(tmp_path / "w", name="x", shard_size=2)
    with pytest.raises(DataError, match="no samples"):
        writer.close()
    writer.add(_decimal_signals(1))
    with pytest.raises(DataError, match="same channels"):
        writer.add(_decimal_signals(1).select_channels("a|*"))


def test_streaming_from_urls_verifies_and_cleans_up(tmp_path: Path) -> None:
    data = _decimal_signals(4)
    src = tmp_path / "remote"
    with ShardWriter(src, name="dec", shard_size=2) as writer:
        writer.add(data)
    remote = ShardManifest.from_dict(
        {**ShardManifest.load(src / "manifest.json").to_dict(), "base_url": src.as_uri()}
    )
    assert "local_dir" not in remote.metadata
    cache = tmp_path / "cache"
    shards = list(iter_shards(remote, cache_directory=cache))
    assert sum(s.n_samples for s in shards) == 4
    assert list(cache.iterdir()) == []  # downloads are deleted after use
    kept = list(iter_shards(remote, cache_directory=cache, keep_downloads=True, shards=[1]))
    assert len(kept) == 1
    assert len(list(cache.iterdir())) == 1
    (src / remote.shards[0].filename).write_bytes(b"tampered")
    local = ShardManifest.load(src / "manifest.json")
    with pytest.raises(ChecksumError):
        next(iter_shards(local))
    no_url = ShardManifest.from_dict({**local.to_dict(), "base_url": None})
    with pytest.raises(DataError, match="neither local nor has a URL"):
        next(iter_shards(no_url))


def test_manifest_validation_and_loading_errors(tmp_path: Path) -> None:
    data = ShardManifest.load
    with pytest.raises(DataError, match="cannot read"):
        data(tmp_path / "missing.json")
    (tmp_path / "old.json").write_text(json.dumps({"manifest_version": 0}))
    with pytest.raises(DataError, match="unsupported manifest"):
        data(tmp_path / "old.json")
    from tranpy_core.data import ShardInfo

    dup = ShardInfo(filename="a.npz", sha256="0" * 64, n_samples=1, sample_ids=(0, 0))
    with pytest.raises(ConfigurationError, match="twice"):
        ShardManifest(name="x", channel_names=(), label_names=(), shards=(dup, dup))
    m = ShardManifest(
        name="x", channel_names=(), label_names=(), shards=(dup,), base_url="https://h/"
    )
    assert m.shard_url(dup) == "https://h/a.npz"


def test_map_shards_and_concat(tmp_path: Path) -> None:
    data = _decimal_signals(5)
    with ShardWriter(tmp_path, name="dec", shard_size=2) as writer:
        writer.add(data)
    manifest = ShardManifest.load(tmp_path / "manifest.json")
    stats = WindowStatistics(statistics=("mean",))
    table = map_shards(manifest, stats.transform)
    np.testing.assert_allclose(table.features, stats.transform(data).features)
    both = map_shards(manifest, lambda s: {"a": stats.transform(s), "b": stats.transform(s)})
    assert set(both) == {"a", "b"}
    assert both["a"].n_samples == 5
    with pytest.raises(DataError, match="nothing"):
        concat_tabular([])
    other = TabularDataset(
        features=[[1.0]], labels=[0], feature_names=("z",), label_names=("neg", "pos"), name="o"
    )
    with pytest.raises(DataError, match="different feature names"):
        concat_tabular([table, other])

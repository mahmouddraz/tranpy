from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tranpy_core.data import (
    RemoteFile,
    SignalDataset,
    TabularDataset,
    cache_dir,
    fetch,
    load_dataset_file,
    load_signals,
    load_tabular,
    save_signals,
    save_tabular,
    sha256_file,
    verify_sha256,
)
from tranpy_core.errors import ChecksumError, ConfigurationError, DataError


def test_tabular_round_trip_is_exact_and_deterministic(toy: TabularDataset, tmp_path: Path) -> None:
    a = save_tabular(toy, tmp_path / "a.npz")
    b = save_tabular(toy, tmp_path / "b.npz")
    assert sha256_file(a) == sha256_file(b)
    back = load_tabular(a)
    assert np.array_equal(back.features, toy.features)
    assert np.array_equal(back.labels, toy.labels)
    assert back.feature_names == toy.feature_names
    assert back.name == toy.name
    assert isinstance(load_dataset_file(a), TabularDataset)


def test_signal_round_trip(mock_events: SignalDataset, tmp_path: Path) -> None:
    path = save_signals(mock_events, tmp_path / "s.npz")
    back = load_signals(path)
    assert np.array_equal(back.values, mock_events.values)
    assert np.array_equal(back.offsets, mock_events.offsets)
    assert back.channel_names == mock_events.channel_names
    assert set(back.annotations) == set(mock_events.annotations)
    assert dict(back.metadata) == dict(mock_events.metadata)
    assert isinstance(load_dataset_file(path), SignalDataset)


def test_reading_rejects_foreign_and_wrong_files(toy: TabularDataset, tmp_path: Path) -> None:
    path = save_tabular(toy, tmp_path / "t.npz")
    with pytest.raises(DataError, match="holds a 'tabular' dataset"):
        load_signals(path)
    plain = tmp_path / "plain.npz"
    np.savez(plain, x=np.ones(2))
    with pytest.raises(DataError, match="no header"):
        load_tabular(plain)
    with pytest.raises(DataError, match="cannot read"):
        load_tabular(tmp_path / "missing.npz")
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, __header__=np.array(["{}"]), obj=np.array([{"a": 1}], dtype=object))
    with pytest.raises(DataError, match="cannot read"):
        load_tabular(pickled)
    with pytest.raises(DataError, match="no header"):
        load_dataset_file(plain)


def test_format_version_is_checked(toy: TabularDataset, tmp_path: Path) -> None:
    path = save_tabular(toy, tmp_path / "t.npz")
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    import io
    import json

    header = json.loads(str(np.load(io.BytesIO(members["__header__.npy"])).reshape(-1)[0]))
    header["format_version"] = 99
    buf = io.BytesIO()
    np.save(buf, np.array([json.dumps(header)]))
    members["__header__.npy"] = buf.getvalue()
    with zipfile.ZipFile(path, "w") as zf:
        for n, data in members.items():
            zf.writestr(n, data)
    with pytest.raises(DataError, match="format version 99"):
        load_tabular(path)


def test_failed_write_leaves_no_partial_file(
    toy: TabularDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(np.lib.format, "write_array", boom)
    with pytest.raises(OSError, match="disk full"):
        save_tabular(toy, tmp_path / "t.npz")
    assert list(tmp_path.iterdir()) == []


def test_verify_sha256(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    verify_sha256(f, digest.upper())
    with pytest.raises(ChecksumError):
        verify_sha256(f, "0" * 64)


def _remote(path: Path, digest: str | None = None) -> RemoteFile:
    return RemoteFile(filename="data.bin", url=path.as_uri(), sha256=digest or sha256_file(path))


def test_fetch_downloads_verifies_and_caches(tmp_path: Path, isolated_cache: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    remote = _remote(src)
    got = fetch(remote)
    assert got.parent == isolated_cache == cache_dir()
    assert got.read_bytes() == b"payload"
    src.unlink()
    assert fetch(remote) == got  # cached, no download needed
    got.write_bytes(b"tampered")
    with pytest.raises(DataError, match="failed"):
        fetch(remote)  # re-download attempted because the cache fails its checksum


def test_fetch_rejects_wrong_checksum(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    with pytest.raises(ChecksumError, match="discarded"):
        fetch(_remote(src, "a" * 64), directory=tmp_path / "c")
    assert list((tmp_path / "c").iterdir()) == []


def test_remote_file_validation() -> None:
    with pytest.raises(ConfigurationError) as info:
        RemoteFile(filename="../x", url="http://insecure", sha256="xyz")
    assert len(info.value.problems) == 3


def test_cache_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRANPY_CACHE_DIR", raising=False)
    assert "tranpy" in str(cache_dir())
    assert not str(cache_dir()).startswith(str(Path(__file__).parents[3]))

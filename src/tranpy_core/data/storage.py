"""Safe on-disk format for datasets: compressed ``.npz`` arrays plus JSON metadata.

Files are read with ``allow_pickle=False``, so loading a dataset can never execute code.
The JSON header records a format name and version; incompatible files are rejected with
a clear message instead of being half-read.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from tranpy_core.data.dataset import SignalDataset, TabularDataset
from tranpy_core.errors import ChecksumError, DataError

__all__ = [
    "FORMAT_VERSION",
    "load_dataset_file",
    "load_signals",
    "load_tabular",
    "save_signals",
    "save_tabular",
    "sha256_file",
    "verify_sha256",
]

FORMAT_VERSION = 1
_HEADER = "__header__"
_ANNOTATION_PREFIX = "annotation/"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    """Raise :class:`ChecksumError` unless ``path`` has digest ``expected``."""
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ChecksumError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def _write(path: Path, header: dict[str, Any], arrays: dict[str, np.ndarray]) -> Path:
    """Write an ``.npz`` byte-for-byte reproducibly (fixed member order and timestamps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    members = {_HEADER: np.array(json.dumps(header, sort_keys=True)), **arrays}
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, array in members.items():
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with zf.open(info, "w", force_zip64=True) as member:
                    np.lib.format.write_array(
                        member, np.ascontiguousarray(array), allow_pickle=False
                    )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)
    return path


def _read(path: Path, kind: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read dataset file {path}: {exc}") from exc
    if _HEADER not in arrays:
        raise DataError(f"{path} is not a tranpy dataset file (no header)")
    header = json.loads(str(arrays.pop(_HEADER).reshape(-1)[0]))
    if header.get("kind") != kind:
        raise DataError(f"{path} holds a {header.get('kind')!r} dataset, expected {kind!r}")
    if header.get("format_version") != FORMAT_VERSION:
        raise DataError(
            f"{path} uses format version {header.get('format_version')}, "
            f"this version of tranpy reads version {FORMAT_VERSION}"
        )
    return header, arrays


def save_tabular(dataset: TabularDataset, path: str | Path) -> Path:
    """Write ``dataset`` to ``path`` (``.npz``)."""
    header = {
        "kind": "tabular",
        "format_version": FORMAT_VERSION,
        "name": dataset.name,
        "description": dataset.description,
        "feature_names": list(dataset.feature_names),
        "label_names": list(dataset.label_names),
        "metadata": dict(dataset.metadata),
    }
    assert dataset.sample_ids is not None
    arrays = {
        "features": dataset.features,
        "labels": dataset.labels,
        "sample_ids": dataset.sample_ids,
    }
    return _write(Path(path), header, arrays)


def load_tabular(path: str | Path) -> TabularDataset:
    """Read a dataset written by :func:`save_tabular`."""
    header, arrays = _read(Path(path), "tabular")
    return TabularDataset(
        features=arrays["features"],
        labels=arrays["labels"],
        sample_ids=arrays["sample_ids"],
        feature_names=tuple(header["feature_names"]),
        label_names=tuple(header["label_names"]),
        name=header["name"],
        description=header["description"],
        metadata=header["metadata"],
    )


def save_signals(dataset: SignalDataset, path: str | Path) -> Path:
    """Write ``dataset`` to ``path`` (``.npz``)."""
    header = {
        "kind": "signals",
        "format_version": FORMAT_VERSION,
        "name": dataset.name,
        "description": dataset.description,
        "channel_names": list(dataset.channel_names),
        "label_names": list(dataset.label_names),
        "time_unit": dataset.time_unit,
        "metadata": dict(dataset.metadata),
    }
    assert dataset.sample_ids is not None
    arrays = {
        "time": dataset.time,
        "values": dataset.values,
        "offsets": dataset.offsets,
        "labels": dataset.labels,
        "sample_ids": dataset.sample_ids,
        **{_ANNOTATION_PREFIX + k: v for k, v in dataset.annotations.items()},
    }
    return _write(Path(path), header, arrays)


def load_signals(path: str | Path) -> SignalDataset:
    """Read a dataset written by :func:`save_signals`."""
    header, arrays = _read(Path(path), "signals")
    annotations = {
        k[len(_ANNOTATION_PREFIX) :]: v
        for k, v in arrays.items()
        if k.startswith(_ANNOTATION_PREFIX)
    }
    return SignalDataset(
        time=arrays["time"],
        values=arrays["values"],
        offsets=arrays["offsets"],
        labels=arrays["labels"],
        sample_ids=arrays["sample_ids"],
        annotations=annotations,
        channel_names=tuple(header["channel_names"]),
        label_names=tuple(header["label_names"]),
        name=header["name"],
        description=header["description"],
        time_unit=header["time_unit"],
        metadata=header["metadata"],
    )


def load_dataset_file(path: str | Path) -> TabularDataset | SignalDataset:
    """Read either kind of dataset file."""
    try:
        return load_tabular(path)
    except DataError as exc:
        if "holds a 'signals' dataset" not in str(exc):
            raise
    return load_signals(path)

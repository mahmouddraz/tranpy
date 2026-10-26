"""Sharded signal datasets: many events, streamed one shard at a time.

Large datasets are split into *shards* (``.npz`` files of a few hundred samples each) and
described by a *manifest* (JSON: channels, labels, and per shard the file name, SHA-256,
sample count and URL). Analyses iterate over shards: each shard is fetched (or read from
a local folder), verified, processed and released, so neither memory nor local disk ever
holds the whole dataset. Only the small derived results need to be kept.

Typical use::

    manifest = ShardManifest.load("new_england_events/manifest.json")
    table = map_shards(manifest, TrajectoryFeatures(...).transform)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from tranpy_core.data.dataset import SignalDataset, TabularDataset
from tranpy_core.data.download import RemoteFile, cache_dir, fetch
from tranpy_core.data.storage import load_signals, save_signals, sha256_file, verify_sha256
from tranpy_core.errors import ConfigurationError, DataError

__all__ = [
    "ShardInfo",
    "ShardManifest",
    "ShardWriter",
    "concat_tabular",
    "iter_shards",
    "map_shards",
]

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ShardInfo:
    """One shard file."""

    filename: str
    sha256: str
    n_samples: int
    sample_ids: tuple[int, int]
    """First and last sample id in the shard."""
    url: str | None = None


@dataclass(frozen=True)
class ShardManifest:
    """Description of a sharded signal dataset.

    Attributes:
        name: dataset name.
        channel_names: channels of every shard (identical across shards).
        label_names: class names.
        shards: shard files in order.
        base_url: prefix for shards without their own URL (``base_url + filename``).
        metadata: free JSON-serialisable information (system, source, provenance).
    """

    name: str
    channel_names: tuple[str, ...]
    label_names: tuple[str, ...]
    shards: tuple[ShardInfo, ...]
    base_url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [s.filename for s in self.shards]
        if len(set(names)) != len(names):
            raise ConfigurationError(f"manifest {self.name!r} lists a shard twice")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_samples(self) -> int:
        return sum(s.n_samples for s in self.shards)

    def label_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self.metadata.get("label_counts", {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "name": self.name,
            "channel_names": list(self.channel_names),
            "label_names": list(self.label_names),
            "base_url": self.base_url,
            "metadata": {k: v for k, v in self.metadata.items() if k != "local_dir"},
            "shards": [
                {
                    "filename": s.filename,
                    "sha256": s.sha256,
                    "n_samples": s.n_samples,
                    "sample_ids": list(s.sample_ids),
                    "url": s.url,
                }
                for s in self.shards
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShardManifest:
        if data.get("manifest_version") != MANIFEST_VERSION:
            raise DataError(f"unsupported manifest version {data.get('manifest_version')!r}")
        return cls(
            name=data["name"],
            channel_names=tuple(data["channel_names"]),
            label_names=tuple(data["label_names"]),
            base_url=data.get("base_url"),
            metadata=data.get("metadata", {}),
            shards=tuple(
                ShardInfo(
                    filename=s["filename"],
                    sha256=s["sha256"],
                    n_samples=int(s["n_samples"]),
                    sample_ids=(int(s["sample_ids"][0]), int(s["sample_ids"][1])),
                    url=s.get("url"),
                )
                for s in data["shards"]
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> ShardManifest:
        """Read a manifest file; shards are looked up next to it unless they have URLs."""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataError(f"cannot read manifest {path}: {exc}") from exc
        return cls.from_dict(data).with_local_dir(path.parent)

    def with_local_dir(self, directory: str | Path) -> ShardManifest:
        """Copy that reads shards from ``directory`` when they exist there."""
        return ShardManifest(
            name=self.name,
            channel_names=self.channel_names,
            label_names=self.label_names,
            shards=self.shards,
            base_url=self.base_url,
            metadata={**self.metadata, "local_dir": str(directory)},
        )

    def shard_url(self, shard: ShardInfo) -> str | None:
        if shard.url:
            return shard.url
        if self.base_url:
            return self.base_url.rstrip("/") + "/" + shard.filename
        return None


class ShardWriter:
    """Collect samples and write them as shards of ``shard_size`` plus a manifest.

    Resumable: shards already listed in an existing ``manifest.json`` in ``directory`` are
    kept, and :meth:`done_sample_ids` tells the caller what to skip.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        name: str,
        shard_size: int = 250,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if shard_size < 1:
            raise ConfigurationError("shard_size must be >= 1")
        self.directory = Path(directory)
        self.name = name
        self.shard_size = shard_size
        self.metadata = dict(metadata or {})
        self._buffer: list[SignalDataset] = []
        self._shards: list[ShardInfo] = []
        self._channels: tuple[str, ...] | None = None
        self._labels: tuple[str, ...] | None = None
        self._counts: dict[str, int] = {}
        existing = self.directory / "manifest.json"
        if existing.exists():
            old = ShardManifest.load(existing)
            self._shards = list(old.shards)
            self._channels, self._labels = old.channel_names, old.label_names
            self._counts = dict(old.metadata.get("label_counts", {}))
            self.metadata = {
                **{k: v for k, v in old.metadata.items() if k != "local_dir"},
                **self.metadata,
            }

    def done_sample_ids(self) -> set[int]:
        """Sample ids already written (inclusive id ranges of finished shards)."""
        ids: set[int] = set(self.metadata.get("sample_ids_written", []))
        return ids

    def add(self, dataset: SignalDataset) -> None:
        if self._channels is None:
            self._channels, self._labels = dataset.channel_names, dataset.label_names
        elif dataset.channel_names != self._channels or dataset.label_names != self._labels:
            raise DataError("all samples of a sharded dataset need the same channels and labels")
        self._buffer.append(dataset)
        while sum(d.n_samples for d in self._buffer) >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, n: int) -> None:
        merged = _concat_signals(self._buffer)
        take = min(n, merged.n_samples)
        shard, rest = (
            merged.subset(np.arange(take)),
            merged.subset(np.arange(take, merged.n_samples)),
        )
        self._buffer = [rest] if rest.n_samples else []
        index = len(self._shards)
        path = save_signals(shard, self.directory / f"{self.name}-{index:05d}.npz")
        assert shard.sample_ids is not None
        self._shards.append(
            ShardInfo(
                filename=path.name,
                sha256=sha256_file(path),
                n_samples=shard.n_samples,
                sample_ids=(int(shard.sample_ids[0]), int(shard.sample_ids[-1])),
            )
        )
        written = set(self.metadata.get("sample_ids_written", []))
        written.update(int(i) for i in shard.sample_ids)
        self.metadata["sample_ids_written"] = sorted(written)
        for label, count in zip(shard.label_names, shard.class_counts(), strict=True):
            self._counts[label] = self._counts.get(label, 0) + int(count)
        self._write_manifest()
        logger.info("wrote %s (%d samples)", path.name, shard.n_samples)

    def _write_manifest(self) -> Path:
        assert self._channels is not None
        assert self._labels is not None
        manifest = ShardManifest(
            name=self.name,
            channel_names=self._channels,
            label_names=self._labels,
            shards=tuple(self._shards),
            metadata={**self.metadata, "label_counts": self._counts},
        )
        return manifest.save(self.directory / "manifest.json")

    def close(self) -> ShardManifest:
        """Write the remaining samples as a last (smaller) shard and return the manifest."""
        if self._buffer and sum(d.n_samples for d in self._buffer):
            self._flush(self.shard_size)
        if self._channels is None:
            raise DataError("no samples were written")
        return ShardManifest.load(self._write_manifest())

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        if exc[0] is None:
            self.close()


def _concat_signals(parts: Sequence[SignalDataset]) -> SignalDataset:
    first = parts[0]
    if len(parts) == 1:
        return first
    offsets = [first.offsets]
    base = first.time.shape[0]
    for p in parts[1:]:
        offsets.append(p.offsets[1:] + base)
        base += p.time.shape[0]
    keys = set(first.annotations)
    return first.replace(
        time=np.concatenate([p.time for p in parts]),
        values=np.concatenate([p.values for p in parts]),
        offsets=np.concatenate(offsets),
        labels=np.concatenate([p.labels for p in parts]),
        sample_ids=np.concatenate([p.sample_ids for p in parts if p.sample_ids is not None]),
        annotations={k: np.concatenate([p.annotations[k] for p in parts]) for k in keys},
    )


def iter_shards(
    manifest: ShardManifest,
    *,
    cache_directory: Path | None = None,
    keep_downloads: bool = False,
    shards: Iterable[int] | None = None,
) -> Iterator[SignalDataset]:
    """Yield the shards of ``manifest`` one at a time, verified by SHA-256.

    Shards found in the manifest's local folder are read in place. Others are downloaded
    to ``cache_directory`` (default: user cache) and deleted after use unless
    ``keep_downloads`` is set, so local disk use stays at about one shard.
    """
    local_dir = manifest.metadata.get("local_dir")
    target = (
        cache_directory if cache_directory is not None else cache_dir() / "shards" / manifest.name
    )
    for index in shards if shards is not None else range(len(manifest.shards)):
        shard = manifest.shards[index]
        local = Path(local_dir) / shard.filename if local_dir else None
        downloaded = False
        if local is not None and local.exists():
            verify_sha256(local, shard.sha256)
            path = local
        else:
            url = manifest.shard_url(shard)
            if url is None:
                raise DataError(f"shard {shard.filename} is neither local nor has a URL")
            path = fetch(
                RemoteFile(filename=shard.filename, url=url, sha256=shard.sha256), directory=target
            )
            downloaded = True
        try:
            yield load_signals(path)
        finally:
            if downloaded and not keep_downloads:
                path.unlink(missing_ok=True)


def concat_tabular(parts: Sequence[TabularDataset]) -> TabularDataset:
    """Stack tables with identical columns (e.g. per-shard feature tables)."""
    if not parts:
        raise DataError("nothing to concatenate")
    first = parts[0]
    if any(p.feature_names != first.feature_names for p in parts):
        raise DataError("tables have different feature names")
    return TabularDataset(
        features=np.concatenate([p.features for p in parts]),
        labels=np.concatenate([p.labels for p in parts]),
        feature_names=first.feature_names,
        label_names=first.label_names,
        name=first.name,
        description=first.description,
        sample_ids=np.concatenate([p.sample_ids for p in parts if p.sample_ids is not None]),
        metadata=first.metadata,
    )


def map_shards(
    manifest: ShardManifest,
    transform: Callable[[SignalDataset], TabularDataset | Mapping[Any, TabularDataset]],
    **kwargs: Any,
) -> Any:
    """Apply ``transform`` to every shard and concatenate the resulting tables.

    ``transform`` may return one table or a mapping of tables (e.g. one per window); in the
    latter case a mapping of concatenated tables is returned. ``kwargs`` go to
    :func:`iter_shards`.
    """
    single: list[TabularDataset] = []
    many: dict[Any, list[TabularDataset]] = {}
    for shard in iter_shards(manifest, **kwargs):
        out = transform(shard)
        if isinstance(out, TabularDataset):
            single.append(out)
        else:
            for key, table in out.items():
                many.setdefault(key, []).append(table)
    if many:
        return {key: concat_tabular(tables) for key, tables in many.items()}
    return concat_tabular(single)

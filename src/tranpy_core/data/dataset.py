"""Typed, immutable dataset containers.

* :class:`TabularDataset` - one feature vector and one label per sample.
* :class:`SignalDataset` - one multichannel time series and one label per sample. Samples
  may have different lengths; they are stored back to back in a single array, with
  ``offsets[i]:offsets[i + 1]`` selecting the rows of sample ``i``. This layout lets
  feature extractors work on all samples at once instead of looping in Python.

Arrays are copied on construction and made read-only, so a dataset cannot be changed by
accident after it has been split or fed to a model.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy_core.errors import DataError

__all__ = ["Signal", "SignalDataset", "TabularDataset"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _frozen(array: npt.ArrayLike, dtype: npt.DTypeLike, name: str, ndim: int) -> Any:
    out = np.array(array, dtype=dtype, copy=True)
    if out.ndim != ndim:
        raise DataError(f"{name} must be {ndim}-dimensional, got shape {out.shape}")
    out.setflags(write=False)
    return out


def _frozen_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, eq=False)
class TabularDataset:
    """Feature matrix with integer labels.

    Attributes:
        features: ``(n_samples, n_features)`` float array.
        labels: ``(n_samples,)`` integer class indices into ``label_names``.
        feature_names: one name per feature column.
        label_names: class names, indexed by label value.
        name: short identifier of the dataset.
        description: free-text provenance.
        sample_ids: ``(n_samples,)`` identifiers that survive subsetting.
        metadata: additional JSON-serialisable information.
    """

    features: FloatArray
    labels: IntArray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    name: str
    description: str = ""
    sample_ids: IntArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = _frozen(self.features, np.float64, "features", 2)
        labels = _frozen(self.labels, np.int64, "labels", 1)
        n = features.shape[0]
        ids = np.arange(n) if self.sample_ids is None else self.sample_ids
        sample_ids = _frozen(ids, np.int64, "sample_ids", 1)
        problems = []
        if labels.shape[0] != n:
            problems.append(f"{labels.shape[0]} labels for {n} samples")
        if sample_ids.shape[0] != n:
            problems.append(f"{sample_ids.shape[0]} sample_ids for {n} samples")
        if len(self.feature_names) != features.shape[1]:
            problems.append(
                f"{len(self.feature_names)} feature names for {features.shape[1]} columns"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            problems.append("feature names are not unique")
        if n and (labels.min() < 0 or labels.max() >= len(self.label_names)):
            problems.append(f"labels must lie in [0, {len(self.label_names) - 1}]")
        if problems:
            raise DataError(f"inconsistent dataset '{self.name}': " + "; ".join(problems))
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "label_names", tuple(self.label_names))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def n_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{n}={c}" for n, c in zip(self.label_names, self.class_counts(), strict=True)
        )
        return (
            f"TabularDataset({self.name!r}, samples={self.n_samples}, "
            f"features={self.n_features}, {counts})"
        )

    def class_counts(self) -> IntArray:
        """Number of samples per class, indexed by label."""
        return np.bincount(self.labels, minlength=len(self.label_names)).astype(np.int64)

    def subset(self, indices: npt.ArrayLike) -> TabularDataset:
        """Return the samples at ``indices`` (order preserved)."""
        idx = np.asarray(indices, dtype=np.int64)
        assert self.sample_ids is not None
        return TabularDataset(
            features=self.features[idx],
            labels=self.labels[idx],
            feature_names=self.feature_names,
            label_names=self.label_names,
            name=self.name,
            description=self.description,
            sample_ids=self.sample_ids[idx],
            metadata=self.metadata,
        )

    def select_features(self, names: Sequence[str]) -> TabularDataset:
        """Return a dataset restricted to the named feature columns."""
        lookup = {n: i for i, n in enumerate(self.feature_names)}
        missing = [n for n in names if n not in lookup]
        if missing:
            raise DataError(f"unknown features: {missing}")
        cols = [lookup[n] for n in names]
        return TabularDataset(
            features=self.features[:, cols],
            labels=self.labels,
            feature_names=tuple(names),
            label_names=self.label_names,
            name=self.name,
            description=self.description,
            sample_ids=self.sample_ids,
            metadata=self.metadata,
        )

    def duplicate_groups(self) -> IntArray:
        """Group index per sample; samples with identical feature vectors share a group.

        Pass the result as ``groups`` to a splitter to keep duplicates on one side.
        """
        _, inverse = np.unique(self.features, axis=0, return_inverse=True)
        return inverse.reshape(-1).astype(np.int64)

    def to_frame(self, label_column: str = "label") -> pd.DataFrame:
        """Features and labels as a DataFrame indexed by ``sample_ids``."""
        frame = pd.DataFrame(self.features, columns=list(self.feature_names), index=self.sample_ids)
        frame[label_column] = self.labels
        return frame


@dataclass(frozen=True, eq=False)
class Signal:
    """One sample of a :class:`SignalDataset` (read-only views)."""

    time: FloatArray
    values: FloatArray
    channel_names: tuple[str, ...]
    label: int
    sample_id: int
    annotations: Mapping[str, float]

    def channel(self, name: str) -> FloatArray:
        """Values of one channel."""
        try:
            return self.values[:, self.channel_names.index(name)]
        except ValueError:
            raise DataError(f"unknown channel {name!r}") from None


@dataclass(frozen=True, eq=False)
class SignalDataset:
    """Labelled multichannel time series of possibly different lengths.

    Attributes:
        time: ``(n_rows,)`` time stamps of all samples, concatenated.
        values: ``(n_rows, n_channels)`` signal values of all samples, concatenated.
        offsets: ``(n_samples + 1,)`` row boundaries; sample ``i`` is
            ``offsets[i]:offsets[i + 1]``.
        channel_names: one name per column of ``values``.
        labels: ``(n_samples,)`` integer class indices into ``label_names``.
        label_names: class names, indexed by label value.
        name: short identifier of the dataset.
        sample_ids: ``(n_samples,)`` identifiers that survive subsetting.
        annotations: per-sample scalar arrays, e.g. event times (NaN where unknown).
        time_unit: unit of ``time``.
        description: free-text provenance.
        metadata: additional JSON-serialisable information.
    """

    time: FloatArray
    values: FloatArray
    offsets: IntArray
    channel_names: tuple[str, ...]
    labels: IntArray
    label_names: tuple[str, ...]
    name: str
    sample_ids: IntArray | None = None
    annotations: Mapping[str, FloatArray] = field(default_factory=dict)
    time_unit: str = "s"
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _frozen(self.time, np.float64, "time", 1)
        values = _frozen(self.values, np.float64, "values", 2)
        offsets = _frozen(self.offsets, np.int64, "offsets", 1)
        labels = _frozen(self.labels, np.int64, "labels", 1)
        n = labels.shape[0]
        ids = np.arange(n) if self.sample_ids is None else self.sample_ids
        sample_ids = _frozen(ids, np.int64, "sample_ids", 1)
        annotations = {
            k: _frozen(v, np.float64, f"annotation {k!r}", 1) for k, v in self.annotations.items()
        }
        problems = []
        if values.shape[0] != time.shape[0]:
            problems.append(f"{values.shape[0]} value rows for {time.shape[0]} time stamps")
        if values.shape[1] != len(self.channel_names):
            problems.append(
                f"{len(self.channel_names)} channel names for {values.shape[1]} columns"
            )
        if offsets.shape[0] != n + 1 or offsets[0] != 0 or offsets[-1] != time.shape[0]:
            problems.append("offsets must start at 0, end at n_rows and have n_samples + 1 entries")
        elif np.any(np.diff(offsets) < 2):
            problems.append("every sample needs at least two time steps")
        if sample_ids.shape[0] != n:
            problems.append(f"{sample_ids.shape[0]} sample_ids for {n} samples")
        for k, v in annotations.items():
            if v.shape[0] != n:
                problems.append(f"annotation {k!r} has {v.shape[0]} values for {n} samples")
        if n and (labels.min() < 0 or labels.max() >= len(self.label_names)):
            problems.append(f"labels must lie in [0, {len(self.label_names) - 1}]")
        if problems:
            raise DataError(f"inconsistent signal dataset '{self.name}': " + "; ".join(problems))
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        object.__setattr__(self, "label_names", tuple(self.label_names))
        object.__setattr__(self, "annotations", _frozen_mapping(annotations))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @classmethod
    def from_samples(
        cls,
        times: Sequence[npt.ArrayLike],
        values: Sequence[npt.ArrayLike],
        *,
        channel_names: Sequence[str],
        labels: npt.ArrayLike,
        label_names: Sequence[str],
        name: str,
        **kwargs: Any,
    ) -> SignalDataset:
        """Build a dataset from per-sample arrays."""
        if len(times) != len(values):
            raise DataError(f"{len(times)} time vectors for {len(values)} value arrays")
        lengths = [np.asarray(t).shape[0] for t in times]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        n_channels = len(channel_names)
        stacked = (
            np.concatenate([np.asarray(v, dtype=np.float64) for v in values])
            if values
            else np.empty((0, n_channels))
        )
        return cls(
            time=np.concatenate([np.asarray(t, dtype=np.float64) for t in times])
            if times
            else np.empty(0),
            values=stacked,
            offsets=offsets,
            channel_names=tuple(channel_names),
            labels=labels,  # type: ignore[arg-type]
            label_names=tuple(label_names),
            name=name,
            **kwargs,
        )

    @property
    def n_samples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[1])

    @property
    def lengths(self) -> IntArray:
        """Number of time steps per sample."""
        return np.diff(self.offsets)

    @property
    def sample_index(self) -> IntArray:
        """``(n_rows,)`` sample index of every row."""
        return np.repeat(np.arange(self.n_samples), self.lengths)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{n}={c}" for n, c in zip(self.label_names, self.class_counts(), strict=True)
        )
        return (
            f"SignalDataset({self.name!r}, samples={self.n_samples}, channels={self.n_channels}, "
            f"rows={self.time.shape[0]}, {counts})"
        )

    def __getitem__(self, i: int) -> Signal:
        if not -self.n_samples <= i < self.n_samples:
            raise IndexError(i)
        i %= self.n_samples
        rows = slice(int(self.offsets[i]), int(self.offsets[i + 1]))
        assert self.sample_ids is not None
        return Signal(
            time=self.time[rows],
            values=self.values[rows],
            channel_names=self.channel_names,
            label=int(self.labels[i]),
            sample_id=int(self.sample_ids[i]),
            annotations=MappingProxyType({k: float(v[i]) for k, v in self.annotations.items()}),
        )

    def class_counts(self) -> IntArray:
        """Number of samples per class, indexed by label."""
        return np.bincount(self.labels, minlength=len(self.label_names)).astype(np.int64)

    def channel_index(self, patterns: str | Sequence[str]) -> IntArray:
        """Column indices of channels matching any shell-style pattern, in dataset order."""
        pats = [patterns] if isinstance(patterns, str) else list(patterns)
        return np.array(
            [
                i
                for i, c in enumerate(self.channel_names)
                if any(fnmatch.fnmatchcase(c, p) for p in pats)
            ],
            dtype=np.int64,
        )

    def select_channels(self, patterns: str | Sequence[str]) -> SignalDataset:
        """Return a dataset with only the channels matching ``patterns``."""
        cols = self.channel_index(patterns)
        if cols.size == 0:
            raise DataError(f"no channel matches {patterns!r}")
        return self.replace(
            values=self.values[:, cols], channel_names=tuple(self.channel_names[i] for i in cols)
        )

    def drop_channels(self, patterns: str | Sequence[str]) -> SignalDataset:
        """Return a dataset without the channels matching ``patterns``."""
        drop = set(self.channel_index(patterns).tolist())
        keep = np.array([i for i in range(self.n_channels) if i not in drop], dtype=np.int64)
        return self.replace(
            values=self.values[:, keep], channel_names=tuple(self.channel_names[i] for i in keep)
        )

    def subset(self, indices: npt.ArrayLike) -> SignalDataset:
        """Return the samples at ``indices`` (order preserved)."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        starts, stops = self.offsets[idx], self.offsets[idx + 1]
        lengths = stops - starts
        rows = np.repeat(
            starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths
        ) + np.arange(int(lengths.sum()))
        assert self.sample_ids is not None
        return self.replace(
            time=self.time[rows],
            values=self.values[rows],
            offsets=np.concatenate([[0], np.cumsum(lengths)]),
            labels=self.labels[idx],
            sample_ids=self.sample_ids[idx],
            annotations={k: v[idx] for k, v in self.annotations.items()},
        )

    def with_annotations(self, **annotations: npt.ArrayLike) -> SignalDataset:
        """Return a copy with added or replaced per-sample annotations."""
        merged = dict(self.annotations)
        merged.update({k: np.asarray(v, dtype=np.float64) for k, v in annotations.items()})
        return self.replace(annotations=merged)

    def replace(self, **changes: Any) -> SignalDataset:
        """Return a copy with some fields replaced (validated again)."""
        fields: dict[str, Any] = {
            "time": self.time,
            "values": self.values,
            "offsets": self.offsets,
            "channel_names": self.channel_names,
            "labels": self.labels,
            "label_names": self.label_names,
            "name": self.name,
            "sample_ids": self.sample_ids,
            "annotations": self.annotations,
            "time_unit": self.time_unit,
            "description": self.description,
            "metadata": self.metadata,
        }
        fields.update(changes)
        return SignalDataset(**fields)

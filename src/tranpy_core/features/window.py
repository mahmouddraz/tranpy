"""Per-sample time windows and a generic statistics extractor.

Feature extraction maps a :class:`SignalDataset` to a :class:`TabularDataset`. Extractors
in this package are stateless (nothing is fitted), so they cannot leak information from
test samples into training samples; anything learned from data belongs in a Pipeline
after extraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from tranpy_core.data.dataset import SignalDataset, TabularDataset
from tranpy_core.errors import ConfigurationError, DataError
from tranpy_core.features import segments as seg

__all__ = ["SignalFeatureExtractor", "WindowStatistics", "crop_to_window"]

Statistic = Literal["mean", "std", "min", "max", "first", "last", "range"]


class SignalFeatureExtractor(Protocol):
    """Anything that turns signals into a feature table."""

    def transform(self, dataset: SignalDataset) -> TabularDataset: ...


def crop_to_window(
    dataset: SignalDataset,
    end: npt.ArrayLike,
    start: npt.ArrayLike | None = None,
    *,
    min_steps: int = 2,
) -> SignalDataset:
    """Keep, per sample, the rows with ``start <= time <= end`` (bounds per sample or scalar).

    Raises :class:`DataError` if a sample keeps fewer than ``min_steps`` rows.
    """
    n = dataset.n_samples
    end_arr = np.broadcast_to(np.asarray(end, dtype=np.float64), (n,))
    seg_index = dataset.sample_index
    mask = dataset.time <= end_arr[seg_index]
    if start is not None:
        start_arr = np.broadcast_to(np.asarray(start, dtype=np.float64), (n,))
        mask &= dataset.time >= start_arr[seg_index]
    rows, offsets = seg.compact(mask, dataset.offsets)
    short = np.flatnonzero(np.diff(offsets) < min_steps)
    if short.size:
        assert dataset.sample_ids is not None
        raise DataError(
            f"{short.size} samples keep fewer than {min_steps} time steps in the window "
            f"(first sample ids: {dataset.sample_ids[short[:5]].tolist()})"
        )
    return dataset.replace(time=dataset.time[rows], values=dataset.values[rows], offsets=offsets)


@dataclass(frozen=True)
class WindowStatistics:
    """Summary statistics of selected channels, optionally inside a fixed time window.

    Feature names are ``"<channel>__<statistic>"``.
    """

    channels: Sequence[str] = ("*",)
    statistics: Sequence[Statistic] = ("mean", "std", "min", "max", "last")
    end: float | None = None
    start: float | None = None

    def __post_init__(self) -> None:
        allowed = {"mean", "std", "min", "max", "first", "last", "range"}
        bad = [s for s in self.statistics if s not in allowed]
        if bad:
            raise ConfigurationError(f"unknown statistics {bad}; allowed: {sorted(allowed)}")
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "statistics", tuple(self.statistics))

    def transform(self, dataset: SignalDataset) -> TabularDataset:
        data = dataset.select_channels(self.channels)
        if self.end is not None or self.start is not None:
            end = np.inf if self.end is None else self.end
            data = crop_to_window(data, end, self.start)
        v, off = data.values, data.offsets
        columns: list[npt.NDArray[np.float64]] = []
        names: list[str] = []
        for stat in self.statistics:
            if stat == "mean":
                block = seg.segment_mean(v, off)
            elif stat == "std":
                mean = seg.segment_mean(v, off)
                block = np.sqrt(np.maximum(seg.segment_mean(v * v, off) - mean * mean, 0.0))
            elif stat == "min":
                block = seg.segment_min(v, off)
            elif stat == "max":
                block = seg.segment_max(v, off)
            elif stat == "first":
                block = seg.segment_first(v, off)
            elif stat == "last":
                block = seg.segment_last(v, off)
            else:
                block = seg.segment_max(v, off) - seg.segment_min(v, off)
            columns.append(block)
            names.extend(f"{c}__{stat}" for c in data.channel_names)
        return TabularDataset(
            features=np.concatenate(columns, axis=1),
            labels=data.labels,
            feature_names=tuple(names),
            label_names=data.label_names,
            name=data.name,
            description=f"window statistics of {data.name}",
            sample_ids=data.sample_ids,
        )

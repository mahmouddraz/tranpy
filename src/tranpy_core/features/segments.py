"""Vectorised reductions over ragged segments of a concatenated array.

A *segmented array* is a ``(n_rows, ...)`` array plus ``offsets`` of length
``n_segments + 1``; segment ``i`` is ``rows[offsets[i]:offsets[i + 1]]``. These helpers
compute one value per segment for all segments at once, which replaces per-sample
Python loops in feature extraction.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "compact",
    "first_true_index",
    "segment_any",
    "segment_diff",
    "segment_first",
    "segment_last",
    "segment_last_where",
    "segment_max",
    "segment_mean",
    "segment_min",
    "segment_sum",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def _check(values: npt.NDArray[np.generic], offsets: IntArray) -> None:
    if offsets[-1] != values.shape[0]:
        raise ValueError(f"offsets end at {offsets[-1]} but there are {values.shape[0]} rows")
    if np.any(np.diff(offsets) < 1):
        raise ValueError("every segment needs at least one row")


def segment_max(values: FloatArray, offsets: IntArray) -> FloatArray:
    _check(values, offsets)
    return np.asarray(np.maximum.reduceat(values, offsets[:-1], axis=0), dtype=values.dtype)


def segment_min(values: FloatArray, offsets: IntArray) -> FloatArray:
    _check(values, offsets)
    return np.asarray(np.minimum.reduceat(values, offsets[:-1], axis=0), dtype=values.dtype)


def segment_sum(values: FloatArray, offsets: IntArray) -> FloatArray:
    _check(values, offsets)
    return np.asarray(np.add.reduceat(values, offsets[:-1], axis=0), dtype=values.dtype)


def segment_mean(values: FloatArray, offsets: IntArray) -> FloatArray:
    counts = np.diff(offsets).astype(np.float64)
    sums = segment_sum(values, offsets)
    return np.asarray(sums / counts.reshape((-1,) + (1,) * (values.ndim - 1)), dtype=np.float64)


def segment_any(mask: BoolArray, offsets: IntArray) -> BoolArray:
    _check(mask, offsets)
    return np.asarray(np.logical_or.reduceat(mask, offsets[:-1], axis=0), dtype=np.bool_)


def segment_first(values: FloatArray, offsets: IntArray) -> FloatArray:
    _check(values, offsets)
    return values[offsets[:-1]]


def segment_last(values: FloatArray, offsets: IntArray) -> FloatArray:
    _check(values, offsets)
    return values[offsets[1:] - 1]


def first_true_index(mask: BoolArray, offsets: IntArray) -> IntArray:
    """Index (within its segment) of the first True row per segment, or -1 if none."""
    _check(mask, offsets)
    n_seg = offsets.shape[0] - 1
    seg = np.repeat(np.arange(n_seg), np.diff(offsets))
    local = np.arange(mask.shape[0]) - offsets[seg]
    big = np.iinfo(np.int64).max
    candidate = np.where(mask, local, big)
    first = np.minimum.reduceat(candidate, offsets[:-1])
    return np.where(first == big, -1, first).astype(np.int64)


def segment_last_where(
    values: FloatArray, offsets: IntArray, mask: BoolArray, *, default: str = "first"
) -> FloatArray:
    """Value at the last row where ``mask`` is True, per segment.

    Segments without a True row fall back to their first row (``default="first"``).
    """
    if default != "first":
        raise ValueError("only default='first' is supported")
    _check(values, offsets)
    candidate = np.where(mask, np.arange(values.shape[0]), -1)
    last = np.maximum.reduceat(candidate, offsets[:-1])
    return values[np.where(last < 0, offsets[:-1], last)]


def segment_diff(values: FloatArray, offsets: IntArray) -> tuple[FloatArray, IntArray]:
    """First differences within each segment (one row shorter per segment)."""
    _check(values, offsets)
    keep = np.ones(values.shape[0], dtype=bool)
    keep[offsets[:-1]] = False
    diffs = np.diff(values, axis=0, prepend=values[:1])[keep]
    return diffs, (offsets - np.arange(offsets.shape[0])).astype(np.int64)


def compact(mask: BoolArray, offsets: IntArray) -> tuple[IntArray, IntArray]:
    """Rows selected by ``mask`` and the offsets of the compacted segments.

    Returns ``(rows, new_offsets)``; segments may become empty.
    """
    counts = (
        np.add.reduceat(mask.astype(np.int64), offsets[:-1])
        if mask.shape[0]
        else np.zeros(offsets.shape[0] - 1, dtype=np.int64)
    )
    # reduceat over an empty trailing segment returns the element at the start; guard it
    counts = np.where(np.diff(offsets) == 0, 0, counts)
    rows = np.flatnonzero(mask).astype(np.int64)
    return rows, np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

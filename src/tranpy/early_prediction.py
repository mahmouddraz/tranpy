"""How much post-fault observation is needed? Accuracy versus observation window.

For each window length, trajectory features are extracted from the windowed events and
every model is cross-validated (stratified, seeded). Scaling lives inside each model's
pipeline, so it is fitted on the training folds only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tranpy.features import TrajectoryFeatures, WindowReference, window_end_times
from tranpy_core.data import SignalDataset
from tranpy_core.errors import ConfigurationError
from tranpy_core.evaluation import METRICS, cross_validate
from tranpy_core.features import segments as seg
from tranpy_core.random import Seed

__all__ = ["find_knee_point", "run_early_prediction"]

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS_S = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def _with_scaler(model: BaseEstimator) -> Pipeline:
    if isinstance(model, Pipeline):
        return clone(model)
    return Pipeline([("scale", StandardScaler()), ("model", clone(model))])


def run_early_prediction(
    events: SignalDataset,
    models: Mapping[str, BaseEstimator],
    *,
    reference: WindowReference,
    windows_s: Sequence[float] = DEFAULT_WINDOWS_S,
    n_splits: int = 5,
    seed: Seed = 42,
    min_events: int = 10,
    feature_settings: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Cross-validated scores for every (window, model) pair.

    Args:
        events: event trajectories with fault-timing annotations.
        models: estimators by name; each is put behind a StandardScaler unless it already
            is a Pipeline.
        reference: instant the window is measured from (see :mod:`tranpy.features`).
        windows_s: observation windows in seconds after ``reference``.
        n_splits: cross-validation folds (reduced to the smallest class size if needed).
        seed: seed of the fold assignment.
        min_events: windows with fewer usable events are skipped.
        feature_settings: extra keyword arguments for :class:`TrajectoryFeatures`.

    Events whose recording ends before the window ends are left out of that window.
    Returns one row per (window, model) with ``<metric>_mean`` / ``<metric>_std``.
    """
    if not models:
        raise ConfigurationError("at least one model is required")
    t_end = seg.segment_last(events.time, events.offsets)
    rows: list[dict[str, Any]] = []
    for window in windows_s:
        if reference == "absolute":
            usable = np.ones(events.n_samples, dtype=bool)
        else:
            usable = t_end >= window_end_times(events, window, reference)
        subset = events.subset(np.flatnonzero(usable))
        if subset.n_samples < min_events:
            logger.info("window %.3f s: only %d usable events, skipped", window, subset.n_samples)
            continue
        table = TrajectoryFeatures(
            window_s=window, reference=reference, **(feature_settings or {})
        ).transform(subset)
        X = table.features
        n_bad = int((~np.isfinite(X)).sum())
        if n_bad:
            logger.warning(
                "window %.3f s: %d non-finite feature values replaced by 0", window, n_bad
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        splits = min(
            n_splits, int(np.min(np.bincount(table.labels, minlength=len(table.label_names))))
        )
        if splits < 2:
            logger.info("window %.3f s: fewer than 2 events in a class, skipped", window)
            continue
        for name, model in models.items():
            summary = cross_validate(
                _with_scaler(model), X, table.labels, n_splits=splits, seed=seed
            ).summary()
            rows.append(
                {
                    "window_s": float(window),
                    "model": name,
                    "n_events": subset.n_samples,
                    "n_features": X.shape[1],
                    "n_splits": splits,
                    **summary,
                }
            )
    columns = ["window_s", "model", "n_events", "n_features", "n_splits"] + [
        f"{m}_{s}" for m in METRICS for s in ("mean", "std")
    ]
    return pd.DataFrame(rows, columns=columns)


def find_knee_point(results: pd.DataFrame, model: str, metric: str = "accuracy") -> float:
    """Window length at the maximum curvature of ``metric`` versus window (diminishing returns).

    As in earlier versions, derivatives are taken over the *index* of the sorted windows,
    i.e. the windows are treated as equally spaced (they are roughly log-spaced by default).
    """
    curve = results[results["model"] == model].sort_values("window_s")
    x = curve["window_s"].to_numpy(np.float64)
    y = curve[f"{metric}_mean"].to_numpy(np.float64)
    if x.size == 0:
        raise ConfigurationError(f"no results for model {model!r}")
    if x.size < 3:
        return float(x[-1])
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    dy = np.gradient(y_norm)
    curvature = np.abs(np.gradient(dy)) / (1 + dy**2) ** 1.5
    return float(x[int(np.argmax(curvature[1:-1])) + 1])

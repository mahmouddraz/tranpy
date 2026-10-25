"""How much post-fault observation is needed? Accuracy versus observation window.

For each window length two representations of the windowed events are built:

* **features** - :class:`~tranpy.features.TrajectoryFeatures` (summary features per event)
  for ``models``;
* **sequences** - :class:`~tranpy.features.TrajectorySequence` (raw signals on a uniform
  time grid, the time dimension) for ``sequence_models`` such as LSTM, TCN or
  Transformer networks.

All models are cross-validated on the *same* stratified, seeded folds. Scaling lives
inside each model's pipeline, so it is fitted on the training folds only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tranpy.features import (
    TrajectoryFeatures,
    TrajectorySequence,
    WindowReference,
    window_end_times,
)
from tranpy_core.data import SignalDataset, kfold_splits
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


def _with_steps(pipeline: Pipeline, n_steps: int) -> Pipeline:
    """Tell a sequence network how many time steps each row holds."""
    estimator = pipeline.steps[-1][1]
    if "n_steps" in estimator.get_params():
        pipeline.set_params(**{f"{pipeline.steps[-1][0]}__n_steps": n_steps})
    return pipeline


def _finite(X: npt.NDArray[np.float64], window: float, what: str) -> npt.NDArray[np.float64]:
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad:
        logger.warning("window %.3f s: %d non-finite %s values replaced by 0", window, n_bad, what)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def run_early_prediction(
    events: SignalDataset,
    models: Mapping[str, BaseEstimator] | None = None,
    *,
    reference: WindowReference,
    sequence_models: Mapping[str, BaseEstimator] | None = None,
    windows_s: Sequence[float] = DEFAULT_WINDOWS_S,
    n_splits: int = 5,
    seed: Seed = 42,
    min_events: int = 10,
    feature_settings: Mapping[str, Any] | None = None,
    sequence_settings: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Cross-validated scores for every (window, model) pair.

    Args:
        events: event trajectories with fault-timing annotations.
        models: estimators on trajectory features, by name; each is put behind a
            StandardScaler unless it already is a Pipeline.
        reference: instant the window is measured from (see :mod:`tranpy.features`).
        sequence_models: estimators on resampled sequences (e.g. registry pipelines
            ``lstm_network``); their ``n_steps`` is set automatically.
        windows_s: observation windows in seconds after ``reference``.
        n_splits: cross-validation folds (reduced to the smallest class size if needed).
        seed: seed of the fold assignment (shared by all models).
        min_events: windows with fewer usable events are skipped.
        feature_settings: extra keyword arguments for :class:`TrajectoryFeatures`.
        sequence_settings: extra keyword arguments for :class:`TrajectorySequence`.

    Events whose recording ends before the window ends are left out of that window.
    Returns one row per (window, model) with ``input`` (features / sequence),
    ``<metric>_mean`` / ``<metric>_std`` and the mean fit time.
    """
    models = dict(models or {})
    sequence_models = dict(sequence_models or {})
    if not models and not sequence_models:
        raise ConfigurationError("at least one model is required")
    clash = set(models) & set(sequence_models)
    if clash:
        raise ConfigurationError(f"model names used twice: {sorted(clash)}")
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
        counts = np.bincount(subset.labels, minlength=len(subset.label_names))
        splits = min(n_splits, int(np.min(counts)))
        if splits < 2:
            logger.info("window %.3f s: fewer than 2 events in a class, skipped", window)
            continue
        folds = kfold_splits(subset.labels, n_splits=splits, seed=seed)
        jobs: list[tuple[str, str, Pipeline, npt.NDArray[np.float64]]] = []
        if models:
            table = TrajectoryFeatures(
                window_s=window, reference=reference, **(feature_settings or {})
            ).transform(subset)
            X = _finite(table.features, window, "feature")
            jobs += [(name, "features", _with_scaler(m), X) for name, m in models.items()]
        if sequence_models:
            seq = TrajectorySequence(
                window_s=window, reference=reference, **(sequence_settings or {})
            ).transform(subset)
            S = _finite(seq.features, window, "sequence")
            n_steps = int(seq.metadata["n_steps"])
            jobs += [
                (name, "sequence", _with_steps(_with_scaler(m), n_steps), S)
                for name, m in sequence_models.items()
            ]
        for name, kind, pipeline, data in jobs:
            t0 = time.perf_counter()
            result = cross_validate(pipeline, data, subset.labels, splits=folds)
            rows.append(
                {
                    "window_s": float(window),
                    "model": name,
                    "input": kind,
                    "n_events": subset.n_samples,
                    "n_features": data.shape[1],
                    "n_splits": splits,
                    **result.summary(),
                    "fit_time_s": float(np.mean(result.fit_times_s)),
                }
            )
            logger.info("window %.3f s, %s: %.1f s", window, name, time.perf_counter() - t0)
    columns = (
        ["window_s", "model", "input", "n_events", "n_features", "n_splits"]
        + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
        + ["fit_time_s"]
    )
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

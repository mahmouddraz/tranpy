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

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tranpy.datasets import LABEL_CHANNELS
from tranpy.features import (
    TrajectoryFeatures,
    TrajectorySequence,
    WindowReference,
    window_end_times,
)
from tranpy_core.data import (
    ShardManifest,
    SignalDataset,
    TabularDataset,
    load_tabular,
    map_shards,
    repeated_kfold_splits,
    save_tabular,
)
from tranpy_core.errors import ConfigurationError
from tranpy_core.evaluation import METRICS, cross_validate
from tranpy_core.features import segments as seg
from tranpy_core.random import Seed

__all__ = [
    "WindowTables",
    "evaluate_window_tables",
    "extract_window_tables",
    "find_knee_point",
    "run_early_prediction",
]

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


WindowTables = dict[tuple[float, str], TabularDataset]
"""Representation tables keyed by ``(window_s, "features" | "sequence")``."""


def _usable(
    events: SignalDataset, window: float, reference: WindowReference
) -> npt.NDArray[np.int64]:
    if reference == "absolute":
        return np.arange(events.n_samples)
    t_end = seg.segment_last(events.time, events.offsets)
    return np.flatnonzero(t_end >= window_end_times(events, window, reference))


def _tables_for(
    events: SignalDataset,
    *,
    reference: WindowReference,
    windows_s: Sequence[float],
    features: bool,
    sequences: bool,
    feature_settings: Mapping[str, Any],
    sequence_settings: Mapping[str, Any],
) -> WindowTables:
    out: WindowTables = {}
    label_channels = events.channel_index(LABEL_CHANNELS)
    if label_channels.size:  # out-of-step flags define the label; never use them as input
        events = events.drop_channels(LABEL_CHANNELS)
    for window in windows_s:
        subset = events.subset(_usable(events, window, reference))
        if subset.n_samples == 0:
            continue
        if features:
            out[(float(window), "features")] = TrajectoryFeatures(
                window_s=window, reference=reference, **feature_settings
            ).transform(subset)
        if sequences:
            out[(float(window), "sequence")] = TrajectorySequence(
                window_s=window, reference=reference, **sequence_settings
            ).transform(subset)
    return out


def _cache_key(source: SignalDataset | ShardManifest, settings: Mapping[str, Any]) -> str:
    if isinstance(source, ShardManifest):
        identity: Any = [s.sha256 for s in source.shards]
    else:
        digest = hashlib.sha256()
        digest.update(source.values.tobytes())
        digest.update(source.time.tobytes())
        identity = digest.hexdigest()
    blob = json.dumps({"data": identity, "settings": settings}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def extract_window_tables(
    source: SignalDataset | ShardManifest,
    *,
    reference: WindowReference,
    windows_s: Sequence[float] = DEFAULT_WINDOWS_S,
    features: bool = True,
    sequences: bool = False,
    feature_settings: Mapping[str, Any] | None = None,
    sequence_settings: Mapping[str, Any] | None = None,
    cache_directory: str | Path | None = None,
    **shard_options: Any,
) -> WindowTables:
    """Feature and/or sequence tables for every window, in one pass over the data.

    ``source`` is an in-memory dataset or a :class:`~tranpy_core.data.ShardManifest`; shards
    are streamed (see :func:`tranpy_core.data.iter_shards`, which receives
    ``shard_options``), so the raw trajectories never need to fit in memory or on disk.
    Events whose recording ends before a window ends are left out of that window. With
    ``cache_directory`` the tables are stored there, keyed by the data checksums and all
    settings, and reused on the next call.
    """
    feature_kwargs = dict(feature_settings or {})
    sequence_kwargs = dict(sequence_settings or {})
    settings: dict[str, Any] = {
        "reference": reference,
        "windows_s": [float(w) for w in windows_s],
        "features": features,
        "sequences": sequences,
        "feature_settings": dict(feature_settings or {}),
        "sequence_settings": dict(sequence_settings or {}),
    }
    cache_dir_path: Path | None = None
    if cache_directory is not None:
        cache_dir_path = Path(cache_directory) / _cache_key(source, settings)
        index = cache_dir_path / "tables.json"
        if index.exists():
            entries = json.loads(index.read_text(encoding="utf-8"))
            return {(float(w), kind): load_tabular(cache_dir_path / f) for w, kind, f in entries}

    def build(events: SignalDataset) -> WindowTables:
        return _tables_for(
            events,
            reference=reference,
            windows_s=windows_s,
            features=features,
            sequences=sequences,
            feature_settings=feature_kwargs,
            sequence_settings=sequence_kwargs,
        )

    tables: WindowTables = (
        build(source)
        if isinstance(source, SignalDataset)
        else map_shards(source, build, **shard_options)
    )
    if cache_dir_path is not None:
        entries = []
        for (window, kind), table in tables.items():
            filename = f"{kind}_{window:g}.npz"
            save_tabular(table, cache_dir_path / filename)
            entries.append([window, kind, filename])
        (cache_dir_path / "settings.json").write_text(
            json.dumps(settings, indent=1), encoding="utf-8"
        )
        (cache_dir_path / "tables.json").write_text(json.dumps(entries), encoding="utf-8")
    return tables


def evaluate_window_tables(
    tables: WindowTables,
    models: Mapping[str, BaseEstimator] | None = None,
    *,
    sequence_models: Mapping[str, BaseEstimator] | None = None,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: Seed = 42,
    min_events: int = 10,
) -> pd.DataFrame:
    """Cross-validate feature models and sequence models on the same folds per window.

    With ``n_repeats > 1`` the k-fold split is repeated with different shuffles, which
    makes the reported 95 % confidence intervals (``<metric>_ci95``) meaningful.
    """
    models = dict(models or {})
    sequence_models = dict(sequence_models or {})
    if not models and not sequence_models:
        raise ConfigurationError("at least one model is required")
    clash = set(models) & set(sequence_models)
    if clash:
        raise ConfigurationError(f"model names used twice: {sorted(clash)}")
    rows: list[dict[str, Any]] = []
    for window in sorted({w for w, _ in tables}):
        base = tables.get((window, "features")) or tables[(window, "sequence")]
        if base.n_samples < min_events:
            logger.info("window %.3f s: only %d usable events, skipped", window, base.n_samples)
            continue
        counts = np.bincount(base.labels, minlength=len(base.label_names))
        splits = min(n_splits, int(np.min(counts)))
        if splits < 2:
            logger.info("window %.3f s: fewer than 2 events in a class, skipped", window)
            continue
        folds = repeated_kfold_splits(base.labels, n_splits=splits, n_repeats=n_repeats, seed=seed)
        jobs: list[tuple[str, str, Pipeline, TabularDataset]] = []
        if models:
            table = tables[(window, "features")]
            jobs += [(name, "features", _with_scaler(m), table) for name, m in models.items()]
        if sequence_models:
            seq = tables[(window, "sequence")]
            if not np.array_equal(seq.sample_ids, base.sample_ids):
                raise ConfigurationError("feature and sequence tables hold different events")
            n_steps = int(seq.metadata["n_steps"])
            jobs += [
                (name, "sequence", _with_steps(_with_scaler(m), n_steps), seq)
                for name, m in sequence_models.items()
            ]
        for name, kind, pipeline, table in jobs:
            data = _finite(table.features, window, kind)
            t0 = time.perf_counter()
            result = cross_validate(pipeline, data, table.labels, splits=folds)
            rows.append(
                {
                    "window_s": float(window),
                    "model": name,
                    "input": kind,
                    "n_events": table.n_samples,
                    "n_features": data.shape[1],
                    "n_splits": splits,
                    **result.summary(),
                    "fit_time_s": float(np.mean(result.fit_times_s)),
                }
            )
            logger.info("window %.3f s, %s: %.1f s", window, name, time.perf_counter() - t0)
    columns = [
        "window_s",
        "model",
        "input",
        "n_events",
        "n_features",
        "n_splits",
        *(f"{m}_{s}" for m in METRICS for s in ("mean", "std", "ci95")),
        "fit_time_s",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_early_prediction(
    events: SignalDataset | ShardManifest,
    models: Mapping[str, BaseEstimator] | None = None,
    *,
    reference: WindowReference,
    sequence_models: Mapping[str, BaseEstimator] | None = None,
    windows_s: Sequence[float] = DEFAULT_WINDOWS_S,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: Seed = 42,
    min_events: int = 10,
    feature_settings: Mapping[str, Any] | None = None,
    sequence_settings: Mapping[str, Any] | None = None,
    cache_directory: str | Path | None = None,
    **shard_options: Any,
) -> pd.DataFrame:
    """Cross-validated scores for every (window, model) pair.

    Args:
        events: event trajectories with fault-timing annotations, in memory or as a
            shard manifest (streamed; see :func:`extract_window_tables`).
        models: estimators on trajectory features, by name; each is put behind a
            StandardScaler unless it already is a Pipeline.
        reference: instant the window is measured from (see :mod:`tranpy.features`).
        sequence_models: estimators on resampled sequences (e.g. registry pipelines
            ``lstm_network``); their ``n_steps`` is set automatically.
        windows_s: observation windows in seconds after ``reference``.
        n_splits: cross-validation folds (reduced to the smallest class size if needed).
        n_repeats: repetitions of the k-fold split (for confidence intervals).
        seed: seed of the fold assignment (shared by all models).
        min_events: windows with fewer usable events are skipped.
        feature_settings: extra keyword arguments for :class:`TrajectoryFeatures`.
        sequence_settings: extra keyword arguments for :class:`TrajectorySequence`.
        cache_directory: cache the extracted tables there (see :func:`extract_window_tables`).

    Returns one row per (window, model) with ``input`` (features / sequence),
    ``<metric>_mean`` / ``<metric>_std`` and the mean fit time.
    """
    if not models and not sequence_models:
        raise ConfigurationError("at least one model is required")
    tables = extract_window_tables(
        events,
        reference=reference,
        windows_s=windows_s,
        features=bool(models),
        sequences=bool(sequence_models),
        feature_settings=feature_settings,
        sequence_settings=sequence_settings,
        cache_directory=cache_directory,
        **shard_options,
    )
    return evaluate_window_tables(
        tables,
        models,
        sequence_models=sequence_models,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        min_events=min_events,
    )


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

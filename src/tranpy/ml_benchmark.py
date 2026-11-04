"""Configured benchmark of classifiers and imbalance methods on the fault events.

The configuration (YAML, see ``configs/ml/tsa_v4dyn.yaml``) names the feature sets, the
models and the imbalance options (:class:`tranpy_core.imbalance.ImbalanceSpec`); the
benchmark crosses them and evaluates every candidate by leave-one-group-out
cross-validation (by default one operating hour held out per fold, so every prediction is
for an hour the model has not seen). Out-of-fold predictions are kept, so the scores are
reported for the pooled set and per stratum (hour, clearing-time band, event kind, event
type) without refitting.

A candidate becomes the new champion only if it is better on the primary metric (PR-AUC
by default) and does not miss more unstable events (:func:`promotion_decision`).
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from tranpy_core.data.split import Split, leave_one_group_out_splits
from tranpy_core.errors import ConfigurationError
from tranpy_core.evaluation import (
    Scores,
    positive_scores,
    score_predictions,
    scores_by_group,
)
from tranpy_core.imbalance import ImbalanceSpec, make_imbalance_pipeline

__all__ = [
    "BenchmarkSpec",
    "Candidate",
    "CandidateResult",
    "candidates",
    "evaluate_candidate",
    "evaluate_feature_file",
    "promotion_decision",
]


@dataclass(frozen=True)
class Candidate:
    """One model with one imbalance option and fixed hyperparameters."""

    model: str
    imbalance: ImbalanceSpec = field(default_factory=ImbalanceSpec)
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        tag = self.imbalance.method
        if self.imbalance.threshold:
            tag += f"+thr_{self.imbalance.threshold}"
        return f"{self.model}__{tag}"

    def build(self, seed: int) -> Any:
        return make_imbalance_pipeline(self.model, self.imbalance, seed=seed, **self.params)


def _imbalance(entry: Mapping[str, Any] | str) -> ImbalanceSpec:
    if isinstance(entry, str):
        return ImbalanceSpec(method=entry)
    return ImbalanceSpec(**dict(entry))


@dataclass(frozen=True)
class BenchmarkSpec:
    """What to compare and how.

    Attributes:
        feature_sets: names of the feature sets (see :func:`tranpy.ml_events.feature_sets`).
        models: registered model names crossed with every entry of ``imbalance``.
        imbalance: imbalance options crossed with ``models``.
        extra: further (model, imbalance) pairs, e.g. imbalance ensembles with ``none``.
        params: hyperparameters per model name.
        split_by: column whose groups are held out one at a time.
        report_by: columns for the per-stratum reports.
        primary_metric: metric of the promotion gate (higher is better).
        seed: seed of every model and sampler.
    """

    feature_sets: tuple[str, ...]
    models: tuple[str, ...]
    imbalance: tuple[ImbalanceSpec, ...] = (ImbalanceSpec(),)
    extra: tuple[Candidate, ...] = ()
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    split_by: str = "hour"
    report_by: tuple[str, ...] = ("hour", "clearing_band", "kind", "event_type")
    primary_metric: str = "average_precision"
    seed: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BenchmarkSpec:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigurationError(f"unknown benchmark keys: {unknown}")
        params = {k: dict(v) for k, v in dict(data.get("params", {})).items()}
        extra = tuple(
            Candidate(
                model=e["model"],
                imbalance=_imbalance(e.get("imbalance", "none")),
                params=params.get(e["model"], {}),
            )
            for e in data.get("extra", [])
        )
        spec = cls(
            feature_sets=tuple(data["feature_sets"]),
            models=tuple(data.get("models", ())),
            imbalance=tuple(_imbalance(e) for e in data.get("imbalance", ["none"])),
            extra=extra,
            params=params,
            split_by=data.get("split_by", "hour"),
            report_by=tuple(data.get("report_by", cls.report_by)),
            primary_metric=data.get("primary_metric", "average_precision"),
            seed=int(data.get("seed", 0)),
        )
        if spec.primary_metric not in Scores.__dataclass_fields__:
            raise ConfigurationError(f"unknown primary metric {spec.primary_metric!r}")
        return spec


def candidates(spec: BenchmarkSpec) -> Iterator[Candidate]:
    """Every (model, imbalance) pair of the matrix, then the extra pairs."""
    for model in spec.models:
        for imbalance in spec.imbalance:
            yield Candidate(model, imbalance, spec.params.get(model, {}))
    yield from spec.extra


@dataclass
class CandidateResult:
    """Cross-validated result of one candidate on one feature set."""

    candidate: Candidate
    feature_set: str
    pooled: Scores
    folds: list[Scores]
    fold_groups: list[str]
    by_stratum: dict[str, pd.DataFrame]
    oof_score: np.ndarray
    oof_pred: np.ndarray
    fit_time_s: float
    thresholds: list[float]

    def metrics(self) -> dict[str, float]:
        """Flat metrics for tracking: pooled ``<m>`` and fold ``<m>_fold_mean/std``."""
        out = {k: float(v) for k, v in self.pooled.as_dict().items() if isinstance(v, int | float)}
        for m in Scores.__dataclass_fields__:
            values = [getattr(f, m) for f in self.folds]
            if values and isinstance(values[0], int | float):
                arr = np.asarray(values, dtype=np.float64)
                out[f"{m}_fold_mean"] = float(np.nanmean(arr))
                out[f"{m}_fold_std"] = float(np.nanstd(arr))
        out["fit_time_s"] = self.fit_time_s
        return out


def evaluate_candidate(
    candidate: Candidate,
    X: pd.DataFrame,
    y: np.ndarray,
    strata: pd.DataFrame,
    *,
    feature_set: str,
    split_by: str = "hour",
    report_by: Sequence[str] = (),
    seed: int = 0,
    splits: list[Split] | None = None,
    sample_weight: np.ndarray | None = None,
    log: Callable[[str], None] | None = None,
) -> CandidateResult:
    """Leave-one-group-out evaluation of ``candidate`` with out-of-fold predictions.

    ``sample_weight`` weights the evaluation only (e.g. inclusion weights of a subsample).
    """
    X_arr = X.to_numpy(np.float64)
    y_arr = np.asarray(y).astype(int)
    groups = strata[split_by].astype(str).to_numpy()
    folds = splits if splits is not None else leave_one_group_out_splits(groups)
    oof_score = np.full(y_arr.shape[0], np.nan)
    oof_pred = np.full(y_arr.shape[0], -1)
    fold_scores: list[Scores] = []
    fold_groups: list[str] = []
    thresholds: list[float] = []
    probabilities = True
    t_fit = 0.0
    for split in folds:
        model = clone(candidate.build(seed))
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(X_arr[split.train], y_arr[split.train])
        t_fit += time.perf_counter() - t0
        probabilities = probabilities and hasattr(model, "predict_proba")
        oof_score[split.test] = positive_scores(model, X_arr[split.test])
        oof_pred[split.test] = np.asarray(model.predict(X_arr[split.test]))
        thresholds.append(float(getattr(model, "best_threshold_", 0.5)))
        w = None if sample_weight is None else sample_weight[split.test]
        fold_scores.append(
            score_predictions(
                y_arr[split.test],
                oof_pred[split.test],
                oof_score[split.test],
                probabilities=probabilities,
                sample_weight=w,
            )
        )
        fold_groups.append(",".join(sorted(set(groups[split.test]))))
        if log:
            log(f"{feature_set} {candidate.name} held out {fold_groups[-1]}")
    tested = oof_pred >= 0
    pooled = score_predictions(
        y_arr[tested],
        oof_pred[tested],
        oof_score[tested],
        probabilities=probabilities,
        sample_weight=None if sample_weight is None else sample_weight[tested],
    )
    by_stratum = {
        col: scores_by_group(
            y_arr[tested],
            oof_pred[tested],
            oof_score[tested],
            strata[col].astype(str).to_numpy()[tested],
            probabilities=probabilities,
            sample_weight=None if sample_weight is None else sample_weight[tested],
        )
        for col in report_by
        if col in strata
    }
    return CandidateResult(
        candidate=candidate,
        feature_set=feature_set,
        pooled=pooled,
        folds=fold_scores,
        fold_groups=fold_groups,
        by_stratum=by_stratum,
        oof_score=oof_score,
        oof_pred=oof_pred,
        fit_time_s=t_fit,
        thresholds=thresholds,
    )


def promotion_decision(
    challenger: Mapping[str, float],
    champion: Mapping[str, float] | None,
    *,
    primary_metric: str = "average_precision",
) -> tuple[bool, str]:
    """Whether ``challenger`` replaces ``champion`` (metrics as from :meth:`metrics`).

    Promote if there is no champion, or if the challenger is strictly better on
    ``primary_metric`` and its miss rate (unstable events predicted stable) is not higher.
    """
    if champion is None:
        return True, "no champion yet"
    better = challenger[primary_metric] > champion[primary_metric]
    safe = challenger["miss_rate"] <= champion["miss_rate"]
    reason = (
        f"{primary_metric} {challenger[primary_metric]:.4f} vs {champion[primary_metric]:.4f}, "
        f"miss rate {challenger['miss_rate']:.4f} vs {champion['miss_rate']:.4f}"
    )
    return bool(better and safe), reason


def evaluate_feature_file(
    features_dir: str | Path,
    feature_set: str,
    candidate: Candidate,
    seed: int,
    split_by: str,
    report_by: Sequence[str],
) -> tuple[str, Candidate, CandidateResult | None, str | None, float]:
    """:func:`evaluate_candidate` on ``<features_dir>/<feature_set>.parquet`` and
    ``strata.parquet``, for worker processes (single-threaded libraries).

    Returns ``(feature_set, candidate, result or None, error or None, seconds)``; an error
    of one candidate is returned, not raised, so a benchmark records it and goes on.
    """
    for var in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(var, "1")
    base = Path(features_dir)
    X = pd.read_parquet(base / f"{feature_set}.parquet")
    strata = pd.read_parquet(base / "strata.parquet")
    t0 = time.perf_counter()
    try:
        result = evaluate_candidate(
            candidate,
            X,
            strata["label"].to_numpy(),
            strata,
            feature_set=feature_set,
            split_by=split_by,
            report_by=report_by,
            seed=seed,
        )
    except Exception as exc:  # recorded per candidate
        return (
            feature_set,
            candidate,
            None,
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    return feature_set, candidate, result, None, time.perf_counter() - t0

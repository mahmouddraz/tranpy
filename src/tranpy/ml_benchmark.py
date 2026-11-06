"""Configured benchmark of classifiers and imbalance methods on the fault events.

The configuration (YAML, see ``configs/ml/tsa_v4dyn.yaml``) names the feature sets, the
models, the imbalance options (:class:`tranpy_core.imbalance.ImbalanceSpec`) and the
evaluation protocols; the benchmark crosses them and evaluates every candidate by
cross-validation with out-of-fold predictions. Protocols (:data:`PROTOCOLS`):

``hour``
    leave one operating hour out, so every prediction is for an hour the model has not
    seen (generalisation across operating points);
``random``
    stratified 5-fold, shuffled with the seed (the usual i.i.d. estimate, optimistic when
    events of one hour are near duplicates);
``line``
    stratified group 5-fold with the faulted line as the group (generalisation to a line
    never faulted in training).

Out-of-fold predictions are kept, so the scores are reported for the pooled set and per
stratum (hour, clearing-time band, event kind, event type) without refitting. Every pooled
metric comes with a 95 % stratified bootstrap interval
(:func:`tranpy_core.evaluation.bootstrap_ci`).

Candidates are ranked by :func:`select_best`: the primary metric (PR-AUC by default)
decides, but candidates whose point estimate lies inside the leader's interval are ties,
settled by the lower miss rate. :func:`promotion_decision` applies the same rule to a
challenger against the registered champion.

With a ``tuning`` section (:class:`tranpy.ml_tuning.TuningSpec`) the hyperparameters of
every candidate are tuned by an Optuna study inside each outer training fold (nested
cross-validation, :func:`tranpy.ml_tuning.tune_params`); ``tuning.only_sets`` restricts
this to some feature sets. ``tuning: null`` (the default) keeps the configured parameters.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from tranpy.ml_tuning import TuningResult, TuningSpec, tune_params
from tranpy_core.data.split import Split, kfold_splits, leave_one_group_out_splits
from tranpy_core.errors import ConfigurationError
from tranpy_core.evaluation import (
    Scores,
    bootstrap_ci,
    positive_scores,
    score_predictions,
    scores_by_group,
)
from tranpy_core.imbalance import ImbalanceSpec, make_imbalance_pipeline

__all__ = [
    "PROTOCOLS",
    "BenchmarkSpec",
    "Candidate",
    "CandidateResult",
    "TuningSpec",
    "candidates",
    "evaluate_candidate",
    "evaluate_feature_file",
    "promotion_decision",
    "protocol_splits",
    "select_best",
    "tuning_for",
]

PROTOCOLS: Mapping[str, str] = {
    "hour": "leave one operating hour out",
    "random": "stratified 5-fold, shuffled with the seed",
    "line": "stratified group 5-fold, grouped by the faulted line",
}
"""Evaluation protocols: name -> how the folds are made."""

N_FOLDS = 5
N_BOOT = 1000


@dataclass(frozen=True)
class Candidate:
    """One model with one imbalance option and fixed hyperparameters, under one protocol."""

    model: str
    imbalance: ImbalanceSpec = field(default_factory=ImbalanceSpec)
    params: Mapping[str, Any] = field(default_factory=dict)
    protocol: str = "hour"
    tuning: TuningSpec | None = None

    @property
    def name(self) -> str:
        tag = self.imbalance.method
        if self.imbalance.threshold:
            tag += f"+thr_{self.imbalance.threshold}"
        return f"{self.model}__{tag}"

    def build(self, seed: int, **params: Any) -> Any:
        """The pipeline with the configured parameters, overridden by ``params``
        (e.g. the tuned ones)."""
        return make_imbalance_pipeline(
            self.model, self.imbalance, seed=seed, **{**self.params, **params}
        )

    def build_unthresholded(self, seed: int, **params: Any) -> Any:
        """:meth:`build` without the decision-threshold wrapper: what the tuning study
        fits (the threshold is calibrated afterwards on the refit, not per trial)."""
        base = replace(self, imbalance=replace(self.imbalance, threshold=None))
        return base.build(seed, **params)


def tuning_for(candidate: Candidate, feature_set: str) -> TuningSpec | None:
    """The tuning of ``candidate`` on ``feature_set``, or None (not tuned there)."""
    spec = candidate.tuning
    return spec if spec is not None and spec.applies_to(feature_set) else None


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
        protocols: evaluation protocols (:data:`PROTOCOLS`) crossed with every candidate;
            the first one is the protocol on which the best candidate is selected.
        split_by: deprecated alias of a single protocol (used when ``protocols`` is absent).
        report_by: columns for the per-stratum reports.
        primary_metric: metric of the ranking and promotion gate (higher is better).
        seed: seed of every model, sampler, shuffled split and bootstrap.
        tuning: nested hyperparameter tuning of every candidate, or None.
    """

    feature_sets: tuple[str, ...]
    models: tuple[str, ...]
    imbalance: tuple[ImbalanceSpec, ...] = (ImbalanceSpec(),)
    extra: tuple[Candidate, ...] = ()
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    protocols: tuple[str, ...] = ("hour",)
    split_by: str = "hour"
    report_by: tuple[str, ...] = ("hour", "clearing_band", "kind", "event_type")
    primary_metric: str = "average_precision"
    seed: int = 0
    tuning: TuningSpec | None = None

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
        split_by = str(data.get("split_by", "hour"))
        protocols = tuple(str(p) for p in data.get("protocols", [split_by]))
        tuning_data = data.get("tuning")
        tuning = None if tuning_data is None else TuningSpec.from_mapping(dict(tuning_data))
        spec = cls(
            feature_sets=tuple(data["feature_sets"]),
            models=tuple(data.get("models", ())),
            imbalance=tuple(_imbalance(e) for e in data.get("imbalance", ["none"])),
            extra=extra,
            params=params,
            protocols=protocols,
            split_by=split_by,
            report_by=tuple(data.get("report_by", cls.report_by)),
            primary_metric=data.get("primary_metric", "average_precision"),
            seed=int(data.get("seed", 0)),
            tuning=tuning,
        )
        if spec.tuning is not None:
            unknown_sets = sorted(set(spec.tuning.only_sets) - set(spec.feature_sets))
            if unknown_sets:
                raise ConfigurationError(f"tuning.only_sets not in feature_sets: {unknown_sets}")
        if spec.primary_metric not in Scores.__dataclass_fields__:
            raise ConfigurationError(f"unknown primary metric {spec.primary_metric!r}")
        unknown_protocols = sorted(set(spec.protocols) - set(PROTOCOLS))
        if not spec.protocols or unknown_protocols:
            raise ConfigurationError(
                f"unknown protocols {unknown_protocols}; known: {sorted(PROTOCOLS)}"
            )
        return spec


def candidates(spec: BenchmarkSpec) -> Iterator[Candidate]:
    """Every (model, imbalance) pair of the matrix, then the extra pairs, per protocol."""
    for protocol in spec.protocols:
        for model in spec.models:
            for imbalance in spec.imbalance:
                yield Candidate(model, imbalance, spec.params.get(model, {}), protocol, spec.tuning)
        for extra in spec.extra:
            yield Candidate(extra.model, extra.imbalance, extra.params, protocol, spec.tuning)


def protocol_splits(
    protocol: str, y: np.ndarray, strata: pd.DataFrame, *, seed: int = 0
) -> tuple[list[Split], list[str]]:
    """Folds of ``protocol`` (:data:`PROTOCOLS`) and a label per fold (the held-out group(s))."""
    y_arr = np.asarray(y)
    if protocol == "hour":
        groups = strata["hour"].astype(str).to_numpy()
        splits = leave_one_group_out_splits(groups)
        return splits, [",".join(sorted(set(groups[s.test]))) for s in splits]
    if protocol == "random":
        splits = kfold_splits(y_arr, n_splits=N_FOLDS, seed=seed)
        return splits, [f"fold_{i}" for i in range(len(splits))]
    if protocol == "line":
        if "fault_line" not in strata:
            raise ConfigurationError("protocol 'line' needs the strata column 'fault_line'")
        groups = strata["fault_line"].astype(str).to_numpy()
        if np.unique(groups).size < N_FOLDS:
            n_lines = np.unique(groups).size
            raise ConfigurationError(
                f"protocol 'line' needs at least {N_FOLDS} fault lines, got {n_lines}"
            )
        splits = kfold_splits(y_arr, n_splits=N_FOLDS, seed=seed, groups=groups)
        return splits, [
            f"fold_{i}:{np.unique(groups[s.test]).size} lines" for i, s in enumerate(splits)
        ]
    raise ConfigurationError(f"unknown protocol {protocol!r}; known: {sorted(PROTOCOLS)}")


@dataclass
class CandidateResult:
    """Cross-validated result of one candidate on one feature set under one protocol."""

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
    protocol: str = "hour"
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    tuning: dict[str, Any] | None = None
    """Nested tuning record (None when not tuned): ``n_trials``, ``inner_folds``, ``metric``,
    ``inner_groups``, ``best_params`` (per outer fold), ``best_value`` (per outer fold),
    ``trials`` (table of every trial with its ``fold``) and ``time_s``."""

    @property
    def tuned(self) -> bool:
        return self.tuning is not None

    def metrics(self) -> dict[str, float]:
        """Flat metrics for tracking: pooled ``<m>``, its interval ``<m>_lo``/``<m>_hi``
        and the fold ``<m>_fold_mean/std``."""
        out = {k: float(v) for k, v in self.pooled.as_dict().items() if isinstance(v, int | float)}
        for m, (lo, hi) in self.ci.items():
            out[f"{m}_lo"] = float(lo)
            out[f"{m}_hi"] = float(hi)
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
    split_by: str | None = None,
    report_by: Sequence[str] = (),
    seed: int = 0,
    splits: list[Split] | None = None,
    sample_weight: np.ndarray | None = None,
    n_boot: int = N_BOOT,
    log: Callable[[str], None] | None = None,
) -> CandidateResult:
    """Cross-validated evaluation of ``candidate`` with out-of-fold predictions.

    ``split_by`` is the protocol (:data:`PROTOCOLS`); it defaults to ``candidate.protocol``.
    Explicit ``splits`` override the protocol. ``sample_weight`` weights the evaluation
    only (e.g. inclusion weights of a subsample). The pooled metrics get a 95 % stratified
    bootstrap interval of ``n_boot`` resamples (0 skips it).
    """
    protocol = candidate.protocol if split_by is None else split_by
    X_arr = X.to_numpy(np.float64)
    y_arr = np.asarray(y).astype(int)
    if splits is not None:
        folds = splits
        labels = [f"fold_{i}" for i in range(len(splits))]
    else:
        folds, labels = protocol_splits(protocol, y_arr, strata, seed=seed)
    oof_score = np.full(y_arr.shape[0], np.nan)
    oof_pred = np.full(y_arr.shape[0], -1)
    fold_scores: list[Scores] = []
    fold_groups: list[str] = []
    thresholds: list[float] = []
    probabilities = True
    t_fit = 0.0
    tuning = tuning_for(candidate, feature_set)
    studies: list[TuningResult] = []
    groups = strata["hour"].astype(str).to_numpy() if "hour" in strata else None
    for split, label in zip(folds, labels, strict=True):
        tuned_params: dict[str, Any] = {}
        if tuning is not None:
            # nested: the study sees the outer training rows only, grouped by their hour
            study = tune_params(
                candidate.model,
                lambda p: candidate.build_unthresholded(seed, **p),
                X_arr[split.train],
                y_arr[split.train],
                spec=tuning,
                seed=seed,
                groups=None if groups is None else groups[split.train],
            )
            studies.append(study)
            tuned_params = study.best_params
            if log:
                log(
                    f"{feature_set} {candidate.name} [{protocol}] tuned on {label}: "
                    f"{tuning.metric}={study.best_value:.4f} in {study.time_s:.0f} s"
                )
        model = clone(candidate.build(seed, **tuned_params))
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
        fold_groups.append(label)
        if log:
            log(f"{feature_set} {candidate.name} [{protocol}] held out {label}")
    tested = oof_pred >= 0
    w_tested = None if sample_weight is None else sample_weight[tested]
    pooled = score_predictions(
        y_arr[tested],
        oof_pred[tested],
        oof_score[tested],
        probabilities=probabilities,
        sample_weight=w_tested,
    )
    ci = (
        bootstrap_ci(
            y_arr[tested],
            oof_pred[tested],
            oof_score[tested],
            probabilities=probabilities,
            sample_weight=w_tested,
            n_boot=n_boot,
            seed=seed,
        )
        if n_boot > 0
        else {}
    )
    by_stratum = {
        col: scores_by_group(
            y_arr[tested],
            oof_pred[tested],
            oof_score[tested],
            strata[col].astype(str).to_numpy()[tested],
            probabilities=probabilities,
            sample_weight=w_tested,
        )
        for col in report_by
        if col in strata
    }
    tuning_record = None
    if tuning is not None:
        tuning_record = {
            "n_trials": tuning.n_trials,
            "inner_folds": tuning.inner_folds,
            "metric": tuning.metric,
            "inner_groups": studies[0].inner_groups if studies else "",
            "best_params": [s.best_params for s in studies],
            "best_value": [s.best_value for s in studies],
            "trials": pd.concat(
                [s.trials.assign(fold=g) for s, g in zip(studies, fold_groups, strict=True)],
                ignore_index=True,
            ),
            "time_s": float(sum(s.time_s for s in studies)),
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
        protocol=protocol,
        ci=ci,
        tuning=tuning_record,
    )


def _finite(metrics: Mapping[str, float], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _tied(leader: Mapping[str, float], other: Mapping[str, float], metric: str) -> bool | None:
    """Whether ``other``'s point estimate lies inside the leader's 95 % interval.

    Uses the leader's lower bound; if the leader carries no interval, the other's upper
    bound is used instead (``other_hi >= leader``). None when neither has an interval.
    """
    leader_lo = _finite(leader, f"{metric}_lo")
    if leader_lo is not None:
        return float(other[metric]) >= leader_lo
    other_hi = _finite(other, f"{metric}_hi")
    if other_hi is not None:
        return other_hi >= float(leader[metric])
    return None


def select_best(
    results: Sequence[Mapping[str, float]],
    *,
    primary: str = "average_precision",
    tie_metric: str = "miss_rate",
) -> int:
    """Index of the best result (metrics as from :meth:`CandidateResult.metrics`).

    The leader is the highest ``primary``. Every result whose ``primary`` is at least the
    leader's lower interval bound (``<primary>_lo``) is a tie with it; among the ties the
    lowest ``tie_metric`` wins, the higher ``primary`` breaking equal tie metrics. Without
    interval columns only exact equality of ``primary`` is a tie.
    """
    if not results:
        raise ValueError("no results to select from")
    valid = [i for i, r in enumerate(results) if _finite(r, primary) is not None]
    if not valid:
        raise ValueError(f"no result has a finite {primary!r}")
    leader = max(valid, key=lambda i: float(results[i][primary]))
    ties = [
        i
        for i in valid
        if i == leader
        or (
            (tied := _tied(results[leader], results[i], primary)) is True
            or (tied is None and float(results[i][primary]) == float(results[leader][primary]))
        )
    ]

    def key(i: int) -> tuple[float, float]:
        tie_value = _finite(results[i], tie_metric)
        return (math.inf if tie_value is None else tie_value, -float(results[i][primary]))

    return min(ties, key=key)


def promotion_decision(
    challenger: Mapping[str, float],
    champion: Mapping[str, float] | None,
    *,
    primary_metric: str = "average_precision",
) -> tuple[bool, str]:
    """Whether ``challenger`` replaces ``champion`` (metrics as from :meth:`metrics`).

    Promote if there is no champion. With confidence intervals (``<metric>_lo``/``_hi``):
    promote if the challenger is better on ``primary_metric`` beyond the interval, keep the
    champion if it is better beyond the interval, and break a tie within the interval by the
    Matthews correlation coefficient (``mcc``) at the models' operating points, which
    penalises misses and false alarms together. The miss rate alone is not a tie-breaker:
    it compares different operating points (a model with 27 % precision reaches a low miss
    rate that one with 68 % precision does not). Without intervals: promote if the
    challenger is strictly better on ``primary_metric`` and its MCC is not lower. When a
    side carries no ``mcc`` (versions registered before it was stored) the lower miss rate
    decides the tie instead.
    """
    if champion is None:
        return True, "no champion yet"
    c_primary, k_primary = float(challenger[primary_metric]), float(champion[primary_metric])
    c_miss, k_miss = float(challenger["miss_rate"]), float(champion["miss_rate"])
    c_mcc, k_mcc = _finite(challenger, "mcc"), _finite(champion, "mcc")
    c_prec, k_prec = _finite(challenger, "precision"), _finite(champion, "precision")
    reason = (
        f"{primary_metric} {c_primary:.4f} vs {k_primary:.4f}, "
        f"miss rate {c_miss:.4f} vs {k_miss:.4f}"
    )
    if c_prec is not None and k_prec is not None:
        reason += f", precision {c_prec:.4f} vs {k_prec:.4f}"
    if c_mcc is not None and k_mcc is not None:
        reason += f", mcc {c_mcc:.4f} vs {k_mcc:.4f}"
        tie_break, tie_name = c_mcc > k_mcc, "mcc"
        not_worse = c_mcc >= k_mcc
    else:
        tie_break, tie_name = c_miss < k_miss, "miss rate"
        not_worse = c_miss <= k_miss
    leader, other = (challenger, champion) if c_primary >= k_primary else (champion, challenger)
    tied = _tied(leader, other, primary_metric)
    if tied is None:
        return bool(c_primary > k_primary and not_worse), reason
    if tied:
        verdict = "promoted" if tie_break else "kept"
        tie = f"tied within the {primary_metric} interval, {verdict} by {tie_name}"
        return tie_break, f"{reason}; {tie}"
    if c_primary < k_primary:
        return False, f"{reason}; champion better beyond the {primary_metric} interval"
    return True, f"{reason}; better beyond the {primary_metric} interval"


def evaluate_feature_file(
    features_dir: str | Path,
    feature_set: str,
    candidate: Candidate,
    seed: int,
    split_by: str | None,
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

"""Metrics for binary classifiers and a seeded cross-validation runner.

The *positive* class is label 1 (for transient stability: unstable). Besides the usual
metrics, :class:`Scores` carries the standard measures for imbalanced data: PR-AUC
(average precision), Matthews correlation, the geometric mean of sensitivity and
specificity, the Brier score and the miss rate (positives predicted negative). Every
metric takes optional sample weights, e.g. inverse inclusion-probability weights of a
stratified subsample. ROC-AUC uses ``predict_proba`` where available and
``decision_function`` otherwise; it is NaN (not 0) when a fold contains one class only.
Cross-validation clones the estimator for every fold, so a Pipeline's preprocessing is
refitted on each training fold and never sees the test fold.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tranpy_core.data.split import Split, repeated_kfold_splits
from tranpy_core.errors import ModelError
from tranpy_core.random import Seed

__all__ = [
    "CrossValidationResult",
    "Scores",
    "cross_validate",
    "evaluate",
    "evaluate_by_group",
    "positive_scores",
    "score_predictions",
    "scores_by_group",
]

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "mcc",
    "g_mean",
    "miss_rate",
    "brier",
)


@dataclass(frozen=True)
class Scores:
    """Binary classification metrics on one evaluation set."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    n_samples: int
    average_precision: float = float("nan")
    mcc: float = float("nan")
    g_mean: float = float("nan")
    miss_rate: float = float("nan")
    brier: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["confusion_matrix"] = [list(r) for r in self.confusion_matrix]
        return out


def positive_scores(model: Any, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Continuous score for the positive class: probability if available, else margin."""
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        except (AttributeError, NotImplementedError):
            proba = None
        if proba is not None:
            if proba.ndim == 2 and proba.shape[1] == 1 and hasattr(model, "classes_"):
                # trained on one class only: P(positive) is 1 or 0 for every sample
                return np.full(proba.shape[0], float(np.asarray(model.classes_)[0] == 1))
            if proba.ndim != 2 or proba.shape[1] != 2:
                raise ModelError(f"expected probabilities of shape (n, 2), got {proba.shape}")
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
    raise ModelError(f"{type(model).__name__} has neither predict_proba nor decision_function")


def score_predictions(
    y: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    scores: npt.ArrayLike,
    *,
    probabilities: bool = True,
    sample_weight: npt.ArrayLike | None = None,
) -> Scores:
    """Metrics from labels, predicted labels and continuous positive-class scores.

    ``sample_weight`` weights every metric; the confusion matrix then holds weighted
    counts rounded to integers. ROC-AUC and PR-AUC are NaN for single-class sets; the
    Brier score is NaN unless ``scores`` are probabilities.
    """
    y_true = np.asarray(y)
    y_hat = np.asarray(y_pred)
    s = np.asarray(scores, dtype=np.float64)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    two_classes = np.unique(y_true).size == 2
    auc = float(roc_auc_score(y_true, s, sample_weight=w)) if two_classes else float("nan")
    ap = float(average_precision_score(y_true, s, sample_weight=w)) if two_classes else float("nan")
    brier = (
        float(brier_score_loss(y_true, s, sample_weight=w))
        if probabilities and two_classes
        else float("nan")
    )
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1], sample_weight=w)
    tn, fp, fn, tp = (float(v) for v in cm.ravel())
    sensitivity = tp / (tp + fn) if tp + fn > 0 else float("nan")
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    with warnings.catch_warnings():
        # single-class evaluation sets are legitimate (e.g. small folds); metrics stay defined
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        balanced = float(balanced_accuracy_score(y_true, y_hat, sample_weight=w))
        mcc = float(matthews_corrcoef(y_true, y_hat, sample_weight=w))
    return Scores(
        accuracy=float(accuracy_score(y_true, y_hat, sample_weight=w)),
        balanced_accuracy=balanced,
        precision=float(precision_score(y_true, y_hat, zero_division=0, sample_weight=w)),
        recall=float(recall_score(y_true, y_hat, zero_division=0, sample_weight=w)),
        f1=float(f1_score(y_true, y_hat, zero_division=0, sample_weight=w)),
        roc_auc=auc,
        confusion_matrix=((round(tn), round(fp)), (round(fn), round(tp))),
        n_samples=int(y_true.shape[0]),
        average_precision=ap,
        mcc=mcc,
        g_mean=float(np.sqrt(sensitivity * specificity)),
        miss_rate=1.0 - sensitivity,
        brier=brier,
    )


def evaluate(
    model: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    sample_weight: npt.ArrayLike | None = None,
) -> Scores:
    """Score a fitted binary classifier on ``(X, y)`` (see :func:`score_predictions`)."""
    return score_predictions(
        y,
        model.predict(X),
        positive_scores(model, X),
        probabilities=hasattr(model, "predict_proba"),
        sample_weight=sample_weight,
    )


def scores_by_group(
    y: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    scores: npt.ArrayLike,
    groups: npt.ArrayLike,
    *,
    probabilities: bool = True,
    sample_weight: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """:func:`score_predictions` per group (e.g. hour or clearing-time band), one row each.

    The row ``__all__`` is the score on the whole set. ``n_positive`` is the number of
    label-1 samples of the group, so groups too small to trust are visible.
    """
    y_arr, p_arr, s_arr = np.asarray(y), np.asarray(y_pred), np.asarray(scores)
    g = np.asarray(groups, dtype=object)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    rows: dict[Any, dict[str, Any]] = {}
    for name in ["__all__", *sorted(set(g.tolist()), key=str)]:
        mask = np.ones(g.shape[0], dtype=bool) if name == "__all__" else g == name
        scores_ = score_predictions(
            y_arr[mask],
            p_arr[mask],
            s_arr[mask],
            probabilities=probabilities,
            sample_weight=None if w is None else w[mask],
        )
        rows[name] = {**scores_.as_dict(), "n_positive": int((y_arr[mask] == 1).sum())}
    return pd.DataFrame(rows).T


def evaluate_by_group(
    model: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    groups: npt.ArrayLike,
    *,
    sample_weight: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """:func:`evaluate` on every group of a fitted model (see :func:`scores_by_group`)."""
    return scores_by_group(
        y,
        model.predict(X),
        positive_scores(model, X),
        groups,
        probabilities=hasattr(model, "predict_proba"),
        sample_weight=sample_weight,
    )


@dataclass(frozen=True)
class CrossValidationResult:
    """Per-fold scores and fit times.

    ``test_to_train`` is the mean ratio of test to training samples; it enters the
    corrected confidence interval in :meth:`summary`.
    """

    folds: tuple[Scores, ...]
    fit_times_s: tuple[float, ...]
    test_to_train: float = 0.25

    def summary(self) -> dict[str, float]:
        """Mean, standard deviation and 95 % confidence half-width per metric.

        NaN folds are ignored. The interval is the corrected resampled t interval of
        Nadeau and Bengio (2003): the variance across folds is inflated by
        ``1/n + n_test/n_train`` because folds share training data, which plain
        ``std/sqrt(n)`` ignores. Use repeated cross-validation for a usable interval.
        """
        out: dict[str, float] = {}
        for metric in METRICS:
            values = np.array([getattr(f, metric) for f in self.folds], dtype=np.float64)
            finite = values[np.isfinite(values)]
            n = finite.size
            out[f"{metric}_mean"] = float(finite.mean()) if n else float("nan")
            out[f"{metric}_std"] = float(finite.std()) if n else float("nan")
            if n >= 2:
                variance = float(finite.var(ddof=1)) * (1.0 / n + self.test_to_train)
                out[f"{metric}_ci95"] = float(student_t.ppf(0.975, n - 1) * np.sqrt(variance))
            else:
                out[f"{metric}_ci95"] = float("nan")
        return out

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"fold": i, **f.as_dict(), "fit_time_s": t}
            for i, (f, t) in enumerate(zip(self.folds, self.fit_times_s, strict=True))
        ]
        return pd.DataFrame(rows)


def cross_validate(
    model: Any,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    n_splits: int = 5,
    seed: Seed = 0,
    groups: npt.ArrayLike | None = None,
    splits: list[Split] | None = None,
    n_repeats: int = 1,
    sample_weight: npt.ArrayLike | None = None,
) -> CrossValidationResult:
    """Stratified (repeated) k-fold cross-validation of a (cloned) estimator or pipeline.

    ``sample_weight`` weights the *evaluation* of every test fold only (e.g. inclusion
    weights of a subsample); training is unweighted, so imbalance handling stays in the
    model (see :mod:`tranpy_core.imbalance`).
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    folds = (
        splits
        if splits is not None
        else repeated_kfold_splits(
            y_arr, n_splits=n_splits, n_repeats=n_repeats, seed=seed, groups=groups
        )
    )
    scores: list[Scores] = []
    times: list[float] = []
    for split in folds:
        fold_model = clone(model)
        t0 = time.perf_counter()
        fold_model.fit(X_arr[split.train], y_arr[split.train])
        times.append(time.perf_counter() - t0)
        w = None if weights is None else weights[split.test]
        scores.append(evaluate(fold_model, X_arr[split.test], y_arr[split.test], sample_weight=w))
    ratio = float(np.mean([s.test.size / max(s.train.size, 1) for s in folds]))
    return CrossValidationResult(folds=tuple(scores), fit_times_s=tuple(times), test_to_train=ratio)


def scores_frame(results: Mapping[str, Scores]) -> pd.DataFrame:
    """One row per named result."""
    return pd.DataFrame({name: s.as_dict() for name, s in results.items()}).T

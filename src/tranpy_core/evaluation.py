"""Metrics for binary classifiers and a seeded cross-validation runner.

The *positive* class is label 1. ROC-AUC uses ``predict_proba`` where available and
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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tranpy_core.data.split import Split, repeated_kfold_splits
from tranpy_core.errors import ModelError
from tranpy_core.random import Seed

__all__ = ["CrossValidationResult", "Scores", "cross_validate", "evaluate", "positive_scores"]

METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")


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


def evaluate(model: Any, X: npt.ArrayLike, y: npt.ArrayLike) -> Scores:
    """Score a fitted binary classifier on ``(X, y)``."""
    y_true = np.asarray(y)
    y_pred = np.asarray(model.predict(X))
    scores = positive_scores(model, X)
    auc = float(roc_auc_score(y_true, scores)) if np.unique(y_true).size == 2 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with warnings.catch_warnings():
        # single-class evaluation sets are legitimate (e.g. small folds); metrics stay defined
        warnings.simplefilter("ignore", category=UserWarning)
        balanced = float(balanced_accuracy_score(y_true, y_pred))
    return Scores(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=balanced,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        confusion_matrix=((int(cm[0, 0]), int(cm[0, 1])), (int(cm[1, 0]), int(cm[1, 1]))),
        n_samples=int(y_true.shape[0]),
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
) -> CrossValidationResult:
    """Stratified (repeated) k-fold cross-validation of a (cloned) estimator or pipeline."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
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
        scores.append(evaluate(fold_model, X_arr[split.test], y_arr[split.test]))
    ratio = float(np.mean([s.test.size / max(s.train.size, 1) for s in folds]))
    return CrossValidationResult(folds=tuple(scores), fit_times_s=tuple(times), test_to_train=ratio)


def scores_frame(results: Mapping[str, Scores]) -> pd.DataFrame:
    """One row per named result."""
    return pd.DataFrame({name: s.as_dict() for name, s in results.items()}).T

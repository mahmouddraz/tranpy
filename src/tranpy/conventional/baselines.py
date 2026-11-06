"""Single-score threshold baselines, evaluated with the protocol of the ML benchmark.

A baseline classifies an event as unstable when one score (a clearing-time quantity such
as the largest speed deviation, or the clearing time itself) reaches a threshold chosen on
the *training* events only, by one of two standard criteria: Youden's index
``J = sensitivity + specificity - 1`` (Youden, W. J. (1950), "Index for rating diagnostic
tests", Cancer 3(1), 32-35), the ROC point farthest above the diagonal; or the F1 score,
the criterion of the ML benchmark's ``thr_f1`` candidates. The direction of the score
(higher = more unstable) is fixed in advance, not fitted.

:class:`ThresholdRule` is a scikit-learn classifier, so :class:`RuleCandidate` plugs it into
:func:`tranpy.ml_benchmark.evaluate_candidate` unchanged: the same leave-one-hour-out
splits, out-of-fold predictions, pooled and per-stratum scores as the ML candidates. Its
``decision_function`` is the raw score, so ROC-AUC and PR-AUC are those of the score itself
and do not depend on the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_curve
from sklearn.utils.validation import check_is_fitted

from tranpy.ml_benchmark import Candidate
from tranpy_core.errors import ConfigurationError

__all__ = ["CRITERIA", "RuleCandidate", "ThresholdRule", "f1_threshold", "youden_threshold"]

CRITERIA = ("youden", "f1")


def youden_threshold(y: npt.ArrayLike, score: npt.ArrayLike) -> float:
    """Threshold on ``score`` (predict 1 when ``score >= threshold``) maximising Youden's J.

    Ties are resolved towards the highest threshold among the maximisers (the first ROC
    point reached). With one class only, returns ``+inf`` (all negative) if that class is 0
    and ``-inf`` (all positive) if it is 1.
    """
    y_arr = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=np.float64)
    if y_arr.shape != s.shape:
        raise ConfigurationError("one score per label")
    if not np.all(np.isfinite(s)):
        raise ConfigurationError("scores must be finite")
    classes = np.unique(y_arr)
    if classes.size < 2:
        return float("inf") if classes.size == 0 or classes[0] == 0 else float("-inf")
    fpr, tpr, thresholds = roc_curve(y_arr, s, drop_intermediate=False)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thresholds[k])


def f1_threshold(y: npt.ArrayLike, score: npt.ArrayLike) -> float:
    """Threshold on ``score`` (predict 1 when ``score >= threshold``) maximising F1.

    The criterion of the ML benchmark's ``thr_f1`` candidates
    (``TunedThresholdClassifierCV(scoring="f1")``), here on the training events directly.
    Single-class inputs are handled as in :func:`youden_threshold`.
    """
    y_arr = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=np.float64)
    if y_arr.shape != s.shape:
        raise ConfigurationError("one score per label")
    if not np.all(np.isfinite(s)):
        raise ConfigurationError("scores must be finite")
    classes = np.unique(y_arr)
    if classes.size < 2:
        return float("inf") if classes.size == 0 or classes[0] == 0 else float("-inf")
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y_arr[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]  # predict 1 for score >= s_sorted[i]
    tp, fp, cand = tp[last], fp[last], s_sorted[last]
    f1 = 2 * tp / (2 * tp + fp + (y_arr.sum() - tp))
    return float(cand[int(np.argmax(f1))])


class ThresholdRule(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Unstable (1) iff ``direction * X[:, column] >= threshold_``.

    Attributes:
        column: column of ``X`` holding the score.
        direction: +1 if a larger score means less stable, -1 if a smaller one does.
        criterion: ``"youden"`` (:func:`youden_threshold`) or ``"f1"`` (:func:`f1_threshold`).
        threshold_ / best_threshold_: the fitted threshold (on the signed score).
    """

    def __init__(self, column: int = 0, direction: int = 1, criterion: str = "youden") -> None:
        self.column = column
        self.direction = direction
        self.criterion = criterion

    def _score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        if self.direction not in (1, -1):
            raise ConfigurationError("direction must be +1 or -1")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        return float(self.direction) * arr[:, self.column]

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> ThresholdRule:
        if self.criterion not in CRITERIA:
            raise ConfigurationError(f"unknown criterion {self.criterion!r}; use {CRITERIA}")
        rule = youden_threshold if self.criterion == "youden" else f1_threshold
        self.threshold_ = rule(y, self._score(X))
        self.best_threshold_ = self.threshold_
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The signed raw score (threshold-free)."""
        return self._score(X)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        check_is_fitted(self, "threshold_")
        return (self._score(X) >= self.threshold_).astype(np.int64)


@dataclass(frozen=True)
class RuleCandidate(Candidate):
    """A fixed (unregistered) estimator as a benchmark candidate.

    ``model`` is the candidate's name in reports; ``build`` returns a fresh clone of
    ``estimator`` (no imbalance handling: a threshold rule has nothing to rebalance).
    """

    estimator: Any = None

    @property
    def name(self) -> str:
        return self.model

    def build(self, seed: int, **params: Any) -> Any:
        if params:
            raise ConfigurationError(f"candidate {self.model!r} takes no tuned parameters")
        if self.estimator is None:
            raise ConfigurationError(f"candidate {self.model!r} has no estimator")
        return clone(self.estimator)

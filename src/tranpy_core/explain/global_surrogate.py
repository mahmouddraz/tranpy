"""Global surrogate: a logistic regression trained to mimic a model's predictions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from tranpy_core.errors import ModelError
from tranpy_core.explain.base import feature_names_or_default

__all__ = ["GlobalSurrogate", "fit_surrogate"]


@dataclass(frozen=True, eq=False)
class GlobalSurrogate:
    """Fitted surrogate and how faithfully it reproduces the model.

    Attributes:
        pipeline: StandardScaler + LogisticRegression trained on the model's predictions.
        fidelity: fraction of evaluation samples where surrogate and model agree.
        accuracy: surrogate accuracy against true labels (NaN if none were given).
        coefficients: logistic-regression coefficients on standardised features, so their
            magnitudes are comparable across features with different units.
        feature_names: one name per feature.
    """

    pipeline: Pipeline
    fidelity: float
    accuracy: float
    coefficients: npt.NDArray[np.float64]
    feature_names: tuple[str, ...]

    def importance(self) -> npt.NDArray[np.float64]:
        return np.abs(self.coefficients)

    def ranking(self) -> tuple[str, ...]:
        order = np.argsort(-self.importance(), kind="stable")
        return tuple(self.feature_names[i] for i in order)

    def top_features(self, k: int = 10) -> pd.DataFrame:
        imp = self.importance()
        order = np.argsort(-imp, kind="stable")[:k]
        return pd.DataFrame(
            {
                "feature": [self.feature_names[i] for i in order],
                "importance": imp[order],
                "coefficient": self.coefficients[order],
            }
        )


def fit_surrogate(
    model: Any,
    X_train: npt.ArrayLike,
    X_eval: npt.ArrayLike,
    *,
    y_eval: npt.ArrayLike | None = None,
    feature_names: Sequence[str] | None = None,
    max_iter: int = 5000,
) -> GlobalSurrogate:
    """Fit a logistic-regression surrogate on ``model.predict(X_train)``.

    Fidelity is the agreement rate between surrogate and model on ``X_eval``.
    """
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xev = np.asarray(X_eval, dtype=np.float64)
    target = np.asarray(model.predict(Xtr))
    if np.unique(target).size < 2:
        raise ModelError(
            "the model predicts a single class on X_train; a surrogate cannot be fitted"
        )
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    pipeline.fit(Xtr, target)
    surrogate_pred = pipeline.predict(Xev)
    fidelity = float(np.mean(surrogate_pred == np.asarray(model.predict(Xev))))
    accuracy = (
        float(np.mean(surrogate_pred == np.asarray(y_eval))) if y_eval is not None else float("nan")
    )
    coef = np.asarray(pipeline[-1].coef_[0], dtype=np.float64)
    return GlobalSurrogate(
        pipeline=pipeline,
        fidelity=fidelity,
        accuracy=accuracy,
        coefficients=coef,
        feature_names=feature_names_or_default(feature_names, Xtr.shape[1]),
    )

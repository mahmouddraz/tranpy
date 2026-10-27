"""Common result type of all explanation methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy_core.errors import ModelError

__all__ = ["Explanation", "feature_names_or_default", "probability_function"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class Explanation:
    """Per-sample feature attributions for the positive class.

    Attributes:
        method: name of the method that produced the attributions.
        attributions: ``(n_samples, n_features)`` contribution of each feature.
        feature_names: one name per feature.
        base_values: ``(n_samples,)`` model output with no feature information
            (SHAP / break-down), or None if the method has no such notion (LIME).
        data: the explained inputs, ``(n_samples, n_features)``.
        metadata: method parameters (seed, sample counts, ...).
    """

    method: str
    attributions: FloatArray
    feature_names: tuple[str, ...]
    data: FloatArray
    base_values: FloatArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        attr = np.array(self.attributions, dtype=np.float64)
        data = np.array(self.data, dtype=np.float64)
        if attr.ndim != 2 or attr.shape != data.shape:
            raise ModelError(
                f"attributions {attr.shape} and data {data.shape} must have the same 2-D shape"
            )
        if attr.shape[1] != len(self.feature_names):
            raise ModelError(
                f"{len(self.feature_names)} feature names for {attr.shape[1]} attributions"
            )
        attr.setflags(write=False)
        data.setflags(write=False)
        object.__setattr__(self, "attributions", attr)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if self.base_values is not None:
            base = np.array(self.base_values, dtype=np.float64).reshape(-1)
            base.setflags(write=False)
            object.__setattr__(self, "base_values", base)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_samples(self) -> int:
        return int(self.attributions.shape[0])

    def importance(self) -> FloatArray:
        """Global importance: mean absolute attribution per feature."""
        return np.asarray(np.abs(self.attributions).mean(axis=0), dtype=np.float64)

    def ranking(self) -> tuple[str, ...]:
        """Feature names by decreasing importance (ties keep feature order)."""
        order = np.argsort(-self.importance(), kind="stable")
        return tuple(self.feature_names[i] for i in order)

    def top_features(self, k: int = 10) -> pd.DataFrame:
        """The ``k`` most important features with their importance."""
        imp = self.importance()
        order = np.argsort(-imp, kind="stable")[:k]
        return pd.DataFrame(
            {"feature": [self.feature_names[i] for i in order], "importance": imp[order]}
        )

    def to_frame(self) -> pd.DataFrame:
        """Attributions as a DataFrame (rows = samples, columns = features)."""
        return pd.DataFrame(self.attributions, columns=list(self.feature_names))


def feature_names_or_default(names: Sequence[str] | None, n_features: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"x{i}" for i in range(n_features))
    if len(names) != n_features:
        raise ModelError(f"{len(names)} feature names for {n_features} features")
    return tuple(names)


def probability_function(model: Any) -> Any:
    """``X -> P(positive)`` for a fitted classifier with ``predict_proba``."""
    if not hasattr(model, "predict_proba"):
        raise ModelError(
            f"{type(model).__name__} has no predict_proba; "
            "this explanation method needs probabilities"
        )

    def positive_probability(X: npt.ArrayLike) -> FloatArray:
        return np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1]

    return positive_probability

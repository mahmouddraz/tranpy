"""SHAP attributions (optional ``[explain]`` extra)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from tranpy_core._optional import require
from tranpy_core.explain.base import Explanation, feature_names_or_default, probability_function
from tranpy_core.random import Seed, spawn_seed

__all__ = ["explain_shap"]


def explain_shap(
    model: Any,
    X: npt.ArrayLike,
    *,
    background: npt.ArrayLike,
    n_background: int = 100,
    feature_names: Sequence[str] | None = None,
    seed: Seed = 0,
) -> Explanation:
    """SHAP values of ``P(positive)`` for the rows of ``X``.

    The background distribution is a seeded sample of ``n_background`` rows from
    ``background`` (typically the training data). Attributions satisfy local accuracy:
    ``base_values + attributions.sum(axis=1)`` equals the predicted probability.
    """
    shap = require("shap", "explain")
    X_arr = np.asarray(X, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    random_state = spawn_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        sample = shap.sample(bg, n_background, random_state=random_state)
        explainer = shap.Explainer(probability_function(model), sample, seed=random_state)
        values = explainer(X_arr)
    return Explanation(
        method="shap",
        attributions=np.asarray(values.values, dtype=np.float64),
        base_values=np.asarray(values.base_values, dtype=np.float64),
        data=X_arr,
        feature_names=feature_names_or_default(feature_names, X_arr.shape[1]),
        metadata={
            "n_background": int(sample.shape[0]),
            "seed": random_state,
            "algorithm": type(explainer).__name__,
        },
    )

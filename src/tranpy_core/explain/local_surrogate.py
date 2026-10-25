"""LIME attributions (optional ``[explain]`` extra)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from tranpy_core._optional import require
from tranpy_core.errors import ModelError
from tranpy_core.explain.base import Explanation, feature_names_or_default
from tranpy_core.random import Seed, spawn_seed

__all__ = ["explain_lime"]


def explain_lime(
    model: Any,
    X: npt.ArrayLike,
    *,
    background: npt.ArrayLike,
    num_samples: int = 5000,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] = ("negative", "positive"),
    seed: Seed = 0,
) -> Explanation:
    """LIME weights for the positive class, one local linear model per row of ``X``.

    ``background`` (typically the training data) sets LIME's discretisation and
    perturbation statistics. Rows are explained in order with one seeded explainer, so the
    result is reproducible for a given seed and row order. A global ranking is the mean
    absolute weight over the explained rows (:meth:`Explanation.importance`).
    """
    lime_tabular = require("lime.lime_tabular", "explain")
    if not hasattr(model, "predict_proba"):
        raise ModelError(f"{type(model).__name__} has no predict_proba; LIME needs probabilities")
    X_arr = np.asarray(X, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    names = feature_names_or_default(feature_names, X_arr.shape[1])
    random_state = spawn_seed(seed)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=bg,
        feature_names=list(names),
        class_names=list(class_names),
        mode="classification",
        random_state=random_state,
    )
    weights = np.zeros_like(X_arr)
    intercepts = np.zeros(X_arr.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        for i, row in enumerate(X_arr):
            exp = explainer.explain_instance(
                row, model.predict_proba, num_features=X_arr.shape[1], num_samples=num_samples
            )
            for j, w in exp.local_exp[1]:
                weights[i, j] = w
            intercepts[i] = exp.intercept[1]
    return Explanation(
        method="lime",
        attributions=weights,
        data=X_arr,
        feature_names=names,
        metadata={
            "num_samples": num_samples,
            "seed": random_state,
            "intercepts": intercepts.tolist(),
        },
    )

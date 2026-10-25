"""Break-down attributions via DALEX (optional ``[explain]`` extra)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy_core._optional import require
from tranpy_core.explain.base import Explanation, feature_names_or_default, probability_function

__all__ = ["explain_breakdown"]


def explain_breakdown(
    model: Any,
    X: npt.ArrayLike,
    *,
    background: npt.ArrayLike,
    feature_names: Sequence[str] | None = None,
) -> Explanation:
    """Break-down contributions of each feature to ``P(positive)`` for the rows of ``X``.

    ``background`` is the reference data; ``base_values`` is the mean prediction on it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        dx = require("dalex", "explain")
    X_arr = np.asarray(X, dtype=np.float64)
    names = feature_names_or_default(feature_names, X_arr.shape[1])
    bg = pd.DataFrame(np.asarray(background, dtype=np.float64), columns=list(names))
    predict = probability_function(model)
    explainer = dx.Explainer(
        model,
        data=bg,
        predict_function=lambda m, d: predict(np.asarray(d)),
        label="model",
        verbose=False,
    )
    attributions = np.zeros_like(X_arr)
    base = np.zeros(X_arr.shape[0])
    lookup = {n: j for j, n in enumerate(names)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for i, row in enumerate(X_arr):
            parts = explainer.predict_parts(
                pd.DataFrame([row], columns=list(names)), type="break_down"
            )
            result = parts.result
            base[i] = float(
                result.loc[result["variable_name"] == "intercept", "contribution"].iloc[0]
            )
            for name, contribution in zip(
                result["variable_name"], result["contribution"], strict=True
            ):
                if name in lookup:
                    attributions[i, lookup[name]] = contribution
    return Explanation(
        method="breakdown",
        attributions=attributions,
        base_values=base,
        data=X_arr,
        feature_names=names,
    )

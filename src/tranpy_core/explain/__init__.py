"""Model explanations. SHAP, LIME and break-down need the ``[explain]`` extra."""

from tranpy_core.explain.base import Explanation
from tranpy_core.explain.breakdown import explain_breakdown
from tranpy_core.explain.global_surrogate import GlobalSurrogate, fit_surrogate
from tranpy_core.explain.local_surrogate import explain_lime
from tranpy_core.explain.shapley import explain_shap

__all__ = [
    "Explanation",
    "GlobalSurrogate",
    "explain_breakdown",
    "explain_lime",
    "explain_shap",
    "fit_surrogate",
]

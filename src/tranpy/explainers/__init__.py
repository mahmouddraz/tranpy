"""
TranPy Explainers Module

Provides explainability methods for understanding model predictions.
"""

try:
    from .lime import LIMEExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

try:
    from .shap import SHAPExplainer
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from .dalex import BreakdownExplainer, SurrogateExplainer
    _DALEX_AVAILABLE = True
except ImportError:
    _DALEX_AVAILABLE = False

__all__ = []

if _LIME_AVAILABLE:
    __all__.append('LIMEExplainer')

if _SHAP_AVAILABLE:
    __all__.append('SHAPExplainer')

if _DALEX_AVAILABLE:
    __all__.extend(['BreakdownExplainer', 'SurrogateExplainer'])

"""
TranPy: Power System Transient Stability Analysis

A modern Python library for power system stability prediction and explainability.

Example usage:
    >>> from tranpy.datasets import load_newengland
    >>> from tranpy.models import SVMClassifier
    >>>
    >>> # Load data
    >>> X_train, X_test, y_train, y_test = load_newengland(test_size=0.2)
    >>>
    >>> # Train model
    >>> model = SVMClassifier()
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Evaluate
    >>> accuracy = model.score(X_test, y_test)
"""

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("tranpy-ai")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except ImportError:
    # Python < 3.8
    __version__ = "0.0.0+unknown"

# Import main modules
from . import datasets
from . import models
from . import explainers

__all__ = ['datasets', 'models', 'explainers', '__version__']

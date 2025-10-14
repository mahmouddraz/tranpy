"""
TranPy Models Module

Provides sklearn-compatible classifiers for power system stability prediction.
"""

from .base import StabilityClassifier
from .classical import (
    SVMClassifier,
    MLPClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GaussianNB,
    KNeighborsClassifier,
    GaussianProcessClassifier,
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    DummyClassifier,
    _XGB_AVAILABLE,
    _LGBM_AVAILABLE
)
from .pretrained import load_pretrained, list_pretrained_models, save_model, load_model

# Import optional models if available
if _XGB_AVAILABLE:
    from .classical import XGBClassifier

if _LGBM_AVAILABLE:
    from .classical import LGBMClassifier

# Try to import neural network models (require TensorFlow)
try:
    from .neural import DNNClassifier, RNNClassifier
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

# Try to import ensemble models
try:
    from .ensemble import EnsembleClassifier
    _ENSEMBLE_AVAILABLE = True
except ImportError:
    _ENSEMBLE_AVAILABLE = False

__all__ = [
    'StabilityClassifier',
    # Classical Models
    'SVMClassifier',
    'MLPClassifier',
    'DecisionTreeClassifier',
    # Ensemble Models
    'RandomForestClassifier',
    'AdaBoostClassifier',
    'ExtraTreesClassifier',
    'GradientBoostingClassifier',
    # Naive Bayes & Neighbors
    'GaussianNB',
    'KNeighborsClassifier',
    'GaussianProcessClassifier',
    # Linear Models
    'LogisticRegression',
    'RidgeClassifier',
    'SGDClassifier',
    # Discriminant Analysis
    'LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis',
    # Baseline
    'DummyClassifier',
    # Pretrained
    'load_pretrained',
    'list_pretrained_models',
    'save_model',
    'load_model'
]

if _XGB_AVAILABLE:
    __all__.append('XGBClassifier')

if _LGBM_AVAILABLE:
    __all__.append('LGBMClassifier')

if _NEURAL_AVAILABLE:
    __all__.extend(['DNNClassifier', 'RNNClassifier'])

if _ENSEMBLE_AVAILABLE:
    __all__.append('EnsembleClassifier')

"""Domain-agnostic pipeline for labelled tabular and time-series data.

Layers: :mod:`~tranpy_core.data` (containers, storage, splits) ->
:mod:`~tranpy_core.features` (signal -> feature table) -> :mod:`~tranpy_core.models`
(registry of scikit-learn estimators) -> :mod:`~tranpy_core.evaluation` ->
:mod:`~tranpy_core.explain`; :mod:`~tranpy_core.experiment` ties them together.

This package never imports :mod:`tranpy` (the power-system layer built on top of it).
"""

from tranpy_core.data import SignalDataset, Split, TabularDataset, kfold_splits, train_test_split
from tranpy_core.errors import (
    ChecksumError,
    ConfigurationError,
    DataError,
    MissingDependencyError,
    ModelError,
    SimulationError,
    TranpyError,
)
from tranpy_core.evaluation import Scores, cross_validate, evaluate
from tranpy_core.experiment import ExperimentResult, ExperimentSpec, ModelConfig, run_experiment
from tranpy_core.models import available_models, make_estimator, make_pipeline

__all__ = [
    "ChecksumError",
    "ConfigurationError",
    "DataError",
    "ExperimentResult",
    "ExperimentSpec",
    "MissingDependencyError",
    "ModelConfig",
    "ModelError",
    "Scores",
    "SignalDataset",
    "SimulationError",
    "Split",
    "TabularDataset",
    "TranpyError",
    "available_models",
    "cross_validate",
    "evaluate",
    "kfold_splits",
    "make_estimator",
    "make_pipeline",
    "run_experiment",
    "train_test_split",
]

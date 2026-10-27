"""Model registry: name -> estimator class, default hyperparameters and seed parameter.

The registry hands out *plain* scikit-learn compatible estimators (or a Pipeline with a
scaler in front), so ``clone``, ``set_params``, cross-validation and grid search behave
exactly as documented by scikit-learn. There is no wrapper class per model.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tranpy_core._optional import is_available
from tranpy_core.errors import ConfigurationError, MissingDependencyError, ModelError
from tranpy_core.random import Seed, spawn_seed

__all__ = [
    "ModelSpec",
    "available_models",
    "get_model_spec",
    "make_estimator",
    "make_pipeline",
    "register_model",
]


@dataclass(frozen=True)
class ModelSpec:
    """How to build one model.

    Attributes:
        name: registry key, lower_snake_case.
        estimator: dotted path of the estimator class, imported lazily.
        defaults: hyperparameters that differ from the library defaults.
        seed_param: constructor argument that receives the seed, or None if deterministic.
        extra: optional-dependency extra providing the backend, or None for scikit-learn.
        requires: importable module that must be installed (defaults to the estimator's
            top-level package).
        scale_features: whether :func:`make_pipeline` puts a StandardScaler in front.
        strict_params: reject parameter names the estimator does not declare. Disable for
            libraries that accept arbitrary keyword arguments (XGBoost, LightGBM).
        description: one line for listings.
    """

    name: str
    estimator: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    seed_param: str | None = "random_state"
    extra: str | None = None
    requires: str | None = None
    scale_features: bool = True
    strict_params: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        problems = []
        if (
            not self.name
            or not self.name.replace("_", "").isalnum()
            or self.name != self.name.lower()
        ):
            problems.append(f"name must be lower_snake_case, got {self.name!r}")
        if "." not in self.estimator:
            problems.append(f"estimator must be a dotted path, got {self.estimator!r}")
        if problems:
            raise ConfigurationError("invalid model spec:", tuple(problems))
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))

    @property
    def module(self) -> str:
        return self.estimator.rsplit(".", 1)[0]

    def is_available(self) -> bool:
        return is_available(self.requires or self.module.split(".")[0])

    def load_class(self) -> type[BaseEstimator]:
        module_name, class_name = self.estimator.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            hint = f" Install it with: pip install 'tranpy-ai[{self.extra}]'" if self.extra else ""
            raise MissingDependencyError(
                f"model {self.name!r} needs {module_name!r}, which is not installed.{hint}"
            ) from exc
        cls: type[BaseEstimator] = getattr(module, class_name)
        return cls


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec, *, replace: bool = False) -> ModelSpec:
    """Add ``spec`` to the registry (extension point for new models)."""
    if spec.name in _REGISTRY and not replace:
        raise ConfigurationError(f"model {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ConfigurationError(
            f"unknown model {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def available_models(*, installed_only: bool = False) -> tuple[str, ...]:
    """Registered model names, optionally only those whose backend is installed."""
    names = sorted(_REGISTRY)
    if installed_only:
        names = [n for n in names if _REGISTRY[n].is_available()]
    return tuple(names)


def make_estimator(name: str, *, seed: Seed = 0, **params: Any) -> BaseEstimator:
    """Build the estimator registered as ``name``.

    ``params`` override the registered defaults. Unknown parameter names raise
    :class:`ConfigurationError` instead of being ignored. The seed is passed to the
    estimator's seed parameter unless that parameter is given explicitly.
    """
    spec = get_model_spec(name)
    cls = spec.load_class()
    merged = {**spec.defaults, **params}
    if spec.seed_param is not None and spec.seed_param not in params:
        merged[spec.seed_param] = spawn_seed(seed)
    unknown = sorted(set(merged) - set(cls().get_params(deep=False))) if spec.strict_params else []
    if unknown:
        raise ConfigurationError(f"unknown parameters for model {name!r}: {', '.join(unknown)}")
    try:
        return cls(**merged)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"cannot build model {name!r}: {exc}") from exc


def make_pipeline(
    name: str, *, seed: Seed = 0, scale: bool | None = None, **params: Any
) -> Pipeline:
    """Estimator ``name`` behind a StandardScaler (fitted on training data only).

    Args:
        name: registered model name.
        seed: seed for the estimator.
        scale: include the scaler; defaults to the model's ``scale_features``.
        **params: estimator hyperparameters (address them later as ``model__<param>``).
    """
    spec = get_model_spec(name)
    estimator = make_estimator(name, seed=seed, **params)
    use_scaler = spec.scale_features if scale is None else scale
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())] if use_scaler else []
    steps.append(("model", estimator))
    return Pipeline(steps)


def _builtin(name: str, estimator: str, description: str, **kwargs: Any) -> None:
    register_model(ModelSpec(name=name, estimator=estimator, description=description, **kwargs))


_builtin(
    "svm", "sklearn.svm.SVC", "Support vector machine (RBF kernel)", defaults={"probability": True}
)
_builtin("mlp", "sklearn.neural_network.MLPClassifier", "Multi-layer perceptron")
_builtin("decision_tree", "sklearn.tree.DecisionTreeClassifier", "Decision tree")
_builtin("random_forest", "sklearn.ensemble.RandomForestClassifier", "Random forest")
_builtin("adaboost", "sklearn.ensemble.AdaBoostClassifier", "AdaBoost")
_builtin("extra_trees", "sklearn.ensemble.ExtraTreesClassifier", "Extremely randomised trees")
_builtin("gradient_boosting", "sklearn.ensemble.GradientBoostingClassifier", "Gradient boosting")
_builtin("gaussian_nb", "sklearn.naive_bayes.GaussianNB", "Gaussian naive Bayes", seed_param=None)
_builtin("knn", "sklearn.neighbors.KNeighborsClassifier", "k-nearest neighbours", seed_param=None)
_builtin(
    "gaussian_process", "sklearn.gaussian_process.GaussianProcessClassifier", "Gaussian process"
)
_builtin("logistic_regression", "sklearn.linear_model.LogisticRegression", "Logistic regression")
_builtin(
    "ridge",
    "sklearn.linear_model.RidgeClassifier",
    "Ridge classifier (no probabilities)",
    seed_param=None,
)
_builtin(
    "sgd",
    "sklearn.linear_model.SGDClassifier",
    "Linear model trained by SGD",
    defaults={"loss": "log_loss"},
)
_builtin(
    "lda",
    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "Linear discriminant analysis",
    seed_param=None,
)
_builtin(
    "qda",
    "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
    "Quadratic discriminant analysis (regularised)",
    defaults={"reg_param": 0.01},
    seed_param=None,
)
_builtin(
    "dummy",
    "sklearn.dummy.DummyClassifier",
    "Majority-class baseline",
    defaults={"strategy": "most_frequent"},
    scale_features=False,
)
_builtin(
    "xgboost",
    "xgboost.XGBClassifier",
    "XGBoost gradient boosting",
    defaults={"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3},
    extra="boosting",
    strict_params=False,
)
_builtin(
    "lightgbm",
    "lightgbm.LGBMClassifier",
    "LightGBM gradient boosting",
    defaults={"n_estimators": 100, "max_depth": -1, "learning_rate": 0.1, "verbose": -1},
    extra="boosting",
    strict_params=False,
)

for _name, _cls, _desc in [
    ("dense_network", "DenseNetworkClassifier", "Fully connected neural network (Keras)"),
    ("lstm_network", "LSTMNetworkClassifier", "LSTM with attention pooling (Keras)"),
    ("tcn_network", "TCNNetworkClassifier", "Temporal convolutional network (Keras)"),
    ("transformer_network", "TransformerNetworkClassifier", "Transformer encoder (Keras)"),
]:
    _builtin(
        _name,
        f"tranpy_core.models.neural.{_cls}",
        _desc,
        extra="neural",
        requires="tensorflow",
    )

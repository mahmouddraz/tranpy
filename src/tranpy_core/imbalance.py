"""Class-imbalance handling with established library methods (configuration only).

Nothing here is a new method: every option maps one-to-one to scikit-learn or
imbalanced-learn (optional ``[imbalance]`` extra), so the options can be compared under the
same cross-validation and the best one kept.

==================  ==========================================================================
``method``          library object
==================  ==========================================================================
``none``            the registered estimator unchanged
``class_weight``    ``class_weight="balanced"`` where the estimator has it (scikit-learn,
                    LightGBM, the Keras models); otherwise the fit is given
                    ``sklearn.utils.class_weight.compute_sample_weight("balanced", y)``
                    (e.g. XGBoost), which weights the loss identically
``undersample``     ``imblearn.under_sampling.RandomUnderSampler``
``smote``           ``imblearn.over_sampling.SMOTE``
``smote_tomek``     ``imblearn.combine.SMOTETomek``
==================  ==========================================================================

Imbalance-specific ensembles (``balanced_random_forest``, ``easy_ensemble``) are
registered models and used with ``method="none"``.

Resamplers sit inside an ``imblearn.pipeline.Pipeline``, which applies them in ``fit``
only, so they see the training fold and never the test fold.

``threshold`` wraps the result in ``sklearn.model_selection.TunedThresholdClassifierCV``,
which picks the decision threshold that maximises a scikit-learn scorer on inner folds of
the training data, instead of the fixed 0.5.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted, has_fit_parameter

from tranpy_core._optional import require
from tranpy_core.errors import ConfigurationError
from tranpy_core.models.registry import get_model_spec, make_estimator
from tranpy_core.random import Seed, spawn_seed

__all__ = [
    "METHODS",
    "BalancedSampleWeight",
    "ImbalanceSpec",
    "inclusion_weights",
    "make_imbalance_pipeline",
]

METHODS = ("none", "class_weight", "undersample", "smote", "smote_tomek")

_SAMPLERS = {
    "undersample": "imblearn.under_sampling.RandomUnderSampler",
    "smote": "imblearn.over_sampling.SMOTE",
    "smote_tomek": "imblearn.combine.SMOTETomek",
}


@dataclass(frozen=True)
class ImbalanceSpec:
    """How to handle class imbalance for one model.

    Attributes:
        method: one of :data:`METHODS`.
        threshold: scikit-learn scorer name to tune the decision threshold on (for example
            ``"f1"``, ``"balanced_accuracy"``, ``"matthews_corrcoef"``), or None for 0.5.
        threshold_folds: inner stratified folds used by ``TunedThresholdClassifierCV``.
        sampler_params: keyword arguments of the imbalanced-learn sampler.
    """

    method: str = "none"
    threshold: str | None = None
    threshold_folds: int = 5
    sampler_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ConfigurationError(f"unknown imbalance method {self.method!r}; use {METHODS}")
        if self.threshold_folds < 2:
            raise ConfigurationError("threshold_folds must be at least 2")
        if self.sampler_params and self.method not in _SAMPLERS:
            raise ConfigurationError(f"sampler_params given but method {self.method!r} has none")


class BalancedSampleWeight(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Fit ``estimator`` with ``compute_sample_weight("balanced", y)``.

    For estimators without a ``class_weight`` parameter but with ``sample_weight`` in
    ``fit`` (XGBoost, gradient boosting). The weights are those of
    ``class_weight="balanced"``: ``n_samples / (n_classes * count(class))``.
    """

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> BalancedSampleWeight:
        y_arr = np.asarray(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_arr, sample_weight=compute_sample_weight("balanced", y_arr))
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: npt.ArrayLike) -> Any:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X: npt.ArrayLike) -> Any:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


def _class_weighted(estimator: Any, name: str) -> Any:
    if "class_weight" in estimator.get_params(deep=False):
        return estimator.set_params(class_weight="balanced")
    if has_fit_parameter(estimator, "sample_weight"):
        return BalancedSampleWeight(estimator)
    raise ConfigurationError(
        f"model {name!r} accepts neither class_weight nor sample_weight; "
        f"use a resampling method ({', '.join(_SAMPLERS)}) instead"
    )


def _sampler(spec: ImbalanceSpec, seed: Seed) -> Any:
    module, _, cls_name = _SAMPLERS[spec.method].rpartition(".")
    cls = getattr(require(module, "imbalance"), cls_name)
    return cls(random_state=spawn_seed(seed), **dict(spec.sampler_params))


def make_imbalance_pipeline(
    name: str,
    spec: ImbalanceSpec | None = None,
    *,
    seed: Seed = 0,
    scale: bool | None = None,
    **params: Any,
) -> Any:
    """Registered model ``name`` with the imbalance handling of ``spec``.

    The steps are ``scale`` (if the model asks for it), the sampler (resampling methods
    only) and ``model``, so model hyperparameters stay addressable as ``model__<param>``.
    With ``spec.threshold`` the pipeline is wrapped in ``TunedThresholdClassifierCV``
    (address them as ``estimator__model__<param>``).
    """
    spec = spec or ImbalanceSpec()
    estimator = make_estimator(name, seed=seed, **params)
    if spec.method == "class_weight":
        estimator = _class_weighted(estimator, name)
    use_scaler = get_model_spec(name).scale_features if scale is None else scale
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())] if use_scaler else []
    if spec.method in _SAMPLERS:
        steps.append(("sample", _sampler(spec, seed)))
        pipeline_cls = require("imblearn.pipeline", "imbalance").Pipeline
    else:
        pipeline_cls = Pipeline
    steps.append(("model", estimator))
    pipeline = pipeline_cls(steps)
    if spec.threshold is None:
        return pipeline
    inner = StratifiedKFold(spec.threshold_folds, shuffle=True, random_state=spawn_seed(seed))
    return TunedThresholdClassifierCV(pipeline, scoring=spec.threshold, cv=inner)


def inclusion_weights(
    strata: npt.ArrayLike,
    labels: npt.ArrayLike,
    population: Mapping[tuple[Any, int], int],
) -> npt.NDArray[np.float64]:
    """Inverse inclusion-probability (Horvitz-Thompson) weights of a stratified subsample.

    A sample that keeps ``n_s`` of the ``N_s`` events of stratum ``s`` (one stratum is
    a ``(group, label)`` pair, e.g. ``(case, 0)`` for the stable events of one hour) gives
    each kept event the weight ``N_s / n_s``. Metrics computed with these weights estimate
    the metric on the full population, not on the subsample's class mix.

    Args:
        strata: group of every sampled event (e.g. the case).
        labels: label of every sampled event.
        population: number of events per ``(group, label)`` in the full population.

    Raises:
        ConfigurationError: if a sampled stratum is missing from ``population`` or would
            have more sampled events than population events.
    """
    groups = np.asarray(strata, dtype=object)
    y = np.asarray(labels).astype(int)
    keys = list(zip(groups.tolist(), y.tolist(), strict=True))
    counts: dict[tuple[Any, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = np.empty(len(keys), dtype=np.float64)
    for key, n in counts.items():
        if key not in population:
            raise ConfigurationError(f"stratum {key!r} missing from the population counts")
        if population[key] < n:
            raise ConfigurationError(
                f"stratum {key!r}: {n} sampled but only {population[key]} in the population"
            )
    for i, key in enumerate(keys):
        weights[i] = population[key] / counts[key]
    return weights

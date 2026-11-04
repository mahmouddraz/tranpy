"""Class-imbalance options map to library objects and never touch the test data."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

from tranpy_core.errors import ConfigurationError
from tranpy_core.evaluation import evaluate, evaluate_by_group
from tranpy_core.imbalance import (
    BalancedSampleWeight,
    ImbalanceSpec,
    inclusion_weights,
    make_imbalance_pipeline,
)
from tranpy_core.models.neural import _keras_class_weight

imblearn = pytest.importorskip("imblearn")


@pytest.fixture(scope="module")
def skewed() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=4, weights=[0.9], random_state=3
    )
    return X, y


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ImbalanceSpec(method="magic")
    with pytest.raises(ConfigurationError):
        ImbalanceSpec(method="class_weight", sampler_params={"k_neighbors": 3})


def test_none_is_the_plain_registered_pipeline() -> None:
    pipe = make_imbalance_pipeline("random_forest", seed=0)
    assert isinstance(pipe, Pipeline)
    assert pipe.named_steps["model"].class_weight is None


def test_class_weight_uses_the_estimator_parameter() -> None:
    pipe = make_imbalance_pipeline("random_forest", ImbalanceSpec(method="class_weight"))
    assert pipe.named_steps["model"].class_weight == "balanced"


def test_class_weight_falls_back_to_balanced_sample_weight(skewed) -> None:
    X, y = skewed
    pipe = make_imbalance_pipeline("gradient_boosting", ImbalanceSpec(method="class_weight"))
    assert isinstance(pipe.named_steps["model"], BalancedSampleWeight)
    pipe.fit(X, y)
    assert pipe.predict_proba(X).shape == (len(y), 2)


def test_class_weight_without_support_points_to_resampling() -> None:
    with pytest.raises(ConfigurationError, match="resampling"):
        make_imbalance_pipeline("knn", ImbalanceSpec(method="class_weight"))


@pytest.mark.parametrize(
    ("method", "cls"),
    [
        ("undersample", "RandomUnderSampler"),
        ("smote", "SMOTE"),
        ("smote_tomek", "SMOTETomek"),
    ],
)
def test_resampling_methods_use_imblearn_pipeline(method: str, cls: str, skewed) -> None:
    X, y = skewed
    pipe = make_imbalance_pipeline("logistic_regression", ImbalanceSpec(method=method))
    assert isinstance(pipe, imblearn.pipeline.Pipeline)
    assert type(pipe.named_steps["sample"]).__name__ == cls
    pipe.fit(X[:400], y[:400])
    # samplers act in fit only: prediction returns one row per test sample
    assert pipe.predict(X[400:]).shape == (200,)


def test_undersampler_sees_training_rows_only(skewed) -> None:
    X, y = skewed
    pipe = make_imbalance_pipeline("logistic_regression", ImbalanceSpec(method="undersample"))
    pipe.fit(X[:400], y[:400])
    kept = pipe.named_steps["sample"].sample_indices_
    assert kept.max() < 400
    assert np.bincount(y[:400][kept]).tolist() == [int(y[:400].sum())] * 2


def test_threshold_is_tuned_on_inner_folds(skewed) -> None:
    X, y = skewed
    model = make_imbalance_pipeline(
        "logistic_regression", ImbalanceSpec(method="class_weight", threshold="f1")
    ).fit(X, y)
    assert isinstance(model, TunedThresholdClassifierCV)
    assert 0.0 < model.best_threshold_ < 1.0


def test_imbalance_ensembles_are_registered(skewed) -> None:
    X, y = skewed
    for name in ("balanced_random_forest", "easy_ensemble"):
        pipe = make_imbalance_pipeline(name, seed=1, n_estimators=5).fit(X, y)
        assert pipe.predict(X).shape == y.shape


def test_inclusion_weights_restore_population_counts() -> None:
    cases = np.array(["a", "a", "a", "b", "b", "b"])
    labels = np.array([0, 0, 1, 0, 1, 1])
    population = {("a", 0): 10, ("a", 1): 1, ("b", 0): 7, ("b", 1): 2}
    w = inclusion_weights(cases, labels, population)
    for (case, label), n in population.items():
        mask = (cases == case) & (labels == label)
        assert w[mask].sum() == pytest.approx(n)


def test_inclusion_weights_reject_inconsistent_counts() -> None:
    with pytest.raises(ConfigurationError, match="missing"):
        inclusion_weights(["a"], [1], {("a", 0): 3})
    with pytest.raises(ConfigurationError, match="only"):
        inclusion_weights(["a", "a"], [1, 1], {("a", 1): 1})


def test_integer_weights_equal_repeated_rows(skewed) -> None:
    X, y = skewed
    model = make_imbalance_pipeline("logistic_regression").fit(X, y)
    w = np.arange(len(y)) % 3 + 1
    weighted = evaluate(model, X, y, sample_weight=w)
    repeated = evaluate(model, np.repeat(X, w, axis=0), np.repeat(y, w))
    for metric in ("accuracy", "recall", "precision", "mcc", "average_precision", "brier"):
        assert getattr(weighted, metric) == pytest.approx(getattr(repeated, metric))
    assert weighted.confusion_matrix == repeated.confusion_matrix
    assert weighted.miss_rate == pytest.approx(1 - weighted.recall)


def test_evaluate_by_group_has_overall_and_group_rows(skewed) -> None:
    X, y = skewed
    model = make_imbalance_pipeline("logistic_regression").fit(X, y)
    groups = np.where(np.arange(len(y)) % 2, "odd", "even")
    table = evaluate_by_group(model, X, y, groups)
    assert list(table.index) == ["__all__", "even", "odd"]
    assert table.loc["__all__", "n_positive"] == int(y.sum())


def test_keras_class_weight_follows_sklearn_convention() -> None:
    target = np.array([0, 0, 0, 1], dtype=np.float32)
    assert _keras_class_weight(None, target) is None
    assert _keras_class_weight("balanced", target) == pytest.approx({0: 4 / 6, 1: 2.0})
    assert _keras_class_weight({0: 1, 1: 5}, target) == {0: 1.0, 1: 5.0}
    with pytest.raises(ConfigurationError):
        _keras_class_weight("auto", target)

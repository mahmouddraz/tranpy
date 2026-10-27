from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from tranpy_core.data import TabularDataset
from tranpy_core.errors import ConfigurationError, MissingDependencyError, ModelError
from tranpy_core.evaluation import positive_scores
from tranpy_core.models import (
    ModelSpec,
    available_models,
    get_model_spec,
    make_estimator,
    make_pipeline,
    register_model,
)

CLASSICAL = [m for m in available_models(installed_only=True) if not m.endswith("_network")]


@pytest.mark.parametrize("name", CLASSICAL)
def test_every_installed_model_fits_and_scores(name: str, toy: TabularDataset) -> None:
    if name == "gaussian_process":
        pytest.importorskip("sklearn")
    pipe = make_pipeline(name, seed=0)
    pipe.fit(toy.features[:160], toy.labels[:160])
    pred = pipe.predict(toy.features[160:])
    assert set(np.unique(pred)) <= {0, 1}
    scores = positive_scores(pipe, toy.features[160:])
    assert scores.shape == (80,) and np.isfinite(scores).all()
    if hasattr(pipe, "predict_proba") and name not in ("ridge",):
        proba = pipe.predict_proba(toy.features[160:])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", [m for m in CLASSICAL if get_model_spec(m).seed_param])
def test_seed_is_applied_and_clone_keeps_it(name: str, toy: TabularDataset) -> None:
    spec = get_model_spec(name)
    est = make_estimator(name, seed=123)
    assert est.get_params()[spec.seed_param] == 123
    assert clone(est).get_params()[spec.seed_param] == 123
    a = make_pipeline(name, seed=5).fit(toy.features, toy.labels)
    b = make_pipeline(name, seed=5).fit(toy.features, toy.labels)
    np.testing.assert_array_equal(
        positive_scores(a, toy.features), positive_scores(b, toy.features)
    )


def test_set_params_takes_effect(toy: TabularDataset) -> None:
    """The old wrappers ignored set_params; registry estimators are plain sklearn objects."""
    pipe = make_pipeline("decision_tree", seed=0)
    pipe.set_params(model__max_depth=1).fit(toy.features, toy.labels)
    assert pipe[-1].get_depth() == 1
    grid = GridSearchCV(make_pipeline("knn"), {"model__n_neighbors": [1, 15]}, cv=3).fit(
        toy.features, toy.labels
    )
    assert grid.best_params_["model__n_neighbors"] in (1, 15)


def test_defaults_and_overrides() -> None:
    assert make_estimator("svm").get_params()["probability"] is True
    assert make_estimator("qda").get_params()["reg_param"] == 0.01
    assert make_estimator("dummy").get_params()["strategy"] == "most_frequent"
    assert make_estimator("svm", C=3.0).get_params()["C"] == 3.0
    explicit = make_estimator("random_forest", seed=1, random_state=9)
    assert explicit.get_params()["random_state"] == 9
    assert isinstance(make_pipeline("svm"), Pipeline)
    assert [n for n, _ in make_pipeline("svm").steps] == ["scale", "model"]
    assert [n for n, _ in make_pipeline("dummy").steps] == ["model"]
    assert [n for n, _ in make_pipeline("svm", scale=False).steps] == ["model"]


def test_unknown_names_and_parameters_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown model"):
        make_estimator("nope")
    with pytest.raises(ConfigurationError, match="unknown parameters for model 'svm': kernal"):
        make_estimator("svm", kernal="rbf")
    broken = ModelSpec(
        name="test_only_broken",
        estimator="sklearn.pipeline.Pipeline",
        seed_param=None,
        strict_params=False,
    )
    register_model(broken)
    try:
        with pytest.raises(ModelError, match="cannot build"):
            make_estimator("test_only_broken")
    finally:
        from tranpy_core.models import registry

        registry._REGISTRY.pop("test_only_broken")


def test_register_custom_model_and_missing_backend() -> None:
    spec = ModelSpec(
        name="test_only_lr", estimator="sklearn.linear_model.LogisticRegression", description="x"
    )
    register_model(spec)
    try:
        assert "test_only_lr" in available_models()
        with pytest.raises(ConfigurationError, match="already registered"):
            register_model(spec)
        register_model(spec, replace=True)
        missing = ModelSpec(
            name="test_only_missing", estimator="nonexistent_pkg.Model", extra="boosting"
        )
        register_model(missing)
        assert not missing.is_available()
        assert "test_only_missing" not in available_models(installed_only=True)
        with pytest.raises(MissingDependencyError, match=r"tranpy-ai\[boosting\]"):
            make_estimator("test_only_missing")
    finally:
        from tranpy_core.models import registry

        registry._REGISTRY.pop("test_only_lr", None)
        registry._REGISTRY.pop("test_only_missing", None)


def test_model_spec_validation() -> None:
    with pytest.raises(ConfigurationError) as info:
        ModelSpec(name="Bad Name", estimator="nodots")
    assert len(info.value.problems) == 2

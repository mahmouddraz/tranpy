from __future__ import annotations

import numpy as np
import pytest

from tranpy_core.data import TabularDataset
from tranpy_core.errors import ModelError
from tranpy_core.explain import (
    Explanation,
    explain_breakdown,
    explain_lime,
    explain_shap,
    fit_surrogate,
)
from tranpy_core.models import make_pipeline


@pytest.fixture
def fitted(toy: TabularDataset):  # type: ignore[no-untyped-def]
    return make_pipeline("random_forest", seed=0, n_estimators=30).fit(
        toy.features[:180], toy.labels[:180]
    )


def test_explanation_container() -> None:
    e = Explanation(
        method="m",
        attributions=[[1.0, -3.0], [0.0, 1.0]],
        data=[[0.0, 0.0], [1.0, 1.0]],
        feature_names=("a", "b"),
        base_values=[0.5, 0.5],
    )
    np.testing.assert_allclose(e.importance(), [0.5, 2.0])
    assert e.ranking() == ("b", "a")
    assert e.top_features(1)["feature"].tolist() == ["b"]
    assert e.to_frame().shape == (2, 2)
    assert e.n_samples == 2
    with pytest.raises(ModelError, match="same 2-D shape"):
        Explanation(method="m", attributions=[[1.0]], data=[[1.0, 2.0]], feature_names=("a",))
    with pytest.raises(ModelError, match="feature names"):
        Explanation(method="m", attributions=[[1.0]], data=[[1.0]], feature_names=("a", "b"))


@pytest.mark.requires_explain
def test_shap_local_accuracy_and_reproducibility(fitted, toy: TabularDataset) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("shap")
    X = toy.features[180:190]
    e = explain_shap(
        fitted,
        X,
        background=toy.features[:180],
        n_background=40,
        feature_names=toy.feature_names,
        seed=1,
    )
    assert e.attributions.shape == X.shape
    # local accuracy: base value + sum of attributions = predicted probability
    np.testing.assert_allclose(
        e.base_values + e.attributions.sum(axis=1), fitted.predict_proba(X)[:, 1], atol=1e-6
    )
    again = explain_shap(fitted, X, background=toy.features[:180], n_background=40, seed=1)
    np.testing.assert_array_equal(again.attributions, e.attributions)
    assert again.feature_names[0] == "x0"
    assert e.ranking()[0] in ("f0", "f1")


@pytest.mark.requires_explain
def test_lime_shapes_and_reproducibility(fitted, toy: TabularDataset) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("lime")
    X = toy.features[180:183]
    e = explain_lime(
        fitted,
        X,
        background=toy.features[:180],
        num_samples=300,
        feature_names=toy.feature_names,
        seed=2,
    )
    assert e.attributions.shape == X.shape
    assert len(e.metadata["intercepts"]) == 3
    again = explain_lime(fitted, X, background=toy.features[:180], num_samples=300, seed=2)
    np.testing.assert_array_equal(again.attributions, e.attributions)
    ridge = make_pipeline("ridge").fit(toy.features, toy.labels)
    with pytest.raises(ModelError, match="predict_proba"):
        explain_lime(ridge, X, background=toy.features)


@pytest.mark.requires_explain
def test_breakdown_contributions_add_up(fitted, toy: TabularDataset) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("dalex")
    X = toy.features[180:182]
    e = explain_breakdown(fitted, X, background=toy.features[:180], feature_names=toy.feature_names)
    np.testing.assert_allclose(
        e.base_values + e.attributions.sum(axis=1), fitted.predict_proba(X)[:, 1], atol=1e-6
    )


def test_surrogate(fitted, toy: TabularDataset) -> None:  # type: ignore[no-untyped-def]
    s = fit_surrogate(
        fitted,
        toy.features[:180],
        toy.features[180:],
        y_eval=toy.labels[180:],
        feature_names=toy.feature_names,
    )
    assert 0.5 < s.fidelity <= 1.0
    assert 0.0 <= s.accuracy <= 1.0
    assert s.coefficients.shape == (toy.n_features,)
    assert s.ranking()[0] in ("f0", "f1")
    assert list(s.top_features(2).columns) == ["feature", "importance", "coefficient"]
    assert np.isnan(fit_surrogate(fitted, toy.features[:180], toy.features[180:]).accuracy)
    constant = make_pipeline("dummy").fit(toy.features, np.zeros(toy.n_samples, dtype=int))
    with pytest.raises(ModelError, match="single class"):
        fit_surrogate(constant, toy.features, toy.features)


def test_methods_require_probabilities(toy: TabularDataset) -> None:
    from tranpy_core.explain.base import probability_function

    with pytest.raises(ModelError, match="predict_proba"):
        probability_function(make_pipeline("ridge").fit(toy.features, toy.labels))

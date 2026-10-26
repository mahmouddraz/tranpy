"""Metrics, cross-validation and the guarantees that test data never influence fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tranpy_core.data import TabularDataset, kfold_splits
from tranpy_core.errors import ModelError
from tranpy_core.evaluation import cross_validate, evaluate, positive_scores, scores_frame
from tranpy_core.models import available_models, make_pipeline


def test_evaluate_values(toy: TabularDataset) -> None:
    pipe = make_pipeline("logistic_regression").fit(toy.features, toy.labels)
    s = evaluate(pipe, toy.features, toy.labels)
    assert 0.8 < s.accuracy <= 1.0
    assert s.n_samples == toy.n_samples
    assert sum(map(sum, s.confusion_matrix)) == toy.n_samples
    assert set(s.as_dict()) >= {"accuracy", "roc_auc", "confusion_matrix"}
    assert scores_frame({"lr": s}).loc["lr", "accuracy"] == s.accuracy


def test_auc_is_nan_for_single_class(toy: TabularDataset) -> None:
    pipe = make_pipeline("logistic_regression").fit(toy.features, toy.labels)
    ones = toy.labels == 1
    assert math.isnan(evaluate(pipe, toy.features[ones], toy.labels[ones]).roc_auc)


def test_positive_scores_falls_back_to_decision_function(toy: TabularDataset) -> None:
    ridge = RidgeClassifier().fit(toy.features, toy.labels)
    np.testing.assert_array_equal(
        positive_scores(ridge, toy.features), ridge.decision_function(toy.features)
    )
    with pytest.raises(ModelError, match="neither"):
        positive_scores(object(), toy.features)

    class ThreeClass:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.ones((len(X), 3)) / 3

    with pytest.raises(ModelError, match=r"\(n, 2\)"):
        positive_scores(ThreeClass(), toy.features)


def test_cross_validate_summary_and_frame(toy: TabularDataset) -> None:
    cv = cross_validate(
        make_pipeline("random_forest"), toy.features, toy.labels, n_splits=4, seed=0
    )
    assert len(cv.folds) == 4
    summary = cv.summary()
    assert 0.5 < summary["accuracy_mean"] <= 1.0
    assert summary["roc_auc_std"] >= 0
    assert len(cv.to_frame()) == 4
    explicit = cross_validate(
        make_pipeline("random_forest"),
        toy.features,
        toy.labels,
        splits=kfold_splits(toy.labels, n_splits=4, seed=0),
    )
    assert explicit.summary() == pytest.approx(summary)


def test_cross_validation_ignores_undefined_auc_folds() -> None:
    X = np.arange(20, dtype=float)[:, None]
    y = np.array([0] * 18 + [1] * 2)
    from tranpy_core.data import Split

    splits = [
        Split(train=np.arange(0, 20, 2), test=np.arange(1, 18, 2)),
        Split(train=np.arange(1, 20, 2), test=np.arange(0, 20, 2)),
    ]
    cv = cross_validate(DummyClassifier(strategy="prior"), X, y, splits=splits)
    assert math.isnan(cv.folds[0].roc_auc)
    assert not math.isnan(cv.summary()["roc_auc_mean"])
    assert (
        cross_validate(
            DummyClassifier(), X[:2], [0, 0], splits=[Split(train=[0], test=[1])]
        ).summary()["roc_auc_mean"]
        != 0.0
    )


# ---------------------------------------------------------------- leakage guarantees

SCALED = [
    m for m in available_models(installed_only=True) if not m.endswith("_network") and m != "dummy"
]


@pytest.mark.parametrize("name", SCALED)
def test_pipeline_preprocessing_sees_training_data_only(name: str, toy: TabularDataset) -> None:
    """The scaler's statistics equal those of the training rows, whatever the test rows are."""
    train, test = np.arange(160), np.arange(160, 240)
    pipe = make_pipeline(name, seed=0).fit(toy.features[train], toy.labels[train])
    scaler = pipe.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, toy.features[train].mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, toy.features[train].std(axis=0))
    shifted = toy.features.copy()
    shifted[test] += 1e6
    other = make_pipeline(name, seed=0).fit(shifted[train], toy.labels[train])
    np.testing.assert_allclose(other.named_steps["scale"].mean_, scaler.mean_)


def test_cross_validation_refits_preprocessing_per_fold(toy: TabularDataset) -> None:
    seen: list[np.ndarray] = []

    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):  # type: ignore[no-untyped-def]
            seen.append(np.asarray(X).copy())
            return super().fit(X, y, sample_weight)

    pipe = Pipeline([("scale", RecordingScaler()), ("model", DummyClassifier())])
    folds = kfold_splits(toy.labels, n_splits=3, seed=0)
    cross_validate(pipe, toy.features, toy.labels, splits=folds)
    assert len(seen) == 3
    for fold, X_fit in zip(folds, seen, strict=True):
        np.testing.assert_array_equal(X_fit, toy.features[fold.train])


@settings(max_examples=15, deadline=None)
@given(st.permutations(list(range(80))))
def test_training_predictions_do_not_depend_on_test_rows(perm: list[int]) -> None:
    """Fit on train; permuting or replacing the test rows leaves train predictions unchanged."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 4))
    y = (X[:, 0] > 0).astype(int)
    X_test = X[160:][perm]
    base = make_pipeline("logistic_regression").fit(X[:160], y[:160])
    stacked = np.vstack([X[:160], X_test])
    np.testing.assert_array_equal(base.predict_proba(stacked)[:160], base.predict_proba(X[:160]))
    # predictions are row-wise: permuting the test rows permutes their predictions
    np.testing.assert_allclose(base.predict_proba(X_test), base.predict_proba(X[160:])[perm])


def test_repeated_cross_validation_and_corrected_interval(toy: TabularDataset) -> None:
    from tranpy_core.data import repeated_kfold_splits
    from tranpy_core.errors import ConfigurationError

    single = repeated_kfold_splits(toy.labels, n_splits=4, n_repeats=1, seed=3)
    plain = kfold_splits(toy.labels, n_splits=4, seed=3)
    assert all(np.array_equal(a.test, b.test) for a, b in zip(single, plain, strict=True))
    repeated = repeated_kfold_splits(toy.labels, n_splits=4, n_repeats=3, seed=3)
    assert len(repeated) == 12
    assert not np.array_equal(repeated[0].test, repeated[4].test)
    with pytest.raises(ConfigurationError):
        repeated_kfold_splits(toy.labels, n_splits=4, n_repeats=0, seed=3)
    cv = cross_validate(
        make_pipeline("logistic_regression"), toy.features, toy.labels, n_splits=4, n_repeats=3
    )
    summary = cv.summary()
    assert len(cv.folds) == 12
    assert cv.test_to_train == pytest.approx(1 / 3, rel=0.05)
    acc = np.array([f.accuracy for f in cv.folds])
    naive = 2.2 * acc.std(ddof=1) / np.sqrt(acc.size)
    assert summary["accuracy_ci95"] > naive  # corrected interval is wider than the naive one
    one_fold = cross_validate(
        DummyClassifier(),
        toy.features,
        toy.labels,
        splits=[kfold_splits(toy.labels, n_splits=2, seed=0)[0]],
    )
    assert math.isnan(one_fold.summary()["accuracy_ci95"])

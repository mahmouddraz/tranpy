from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from tranpy_core.data import TabularDataset
from tranpy_core.errors import ChecksumError, ConfigurationError, ModelError
from tranpy_core.models import make_estimator, make_pipeline

pytestmark = pytest.mark.requires_neural
tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("dense_network", {"hidden_units": (8,), "dropout": 0.1}),
        ("lstm_network", {"n_steps": 3, "lstm_units": (4,), "dense_units": 4}),
        (
            "lstm_network",
            {"n_steps": 2, "lstm_units": (4, 4), "use_attention": False, "dense_units": 4},
        ),
        ("tcn_network", {"n_steps": 3, "n_filters": 4, "dilation_rates": (1, 2), "dense_units": 4}),
        (
            "transformer_network",
            {
                "n_steps": 3,
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "ff_dim": 8,
                "dense_units": 4,
            },
        ),
    ],
)
def test_networks_fit_predict_and_are_reproducible(
    name: str, params: dict, toy: TabularDataset
) -> None:
    pipe = make_pipeline(name, seed=0, epochs=2, early_stopping_patience=None, **params)
    pipe.fit(toy.features, toy.labels)
    proba = pipe.predict_proba(toy.features[:20])
    assert proba.shape == (20, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
    assert set(np.unique(pipe.predict(toy.features))) <= {0, 1}
    again = make_pipeline(name, seed=0, epochs=2, early_stopping_patience=None, **params).fit(
        toy.features, toy.labels
    )
    np.testing.assert_allclose(again.predict_proba(toy.features[:20]), proba, rtol=1e-6)
    assert "loss" in pipe[-1].history_


def test_networks_are_sklearn_estimators(toy: TabularDataset) -> None:
    est = make_estimator("dense_network", seed=3, epochs=1)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params()
    est.set_params(epochs=2)
    assert est.epochs == 2
    est.fit(toy.features, toy.labels)
    assert est.n_features_in_ == toy.n_features
    assert est.__sklearn_tags__().non_deterministic is False
    with pytest.raises(ModelError, match="expected 6 features"):
        est.predict(toy.features[:, :3])
    with pytest.raises(ModelError, match="2-D or 3-D"):
        est.predict(np.ones(3))
    lstm = make_estimator("lstm_network", n_steps=4, epochs=1)
    with pytest.raises(ConfigurationError, match="cannot be split"):
        lstm.fit(toy.features, toy.labels)
    with pytest.raises(ModelError, match="binary"):
        make_estimator("dense_network", epochs=1).fit(toy.features[:9], np.arange(9) % 3)
    with pytest.raises(ConfigurationError, match="divisible"):
        make_estimator("transformer_network", n_steps=3, d_model=6, n_heads=4, epochs=1).fit(
            toy.features, toy.labels
        )


def test_three_dimensional_input(toy: TabularDataset) -> None:
    X = toy.features.reshape(toy.n_samples, 3, 2)
    est = make_estimator(
        "tcn_network",
        n_filters=4,
        dilation_rates=(1,),
        dense_units=4,
        epochs=1,
        early_stopping_patience=2,
    )
    est.fit(X, toy.labels)
    assert est.predict_proba(X).shape == (toy.n_samples, 2)


def test_global_rng_untouched_by_fit(toy: TabularDataset) -> None:
    state = np.random.get_state()[1].copy()  # noqa: NPY002
    make_estimator("dense_network", epochs=1).fit(toy.features, toy.labels)
    assert np.array_equal(np.random.get_state()[1], state)  # noqa: NPY002


@pytest.mark.requires_persistence
def test_skops_round_trip(toy: TabularDataset, tmp_path: Path) -> None:
    pytest.importorskip("skops")
    from tranpy_core.data import sha256_file
    from tranpy_core.models.persistence import load_model, save_model

    pipe = make_pipeline("random_forest", seed=0).fit(toy.features, toy.labels)
    path = save_model(pipe, tmp_path / "m" / "rf.skops")
    back = load_model(path, sha256=sha256_file(path))
    np.testing.assert_array_equal(
        back.predict_proba(toy.features), pipe.predict_proba(toy.features)
    )
    with pytest.raises(ChecksumError):
        load_model(path, sha256="0" * 64)


@pytest.mark.requires_persistence
def test_untrusted_types_are_refused(tmp_path: Path) -> None:
    pytest.importorskip("skops")
    from fractions import Fraction

    from tranpy_core.models.persistence import load_model, save_model

    path = save_model({"x": Fraction(1, 3)}, tmp_path / "odd.skops")
    with pytest.raises(ModelError, match="untrusted types"):
        load_model(path)
    assert load_model(path, trusted=["fractions.Fraction"])["x"] == Fraction(1, 3)

"""Nested tuning: valid search spaces, the inner study never sees the outer test fold, and
``tuning: null`` reproduces the untuned benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tranpy import ml_benchmark
from tranpy.ml_benchmark import BenchmarkSpec, candidates, evaluate_candidate, protocol_splits
from tranpy.ml_tuning import (
    SEARCH_SPACES,
    TuningResult,
    TuningSpec,
    inner_splitter,
    suggest_params,
    tune_params,
)
from tranpy_core._optional import is_available
from tranpy_core.errors import ConfigurationError
from tranpy_core.models.registry import get_model_spec, make_estimator

optuna = pytest.importorskip("optuna")
pytest.importorskip("imblearn")


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=3, weights=[0.8], random_state=1
    )
    strata = pd.DataFrame(
        {
            "hour": np.repeat(["a", "b", "c", "d", "e"], 60),
            "clearing_band": np.tile(["0.0-0.1s", "0.1-0.2s"], 150),
        }
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), y, strata


def _sampled(model: str, seed: int = 0) -> dict:
    """Parameters of one random trial of ``model``'s search space."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(lambda t: float(len(suggest_params(model, t))), n_trials=1)
    return suggest_params(model, optuna.trial.FixedTrial(study.best_trial.params))


@pytest.mark.parametrize("model", sorted(SEARCH_SPACES))
def test_search_space_parameters_are_accepted_by_the_estimator(model: str) -> None:
    spec = get_model_spec(model)
    if not is_available(spec.requires or spec.estimator.split(".")[0]):
        pytest.skip(f"{model} needs the [{spec.extra}] extra")
    for seed in range(3):
        params = _sampled(model, seed)
        assert params
        est = make_estimator(model, seed=0, **params)
        got = est.get_params()
        for k, v in params.items():
            assert k in got, f"{model}: {k} is not an estimator parameter"
            assert got[k] == v


def test_unknown_model_has_no_search_space() -> None:
    with pytest.raises(ConfigurationError, match="no search space"):
        suggest_params("knn", optuna.trial.FixedTrial({}))


def test_tuning_spec_validation() -> None:
    assert TuningSpec().n_trials == 30 and TuningSpec().inner_folds == 3
    assert TuningSpec.from_mapping({"only_sets": "line_pre"}).only_sets == ("line_pre",)
    with pytest.raises(ConfigurationError):
        TuningSpec(n_trials=0)
    with pytest.raises(ConfigurationError):
        TuningSpec(inner_folds=1)
    with pytest.raises(ConfigurationError):
        TuningSpec(metric="luck")
    with pytest.raises(ConfigurationError, match="unknown tuning keys"):
        TuningSpec.from_mapping({"trials": 3})


def test_inner_splitter_groups_by_hour_and_falls_back(data) -> None:
    _, y, strata = data
    groups = strata["hour"].to_numpy()
    splitter, kind = inner_splitter(y, groups, 3, 0)
    assert kind == "group"
    for tr, te in splitter.split(np.zeros((len(y), 1)), y, groups=groups):
        assert not set(groups[tr]) & set(groups[te])
    _, kind = inner_splitter(y, None, 3, 0)
    assert kind == "stratified"
    _, kind = inner_splitter(y, groups[:120], 5, 0)  # only two hours: fewer than the folds
    assert kind == "stratified"


def test_tune_params_logistic_is_seeded(data) -> None:
    X, y, _ = data
    spec = TuningSpec(n_trials=3, inner_folds=3)

    def build(p):
        return make_estimator("logistic_regression", seed=0, max_iter=500, **p)

    a = tune_params("logistic_regression", build, X.to_numpy(), y, spec=spec, seed=1)
    b = tune_params("logistic_regression", build, X.to_numpy(), y, spec=spec, seed=1)
    assert isinstance(a, TuningResult)
    assert a.n_trials == 3 and len(a.trials) == 3 and "C" in a.best_params
    assert a.best_params == b.best_params and a.best_value == b.best_value
    assert a.trials["state"].eq("COMPLETE").all()


def test_inner_study_never_sees_the_outer_test_rows(data, monkeypatch) -> None:
    X, y, strata = data
    X = X.assign(row=np.arange(len(X), dtype=float))  # a unique id per row
    seen: list[set[int]] = []

    def fake_tune(model, build, X_tr, y_tr, *, spec, seed, groups):
        seen.append(set(np.asarray(X_tr)[:, -1].astype(int)))
        assert groups is not None and len(groups) == len(X_tr)
        return TuningResult(
            best_params={"C": 0.5}, best_value=0.0, trials=pd.DataFrame(), n_trials=0, time_s=0.0
        )

    monkeypatch.setattr(ml_benchmark, "tune_params", fake_tune)
    spec = BenchmarkSpec.from_mapping(
        {
            "feature_sets": ["system_pre"],
            "models": ["logistic_regression"],
            "tuning": {"n_trials": 2, "inner_folds": 2},
        }
    )
    cand = next(iter(candidates(spec)))
    result = evaluate_candidate(cand, X, y, strata, feature_set="system_pre", n_boot=0)
    splits, _ = protocol_splits("hour", y, strata)
    assert len(seen) == len(splits) == 5
    for rows, split in zip(seen, splits, strict=True):
        assert rows == set(split.train.tolist())
        assert not rows & set(split.test.tolist())
    assert result.tuned and result.tuning["best_params"] == [{"C": 0.5}] * 5


def test_only_sets_leaves_the_other_sets_untuned(data) -> None:
    X, y, strata = data
    base = {"feature_sets": ["system_pre", "line_pre"], "models": ["logistic_regression"]}
    untuned = next(iter(candidates(BenchmarkSpec.from_mapping(base))))
    tuned = next(
        iter(
            candidates(
                BenchmarkSpec.from_mapping(
                    {**base, "tuning": {"n_trials": 2, "only_sets": ["line_pre"]}}
                )
            )
        )
    )
    assert untuned.tuning is None and tuned.tuning is not None
    assert ml_benchmark.tuning_for(tuned, "system_pre") is None
    assert ml_benchmark.tuning_for(tuned, "line_pre") is tuned.tuning
    a = evaluate_candidate(untuned, X, y, strata, feature_set="system_pre", n_boot=0)
    b = evaluate_candidate(tuned, X, y, strata, feature_set="system_pre", n_boot=0)
    assert a.tuning is None and b.tuning is None
    np.testing.assert_array_equal(a.oof_score, b.oof_score)


def test_tuning_null_is_the_untuned_benchmark(data) -> None:
    X, y, strata = data
    base = {"feature_sets": ["system_pre"], "models": ["logistic_regression"], "seed": 2}
    a = next(iter(candidates(BenchmarkSpec.from_mapping(base))))
    b = next(iter(candidates(BenchmarkSpec.from_mapping({**base, "tuning": None}))))
    assert a == b and a.tuning is None
    ra = evaluate_candidate(a, X, y, strata, feature_set="system_pre", n_boot=0)
    rb = evaluate_candidate(b, X, y, strata, feature_set="system_pre", n_boot=0)
    np.testing.assert_array_equal(ra.oof_score, rb.oof_score)
    assert "tuned" not in ra.metrics()


@pytest.mark.requires_boosting
def test_lightgbm_tuning_smoke(data) -> None:
    pytest.importorskip("lightgbm")
    X, y, strata = data
    spec = BenchmarkSpec.from_mapping(
        {
            "feature_sets": ["system_pre"],
            "models": ["lightgbm"],
            "imbalance": ["class_weight"],
            "params": {"lightgbm": {"n_jobs": 1}},
            "protocols": ["random"],
            "tuning": {"n_trials": 2, "inner_folds": 2},
        }
    )
    cand = next(iter(candidates(spec)))
    logs: list[str] = []
    result = evaluate_candidate(
        cand, X, y, strata, feature_set="system_pre", n_boot=0, log=logs.append
    )
    assert result.tuned
    rec = result.tuning
    assert rec["n_trials"] == 2 and rec["inner_groups"] == "group"
    assert len(rec["best_params"]) == 5 and all("num_leaves" in p for p in rec["best_params"])
    assert len(rec["trials"]) == 10 and set(rec["trials"]["fold"]) == set(result.fold_groups)
    assert rec["time_s"] > 0
    assert any("tuned on" in m for m in logs)
    m = result.metrics()
    assert 0.0 <= m["average_precision"] <= 1.0
    assert np.isfinite(result.oof_score).all()


def test_threshold_candidates_tune_the_unwrapped_pipeline() -> None:
    from sklearn.model_selection import TunedThresholdClassifierCV
    from sklearn.pipeline import Pipeline

    spec = BenchmarkSpec.from_mapping(
        {
            "feature_sets": ["system_pre"],
            "models": ["logistic_regression"],
            "imbalance": [{"method": "class_weight", "threshold": "f1"}],
            "tuning": {"n_trials": 2},
        }
    )
    cand = next(iter(candidates(spec)))
    assert isinstance(cand.build(0), TunedThresholdClassifierCV)
    base = cand.build_unthresholded(0, C=0.3)
    assert isinstance(base, Pipeline)
    assert base.get_params()["model__C"] == 0.3
    assert cand.imbalance.threshold == "f1"  # the candidate itself is unchanged

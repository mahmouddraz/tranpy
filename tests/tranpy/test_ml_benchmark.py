"""The benchmark crosses models with imbalance options and predicts every event out of fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from tranpy.ml_benchmark import (
    BenchmarkSpec,
    candidates,
    evaluate_candidate,
    promotion_decision,
)
from tranpy_core.errors import ConfigurationError

pytest.importorskip("imblearn")


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=360, n_features=5, n_informative=3, weights=[0.85], random_state=0
    )
    strata = pd.DataFrame(
        {
            "hour": np.repeat(["a", "b", "c", "d"], 90),
            "clearing_band": np.tile(["0.0-0.1s", "0.1-0.2s"], 180),
        }
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y, strata


SPEC = {
    "feature_sets": ["system_pre"],
    "models": ["logistic_regression", "extra_trees"],
    "imbalance": ["none", {"method": "class_weight", "threshold": "f1", "threshold_folds": 3}],
    "extra": [{"model": "balanced_random_forest"}],
    "params": {"extra_trees": {"n_estimators": 20}, "balanced_random_forest": {"n_estimators": 20}},
    "seed": 3,
}


def test_matrix_and_extra_candidates() -> None:
    spec = BenchmarkSpec.from_mapping(SPEC)
    names = [c.name for c in candidates(spec)]
    assert names == [
        "logistic_regression__none",
        "logistic_regression__class_weight+thr_f1",
        "extra_trees__none",
        "extra_trees__class_weight+thr_f1",
        "balanced_random_forest__none",
    ]
    assert next(c for c in candidates(spec) if c.model == "extra_trees").params == {
        "n_estimators": 20
    }


def test_unknown_keys_and_metrics_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown benchmark keys"):
        BenchmarkSpec.from_mapping({**SPEC, "modles": []})
    with pytest.raises(ConfigurationError, match="primary metric"):
        BenchmarkSpec.from_mapping({**SPEC, "primary_metric": "luck"})


def test_every_event_is_predicted_by_a_model_that_never_saw_its_hour(data) -> None:
    X, y, strata = data
    spec = BenchmarkSpec.from_mapping(SPEC)
    cand = next(iter(candidates(spec)))
    result = evaluate_candidate(
        cand, X, y, strata, feature_set="t", report_by=["hour", "clearing_band"], seed=3
    )
    assert (result.oof_pred >= 0).all()
    assert np.isfinite(result.oof_score).all()
    assert sorted(result.fold_groups) == ["a", "b", "c", "d"]
    assert result.pooled.n_samples == len(y)
    assert list(result.by_stratum["hour"].index) == ["__all__", "a", "b", "c", "d"]
    metrics = result.metrics()
    assert {"average_precision", "miss_rate", "mcc_fold_mean", "fit_time_s"} <= set(metrics)


def test_tuned_thresholds_are_recorded(data) -> None:
    X, y, strata = data
    spec = BenchmarkSpec.from_mapping(SPEC)
    cand = next(c for c in candidates(spec) if c.imbalance.threshold)
    result = evaluate_candidate(cand, X, y, strata, feature_set="t", seed=3)
    assert len(result.thresholds) == 4
    assert all(0 < t < 1 for t in result.thresholds)


def test_evaluation_weights_change_scores_not_training(data) -> None:
    X, y, strata = data
    cand = next(iter(candidates(BenchmarkSpec.from_mapping(SPEC))))
    plain = evaluate_candidate(cand, X, y, strata, feature_set="t", seed=3)
    w = np.where(y == 0, 5.0, 1.0)
    weighted = evaluate_candidate(cand, X, y, strata, feature_set="t", seed=3, sample_weight=w)
    assert np.array_equal(plain.oof_pred, weighted.oof_pred)
    assert weighted.pooled.precision <= plain.pooled.precision


def test_promotion_needs_better_metric_and_no_more_misses() -> None:
    champ = {"average_precision": 0.80, "miss_rate": 0.10}
    assert promotion_decision({"average_precision": 0.5, "miss_rate": 0.5}, None)[0]
    assert promotion_decision({"average_precision": 0.85, "miss_rate": 0.10}, champ)[0]
    assert not promotion_decision({"average_precision": 0.85, "miss_rate": 0.12}, champ)[0]
    assert not promotion_decision({"average_precision": 0.80, "miss_rate": 0.05}, champ)[0]

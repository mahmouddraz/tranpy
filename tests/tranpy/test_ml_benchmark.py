"""The benchmark crosses models with imbalance options and protocols, predicts every event out
of fold, attaches bootstrap intervals and ranks candidates with a tie rule inside them."""

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
    protocol_splits,
    select_best,
)
from tranpy_core.errors import ConfigurationError

pytest.importorskip("imblearn")


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(
        n_samples=360, n_features=5, n_informative=3, weights=[0.85], random_state=0
    )
    rng = np.random.default_rng(0)
    strata = pd.DataFrame(
        {
            "hour": np.repeat(["a", "b", "c", "d"], 90),
            "clearing_band": np.tile(["0.0-0.1s", "0.1-0.2s"], 180),
            "fault_line": rng.choice([f"L{i}" for i in range(12)], size=360),
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
    assert spec.protocols == ("hour",)
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


def test_protocols_cross_the_candidates() -> None:
    spec = BenchmarkSpec.from_mapping({**SPEC, "protocols": ["hour", "random"]})
    cands = list(candidates(spec))
    assert len(cands) == 10
    assert [c.protocol for c in cands] == ["hour"] * 5 + ["random"] * 5
    assert cands[4].model == "balanced_random_forest" and cands[9].model == cands[4].model
    legacy = BenchmarkSpec.from_mapping({**SPEC, "split_by": "random"})
    assert legacy.protocols == ("random",)


def test_unknown_keys_metrics_and_protocols_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown benchmark keys"):
        BenchmarkSpec.from_mapping({**SPEC, "modles": []})
    with pytest.raises(ConfigurationError, match="primary metric"):
        BenchmarkSpec.from_mapping({**SPEC, "primary_metric": "luck"})
    with pytest.raises(ConfigurationError, match="unknown protocols"):
        BenchmarkSpec.from_mapping({**SPEC, "protocols": ["hour", "day"]})


def test_hour_protocol_holds_out_whole_hours(data) -> None:
    _, y, strata = data
    splits, labels = protocol_splits("hour", y, strata)
    assert sorted(labels) == ["a", "b", "c", "d"]
    hours = strata["hour"].to_numpy()
    for split in splits:
        assert len(set(hours[split.test])) == 1
        assert not set(hours[split.test]) & set(hours[split.train])


def test_random_protocol_is_stratified_and_seeded(data) -> None:
    _, y, strata = data
    splits, labels = protocol_splits("random", y, strata, seed=3)
    assert len(splits) == 5 and labels == [f"fold_{i}" for i in range(5)]
    tested = np.concatenate([s.test for s in splits])
    assert np.array_equal(np.sort(tested), np.arange(len(y)))
    rate = y.mean()
    for split in splits:
        assert abs(y[split.test].mean() - rate) < 0.05
    again, _ = protocol_splits("random", y, strata, seed=3)
    assert all(np.array_equal(a.test, b.test) for a, b in zip(splits, again, strict=True))
    other, _ = protocol_splits("random", y, strata, seed=4)
    assert not all(np.array_equal(a.test, b.test) for a, b in zip(splits, other, strict=True))


def test_line_protocol_keeps_fault_lines_disjoint(data) -> None:
    _, y, strata = data
    splits, labels = protocol_splits("line", y, strata, seed=3)
    assert len(splits) == 5 and all(label.startswith("fold_") for label in labels)
    lines = strata["fault_line"].to_numpy()
    for split in splits:
        assert not set(lines[split.test]) & set(lines[split.train])
    with pytest.raises(ConfigurationError, match="fault_line"):
        protocol_splits("line", y, strata.drop(columns="fault_line"))
    with pytest.raises(ConfigurationError, match="unknown protocol"):
        protocol_splits("day", y, strata)


def test_every_event_is_predicted_by_a_model_that_never_saw_its_hour(data) -> None:
    X, y, strata = data
    spec = BenchmarkSpec.from_mapping(SPEC)
    cand = next(iter(candidates(spec)))
    result = evaluate_candidate(
        cand, X, y, strata, feature_set="t", report_by=["hour", "clearing_band"], seed=3, n_boot=50
    )
    assert result.protocol == "hour"
    assert (result.oof_pred >= 0).all()
    assert np.isfinite(result.oof_score).all()
    assert sorted(result.fold_groups) == ["a", "b", "c", "d"]
    assert result.pooled.n_samples == len(y)
    assert list(result.by_stratum["hour"].index) == ["__all__", "a", "b", "c", "d"]
    metrics = result.metrics()
    assert {"average_precision", "miss_rate", "mcc_fold_mean", "fit_time_s"} <= set(metrics)
    assert {"accuracy", "balanced_accuracy"} <= set(metrics)
    for m in ("average_precision", "miss_rate", "accuracy", "balanced_accuracy"):
        assert metrics[f"{m}_lo"] <= metrics[m] <= metrics[f"{m}_hi"]


def test_candidate_protocol_drives_the_folds(data) -> None:
    X, y, strata = data
    spec = BenchmarkSpec.from_mapping({**SPEC, "protocols": ["random"]})
    cand = next(iter(candidates(spec)))
    result = evaluate_candidate(cand, X, y, strata, feature_set="t", seed=3, n_boot=0)
    assert result.protocol == "random"
    assert result.fold_groups == [f"fold_{i}" for i in range(5)]
    assert result.ci == {} and "average_precision_lo" not in result.metrics()
    lines = evaluate_candidate(
        cand, X, y, strata, feature_set="t", split_by="line", seed=3, n_boot=0
    )
    assert lines.protocol == "line" and (lines.oof_pred >= 0).all()


def test_tuned_thresholds_are_recorded(data) -> None:
    X, y, strata = data
    spec = BenchmarkSpec.from_mapping(SPEC)
    cand = next(c for c in candidates(spec) if c.imbalance.threshold)
    result = evaluate_candidate(cand, X, y, strata, feature_set="t", seed=3, n_boot=0)
    assert len(result.thresholds) == 4
    assert all(0 < t < 1 for t in result.thresholds)


def test_evaluation_weights_change_scores_not_training(data) -> None:
    X, y, strata = data
    cand = next(iter(candidates(BenchmarkSpec.from_mapping(SPEC))))
    plain = evaluate_candidate(cand, X, y, strata, feature_set="t", seed=3, n_boot=0)
    w = np.where(y == 0, 5.0, 1.0)
    weighted = evaluate_candidate(
        cand, X, y, strata, feature_set="t", seed=3, sample_weight=w, n_boot=0
    )
    assert np.array_equal(plain.oof_pred, weighted.oof_pred)
    assert weighted.pooled.precision <= plain.pooled.precision


def _row(ap: float, miss: float, half: float = 0.05) -> dict[str, float]:
    return {
        "average_precision": ap,
        "average_precision_lo": ap - half,
        "average_precision_hi": ap + half,
        "miss_rate": miss,
    }


def test_select_best_prefers_lower_miss_rate_inside_the_interval() -> None:
    rows = [_row(0.80, 0.10), _row(0.82, 0.15), _row(0.70, 0.01)]
    # 0.80 lies inside [0.77, 0.87] of the leader: tie, lower miss rate wins; 0.70 is outside
    assert select_best(rows) == 0
    rows = [_row(0.80, 0.10), _row(0.90, 0.15), _row(0.70, 0.01)]
    assert select_best(rows) == 1  # outside the interval: the higher PR-AUC wins
    assert select_best([_row(0.80, 0.10), _row(0.84, 0.10)]) == 1  # equal misses: higher AP
    with pytest.raises(ValueError, match="no results"):
        select_best([])


def test_select_best_without_intervals_falls_back_to_exact_ties() -> None:
    rows = [
        {"average_precision": 0.80, "miss_rate": 0.10},
        {"average_precision": 0.80, "miss_rate": 0.05},
        {"average_precision": 0.79, "miss_rate": 0.00},
    ]
    assert select_best(rows) == 1
    nan = {"average_precision": float("nan"), "miss_rate": 0.0}
    assert select_best([nan, *rows]) == 2


def test_promotion_without_intervals_needs_better_metric_and_no_worse_mcc() -> None:
    champ = {"average_precision": 0.80, "miss_rate": 0.10, "mcc": 0.60}
    assert promotion_decision({"average_precision": 0.5, "miss_rate": 0.5}, None)[0]
    assert promotion_decision({"average_precision": 0.85, "miss_rate": 0.1, "mcc": 0.6}, champ)[0]
    assert not promotion_decision({"average_precision": 0.85, "miss_rate": 0.1, "mcc": 0.5}, champ)[
        0
    ]
    assert not promotion_decision({"average_precision": 0.8, "miss_rate": 0.05, "mcc": 0.7}, champ)[
        0
    ]
    # without mcc on both sides the miss rate decides
    bare = {"average_precision": 0.80, "miss_rate": 0.10}
    assert promotion_decision({"average_precision": 0.85, "miss_rate": 0.10}, bare)[0]
    assert not promotion_decision({"average_precision": 0.85, "miss_rate": 0.12}, bare)[0]


def test_promotion_with_intervals() -> None:
    champ = {**_row(0.80, 0.10), "mcc": 0.60}
    # tied within the interval: a higher mcc promotes, whatever the miss rate
    promote, reason = promotion_decision({**_row(0.82, 0.12), "mcc": 0.65}, champ)
    assert promote and "promoted by mcc" in reason
    assert not promotion_decision({**_row(0.82, 0.05), "mcc": 0.60}, champ)[0]
    assert not promotion_decision({**_row(0.78, 0.05), "mcc": 0.55}, champ)[0]
    assert promotion_decision({**_row(0.78, 0.12), "mcc": 0.61}, champ)[0]
    # better beyond the interval: promoted even with more misses (other operating point)
    better = {**_row(0.90, 0.20), "miss_rate_lo": 0.15, "miss_rate_hi": 0.25, "mcc": 0.5}
    promote, reason = promotion_decision(better, champ)
    assert promote and "better beyond" in reason
    # worse beyond the interval: never, whatever the mcc
    assert not promotion_decision({**_row(0.60, 0.00), "mcc": 0.9}, champ)[0]
    # champion registered without intervals or mcc: the challenger's interval decides the
    # tie and the miss rate breaks it
    bare = {"average_precision": 0.80, "miss_rate": 0.10}
    assert promotion_decision(_row(0.78, 0.05), bare)[0]
    assert not promotion_decision(_row(0.78, 0.10), bare)[0]


def test_promotion_regression_tsa_pre_v4_vs_v2() -> None:
    # v4.1 tuned line_pre (v4) against the v4 system_pre champion (v2), real gate metrics
    # of 2026-11-06: better beyond the interval and a higher MCC, blocked only by the old
    # miss-rate rule (26.6 % at 68 % precision vs 13.4 % at 27 % precision)
    v2 = {"average_precision": 0.4356, "miss_rate": 0.1339, "precision": 0.27, "mcc": 0.33}
    v4 = {
        "average_precision": 0.7805,
        "average_precision_lo": 0.7607,
        "average_precision_hi": 0.7997,
        "miss_rate": 0.2664,
        "miss_rate_lo": 0.2422,
        "miss_rate_hi": 0.2898,
        "precision": 0.6797,
        "mcc": 0.6649,
    }
    promote, reason = promotion_decision(v4, v2)
    assert promote and "better beyond the average_precision interval" in reason
    assert "precision 0.6797 vs 0.2700" in reason


def test_tuning_section_is_parsed_and_null_keeps_the_candidates_untuned() -> None:
    from tranpy.ml_tuning import TuningSpec

    untuned = BenchmarkSpec.from_mapping({**SPEC, "tuning": None})
    assert untuned.tuning is None
    assert all(c.tuning is None for c in candidates(untuned))
    spec = BenchmarkSpec.from_mapping(
        {
            **SPEC,
            "feature_sets": ["system_pre", "line_pre"],
            "tuning": {"n_trials": 5, "inner_folds": 2, "metric": "roc_auc", "timeout_s": 60},
        }
    )
    assert spec.tuning == TuningSpec(n_trials=5, inner_folds=2, metric="roc_auc", timeout_s=60)
    assert spec.tuning is not None and spec.tuning.only_sets == ()
    assert all(c.tuning == spec.tuning for c in candidates(spec))
    with pytest.raises(ConfigurationError, match="unknown tuning keys"):
        BenchmarkSpec.from_mapping({**SPEC, "tuning": {"trials": 5}})


def test_tuning_only_sets_must_be_feature_sets() -> None:
    from tranpy.ml_benchmark import tuning_for

    spec = BenchmarkSpec.from_mapping(
        {
            **SPEC,
            "feature_sets": ["system_pre", "line_pre"],
            "tuning": {"n_trials": 2, "only_sets": ["line_pre"]},
        }
    )
    cand = next(iter(candidates(spec)))
    assert tuning_for(cand, "line_pre") is spec.tuning
    assert tuning_for(cand, "system_pre") is None
    with pytest.raises(ConfigurationError, match="only_sets"):
        BenchmarkSpec.from_mapping({**SPEC, "tuning": {"only_sets": ["aligned_pre"]}})

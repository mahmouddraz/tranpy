"""Threshold baselines: thresholds from the training hours only, scored like the ML candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from tranpy.conventional.baselines import (
    RuleCandidate,
    ThresholdRule,
    f1_threshold,
    youden_threshold,
)
from tranpy.ml_benchmark import evaluate_candidate
from tranpy_core.errors import ConfigurationError


def test_youden_threshold_separates_a_separable_score() -> None:
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = youden_threshold(y, s)
    assert 0.3 < t <= 0.8
    assert np.array_equal((s >= t).astype(int), y)


def test_f1_threshold_maximises_f1_over_all_cut_points() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    s = y + rng.normal(0, 0.7, 300)
    t = f1_threshold(y, s)
    best = max(f1_score(y, (s >= c).astype(int)) for c in np.unique(s))
    assert f1_score(y, (s >= t).astype(int)) == pytest.approx(best)


def test_ties_are_one_cut_point() -> None:
    y = np.array([0, 1, 1, 0])
    s = np.array([1.0, 1.0, 2.0, 0.0])
    assert f1_threshold(y, s) in (1.0, 2.0)
    assert youden_threshold(y, s) in (1.0, 2.0)


def test_single_class_training_sets() -> None:
    assert youden_threshold([0, 0], [1.0, 2.0]) == np.inf
    assert f1_threshold([1, 1], [1.0, 2.0]) == -np.inf


def test_rule_direction_and_unknown_options() -> None:
    X = np.array([[5.0], [4.0], [1.0], [0.0]])
    y = np.array([0, 0, 1, 1])
    rule = ThresholdRule(direction=-1).fit(X, y)
    assert np.array_equal(rule.predict(X), y)
    assert np.array_equal(rule.decision_function(X), -X[:, 0])
    with pytest.raises(ConfigurationError):
        ThresholdRule(criterion="accuracy").fit(X, y)
    with pytest.raises(ConfigurationError):
        ThresholdRule(direction=2).fit(X, y)


def test_threshold_comes_from_the_training_hours_only() -> None:
    # hour "b" has its unstable events at a higher score; leave-one-hour-out must not see them
    score = np.r_[np.linspace(0, 1, 50), np.linspace(0, 1, 50)]
    y = np.r_[(np.linspace(0, 1, 50) > 0.5), (np.linspace(0, 1, 50) > 0.8)].astype(int)
    strata = pd.DataFrame({"hour": np.repeat(["a", "b"], 50)})
    cand = RuleCandidate(model="rule:s", estimator=ThresholdRule())
    result = evaluate_candidate(
        cand, pd.DataFrame({"s": score}), y, strata, feature_set="s", report_by=("hour",)
    )
    assert result.candidate.name == "rule:s"
    assert result.fold_groups == ["a", "b"]
    t_test_a, t_test_b = result.thresholds  # each fitted on the other hour
    assert t_test_a == pytest.approx(youden_threshold(y[50:], score[50:]))
    assert t_test_b == pytest.approx(youden_threshold(y[:50], score[:50]))
    # ranking metrics are those of the raw score
    assert np.array_equal(result.oof_score, score)
    assert set(result.by_stratum["hour"].index) == {"__all__", "a", "b"}
    assert np.isnan(result.pooled.brier)  # no probabilities


def test_candidate_without_estimator_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        RuleCandidate(model="empty").build(0)

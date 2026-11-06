"""Bootstrap confidence intervals of the pooled metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tranpy_core.evaluation import METRICS, bootstrap_ci, score_predictions


def _predictions(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    scores = np.clip(0.3 * y + 0.7 * rng.random(n), 0.0, 1.0)  # overlapping: AP < 1
    return y, (scores >= 0.5).astype(int), scores


def test_interval_contains_point_estimate_for_every_metric() -> None:
    y, y_pred, s = _predictions(300)
    point = score_predictions(y, y_pred, s)
    ci = bootstrap_ci(y, y_pred, s, n_boot=200, seed=1)
    assert set(ci) == set(METRICS)
    for metric, (lo, hi) in ci.items():
        value = getattr(point, metric)
        assert lo <= value <= hi, (metric, lo, value, hi)
        assert lo <= hi


def test_interval_width_shrinks_with_sample_size() -> None:
    widths = []
    for n in (100, 1600):
        y, y_pred, s = _predictions(n, seed=3)
        lo, hi = bootstrap_ci(y, y_pred, s, n_boot=300, seed=1)["average_precision"]
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 2


def test_seed_makes_the_interval_reproducible() -> None:
    y, y_pred, s = _predictions(120)
    assert bootstrap_ci(y, y_pred, s, n_boot=50, seed=7) == bootstrap_ci(
        y, y_pred, s, n_boot=50, seed=7
    )
    assert bootstrap_ci(y, y_pred, s, n_boot=50, seed=7) != bootstrap_ci(
        y, y_pred, s, n_boot=50, seed=8
    )


def test_stratified_resamples_keep_rank_metrics_defined_with_few_positives() -> None:
    y = np.array([0] * 40 + [1] * 2)
    s = np.linspace(0, 1, 42)
    y_pred = (s > 0.9).astype(int)
    ci = bootstrap_ci(y, y_pred, s, n_boot=100, seed=0)
    assert all(math.isfinite(v) for v in ci["average_precision"])
    assert all(math.isfinite(v) for v in ci["roc_auc"])


def test_weights_travel_with_their_samples() -> None:
    y, y_pred, s = _predictions(200)
    w = np.where(y == 0, 5.0, 1.0)
    plain = bootstrap_ci(y, y_pred, s, n_boot=100, seed=0)
    weighted = bootstrap_ci(y, y_pred, s, n_boot=100, seed=0, sample_weight=w)
    assert weighted["precision"][1] <= plain["precision"][1]


def test_rejects_zero_resamples() -> None:
    y, y_pred, s = _predictions(20)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(y, y_pred, s, n_boot=0)

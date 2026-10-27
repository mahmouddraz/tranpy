"""Pinned, seeded results of every bundled model/dataset combination.

A failure means results changed. If the change is intended, regenerate the snapshot with
``uv run python -m tests.update_snapshot`` and give the reason in the commit message.

On the platform the snapshot was recorded on (``platform`` in the file) the comparison is
exact. Other platforms use different floating-point libraries (e.g. Apple Accelerate vs
OpenBLAS), which changes the last digits of sums and occasionally resolves near-ties in
trees or SGD differently; there each confusion-matrix cell may differ by at most 1 % of
the test samples, accuracy and ROC-AUC by 0.005 and probability sums by 0.1 %, far below
any real change of method. Precision, recall and F1 follow from the confusion matrix.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from tests.update_snapshot import SNAPSHOT, compute, current_platform

from tranpy_core.models import get_model_spec

RECORD = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
PINNED = RECORD["metrics"]
EXACT = RECORD.get("platform") == current_platform()
CROSS_PLATFORM_TOLERANCE = {"accuracy": 0.005, "roc_auc": 0.005}
CROSS_PLATFORM_CELL_FRACTION = 0.01
FAST = [k for k in PINNED if not k.endswith("/gaussian_process")]
SLOW = [k for k in PINNED if k.endswith("/gaussian_process")]


def _check(key: str) -> None:
    system, model = key.split("/")
    if not get_model_spec(model).is_available():
        pytest.skip(f"{model} backend not installed")
    actual = compute([model], (system,))[key]
    expected = PINNED[key]
    if EXACT:
        assert actual["confusion_matrix"] == expected["confusion_matrix"]
        for metric, value in expected.items():
            if metric != "confusion_matrix":
                assert actual[metric] == pytest.approx(value, rel=1e-9, abs=1e-12), metric
        return
    assert actual["n_samples"] == expected["n_samples"]
    cells = np.abs(np.array(actual["confusion_matrix"]) - np.array(expected["confusion_matrix"]))
    assert cells.max() <= CROSS_PLATFORM_CELL_FRACTION * expected["n_samples"], "confusion_matrix"
    for metric, tolerance in CROSS_PLATFORM_TOLERANCE.items():
        assert actual[metric] == pytest.approx(expected[metric], abs=tolerance), metric
    assert actual["score_sum"] == pytest.approx(expected["score_sum"], rel=1e-3), "score_sum"


@pytest.mark.parametrize("key", FAST)
def test_pinned_metrics(key: str) -> None:
    _check(key)


@pytest.mark.slow
@pytest.mark.parametrize("key", SLOW)
def test_pinned_metrics_slow_models(key: str) -> None:
    _check(key)

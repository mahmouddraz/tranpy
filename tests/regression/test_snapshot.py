"""Pinned, seeded results of every bundled model/dataset combination.

A failure means results changed. If the change is intended, regenerate the snapshot with
``uv run python -m tests.update_snapshot`` and give the reason in the commit message.
"""

from __future__ import annotations

import json

import pytest
from tests.update_snapshot import SNAPSHOT, compute

from tranpy_core.models import get_model_spec

PINNED = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["metrics"]
FAST = [k for k in PINNED if not k.endswith("/gaussian_process")]
SLOW = [k for k in PINNED if k.endswith("/gaussian_process")]


def _check(key: str) -> None:
    system, model = key.split("/")
    if not get_model_spec(model).is_available():
        pytest.skip(f"{model} backend not installed")
    actual = compute([model], (system,))[key]
    expected = PINNED[key]
    assert actual["confusion_matrix"] == expected["confusion_matrix"]
    for metric, value in expected.items():
        if metric != "confusion_matrix":
            assert actual[metric] == pytest.approx(value, rel=1e-9, abs=1e-12), metric


@pytest.mark.parametrize("key", FAST)
def test_pinned_metrics(key: str) -> None:
    _check(key)


@pytest.mark.slow
@pytest.mark.parametrize("key", SLOW)
def test_pinned_metrics_slow_models(key: str) -> None:
    _check(key)

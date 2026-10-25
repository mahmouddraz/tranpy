"""Seeded test metrics of every registered classical model on every bundled dataset.

Regenerate the pinned snapshot (state the reason in the commit message):

    uv run python -m tests.update_snapshot

The values were verified to be identical to the bug-fixed reference implementation
(see docs/validation.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

import tranpy
from tranpy_core.data import train_test_split
from tranpy_core.evaluation import evaluate, positive_scores
from tranpy_core.models import available_models, get_model_spec, make_pipeline

SNAPSHOT = Path(__file__).parent / "snapshots" / "metrics.json"
SYSTEMS = ("new_england", "ieee_9bus")
SPLIT_SEED, MODEL_SEED = 42, 0


def snapshot_models() -> list[str]:
    return [m for m in available_models() if not m.endswith("_network")]


def compute(models: list[str] | None = None, systems: tuple[str, ...] = SYSTEMS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system in systems:
        data = tranpy.load_snapshots(system)
        split = train_test_split(data.labels, test_size=0.2, seed=SPLIT_SEED)
        X_tr, y_tr = data.features[split.train], data.labels[split.train]
        X_te, y_te = data.features[split.test], data.labels[split.test]
        for name in models or snapshot_models():
            if not get_model_spec(name).is_available():
                continue
            pipe = make_pipeline(name, seed=MODEL_SEED, scale=True).fit(X_tr, y_tr)
            scores = evaluate(pipe, X_te, y_te).as_dict()
            scores["score_sum"] = float(np.sum(positive_scores(pipe, X_te)))
            out[f"{system}/{name}"] = scores
    return out


def main() -> None:
    data = {
        "split_seed": SPLIT_SEED,
        "model_seed": MODEL_SEED,
        "scikit_learn": sklearn.__version__,
        "metrics": compute(),
    }
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT} ({len(data['metrics'])} entries)", file=sys.stderr)


if __name__ == "__main__":
    main()

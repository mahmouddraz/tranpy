"""Pretrained model pipelines: seeded training, safe (skops) storage, verified download.

Pretrained pipelines are trained by :func:`train_pretrained` on a fixed, seeded split of
a bundled snapshot dataset, so anyone can rebuild them bit-for-bit. Published files are
listed in ``resources/registry.json`` under ``"pretrained"`` with URL and SHA-256 and are
loaded with :func:`load_pretrained` (needs the ``[persistence]`` extra).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tranpy.datasets import data_registry, load_snapshots
from tranpy.systems import SYSTEMS, get_system
from tranpy_core.data import RemoteFile, fetch, sha256_file, train_test_split
from tranpy_core.errors import DataError
from tranpy_core.evaluation import evaluate
from tranpy_core.models import available_models, make_pipeline
from tranpy_core.models.persistence import load_model, save_model

__all__ = ["list_pretrained", "load_pretrained", "pretrained_name", "train_pretrained"]

SPLIT_SEED = 42
MODEL_SEED = 0
TEST_SIZE = 0.2


def pretrained_name(system: str, model: str) -> str:
    return f"{get_system(system).key}__{model}"


def list_pretrained() -> pd.DataFrame:
    """Published pretrained pipelines with their recorded test scores."""
    rows = [{"name": name, **entry} for name, entry in data_registry()["pretrained"].items()]
    return pd.DataFrame(
        rows, columns=["name", "system", "model", "filename", "url", "sha256", "test_accuracy"]
    )


def load_pretrained(name: str, *, cache_directory: Path | None = None) -> Any:
    """Download (once), verify and load a published pipeline."""
    entry = data_registry()["pretrained"].get(name)
    if entry is None:
        raise DataError(f"unknown pretrained model {name!r}; see list_pretrained()")
    if not entry.get("url"):
        raise DataError(
            f"pretrained model {name!r} has no download URL yet; build it with train_pretrained()"
        )
    path = fetch(
        RemoteFile(filename=entry["filename"], url=entry["url"], sha256=entry["sha256"]),
        directory=cache_directory,
    )
    return load_model(path, sha256=entry["sha256"])


def train_pretrained(
    out_dir: str | Path,
    *,
    systems: Sequence[str] = tuple(SYSTEMS),
    models: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Train, evaluate and save pipelines; return registry entries (url left empty).

    Writes ``<system>__<model>.skops`` files and ``registry_entries.json`` to ``out_dir``.
    Models whose backend is not installed are skipped.
    """
    out = Path(out_dir)
    names = (
        list(models)
        if models is not None
        else [m for m in available_models(installed_only=True) if not m.endswith("_network")]
    )
    entries: dict[str, dict[str, Any]] = {}
    for system in systems:
        data = load_snapshots(system)
        split = train_test_split(data.labels, test_size=TEST_SIZE, seed=SPLIT_SEED)
        X_train, y_train = data.features[split.train], data.labels[split.train]
        for model in names:
            pipeline = make_pipeline(model, seed=MODEL_SEED).fit(X_train, y_train)
            scores = evaluate(pipeline, data.features[split.test], data.labels[split.test])
            name = pretrained_name(system, model)
            path = save_model(pipeline, out / f"{name}.skops")
            entries[name] = {
                "system": get_system(system).key,
                "model": model,
                "filename": path.name,
                "url": None,
                "sha256": sha256_file(path),
                "test_accuracy": scores.accuracy,
                "split_seed": SPLIT_SEED,
                "model_seed": MODEL_SEED,
            }
    (out / "registry_entries.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries

"""Declarative experiments: split a tabular dataset, fit model pipelines, score them.

An :class:`ExperimentSpec` is a frozen, validated description that can be written to and
read from YAML (see :mod:`tranpy_core.config`). :func:`run_experiment` is deterministic
for a given spec and dataset, and returns a JSON-serialisable :class:`ExperimentResult`.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from tranpy_core.config import to_mapping
from tranpy_core.data.dataset import TabularDataset
from tranpy_core.data.split import Split, train_test_split
from tranpy_core.errors import ConfigurationError
from tranpy_core.evaluation import Scores, cross_validate, evaluate
from tranpy_core.models.registry import get_model_spec, make_pipeline

__all__ = ["ExperimentResult", "ExperimentSpec", "ModelConfig", "run_experiment"]


@dataclass(frozen=True)
class ModelConfig:
    """One registered model and its hyperparameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    scale: bool | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        get_model_spec(self.name)  # raises for unknown names
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @property
    def key(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class ExperimentSpec:
    """What to run.

    Attributes:
        name: identifier used in results.
        models: models to fit and evaluate.
        test_size: held-out fraction for the final test scores.
        seed: seed for the split and every model.
        cv_folds: k for k-fold cross-validation on the training part (0 disables it).
        stratify: stratify the split and the folds by label.
        group_duplicates: keep identical feature vectors on the same side of every split.
    """

    name: str
    models: tuple[ModelConfig, ...]
    test_size: float = 0.2
    seed: int = 0
    cv_folds: int = 0
    stratify: bool = True
    group_duplicates: bool = False

    def __post_init__(self) -> None:
        problems = []
        if not self.models:
            problems.append("models: at least one model is required")
        keys = [m.key for m in self.models]
        if len(set(keys)) != len(keys):
            problems.append(
                f"models: duplicate labels {sorted({k for k in keys if keys.count(k) > 1})}"
            )
        if not 0.0 < self.test_size < 1.0:
            problems.append(f"test_size must lie in (0, 1), got {self.test_size}")
        if self.cv_folds == 1 or self.cv_folds < 0:
            problems.append(f"cv_folds must be 0 or at least 2, got {self.cv_folds}")
        if problems:
            raise ConfigurationError(f"invalid experiment {self.name!r}:", tuple(problems))
        object.__setattr__(self, "models", tuple(self.models))


@dataclass(frozen=True, eq=False)
class ExperimentResult:
    """Outcome of :func:`run_experiment`."""

    spec: ExperimentSpec
    dataset: str
    split: Split
    test_scores: Mapping[str, Scores]
    cv_summary: Mapping[str, Mapping[str, float]]
    fit_times_s: Mapping[str, float]
    environment: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": to_mapping(self.spec),
            "dataset": self.dataset,
            "split": {"train": self.split.train.tolist(), "test": self.split.test.tolist()},
            "test_scores": {k: v.as_dict() for k, v in self.test_scores.items()},
            "cv_summary": {k: dict(v) for k, v in self.cv_summary.items()},
            "fit_times_s": dict(self.fit_times_s),
            "environment": dict(self.environment),
        }

    def summary_frame(self) -> pd.DataFrame:
        """One row per model: cross-validation means/stds (if run) and test scores."""
        rows = {}
        for key, scores in self.test_scores.items():
            row: dict[str, Any] = dict(self.cv_summary.get(key, {}))
            row.update(
                {f"test_{k}": v for k, v in scores.as_dict().items() if k != "confusion_matrix"}
            )
            row["fit_time_s"] = self.fit_times_s[key]
            rows[key] = row
        return pd.DataFrame.from_dict(rows, orient="index")

    def save_json(self, path: str | Path) -> Path:
        """Write the result to ``path`` (only when called explicitly)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=True), encoding="utf-8")
        return path


def run_experiment(spec: ExperimentSpec, dataset: TabularDataset) -> ExperimentResult:
    """Split ``dataset``, fit each model pipeline on the training part and score the test part."""
    groups = dataset.duplicate_groups() if spec.group_duplicates else None
    split = train_test_split(
        dataset.labels,
        test_size=spec.test_size,
        seed=spec.seed,
        stratify=spec.stratify,
        groups=groups,
    )
    X_train, y_train = dataset.features[split.train], dataset.labels[split.train]
    X_test, y_test = dataset.features[split.test], dataset.labels[split.test]
    test_scores: dict[str, Scores] = {}
    cv_summary: dict[str, Mapping[str, float]] = {}
    fit_times: dict[str, float] = {}
    for model in spec.models:
        pipeline = make_pipeline(model.name, seed=spec.seed, scale=model.scale, **model.params)
        if spec.cv_folds:
            cv = cross_validate(
                pipeline,
                X_train,
                y_train,
                n_splits=spec.cv_folds,
                seed=spec.seed,
                groups=None if groups is None else groups[split.train],
            )
            cv_summary[model.key] = cv.summary()
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_times[model.key] = time.perf_counter() - t0
        test_scores[model.key] = evaluate(pipeline, X_test, y_test)
    return ExperimentResult(
        spec=spec,
        dataset=dataset.name,
        split=split,
        test_scores=MappingProxyType(test_scores),
        cv_summary=MappingProxyType(cv_summary),
        fit_times_s=MappingProxyType(fit_times),
        environment=MappingProxyType(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            }
        ),
    )

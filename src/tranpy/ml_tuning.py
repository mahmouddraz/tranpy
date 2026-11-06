"""Hyperparameter tuning of a benchmark candidate with Optuna (``[tuning]`` extra).

Established nested cross-validation: :func:`tune_params` is called once per *outer*
training fold of :func:`tranpy.ml_benchmark.evaluate_candidate` with that fold's training
rows only. Inside, an Optuna study (TPE sampler, fixed seed) proposes hyperparameters from
the model's search space (:data:`SEARCH_SPACES`) and scores every trial by an *inner*
``StratifiedGroupKFold`` grouped by the operating hour (``StratifiedKFold`` when no groups
are given or there are fewer groups than folds). The best parameters are then refitted on
the whole outer training fold by the caller; the outer test fold never enters the study.

Search spaces are the usual ranges of the libraries' documentation; nothing model-specific
is invented here. Parameters are those of the registered estimators
(:mod:`tranpy_core.models.registry`), addressed through ``Candidate.params``.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from tranpy_core._optional import require
from tranpy_core.errors import ConfigurationError
from tranpy_core.random import Seed, spawn_seed

__all__ = [
    "SEARCH_SPACES",
    "TuningResult",
    "TuningSpec",
    "inner_splitter",
    "suggest_params",
    "tune_params",
]


@dataclass(frozen=True)
class TuningSpec:
    """How to tune a candidate inside every outer training fold.

    Attributes:
        n_trials: Optuna trials per outer fold.
        inner_folds: folds of the inner ``StratifiedGroupKFold`` (by hour).
        metric: scikit-learn scorer name maximised by the study.
        timeout_s: wall-clock limit of one study (None: trials only).
        only_sets: feature sets that are tuned; empty means every set.
    """

    n_trials: int = 30
    inner_folds: int = 3
    metric: str = "average_precision"
    timeout_s: float | None = None
    only_sets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ConfigurationError(f"tuning needs at least one trial, got {self.n_trials}")
        if self.inner_folds < 2:
            raise ConfigurationError(
                f"tuning needs at least two inner folds, got {self.inner_folds}"
            )
        try:
            get_scorer(self.metric)
        except ValueError as exc:
            raise ConfigurationError(f"unknown tuning metric {self.metric!r}") from exc
        object.__setattr__(self, "only_sets", tuple(str(s) for s in self.only_sets))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TuningSpec:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigurationError(f"unknown tuning keys: {unknown}")
        values = dict(data)
        if "only_sets" in values:
            only = values["only_sets"]
            values["only_sets"] = (only,) if isinstance(only, str) else tuple(only or ())
        return cls(**values)

    def applies_to(self, feature_set: str) -> bool:
        return not self.only_sets or feature_set in self.only_sets


@dataclass
class TuningResult:
    """Outcome of one study: the best parameters and every trial."""

    best_params: dict[str, Any]
    best_value: float
    trials: pd.DataFrame
    n_trials: int
    time_s: float
    inner_groups: str = "group"


# ----------------------------------------------------------------------------- search spaces
def _boosting(trial: Any, leaves: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if leaves:
        out["num_leaves"] = trial.suggest_int("num_leaves", 8, 128, log=True)
        out["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100, log=True)
        # LightGBM applies subsample only every subsample_freq iterations
        out["subsample_freq"] = 1
    else:
        out["max_depth"] = trial.suggest_int("max_depth", 2, 10)
        out["min_child_weight"] = trial.suggest_float("min_child_weight", 0.5, 20.0, log=True)
    return out


def _logistic(trial: Any) -> dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


def _balanced_forest(trial: Any) -> dict[str, Any]:
    depth = trial.suggest_int("max_depth", 3, 30)
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": None if depth >= 30 else depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
    }


def _dense(trial: Any) -> dict[str, Any]:
    layers = trial.suggest_int("n_layers", 1, 3)
    width = trial.suggest_categorical("width", [32, 64, 128, 256])
    return {
        "hidden_units": tuple(max(8, width >> k) for k in range(layers)),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


SEARCH_SPACES: Mapping[str, Callable[[Any], dict[str, Any]]] = {
    "lightgbm": lambda t: _boosting(t, leaves=True),
    "xgboost": lambda t: _boosting(t, leaves=False),
    "logistic_regression": _logistic,
    "balanced_random_forest": _balanced_forest,
    "dense_network": _dense,
}
"""Model name -> function of an Optuna trial returning estimator keyword arguments."""


def suggest_params(model: str, trial: Any) -> dict[str, Any]:
    """Estimator keyword arguments proposed by ``trial`` for ``model``."""
    if model not in SEARCH_SPACES:
        raise ConfigurationError(
            f"no search space for model {model!r}; tunable: {sorted(SEARCH_SPACES)}"
        )
    return SEARCH_SPACES[model](trial)


# ------------------------------------------------------------------------------------ tuning
def inner_splitter(
    y: np.ndarray, groups: np.ndarray | None, n_splits: int, seed: Seed
) -> tuple[Any, str]:
    """``StratifiedGroupKFold`` when ``groups`` has at least ``n_splits`` distinct values,
    else ``StratifiedKFold``; returns the splitter and which one it is."""
    state = spawn_seed(seed)
    if groups is not None and np.unique(groups).size >= n_splits:
        return StratifiedGroupKFold(n_splits, shuffle=True, random_state=state), "group"
    return StratifiedKFold(n_splits, shuffle=True, random_state=state), "stratified"


def tune_params(
    model: str,
    build: Callable[[Mapping[str, Any]], Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    spec: TuningSpec,
    seed: Seed = 0,
    groups: np.ndarray | None = None,
) -> TuningResult:
    """Optuna study of ``model`` on ``(X, y)`` only (one outer training fold).

    ``build(params)`` returns the unfitted estimator (pipeline) for the proposed
    ``params``; the study's objective is the mean inner-fold score of ``spec.metric``.
    """
    optuna = require("optuna", "tuning")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).astype(int)
    if np.unique(y_arr).size < 2:
        raise ConfigurationError("tuning needs both classes in the outer training fold")
    g = None if groups is None else np.asarray(groups)
    splitter, kind = inner_splitter(y_arr, g, spec.inner_folds, seed)
    folds = list(splitter.split(X_arr, y_arr, groups=g if kind == "group" else None))
    scorer = get_scorer(spec.metric)

    def objective(trial: Any) -> float:
        params = suggest_params(model, trial)
        scores = []
        for tr, te in folds:
            est = clone(build(params))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                est.fit(X_arr[tr], y_arr[tr])
            scores.append(float(scorer(est, X_arr[te], y_arr[te])))
        return float(np.mean(scores))

    t0 = time.perf_counter()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=spawn_seed(seed))
    )
    study.optimize(objective, n_trials=spec.n_trials, timeout=spec.timeout_s, n_jobs=1)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise ConfigurationError(f"no completed tuning trial for {model!r}")
    best = max(done, key=lambda t: t.value)
    # the estimator arguments are what the search space builds from the trial's values
    best_params = suggest_params(model, optuna.trial.FixedTrial(best.params))
    trials = pd.DataFrame(
        [
            {
                "number": t.number,
                "value": t.value,
                "state": t.state.name,
                "seconds": (
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_complete and t.datetime_start
                    else np.nan
                ),
                **t.params,
            }
            for t in study.trials
        ]
    )
    return TuningResult(
        best_params=best_params,
        best_value=float(best.value),
        trials=trials,
        n_trials=len(study.trials),
        time_s=time.perf_counter() - t0,
        inner_groups=kind,
    )

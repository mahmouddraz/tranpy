"""MLflow tracking and Unity Catalog registration of the benchmark (needs ``mlflow``).

Every evaluated candidate is one MLflow run (nested under the benchmark run) with its
configuration as parameters, the pooled and per-fold metrics, the per-stratum tables and
the dataset provenance as tags. The chosen model of an assessment time is refitted on all
events, logged with its input signature, registered in Unity Catalog and given the alias
``challenger``; it also gets ``champion`` only if :func:`tranpy.ml_benchmark.promotion_decision`
says so against the current champion. Metrics of a registered version are stored as
version tags, so the gate compares like with like.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from tranpy.ml_benchmark import CandidateResult, promotion_decision
from tranpy_core._optional import require

__all__ = ["log_candidate", "register_model_version"]

CHALLENGER = "challenger"
CHAMPION = "champion"
_METRICS_TAG = "cv_metrics"


def _mlflow() -> Any:
    return require("mlflow", "mlflow")


def log_candidate(
    result: CandidateResult,
    *,
    tags: Mapping[str, str] | None = None,
    nested: bool = True,
) -> str:
    """Log one cross-validated candidate as an MLflow run and return its run id."""
    mlflow = _mlflow()
    cand = result.candidate
    with mlflow.start_run(run_name=f"{result.feature_set}/{cand.name}", nested=nested) as run:
        mlflow.set_tags({**dict(tags or {}), "feature_set": result.feature_set})
        mlflow.log_params(
            {
                "model": cand.model,
                "feature_set": result.feature_set,
                "imbalance_method": cand.imbalance.method,
                "threshold_metric": cand.imbalance.threshold or "fixed_0.5",
                **{f"param_{k}": v for k, v in cand.params.items()},
                **{f"sampler_{k}": v for k, v in cand.imbalance.sampler_params.items()},
            }
        )
        mlflow.log_metrics(result.metrics())
        folds = pd.DataFrame(
            [
                {"held_out": g, "threshold": t, **s.as_dict()}
                for g, t, s in zip(result.fold_groups, result.thresholds, result.folds, strict=True)
            ]
        )
        mlflow.log_text(folds.to_csv(index=False), "folds.csv")
        for col, table in result.by_stratum.items():
            mlflow.log_text(table.to_csv(index_label=col), f"by_{col}.csv")
        return str(run.info.run_id)


def _gate_metrics(primary_metric: str) -> list[str]:
    keys = [primary_metric, "miss_rate", "average_precision", "roc_auc", "recall", "precision"]
    return list(dict.fromkeys([*keys, "mcc", "balanced_accuracy", "brier"]))


def _version_metrics(client: Any, name: str, alias: str) -> dict[str, float] | None:
    try:
        version = client.get_model_version_by_alias(name, alias)
    except Exception as exc:  # MlflowException when the alias (or model) does not exist
        if "RESOURCE_DOES_NOT_EXIST" in str(exc) or "not found" in str(exc).lower():
            return None
        raise
    raw = version.tags.get(_METRICS_TAG)
    return None if raw is None else {k: float(v) for k, v in json.loads(raw).items()}


def register_model_version(
    model: Any,
    X: pd.DataFrame,
    *,
    registered_name: str,
    metrics: Mapping[str, float],
    primary_metric: str,
    tags: Mapping[str, str],
    pip_requirements: list[str] | None = None,
) -> dict[str, Any]:
    """Log ``model`` (fitted on ``X``), register it, set ``challenger``, gate ``champion``.

    Must run inside an active MLflow run. Returns the version, the gate decision and the
    reason.
    """
    mlflow = _mlflow()
    signature = mlflow.models.infer_signature(X.head(50), model.predict_proba(X.head(50))[:, 1])
    # mlflow saves scikit-learn models with skops, which loads only trusted types; the
    # model is our own fit, so the types it uses (scorers, splitters, our wrappers) are listed
    skops_io = require("skops.io", "persistence")
    trusted = skops_io.get_untrusted_types(data=skops_io.dumps(model))
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=trusted,
        signature=signature,
        input_example=X.head(3),
        registered_model_name=registered_name,
        pip_requirements=pip_requirements,
        pyfunc_predict_fn="predict_proba",
    )
    client = mlflow.MlflowClient()
    version = str(info.registered_model_version)
    champion = _version_metrics(client, registered_name, CHAMPION)
    promote, reason = promotion_decision(metrics, champion, primary_metric=primary_metric)
    for key, value in {
        **dict(tags),
        # Unity Catalog tag values hold at most 1000 characters: the gate metrics only
        # (all metrics are in the cross-validation run)
        _METRICS_TAG: json.dumps(
            {k: round(float(metrics[k]), 6) for k in _gate_metrics(primary_metric) if k in metrics}
        ),
        "gate": "promoted" if promote else "kept_champion",
        "gate_reason": reason,
    }.items():
        client.set_model_version_tag(registered_name, version, key, value)
    client.set_registered_model_alias(registered_name, CHALLENGER, version)
    if promote:
        client.set_registered_model_alias(registered_name, CHAMPION, version)
    return {"version": version, "promoted": promote, "reason": reason}

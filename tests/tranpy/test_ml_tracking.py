"""Registered models are saved with skops (safe loading) whenever mlflow supports it; the
fallback to cloudpickle is explicit, recorded and never silent."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tranpy.ml_tracking as tracking

pytest.importorskip("skops")


class _Info:
    registered_model_version = 7


class _Client:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}
        self.aliases: dict[str, str] = {}

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        raise RuntimeError("RESOURCE_DOES_NOT_EXIST: no champion yet")

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.tags[key] = value

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version


def _fake_mlflow(monkeypatch: pytest.MonkeyPatch, *, skops_supported: bool) -> dict[str, Any]:
    """A stand-in for ``mlflow`` recording the ``log_model`` call; no tracking server."""
    seen: dict[str, Any] = {}
    mlflow = types.ModuleType("mlflow")
    sk = types.ModuleType("mlflow.sklearn")
    if skops_supported:
        sk.SERIALIZATION_FORMAT_SKOPS = "skops"  # type: ignore[attr-defined]
    sk.SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"  # type: ignore[attr-defined]

    def log_model(model: Any, **kwargs: Any) -> _Info:
        seen.update(kwargs)
        return _Info()

    sk.log_model = log_model  # type: ignore[attr-defined]
    models = types.ModuleType("mlflow.models")
    models.infer_signature = lambda X, y: "signature"  # type: ignore[attr-defined]
    client = _Client()
    mlflow.sklearn = sk  # type: ignore[attr-defined]
    mlflow.models = models  # type: ignore[attr-defined]
    mlflow.MlflowClient = lambda: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    seen["client"] = client
    return seen


@pytest.fixture
def fitted() -> tuple[Any, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = (X["a"] + 0.1 * rng.normal(size=60) > 0).astype(int)
    return make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y), X


def _register(model: Any, X: pd.DataFrame, **kw: Any) -> dict[str, Any]:
    return tracking.register_model_version(
        model,
        X,
        registered_name="cat.schema.tsa_pre",
        metrics={"average_precision": 0.8, "miss_rate": 0.2},
        primary_metric="average_precision",
        tags={"job": "1"},
        **kw,
    )


def test_models_are_logged_with_skops_when_mlflow_supports_it(monkeypatch, fitted) -> None:
    seen = _fake_mlflow(monkeypatch, skops_supported=True)
    model, X = fitted
    res = _register(model, X, pip_requirements=["tranpy-ai==0.2"])
    # the default of mlflow.sklearn.log_model is cloudpickle on Databricks (runtime or tracking
    # URI) and skops elsewhere; the format must be requested explicitly to be the same everywhere
    assert seen["serialization_format"] == "skops"
    assert isinstance(seen["skops_trusted_types"], list)
    # skops must be installable where the model is loaded
    assert any(r.startswith("skops") for r in seen["pip_requirements"])
    assert "tranpy-ai==0.2" in seen["pip_requirements"]
    assert res["serialization_format"] == "skops"
    assert seen["client"].tags["serialization_format"] == "skops"
    assert seen["client"].aliases == {"challenger": "7", "champion": "7"}
    assert res["version"] == "7"


def test_requirements_left_to_mlflow_when_none_are_given(monkeypatch, fitted) -> None:
    seen = _fake_mlflow(monkeypatch, skops_supported=True)
    model, X = fitted
    _register(model, X)
    # mlflow pins skops itself when it infers the requirements
    assert seen["pip_requirements"] is None
    assert seen["serialization_format"] == "skops"


def test_fallback_to_cloudpickle_is_explicit(monkeypatch, fitted) -> None:
    seen = _fake_mlflow(monkeypatch, skops_supported=False)
    model, X = fitted
    with pytest.warns(RuntimeWarning, match="cloudpickle"):
        res = _register(model, X, pip_requirements=["tranpy-ai==0.2"])
    assert seen["serialization_format"] == "cloudpickle"
    assert "skops_trusted_types" not in seen
    assert seen["pip_requirements"] == ["tranpy-ai==0.2"]
    assert res["serialization_format"] == "cloudpickle"
    assert seen["client"].tags["serialization_format"] == "cloudpickle"


def test_fallback_when_skops_is_not_installed(monkeypatch, fitted) -> None:
    seen = _fake_mlflow(monkeypatch, skops_supported=True)
    monkeypatch.setattr(tracking, "is_available", lambda module: module != "skops.io")
    model, X = fitted
    with pytest.warns(RuntimeWarning, match="skops is not installed"):
        res = _register(model, X)
    assert seen["serialization_format"] == "cloudpickle"
    assert res["serialization_format"] == "cloudpickle"

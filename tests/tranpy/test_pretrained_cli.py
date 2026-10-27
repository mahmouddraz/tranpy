from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tranpy.cli import main
from tranpy.datasets import data_registry
from tranpy.pretrained import list_pretrained, load_pretrained, pretrained_name, train_pretrained
from tranpy_core.data import load_signals
from tranpy_core.errors import DataError

pytestmark = pytest.mark.requires_persistence


def test_train_and_load_pretrained(
    tmp_path: Path, isolated_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("skops")
    entries = train_pretrained(
        tmp_path / "models", systems=["ieee_9bus"], models=["logistic_regression"]
    )
    name = pretrained_name("NineBusSystem", "logistic_regression")
    assert name == "ieee_9bus__logistic_regression"
    entry = entries[name]
    assert entry["test_accuracy"] > 0.9 and entry["url"] is None
    assert (
        json.loads((tmp_path / "models" / "registry_entries.json").read_text())[name]["sha256"]
        == entry["sha256"]
    )
    # identical retraining gives an identical model file
    again = train_pretrained(
        tmp_path / "again", systems=["ieee_9bus"], models=["logistic_regression"]
    )
    assert again[name]["test_accuracy"] == entry["test_accuracy"]

    published = {**entry, "url": (tmp_path / "models" / entry["filename"]).as_uri()}
    monkeypatch.setattr(
        "tranpy.pretrained.data_registry",
        lambda: {**data_registry(), "pretrained": {name: published}},
    )
    assert list_pretrained()["name"].tolist() == [name]
    model = load_pretrained(name)
    assert hasattr(model, "predict_proba")
    monkeypatch.setattr(
        "tranpy.pretrained.data_registry",
        lambda: {**data_registry(), "pretrained": {name: {**entry, "url": None}}},
    )
    with pytest.raises(DataError, match="no download URL"):
        load_pretrained(name)
    with pytest.raises(DataError, match="unknown pretrained"):
        load_pretrained("nope")


def test_cli_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["datasets"]) == 0
    assert "new_england" in capsys.readouterr().out

    config = tmp_path / "exp.yaml"
    experiment = {"name": "cli", "models": [{"name": "logistic_regression"}], "test_size": 0.3}
    config.write_text(yaml.safe_dump({"dataset": "ieee_9bus", "experiment": experiment}))
    out = tmp_path / "result.json"
    assert main(["experiment", str(config), "-o", str(out)]) == 0
    assert "logistic_regression" in capsys.readouterr().out
    assert json.loads(out.read_text())["dataset"] == "ieee_9bus"

    config.write_text(
        yaml.safe_dump(
            {"dataset": "ieee_9bus", "experiment": {"name": "x", "models": [], "tst_size": 1}}
        )
    )
    assert main(["experiment", str(config)]) == 1
    err = capsys.readouterr().err
    assert "tst_size: unknown key" in err

    sim = tmp_path / "sim.npz"
    assert (
        main(["-v", "simulate", "ieee_9bus", str(sim), "--events", "4", "--duration", "1.5"]) == 0
    )
    events = load_signals(sim)
    assert events.n_samples == 4

    from tests.conftest import write_powerfactory_csvs

    csv_dir = write_powerfactory_csvs(events, tmp_path / "csv")
    converted = tmp_path / "converted.npz"
    assert main(["convert-events", str(csv_dir), "ieee_9bus", str(converted)]) == 0
    assert load_signals(converted).n_samples == 4
    assert "sha256=" in capsys.readouterr().out

    pytest.importorskip("skops")
    assert main(["train-pretrained", str(tmp_path / "pre"), "--models", "gaussian_nb"]) == 0
    assert "new_england__gaussian_nb" in capsys.readouterr().out

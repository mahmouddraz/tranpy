from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tranpy_core.config import from_mapping, to_mapping
from tranpy_core.data import TabularDataset
from tranpy_core.errors import ConfigurationError
from tranpy_core.experiment import ExperimentSpec, ModelConfig, run_experiment


def test_run_experiment_is_deterministic(toy: TabularDataset, tmp_path: Path) -> None:
    spec = ExperimentSpec(
        name="e",
        models=(
            ModelConfig("random_forest", params={"n_estimators": 20}),
            ModelConfig("svm", label="svm_c2", params={"C": 2.0}),
        ),
        cv_folds=3,
        seed=1,
    )
    a, b = run_experiment(spec, toy), run_experiment(spec, toy)
    assert a.test_scores["random_forest"] == b.test_scores["random_forest"]
    assert set(a.cv_summary) == {"random_forest", "svm_c2"}
    out = a.save_json(tmp_path / "r" / "result.json")
    data = json.loads(out.read_text())
    assert data["spec"]["models"][1]["label"] == "svm_c2"
    assert len(data["split"]["test"]) == round(0.2 * toy.n_samples)


def test_group_duplicates_keeps_duplicates_together() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 3))
    X = np.vstack([base, base[:30]])
    y = (X[:, 0] > 0).astype(int)
    ds = TabularDataset(
        features=X, labels=y, feature_names=("a", "b", "c"), label_names=("n", "p"), name="dups"
    )
    spec = ExperimentSpec(
        name="g", models=(ModelConfig("logistic_regression"),), group_duplicates=True, cv_folds=2
    )
    result = run_experiment(spec, ds)
    groups = ds.duplicate_groups()
    assert not set(groups[result.split.train]) & set(groups[result.split.test])


def test_spec_validation_and_yaml_round_trip() -> None:
    with pytest.raises(ConfigurationError) as info:
        ExperimentSpec(name="x", models=(), test_size=2.0, cv_folds=1)
    assert len(info.value.problems) == 3
    with pytest.raises(ConfigurationError, match="duplicate labels"):
        ExperimentSpec(name="x", models=(ModelConfig("svm"), ModelConfig("svm")))
    with pytest.raises(ConfigurationError, match="unknown model"):
        ModelConfig("nope")
    spec = ExperimentSpec(
        name="y", models=(ModelConfig("knn", params={"n_neighbors": 3}),), cv_folds=2
    )
    loaded = from_mapping(ExperimentSpec, yaml.safe_load(yaml.safe_dump(to_mapping(spec))))
    assert loaded.models[0].params["n_neighbors"] == 3
    assert loaded.cv_folds == 2


@pytest.mark.requires_plot
def test_plotting(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from tranpy_core.plotting import (
        plot_importance,
        plot_metric_curves,
        publication_style,
        save_figure,
    )

    with publication_style():
        fig = plot_importance(["a", "b", "c"], [0.1, 0.5, 0.2], top=2, title="importance")
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["c", "b"]
        frame = pd.DataFrame(
            {
                "w": [0.1, 1.0, 0.1, 1.0],
                "acc": [0.8, 0.9, 0.7, 0.95],
                "sd": 0.01,
                "m": ["x", "x", "y", "y"],
            }
        )
        fig2 = plot_metric_curves(frame, x="w", y="acc", group="m", band="sd", title="t", logx=True)
        path = save_figure(fig2, tmp_path / "figs" / "curve.pdf")
        assert path.read_bytes()[:4] == b"%PDF"
        fig3, ax = plt.subplots()
        assert plot_importance(["a"], [1.0], ax=ax) is fig3
    plt.close("all")

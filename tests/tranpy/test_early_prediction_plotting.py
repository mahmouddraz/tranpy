from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from tranpy.early_prediction import find_knee_point, run_early_prediction
from tranpy_core.data import SignalDataset
from tranpy_core.errors import ConfigurationError
from tranpy_core.models import make_pipeline


def test_run_early_prediction(mock_events: SignalDataset, caplog: pytest.LogCaptureFixture) -> None:
    models = {
        "lr": LogisticRegression(max_iter=500),
        "rf": make_pipeline("random_forest", n_estimators=10),
    }
    with caplog.at_level(logging.INFO):
        result = run_early_prediction(
            mock_events,
            models,
            reference="fault_start",
            windows_s=(0.05, 0.4, 2.9),
            n_splits=3,
            seed=0,
        )
    assert set(result["model"]) == {"lr", "rf"}
    assert sorted(set(result["window_s"])) == [
        0.05,
        0.4,
    ]  # 2.9 s after the fault exceeds the recording
    assert "only" in caplog.text
    assert (result["n_splits"] == 3).all()
    assert result["accuracy_mean"].between(0, 1).all()
    again = run_early_prediction(
        mock_events, models, reference="fault_start", windows_s=(0.05, 0.4), n_splits=3, seed=0
    )
    pd.testing.assert_frame_equal(
        result.drop(columns="fit_time_s").reset_index(drop=True),
        again.drop(columns="fit_time_s").reset_index(drop=True),
    )
    absolute = run_early_prediction(
        mock_events,
        {"lr": LogisticRegression()},
        reference="absolute",
        windows_s=(2.0,),
        n_splits=2,
    )
    assert absolute["n_events"].iloc[0] == mock_events.n_samples


@pytest.mark.requires_neural
def test_sequence_models_share_folds_with_feature_models(mock_events: SignalDataset) -> None:
    pytest.importorskip("tensorflow")
    result = run_early_prediction(
        mock_events,
        {"lr": LogisticRegression(max_iter=500)},
        sequence_models={
            "dense": make_pipeline("dense_network", hidden_units=(4,), epochs=1),
            "lstm": make_pipeline("lstm_network", lstm_units=(4,), dense_units=4, epochs=1),
        },
        reference="fault_start",
        windows_s=(0.2,),
        n_splits=2,
        sequence_settings={"step_s": 0.05},
    )
    assert result.set_index("model")["input"].to_dict() == {
        "lr": "features",
        "dense": "sequence",
        "lstm": "sequence",
    }
    assert (result["fit_time_s"] > 0).all()
    assert result.loc[result.model == "lstm", "n_features"].iloc[0] % 7 == 0


def test_skips_and_errors(mock_events: SignalDataset, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ConfigurationError):
        run_early_prediction(mock_events, {}, reference="fault_start")
    with pytest.raises(ConfigurationError, match="used twice"):
        run_early_prediction(
            mock_events,
            {"m": LogisticRegression()},
            sequence_models={"m": LogisticRegression()},
            reference="fault_start",
        )
    one_class = mock_events.subset(np.flatnonzero(mock_events.labels == 0)[:12])
    with caplog.at_level(logging.INFO):
        empty = run_early_prediction(
            one_class, {"rf": RandomForestClassifier()}, reference="fault_start", windows_s=(0.2,)
        )
    assert empty.empty and "fewer than 2 events" in caplog.text


def test_non_finite_features_are_replaced_with_warning(
    mock_events: SignalDataset, caplog: pytest.LogCaptureFixture
) -> None:
    values = mock_events.values.copy()
    values[:, mock_events.channel_index("G 01|s:xspeed in p.u.")] = np.nan
    with caplog.at_level(logging.WARNING):
        run_early_prediction(
            mock_events.replace(values=values),
            {"lr": LogisticRegression()},
            reference="fault_start",
            windows_s=(0.3,),
            n_splits=2,
        )
    assert "non-finite" in caplog.text


def test_find_knee_point() -> None:
    windows = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    acc = [0.70, 0.85, 0.95, 0.96, 0.965, 0.97]
    frame = pd.DataFrame({"window_s": windows, "model": "m", "accuracy_mean": acc})
    assert find_knee_point(frame, "m") == 0.5
    assert find_knee_point(frame.iloc[:2], "m") == 0.25
    with pytest.raises(ConfigurationError):
        find_knee_point(frame, "other")


@pytest.mark.requires_plot
def test_domain_plots(mock_events: SignalDataset) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from tranpy.plotting import plot_early_prediction, plot_trajectories
    from tranpy_core.errors import DataError

    fig = plot_trajectories(mock_events, "G 01|c:firot in deg", per_class=2, title="rotor angle")
    assert len(fig.axes[0].lines) >= 4
    plot_trajectories(mock_events, "Bus 01|m:u in p.u.", align=None)
    with pytest.raises(DataError, match="exactly one channel"):
        plot_trajectories(mock_events, "Bus*")
    frame = pd.DataFrame(
        {"window_s": [0.1, 1.0], "model": "m", "accuracy_mean": [0.8, 0.9], "accuracy_std": 0.01}
    )
    plot_early_prediction(frame, title="early")
    plt.close("all")

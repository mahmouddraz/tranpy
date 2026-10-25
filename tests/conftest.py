"""Shared fixtures. No test needs network access or PowerFactory."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")

from tranpy.simulation import MockEngine, ScenarioSampler, generate_events
from tranpy_core.data import SignalDataset, TabularDataset


def _toy(n: int = 240, seed: int = 0) -> TabularDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 1] *= 50.0  # mixed scales, so scaling matters
    y = (X[:, 0] + 0.02 * X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return TabularDataset(
        features=X,
        labels=y,
        feature_names=tuple(f"f{i}" for i in range(6)),
        label_names=("negative", "positive"),
        name="toy",
    )


@pytest.fixture
def toy() -> TabularDataset:
    return _toy()


@pytest.fixture(scope="session")
def mock_events_full() -> SignalDataset:
    """40 seeded 9-bus mock events, including the out-of-step label channels."""
    engine = MockEngine("ieee_9bus", simulation_time_s=3.0)
    sampler = ScenarioSampler(
        n_lines=engine.n_lines, n_loads=engine.n_loads, simulation_time_s=3.0, clearing_cycles=20
    )
    return generate_events(engine, sampler.sample(40, seed=3))


@pytest.fixture(scope="session")
def mock_events(mock_events_full: SignalDataset) -> SignalDataset:
    return mock_events_full.drop_channels("*|s:outofstep")


def write_powerfactory_csvs(dataset: SignalDataset, directory: Path) -> Path:
    """Write ``dataset`` in the two-header-row CSV format of PowerFactory exports."""
    directory.mkdir(parents=True, exist_ok=True)
    elements = ["Results"] + [f"\\grid\\{c.split('|')[0]}.ElmTerm" for c in dataset.channel_names]
    variables = ["b:tnow in s"] + [c.split("|", 1)[1] for c in dataset.channel_names]
    columns = pd.MultiIndex.from_arrays([elements, variables])
    assert dataset.sample_ids is not None
    for i in range(dataset.n_samples):
        sample = dataset[i]
        frame = pd.DataFrame(np.column_stack([sample.time, sample.values]), columns=columns)
        frame.to_csv(directory / f"results_event_{dataset.sample_ids[i]}.csv", index=False)
    return directory


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setenv("TRANPY_CACHE_DIR", str(cache))
    return cache

# TranPy

<!-- intro-start -->
Machine learning for power-system **transient stability**: datasets of simulated fault
events, classifiers, evaluation, explanations and simulation engines, with results that
are reproducible by construction (explicit seeds, checksum-pinned data, a regression
snapshot).

The package has two layers:

* **`tranpy_core`** - a domain-agnostic pipeline for labelled tabular and time-series
  data: datasets, splits, feature extraction, a model registry of scikit-learn
  estimators, evaluation, SHAP/LIME/break-down explanations and an experiment runner.
* **`tranpy`** - the power-system layer on top: the IEEE 9-bus and New England 39-bus
  datasets, trajectory features (rotor angle, voltage, frequency, power, coherency),
  early-prediction experiments, a PowerFactory engine and a mock engine.
<!-- intro-end -->

## Install

```bash
pip install tranpy-ai                    # core: numpy, pandas, scipy, scikit-learn
pip install "tranpy-ai[explain]"         # SHAP, LIME, DALEX
pip install "tranpy-ai[neural]"          # TensorFlow networks
pip install "tranpy-ai[boosting]"        # XGBoost, LightGBM
pip install "tranpy-ai[plot]"            # matplotlib figures
pip install "tranpy-ai[all]"             # everything
```

Python 3.11 to 3.13. The import names are `tranpy` and `tranpy_core`.

## Quick start

```python
import tranpy
from tranpy_core import evaluate, make_pipeline, train_test_split

data = tranpy.load_new_england()                       # 5000 events, 77 features
split = train_test_split(data.labels, test_size=0.2, seed=42)
model = make_pipeline("random_forest", seed=0)         # StandardScaler -> RandomForest
model.fit(data.features[split.train], data.labels[split.train])
scores = evaluate(model, data.features[split.test], data.labels[split.test])
print(scores.accuracy, scores.roc_auc)
```

Event trajectories and early prediction:

```python
from sklearn.linear_model import LogisticRegression
from tranpy.early_prediction import run_early_prediction

events = tranpy.load_events("ieee_9bus", source="path/to/results_event_csvs/")
table = tranpy.TrajectoryFeatures(window_s=0.5, reference="fault_start").transform(events)
curve = run_early_prediction(events, {"lr": LogisticRegression()}, reference="fault_start")
```

The window `reference` is required: `fault_start` is causal; `fault_clear` reproduces the
earlier (non-causal) setting - see [docs/validation.md](docs/validation.md#defects-found).

## Command line

```bash
tranpy datasets                                    # available data
tranpy experiment experiment.yaml -o result.json   # run a YAML experiment
tranpy simulate ieee_9bus events.npz --events 200  # mock-engine dataset
tranpy convert-events CSV_DIR new_england out.npz  # PowerFactory CSVs -> .npz
tranpy train-pretrained models/                    # rebuild pretrained pipelines
```

An experiment file:

```yaml
dataset: new_england
experiment:
  name: baseline
  seed: 0
  test_size: 0.2
  cv_folds: 5
  models:
    - name: random_forest
    - name: svm
      params: {C: 2.0}
```

## Data

The snapshot datasets are bundled. Event trajectories (about 300 MB per system) load
from a folder of PowerFactory result CSVs or from a converted `.npz`; once published, they
are downloaded on first use, verified by SHA-256 and cached (`TRANPY_CACHE_DIR`).

## Development

```bash
uv sync --all-extras --group dev
uv run pytest                      # 266 tests, branch coverage >= 90 %
uv run pytest -m "not slow"        # skips the Gaussian-process snapshot entries
uv run ruff check . && uv run ruff format --check .
uv run mypy                        # strict
```

**Pinned results.** `tests/snapshots/metrics.json` holds the seeded metrics of every
model on every bundled dataset. If a change is intended, run
`uv run python -m tests.update_snapshot` and give the reason in the commit message.

Documentation: [architecture](docs/architecture.md), [validation](docs/validation.md),
[agent guide](AGENTS.md).

## License

MIT

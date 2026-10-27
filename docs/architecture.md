# Architecture

TranPy has two layers in one distribution (`tranpy-ai`):

```
tranpy        power-system layer: systems, vocabulary, datasets, trajectory features,
              early prediction, simulation engines, pretrained models, CLI, plots
   |
   v  (one-way dependency, enforced by tests/test_import_boundaries.py)
tranpy_core   generic layer: labelled tabular / time-series data -> features ->
              models -> evaluation -> explanation, experiment runner
```

`tranpy_core` knows nothing about buses or generators; it could serve any labelled
time-series or tabular classification problem (see notebook 06). Optional backends
(SHAP, LIME, DALEX, TensorFlow, XGBoost, LightGBM, matplotlib, skops, PowerFactory) are
imported only inside the functions that need them.

## Vocabulary

| Generic (`tranpy_core`) | Power system (`tranpy`) |
|---|---|
| sample | simulated fault event |
| `TabularDataset` | snapshot dataset (one feature vector per event) |
| `SignalDataset` | event trajectories (all recorded variables over time) |
| channel `"<element>|<variable>"` | PowerFactory result variable, e.g. `G 01|c:firot in deg` |
| label 0 / 1 | `Stability.STABLE` / `Stability.UNSTABLE` |
| annotation | per-event scalar, e.g. `fault_start_s`, `fault_clear_s` |
| feature extractor | `TrajectoryFeatures` (rotor, voltage, frequency, power, coherency) |
| model registry entry | classifier name, e.g. `random_forest` |

Units are part of names where ambiguous: `_s` seconds, `_hz`, `_pu`, `_deg`, `_ohm`,
`_pct`.

## Modules

| Module | Responsibility |
|---|---|
| `tranpy_core.errors` | `TranpyError` -> `ConfigurationError`, `DataError` (`ChecksumError`), `ModelError`, `SimulationError`, `MissingDependencyError` |
| `tranpy_core.config` | frozen dataclasses from mappings/YAML; unknown keys rejected, all problems reported together |
| `tranpy_core.random` | `make_rng(seed)`; no global RNG anywhere |
| `tranpy_core.data` | immutable containers, reproducible `.npz` storage, checksum-verified downloads, seeded stratified/grouped splits |
| `tranpy_core.features` | vectorised segment reductions over ragged samples, windows, generic statistics |
| `tranpy_core.models` | registry: name -> scikit-learn estimator + defaults + seed parameter; Keras networks as estimators; skops persistence |
| `tranpy_core.evaluation` | metrics, cross-validation that clones the whole pipeline per fold |
| `tranpy_core.explain` | SHAP, LIME, break-down -> `Explanation`; logistic surrogate |
| `tranpy_core.experiment` | `ExperimentSpec` -> results (JSON) |
| `tranpy_core.plotting` | publication style and generic plots |
| `tranpy.systems`, `tranpy.vocabulary` | test systems, stability enum, element kinds, variable names |
| `tranpy.datasets` | bundled snapshots, event loading (npz / PowerFactory CSV / download), labels, fault timing |
| `tranpy.features` | `TrajectoryFeatures` with an explicit window reference |
| `tranpy.early_prediction` | accuracy versus observation window |
| `tranpy.simulation` | `SimulationEngine` protocol, scenarios, `MockEngine`, `PowerFactoryEngine`, `generate_events` |
| `tranpy.pretrained` | seeded training and verified loading of published pipelines |
| `tranpy.cli` | `tranpy datasets | experiment | convert-events | simulate | train-pretrained` |

## Data flow

```
PowerFactory / MockEngine --generate_events--> SignalDataset (.npz)
                                                   |
                         TrajectoryFeatures(window_s, reference)
                                                   v
bundled snapshots ----------------------------> TabularDataset
                                                   |
                         train_test_split / kfold_splits (seeded, stratified)
                                                   v
                         make_pipeline(name) = StandardScaler -> estimator
                                                   |
                         evaluate / cross_validate / run_experiment
                                                   v
                         explain_shap / explain_lime / fit_surrogate
```

Leakage rules built into this flow: feature extraction is stateless and only sees data
up to the window end; anything fitted from data (scaling, models) lives in a pipeline and
is refitted on each training fold; label-defining channels (`s:outofstep`) are removed
when events are loaded.

## Extension points

* **Model**: `register_model(ModelSpec(name="my_model", estimator="pkg.module.Class",
  defaults={...}, seed_param="random_state", extra="..."))`. Any scikit-learn compatible
  class works; implement `__init__` storing parameters only.
* **Feature extractor**: any object with `transform(SignalDataset) -> TabularDataset`
  (`tranpy_core.features.SignalFeatureExtractor`). Build it from
  `tranpy_core.features.segments` to stay vectorised.
* **Dataset**: write a `TabularDataset`/`SignalDataset` with `save_tabular`/`save_signals`,
  then add an entry with filename, SHA-256 and URL to `src/tranpy/resources/registry.json`.
* **Explainer**: a function `(model, X, *, background, seed) -> Explanation`.
* **Simulation engine**: implement the `SimulationEngine` protocol (`system`, `n_lines`,
  `n_loads`, `simulation_time_s`, `simulate(scenarios) -> list[SimulationResult]`);
  `generate_events` then works unchanged.

## Data hosting

See [data.md](data.md): bundled snapshots, sharded event datasets streamed from GitHub release assets,
lossless compact storage, checksums everywhere.

# TranPy Simulation Module

Complete PowerFactory simulation implementation for generating transient stability datasets.

## Overview

This module provides a standalone, fully-featured interface to DIgSILENT PowerFactory for running transient stability simulations and generating training datasets. It replaces the legacy code in `tranpy/src/` with a clean, modular architecture.

## Architecture

```
tranpy/simulation/
├── __init__.py                  # Module exports
├── powerfactory_config.py       # PowerFactory path configuration
├── model.py                     # Grid model configuration
├── events.py                    # Event generation and configuration
├── simulator.py                 # Main simulation orchestration
├── results.py                   # Results data structures
├── dataset_generator.py         # Dataset extraction
└── config.py                    # Configuration management
```

## Quick Start

### 1. Configure PowerFactory Path

```python
from tranpy.simulation import configure_powerfactory_path

# Option 1: Auto-detect
configure_powerfactory_path(version='2019')

# Option 2: Custom path
configure_powerfactory_path(
    custom_path=r"C:\Program Files\DIgSILENT\PowerFactory 2019\Python\3.13"
)
```

### 2. Run Simulation

```python
from tranpy.simulation import PowerSystemSimulator

# Create simulator
simulator = PowerSystemSimulator(
    grid='NewEngland',
    simulation_time=10.0,
    output_dir='simulation_results'
)

# Run simulations
results = simulator.run(
    num_events=100,
    fault_clearing_time=[10, 12],  # cycles for [NineBus, NewEngland]
    max_load_change=[60, 60],      # percentage
    random_seed=42,
    save_results=True,
    export_csv=True
)

# View statistics
print(results.get_statistics())
```

### 3. Generate Dataset

```python
from tranpy.simulation import generate_dataset_from_simulation

# Generate dataset from results
dataset, splits = generate_dataset_from_simulation(results)

print(f"Dataset shape: {dataset.data.shape}")
print(f"Train samples: {splits['X_train'].shape[0]}")
print(f"Test samples: {splits['X_test'].shape[0]}")
```

### 4. Use Configuration Files

```python
from tranpy.simulation import load_config_from_yaml, PowerSystemSimulator

# Load config from YAML (compatible with legacy format)
config = load_config_from_yaml('config.yaml')

# Run with config
simulator = PowerSystemSimulator(
    grid=config.grid,
    simulation_time=config.simulation_time,
    output_dir=config.output_path
)

results = simulator.run(
    num_events=config.number_of_events,
    fault_clearing_time=config.fault_clearing_time_cycles,
    max_load_change=config.max_load_change
)
```

## Module Components

### PowerFactoryModel (`model.py`)

Manages PowerFactory grid configuration:
- Project and study case activation
- Grid element access (buses, lines, generators, loads)
- Results monitoring setup
- Initial conditions configuration
- Simulation execution

### EventGenerator & PowerFactoryEventConfigurator (`events.py`)

Handles event generation:
- **EventGenerator**: Random parameter generation
  - Fault locations, times, impedances
  - Load changes
  - Clearing times

- **PowerFactoryEventConfigurator**: PowerFactory event creation
  - IEEE 9-bus events (3 events: fault, clearing, load)
  - New England 39-bus events (4 events: fault, clearing, line trip, load)

### PowerSystemSimulator (`simulator.py`)

Main simulation orchestrator:
1. Connect to PowerFactory
2. Setup grid model
3. Loop through events:
   - Configure events
   - Execute simulation
   - Extract results
   - Save data
4. Generate summary statistics

### SimulationResults (`results.py`)

Data structures for storing results:
- **EventInfo**: Event metadata
- **BusSnapshot**: Voltage and angle snapshots
- **GeneratorSnapshot**: Generator state
- **EventResult**: Complete event result
- **SimulationResults**: Collection of all events
- **LegacyDataFormat**: Backward compatibility

### DatasetGenerator (`dataset_generator.py`)

Convert simulation results to datasets:
- Extract features (voltage, angle) from snapshots
- Create train/test/validation splits
- Generate legacy format for backward compatibility
- Export to pickle files

### SimulationConfig (`config.py`)

Configuration management:
- YAML file loading/saving
- Configuration validation
- Templates (quick_test, standard, detailed, large_scale)
- Grid-specific parameters

## Testing

### Test PowerFactory Connection

```bash
python scripts/test_simulation.py --test-connection
```

### Run Quick Test (2 events)

```bash
python scripts/test_simulation.py --grid NewEngland --events 2
```

### Test with Config File

```bash
python scripts/test_simulation.py --config tranpy/src/config.yaml
```

### Test with Custom PowerFactory Path

```bash
python scripts/test_simulation.py \
    --pf-path "C:/Program Files/DIgSILENT/PowerFactory 2019/Python/3.13" \
    --grid NewEngland \
    --events 2
```

## Configuration Templates

```python
from tranpy.simulation import create_config_from_template

# Quick test: 10 events, no CSV export
config = create_config_from_template('quick_test', grid='NewEngland')

# Standard: 1000 events
config = create_config_from_template('standard', grid='NewEngland')

# Detailed: 1000 events with CSV export
config = create_config_from_template('detailed', grid='NewEngland')

# Large scale: 10000 events
config = create_config_from_template('large_scale', grid='NewEngland')

# Custom overrides
config = create_config_from_template(
    'standard',
    grid='NewEngland',
    simulation_time=15.0,
    fault_clearing_time_cycles=[8, 10]
)
```

## Supported Grids

### NewEngland (39-bus system)
- 39 buses
- 46 lines
- 10 generators
- Event sequence: Fault → Clearing → Line Trip → Load Change

### NineBusSystem (IEEE 9-bus)
- 9 buses
- 9 lines
- 3 generators
- Event sequence: Fault → Clearing → Load Change

## Output Structure

```
simulation_results/
├── NewEngland/
│   ├── events/
│   │   ├── event_0.csv
│   │   ├── event_1.csv
│   │   └── ...
│   └── pickles/
│       └── NewEngland_results.pickle
├── data/
│   └── NewEngland.pickle (legacy format)
└── data_set/
    └── NewEngland.pickle (dataset format)
```

## Legacy Compatibility

The new code generates output compatible with the old training pipeline:

```python
# Legacy dataset format
from tranpy.simulation import generate_legacy_dataset

generate_legacy_dataset(results, 'data_set/NewEngland.pickle')

# Loads as: [X_train, X_test, y_train, y_test, train, test, val, data]
```

## Differences from Legacy Code

| Aspect | Legacy | New |
|--------|--------|-----|
| Structure | Monolithic single file | Modular components |
| Dependencies | Embedded in models.py | Clean imports |
| Data format | Custom Data class | Dataclass-based |
| Configuration | Global state | Config objects |
| Testing | Manual | Automated test script |
| Documentation | Minimal comments | Full docstrings |
| Type hints | None | Complete typing |
| Error handling | Basic | Comprehensive |

## Migration Guide

### Old Code
```python
from models import Model
from dataset import model_grid, get_dataset

model = Model('NewEngland', simulation_time=10)
model.model()
model.run_model(...)

X_train, X_test, y_train, y_test, ... = get_dataset('NewEngland', path)
```

### New Code
```python
from tranpy.simulation import PowerSystemSimulator, generate_dataset_from_simulation

simulator = PowerSystemSimulator('NewEngland', simulation_time=10)
results = simulator.run(num_events=100)

dataset, splits = generate_dataset_from_simulation(results)
X_train = splits['X_train']
X_test = splits['X_test']
```

## Next Steps

After verifying the new implementation works:
1. Run test script with 2-5 events
2. Compare output pickle structure with legacy format
3. Test with existing training code
4. Generate full datasets (1000+ events)
5. Delete legacy code (tranpy/src/*.py)

## Troubleshooting

### PowerFactory not found
```python
from tranpy.simulation import get_powerfactory_info
print(get_powerfactory_info())
```

### Check if simulation available
```python
from tranpy.simulation import is_powerfactory_available
if not is_powerfactory_available():
    print("Configure PowerFactory path first")
```

### Debug mode
Set `export_csv=True` to save CSV files for debugging:
```python
results = simulator.run(num_events=10, export_csv=True)
```

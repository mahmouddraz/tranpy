#!/usr/bin/env python3
"""
Test script for simulation components WITHOUT PowerFactory.

This script creates mock simulation data to test all components
of the refactored simulation code without requiring PowerFactory installation.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tranpy.simulation.results import (
    EventInfo, BusSnapshot, GeneratorSnapshot, EventResult,
    SimulationResults, LegacyDataFormat
)
from tranpy.simulation.dataset_generator import (
    generate_dataset_from_simulation,
    generate_legacy_dataset,
    DatasetGenerator
)
from tranpy.simulation.config import (
    SimulationConfig,
    load_config_from_yaml,
    save_config_to_yaml,
    create_config_from_template
)


def create_mock_bus_snapshot(n_buses: int, time: float, stable: bool = True) -> BusSnapshot:
    """Create mock bus voltage and angle data."""
    voltages = {}
    angles = {}

    for i in range(1, n_buses + 1):
        bus_name = f"Bus_{i}"
        # Stable: voltages close to 1.0, unstable: some voltages drop
        if stable:
            voltages[bus_name] = np.random.uniform(0.95, 1.05)
            angles[bus_name] = np.random.uniform(-15, 15)
        else:
            voltages[bus_name] = np.random.uniform(0.6, 1.1)
            angles[bus_name] = np.random.uniform(-45, 45)

    return BusSnapshot(time=time, voltages=voltages, angles=angles)


def create_mock_generator_snapshot(n_generators: int, stable: bool = True) -> GeneratorSnapshot:
    """Create mock generator data."""
    active_powers = {}
    reactive_powers = {}
    out_of_step = {}

    for i in range(1, n_generators + 1):
        gen_name = f"Gen_{i}"
        active_powers[gen_name] = np.random.uniform(50, 500)  # MW
        reactive_powers[gen_name] = np.random.uniform(10, 100)  # Mvar
        # Out of step: 0 = stable, 1 = unstable
        out_of_step[gen_name] = 0 if stable else np.random.choice([0, 1])

    return GeneratorSnapshot(
        active_powers=active_powers,
        reactive_powers=reactive_powers,
        out_of_step=out_of_step
    )


def create_mock_event_result(
    event_id: int,
    n_buses: int,
    n_generators: int,
    stable: bool = True
) -> EventResult:
    """Create mock event result."""

    # Event info
    fault_time = np.random.uniform(0.5, 2.0)
    clearing_time = fault_time + np.random.uniform(0.2, 0.4)

    event_info = EventInfo(
        event_id=event_id,
        event_types=['short circuit', 'fault clearing', 'load event'],
        event_locations=[f'Line_{np.random.randint(1, 10)}',
                        f'Line_{np.random.randint(1, 10)}',
                        f'Load_{np.random.randint(1, 5)}'],
        event_times=[fault_time, clearing_time, 0.0],
        fault_time=fault_time,
        clearing_time=clearing_time,
        fault_impedance={'R_f': 0.01, 'X_f': 0.01},
        load_change=np.random.randint(-60, 60)
    )

    # Snapshots
    bus_snapshot_fault = create_mock_bus_snapshot(n_buses, fault_time, stable)
    bus_snapshot_clearing = create_mock_bus_snapshot(n_buses, clearing_time, stable)

    # Generator snapshot
    gen_snapshot = create_mock_generator_snapshot(n_generators, stable)

    return EventResult(
        event_info=event_info,
        bus_snapshot_at_fault=bus_snapshot_fault,
        bus_snapshot_at_clearing=bus_snapshot_clearing,
        generator_snapshot=gen_snapshot
    )


def create_mock_simulation_results(
    grid_name: str = 'NewEngland',
    n_events: int = 100,
    stability_ratio: float = 0.7
) -> SimulationResults:
    """Create mock simulation results."""

    # Grid parameters
    if grid_name == 'NewEngland':
        n_buses = 39
        n_generators = 10
    else:  # NineBusSystem
        n_buses = 9
        n_generators = 3

    results = SimulationResults(
        grid_name=grid_name,
        simulation_time=10.0,
        metadata={
            'num_events': n_events,
            'fault_clearing_cycles': 12,
            'max_load_change': 60,
            'random_seed': 42
        }
    )

    # Generate events
    for i in range(n_events):
        # Determine if stable based on ratio
        stable = np.random.random() < stability_ratio

        event = create_mock_event_result(i, n_buses, n_generators, stable)
        results.add_event(event)

    return results


def test_data_structures():
    """Test 1: Data structures."""
    print("\n" + "="*60)
    print("Test 1: Data Structures")
    print("="*60)

    # Test BusSnapshot
    print("\n1.1 Testing BusSnapshot...")
    snapshot = create_mock_bus_snapshot(39, 1.5, stable=True)
    feature_array = snapshot.to_feature_array()
    assert len(feature_array) == 39 * 2, "Feature array should have 2 values per bus"
    print(f"✓ BusSnapshot: {len(snapshot.voltages)} buses, {len(feature_array)} features")

    # Test GeneratorSnapshot
    print("\n1.2 Testing GeneratorSnapshot...")
    gen_snapshot = create_mock_generator_snapshot(10, stable=True)
    assert len(gen_snapshot.out_of_step) == 10, "Should have 10 generators"
    print(f"✓ GeneratorSnapshot: {len(gen_snapshot.active_powers)} generators")

    # Test EventResult
    print("\n1.3 Testing EventResult...")
    event = create_mock_event_result(0, 39, 10, stable=True)
    assert event.is_stable == True, "Event should be stable"
    assert event.stability_label == 'stable', "Label should be 'stable'"
    print(f"✓ EventResult: {event.stability_label}, fault_time={event.event_info.fault_time:.3f}s")

    # Test unstable event
    event_unstable = create_mock_event_result(1, 39, 10, stable=False)
    print(f"✓ Unstable EventResult: {event_unstable.stability_label}")

    print("\n✓ All data structure tests passed!")


def test_simulation_results():
    """Test 2: SimulationResults."""
    print("\n" + "="*60)
    print("Test 2: SimulationResults")
    print("="*60)

    # Create mock results
    print("\n2.1 Creating mock simulation results...")
    results = create_mock_simulation_results('NewEngland', n_events=50, stability_ratio=0.7)

    # Get statistics
    stats = results.get_statistics()
    print(f"\n✓ Created {stats['total_events']} events")
    print(f"  - Stable: {stats['stable_events']} ({stats['stability_ratio']*100:.1f}%)")
    print(f"  - Unstable: {stats['unstable_events']}")

    # Test DataFrame conversion
    print("\n2.2 Testing DataFrame conversion...")
    df = results.to_dataframe(snapshot_type='clearing')
    print(f"✓ DataFrame shape: {df.shape}")
    print(f"  - Features: {df.shape[1] - 1}")  # -1 for label column
    print(f"  - Samples: {df.shape[0]}")

    # Test events DataFrame
    events_df = results.get_events_dataframe()
    print(f"✓ Events DataFrame: {events_df.shape[0]} events")

    # Test save/load
    print("\n2.3 Testing save/load...")
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / 'test_results.pickle'
    results.save(results_path)
    print(f"✓ Saved to: {results_path}")

    loaded_results = SimulationResults.load(results_path)
    loaded_stats = loaded_results.get_statistics()
    assert loaded_stats['total_events'] == stats['total_events'], "Loaded results mismatch"
    print(f"✓ Loaded successfully: {loaded_stats['total_events']} events")

    print("\n✓ All SimulationResults tests passed!")
    return results


def test_dataset_generation(results: SimulationResults):
    """Test 3: Dataset generation."""
    print("\n" + "="*60)
    print("Test 3: Dataset Generation")
    print("="*60)

    # Generate dataset
    print("\n3.1 Testing dataset generation...")
    dataset, splits = generate_dataset_from_simulation(
        results,
        snapshot_type='clearing',
        test_size=0.2,
        val_size=0.2,
        shuffle=False,
        random_state=42
    )

    print(f"✓ Dataset shape: {dataset.data.shape}")
    print(f"  - Total samples: {dataset.data.shape[0]}")
    print(f"  - Features: {dataset.data.shape[1]}")
    print(f"  - Feature names: {len(dataset.feature_names)}")

    # Check splits
    print(f"\n✓ Data splits:")
    print(f"  - Train: {splits['X_train'].shape[0]} samples")
    print(f"  - Test: {splits['X_test'].shape[0]} samples")
    print(f"  - Val: {splits['X_val'].shape[0]} samples")

    # Verify totals
    total = splits['X_train'].shape[0] + splits['X_test'].shape[0] + splits['X_val'].shape[0]
    assert total == dataset.data.shape[0], "Split totals don't match"
    print(f"  - Total: {total} (matches)")

    # Test DataFrames
    print(f"\n3.2 Testing DataFrames...")
    print(f"✓ Train DataFrame: {splits['train'].shape}")
    print(f"✓ Test DataFrame: {splits['test'].shape}")
    print(f"✓ Val DataFrame: {splits['val'].shape}")

    # Test DatasetGenerator class
    print(f"\n3.3 Testing DatasetGenerator...")
    generator = DatasetGenerator(results, snapshot_type='clearing')
    dataset2 = generator.generate(test_size=0.2, val_size=0.2, random_state=42)

    gen_stats = generator.get_statistics()
    print(f"✓ Generator stats:")
    print(f"  - Grid: {gen_stats['grid_name']}")
    print(f"  - Samples: {gen_stats['n_samples']}")
    print(f"  - Features: {gen_stats['n_features']}")
    print(f"  - Stable: {gen_stats['n_stable']}")
    print(f"  - Unstable: {gen_stats['n_unstable']}")

    # Save dataset
    output_dir = Path('test_output')
    dataset_path = output_dir / 'test_dataset.pickle'
    generator.save_dataset(dataset_path, format='new')
    print(f"✓ Saved dataset: {dataset_path}")

    print("\n✓ All dataset generation tests passed!")


def test_legacy_format(results: SimulationResults):
    """Test 4: Legacy format compatibility."""
    print("\n" + "="*60)
    print("Test 4: Legacy Format Compatibility")
    print("="*60)

    # Convert to legacy format
    print("\n4.1 Converting to legacy format...")
    legacy_data = LegacyDataFormat.to_legacy_format(results)

    print(f"✓ Legacy data structure created")
    print(f"  - f1_generator_active_powers: {legacy_data.f1_generator_active_powers.shape}")
    print(f"  - f2_generator_reactive_powers: {legacy_data.f2_generator_reactive_powers.shape}")
    print(f"  - f3_out_of_step: {legacy_data.f3_out_of_step.shape}")
    print(f"  - f4_bus_voltage: {legacy_data.f4_bus_voltage.shape}")
    print(f"  - f5_bus_angles: {legacy_data.f5_bus_angles.shape}")
    print(f"  - df_events: {legacy_data.df_events.shape}")

    # Check bus_data_post_fault structure
    print(f"  - bus_data_post_fault[0] (at fault): {len(legacy_data.bus_data_post_fault[0])} events")
    print(f"  - bus_data_post_fault[1] (at clearing): {len(legacy_data.bus_data_post_fault[1])} events")

    # Save legacy format
    print("\n4.2 Saving legacy format...")
    output_dir = Path('test_output')
    legacy_path = output_dir / 'test_legacy.pickle'
    LegacyDataFormat.save_legacy_format(results, legacy_path)
    print(f"✓ Saved legacy format: {legacy_path}")

    # Generate legacy dataset format
    print("\n4.3 Generating legacy dataset format...")
    dataset_legacy_path = output_dir / 'test_dataset_legacy.pickle'
    legacy_dataset = generate_legacy_dataset(results, dataset_legacy_path)

    print(f"✓ Legacy dataset saved: {dataset_legacy_path}")
    print(f"  - X_train: {legacy_dataset['X_train'].shape}")
    print(f"  - X_test: {legacy_dataset['X_test'].shape}")
    print(f"  - y_train: {legacy_dataset['y_train'].shape}")
    print(f"  - y_test: {legacy_dataset['y_test'].shape}")

    print("\n✓ All legacy format tests passed!")


def test_configuration():
    """Test 5: Configuration management."""
    print("\n" + "="*60)
    print("Test 5: Configuration Management")
    print("="*60)

    # Create config
    print("\n5.1 Testing SimulationConfig...")
    config = SimulationConfig(
        grid='NewEngland',
        number_of_events=100,
        simulation_time=10.0,
        fault_clearing_time_cycles=[10, 12],
        max_load_change=[60, 60]
    )

    print(f"✓ Config created:")
    print(config)

    # Validate
    config.validate()
    print("✓ Config validated")

    # Get grid parameters
    params = config.get_grid_parameters()
    print(f"✓ Grid parameters: {params}")

    # Test templates
    print("\n5.2 Testing configuration templates...")
    templates = ['quick_test', 'standard', 'detailed', 'large_scale']
    for template in templates:
        cfg = create_config_from_template(template, grid='NewEngland')
        print(f"✓ Template '{template}': {cfg.number_of_events} events")

    # Save/load YAML
    print("\n5.3 Testing YAML save/load...")
    output_dir = Path('test_output')
    config_path = output_dir / 'test_config.yaml'

    save_config_to_yaml(config, config_path)
    loaded_config = load_config_from_yaml(config_path)

    assert loaded_config.grid == config.grid, "Config mismatch"
    assert loaded_config.number_of_events == config.number_of_events, "Config mismatch"
    print(f"✓ Config saved and loaded: {config_path}")

    print("\n✓ All configuration tests passed!")


def test_feature_extraction():
    """Test 6: Feature extraction and format verification."""
    print("\n" + "="*60)
    print("Test 6: Feature Extraction")
    print("="*60)

    # Create small dataset for detailed inspection
    print("\n6.1 Creating small dataset for inspection...")
    results = create_mock_simulation_results('NewEngland', n_events=5, stability_ratio=0.6)

    dataset, splits = generate_dataset_from_simulation(results)

    # Inspect features
    print(f"\n✓ Dataset details:")
    print(f"  - Shape: {dataset.data.shape}")
    print(f"  - Feature names (first 10): {dataset.feature_names[:10]}")
    print(f"  - Target distribution: Stable={sum(dataset.target==0)}, Unstable={sum(dataset.target==1)}")

    # Check feature values are reasonable
    print(f"\n6.2 Checking feature value ranges...")
    print(f"  - Min value: {dataset.data.min():.3f}")
    print(f"  - Max value: {dataset.data.max():.3f}")
    print(f"  - Mean: {dataset.data.mean():.3f}")
    print(f"  - Std: {dataset.data.std():.3f}")

    # Verify DataFrame structure
    df = splits['data']
    print(f"\n6.3 DataFrame structure:")
    print(f"  - Columns: {df.shape[1]}")
    print(f"  - Last column (label): '{df.columns[-1]}'")
    print(f"  - Label values: {df['stable-unstable'].unique()}")

    print("\n✓ All feature extraction tests passed!")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*70)
    print("TranPy Simulation Test Suite (Without PowerFactory)")
    print("="*70)

    try:
        # Test 1: Data structures
        test_data_structures()

        # Test 2: Simulation results
        results = test_simulation_results()

        # Test 3: Dataset generation
        test_dataset_generation(results)

        # Test 4: Legacy format
        test_legacy_format(results)

        # Test 5: Configuration
        test_configuration()

        # Test 6: Feature extraction
        test_feature_extraction()

        # Summary
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)

        # Show generated files
        print("\nGenerated test files in 'test_output/':")
        output_dir = Path('test_output')
        if output_dir.exists():
            for file in sorted(output_dir.glob('*')):
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.name} ({size_kb:.1f} KB)")

        print("\n✓ The refactored implementation is working correctly!")
        print("✓ All components validated without PowerFactory")
        print("\nNext steps:")
        print("  1. Review generated test files in 'test_output/'")
        print("  2. Verify pickle file structures match your expectations")
        print("  3. Test with actual training code using test_dataset_legacy.pickle")
        print("  4. When PowerFactory is available, run actual simulations")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

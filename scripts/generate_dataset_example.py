#!/usr/bin/env python3
"""
Example script for generating datasets using the refactored simulation code.

This demonstrates the complete workflow from configuration to dataset generation.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tranpy.simulation import (
    configure_powerfactory_path,
    PowerSystemSimulator,
    generate_dataset_from_simulation,
    create_config_from_template,
    SimulationConfig
)


def example_basic_simulation():
    """Example 1: Basic simulation with default parameters."""
    print("\n" + "="*60)
    print("Example 1: Basic Simulation")
    print("="*60)

    # Configure PowerFactory (do this once at the start)
    # configure_powerfactory_path(custom_path="your/path/here")

    # Create simulator
    simulator = PowerSystemSimulator(
        grid='NewEngland',
        simulation_time=10.0,
        output_dir='simulation_results'
    )

    # Run simulations
    results = simulator.run(
        num_events=10,
        fault_clearing_time=[10, 12],
        max_load_change=[60, 60],
        random_seed=42
    )

    # Print statistics
    stats = results.get_statistics()
    print(f"\nResults:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Stable: {stats['stable_events']}")
    print(f"  Unstable: {stats['unstable_events']}")


def example_with_config():
    """Example 2: Using configuration templates."""
    print("\n" + "="*60)
    print("Example 2: Using Configuration Template")
    print("="*60)

    # Create configuration from template
    config = create_config_from_template(
        'quick_test',
        grid='NewEngland',
        simulation_time=10.0
    )

    print(f"\nConfiguration:")
    print(config)

    # Create simulator
    simulator = PowerSystemSimulator(
        grid=config.grid,
        simulation_time=config.simulation_time,
        output_dir=config.output_path
    )

    # Run with config parameters
    results = simulator.run(
        num_events=config.number_of_events,
        fault_clearing_time=config.fault_clearing_time_cycles,
        max_load_change=config.max_load_change,
        random_seed=42
    )

    print(f"\n{results}")


def example_dataset_generation():
    """Example 3: Complete workflow with dataset generation."""
    print("\n" + "="*60)
    print("Example 3: Complete Workflow with Dataset Generation")
    print("="*60)

    # Configuration
    config = SimulationConfig(
        grid='NewEngland',
        number_of_events=50,
        simulation_time=10.0,
        fault_clearing_time_cycles=[10, 12],
        max_load_change=[60, 60],
        output_path='my_simulations'
    )

    # Run simulation
    simulator = PowerSystemSimulator(
        grid=config.grid,
        simulation_time=config.simulation_time,
        output_dir=config.output_path
    )

    results = simulator.run(
        num_events=config.number_of_events,
        fault_clearing_time=config.fault_clearing_time_cycles,
        max_load_change=config.max_load_change,
        random_seed=42,
        save_results=True,
        export_csv=False  # Set to True for debugging
    )

    # Generate dataset
    print("\n" + "-"*60)
    print("Generating Dataset")
    print("-"*60)

    dataset, splits = generate_dataset_from_simulation(
        results,
        snapshot_type='clearing',  # Use 'fault' for fault-time snapshot
        test_size=0.2,
        val_size=0.2,
        shuffle=False,
        random_state=42
    )

    print(f"\nDataset Statistics:")
    print(f"  Grid: {dataset.grid_name}")
    print(f"  Total samples: {dataset.data.shape[0]}")
    print(f"  Features: {dataset.data.shape[1]}")
    print(f"  Stable samples: {sum(dataset.target == 0)}")
    print(f"  Unstable samples: {sum(dataset.target == 1)}")

    print(f"\nData Splits:")
    print(f"  Train: {splits['X_train'].shape[0]} samples")
    print(f"  Test: {splits['X_test'].shape[0]} samples")
    print(f"  Validation: {splits['X_val'].shape[0]} samples")

    # Access DataFrames
    train_df = splits['train']
    print(f"\nTrain DataFrame shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns[:5])}... (showing first 5)")


def example_load_existing_results():
    """Example 4: Load existing simulation results."""
    print("\n" + "="*60)
    print("Example 4: Load Existing Results")
    print("="*60)

    from tranpy.simulation import SimulationResults, load_and_generate_dataset

    # Load previously saved results
    results_path = Path('simulation_results/NewEngland/pickles/NewEngland_results.pickle')

    if not results_path.exists():
        print(f"⚠ Results file not found: {results_path}")
        print("Run a simulation first to generate results.")
        return

    # Load results
    results = SimulationResults.load(results_path)
    print(f"\nLoaded results: {results}")

    # Or load and generate dataset in one step
    dataset, splits = load_and_generate_dataset(results_path)
    print(f"\nDataset shape: {dataset.data.shape}")


def example_multiple_grids():
    """Example 5: Run simulations for multiple grids."""
    print("\n" + "="*60)
    print("Example 5: Multiple Grids")
    print("="*60)

    grids = ['NewEngland', 'NineBusSystem']

    for grid in grids:
        print(f"\n{'-'*60}")
        print(f"Processing: {grid}")
        print(f"{'-'*60}")

        simulator = PowerSystemSimulator(
            grid=grid,
            simulation_time=10.0,
            output_dir='simulation_results'
        )

        results = simulator.run(
            num_events=10,
            fault_clearing_time=[10, 12],
            max_load_change=[60, 60],
            random_seed=42
        )

        stats = results.get_statistics()
        print(f"  Events: {stats['total_events']}")
        print(f"  Stable: {stats['stable_events']}")
        print(f"  Unstable: {stats['unstable_events']}")


def example_custom_configuration():
    """Example 6: Custom configuration with all parameters."""
    print("\n" + "="*60)
    print("Example 6: Custom Configuration")
    print("="*60)

    config = SimulationConfig(
        # Grid settings
        grid='NewEngland',
        simulation_time=15.0,

        # Event parameters
        number_of_events=100,
        fault_clearing_time_cycles=[8, 10],  # Faster clearing
        max_load_change=[50, 50],            # Smaller load changes

        # Output settings
        output_path='custom_results',
        random_seed=123,
        export_csv=True,

        # ML parameters (for future training)
        ml_algorithm=['svm', 'mlp', 'dnn'],
        epochs=20,
        batch_size=64,
        optimizer='Adam',
        learning_rate=0.001
    )

    # Validate configuration
    config.validate()
    print(f"\nConfiguration validated:")
    print(config)

    # Save configuration for later use
    from tranpy.simulation import save_config_to_yaml
    save_config_to_yaml(config, 'my_custom_config.yaml')


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("TranPy Simulation Examples")
    print("="*60)

    examples = [
        ("Basic Simulation", example_basic_simulation),
        ("Configuration Template", example_with_config),
        ("Complete Workflow", example_dataset_generation),
        ("Load Existing Results", example_load_existing_results),
        ("Multiple Grids", example_multiple_grids),
        ("Custom Configuration", example_custom_configuration)
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nNote: Uncomment the configure_powerfactory_path() call")
    print("      in the examples before running.")

    # To run specific examples, uncomment:
    # example_basic_simulation()
    # example_with_config()
    # example_dataset_generation()
    # example_load_existing_results()
    # example_multiple_grids()
    # example_custom_configuration()


if __name__ == '__main__':
    main()

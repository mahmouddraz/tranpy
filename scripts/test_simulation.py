#!/usr/bin/env python3
"""
Test script for PowerFactory simulation pipeline.

This script tests the complete workflow:
1. Configure PowerFactory path
2. Load configuration
3. Run simulation
4. Generate dataset
5. Verify results

Usage:
    python scripts/test_simulation.py --grid NewEngland --events 2
    python scripts/test_simulation.py --config path/to/config.yaml
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from tranpy.simulation import (
    configure_powerfactory_path,
    PowerSystemSimulator,
    is_powerfactory_available,
    get_powerfactory_info
)
from tranpy.simulation.config import (
    load_config_from_yaml,
    create_config_from_template,
    SimulationConfig
)
from tranpy.simulation.dataset_generator import (
    generate_dataset_from_simulation,
    generate_legacy_dataset
)


def test_powerfactory_connection():
    """Test PowerFactory connection."""
    print("\n" + "="*60)
    print("Testing PowerFactory Connection")
    print("="*60)

    info = get_powerfactory_info()
    print(f"\nPowerFactory Available: {info['available']}")
    print(f"Platform: {info['platform']}")

    if info['available']:
        print(f"Module Path: {info.get('module_path', 'N/A')}")
        print(f"Connected: {info.get('connected', False)}")
        if 'note' in info:
            print(f"Note: {info['note']}")
    else:
        print("\n⚠ PowerFactory not available")
        print("Configure path with:")
        print("  configure_powerfactory_path(custom_path='your/path/here')")
        return False

    return True


def run_simulation_test(
    grid: str = 'NewEngland',
    num_events: int = 2,
    config_path: str = None,
    pf_path: str = None
):
    """
    Run simulation test.

    Args:
        grid: Grid name
        num_events: Number of events to simulate
        config_path: Path to config YAML (optional)
        pf_path: Custom PowerFactory path (optional)
    """
    print("\n" + "="*60)
    print("PowerFactory Simulation Test")
    print("="*60)

    # Configure PowerFactory path if provided
    if pf_path:
        print(f"\nConfiguring PowerFactory path: {pf_path}")
        try:
            configure_powerfactory_path(custom_path=pf_path)
        except Exception as e:
            print(f"⚠ Failed to configure path: {e}")
            return False

    # Test connection
    if not is_powerfactory_available():
        print("\n⚠ PowerFactory not available. Cannot run simulation test.")
        print("\nTo configure PowerFactory:")
        print("  python -c \"from tranpy.simulation import configure_powerfactory_path; "
              "configure_powerfactory_path(custom_path='your/path')\"")
        return False

    # Load or create configuration
    if config_path:
        print(f"\nLoading configuration from: {config_path}")
        config = load_config_from_yaml(config_path)
    else:
        print(f"\nCreating test configuration for {grid}")
        config = create_config_from_template('quick_test', grid=grid)
        config.number_of_events = num_events

    print(f"\nConfiguration:")
    print(f"  Grid: {config.grid}")
    print(f"  Events: {config.number_of_events}")
    print(f"  Simulation time: {config.simulation_time}s")
    params = config.get_grid_parameters()
    print(f"  Fault clearing: {params['fault_clearing_cycles']} cycles")
    print(f"  Max load change: {params['max_load_change']}%")

    # Create simulator
    print("\n" + "-"*60)
    print("Initializing Simulator")
    print("-"*60)

    try:
        simulator = PowerSystemSimulator(
            grid=config.grid,
            simulation_time=config.simulation_time,
            output_dir=config.output_path
        )
    except Exception as e:
        print(f"⚠ Failed to create simulator: {e}")
        return False

    # Run simulation
    print("\n" + "-"*60)
    print("Running Simulation")
    print("-"*60)

    try:
        results = simulator.run(
            num_events=config.number_of_events,
            fault_clearing_time=config.fault_clearing_time_cycles,
            max_load_change=config.max_load_change,
            random_seed=42,
            save_results=True,
            export_csv=True
        )
    except Exception as e:
        print(f"\n⚠ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print results
    print("\n" + "-"*60)
    print("Simulation Results")
    print("-"*60)
    print(results)

    stats = results.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Stable: {stats['stable_events']}")
    print(f"  Unstable: {stats['unstable_events']}")
    print(f"  Stability ratio: {stats['stability_ratio']*100:.1f}%")

    # Generate dataset
    print("\n" + "-"*60)
    print("Generating Dataset")
    print("-"*60)

    try:
        dataset, splits = generate_dataset_from_simulation(results)
        print(f"\n✓ Dataset generated:")
        print(f"  Shape: {dataset.data.shape}")
        print(f"  Features: {dataset.data.shape[1]}")
        print(f"  Samples: {dataset.data.shape[0]}")
        print(f"  Train: {splits['X_train'].shape[0]}")
        print(f"  Test: {splits['X_test'].shape[0]}")
        print(f"  Val: {splits['X_val'].shape[0]}")

        # Save legacy dataset
        output_dir = Path(config.output_path) / 'data_set'
        legacy_path = output_dir / f'{config.grid}.pickle'
        generate_legacy_dataset(results, legacy_path)

    except Exception as e:
        print(f"⚠ Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify pickle files
    print("\n" + "-"*60)
    print("Verifying Output Files")
    print("-"*60)

    output_dir = Path(config.output_path)
    results_pickle = output_dir / config.grid / 'pickles' / f'{config.grid}_results.pickle'
    legacy_pickle = output_dir / 'data' / f'{config.grid}.pickle'
    dataset_pickle = output_dir / 'data_set' / f'{config.grid}.pickle'

    files_to_check = [
        ('Results pickle', results_pickle),
        ('Legacy pickle', legacy_pickle),
        ('Dataset pickle', dataset_pickle)
    ]

    all_exist = True
    for name, path in files_to_check:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✓ {name}: {path} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {name}: {path} (not found)")
            all_exist = False

    # Summary
    print("\n" + "="*60)
    if all_exist:
        print("✓ TEST PASSED - All files generated successfully")
    else:
        print("⚠ TEST INCOMPLETE - Some files missing")
    print("="*60)

    return all_exist


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test PowerFactory simulation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with NewEngland grid, 2 events
  python scripts/test_simulation.py --grid NewEngland --events 2

  # Test with config file
  python scripts/test_simulation.py --config tranpy/src/config.yaml

  # Test with custom PowerFactory path
  python scripts/test_simulation.py --pf-path "C:/Program Files/DIgSILENT/PowerFactory 2019/Python/3.13"

  # Just test connection
  python scripts/test_simulation.py --test-connection
        """
    )

    parser.add_argument(
        '--grid',
        type=str,
        default='NewEngland',
        choices=['NewEngland', 'NineBusSystem'],
        help='Grid model to test'
    )

    parser.add_argument(
        '--events',
        type=int,
        default=2,
        help='Number of events to simulate (default: 2)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--pf-path',
        type=str,
        help='Custom PowerFactory Python path'
    )

    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Only test PowerFactory connection'
    )

    args = parser.parse_args()

    # Test connection only
    if args.test_connection:
        success = test_powerfactory_connection()
        sys.exit(0 if success else 1)

    # Run full test
    success = run_simulation_test(
        grid=args.grid,
        num_events=args.events,
        config_path=args.config,
        pf_path=args.pf_path
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

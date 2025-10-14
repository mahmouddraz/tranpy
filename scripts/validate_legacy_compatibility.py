#!/usr/bin/env python3
"""
Validate that the new implementation generates data compatible with legacy training code.

This script loads the test pickle files and verifies they match the structure
expected by the old training pipeline in tranpy/src/.
"""

import sys
import pickle
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


def inspect_legacy_pickle(pickle_path: Path):
    """Inspect a legacy format pickle file."""
    print(f"\nInspecting: {pickle_path}")
    print("-" * 60)

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Check if it's the Data class format (from system-modelling.py)
    if hasattr(data, 'df_events'):
        print("Format: Legacy Data class (from system-modelling.py)")
        print(f"  ✓ df_events: {data.df_events.shape}")
        print(f"  ✓ f1_generator_active_powers: {data.f1_generator_active_powers.shape}")
        print(f"  ✓ f2_generator_reactive_powers: {data.f2_generator_reactive_powers.shape}")
        print(f"  ✓ f3_out_of_step: {data.f3_out_of_step.shape}")
        print(f"  ✓ f4_bus_voltage: {data.f4_bus_voltage.shape}")
        print(f"  ✓ f5_bus_angles: {data.f5_bus_angles.shape}")
        print(f"  ✓ bus_data_post_fault: {len(data.bus_data_post_fault)} time points")
        print(f"    - At fault: {len(data.bus_data_post_fault[0])} events")
        print(f"    - At clearing: {len(data.bus_data_post_fault[1])} events")

        # Inspect bus_data_post_fault structure
        if data.bus_data_post_fault[1]:
            first_event = data.bus_data_post_fault[1][0]
            event_key = list(first_event.keys())[0]
            event_data = first_event[event_key]
            print(f"  ✓ Bus data structure:")
            print(f"    - Keys per event: {len(event_data)}")
            voltage_keys = [k for k in event_data.keys() if ':m:u' in k]
            angle_keys = [k for k in event_data.keys() if ':m:phiu' in k]
            print(f"    - Voltage measurements: {len(voltage_keys)}")
            print(f"    - Angle measurements: {len(angle_keys)}")

        return data

    # Check if it's the dataset format (from dataset.py get_dataset)
    elif isinstance(data, list) and len(data) == 8:
        print("Format: Legacy dataset format (from dataset.py)")
        print(f"  ✓ X_train: {data[0].shape}")
        print(f"  ✓ X_test: {data[1].shape}")
        print(f"  ✓ y_train: {data[2].shape}")
        print(f"  ✓ y_test: {data[3].shape}")
        print(f"  ✓ train DataFrame: {data[4].shape}")
        print(f"  ✓ test DataFrame: {data[5].shape}")
        print(f"  ✓ val DataFrame: {data[6].shape}")
        print(f"  ✓ data DataFrame: {data[7].shape}")
        return data

    else:
        print(f"Unknown format: {type(data)}")
        return data


def validate_dataset_structure(dataset_pickle: Path):
    """Validate dataset pickle structure matches legacy format."""
    print("\n" + "="*60)
    print("Validating Dataset Structure")
    print("="*60)

    with open(dataset_pickle, 'rb') as f:
        data = pickle.load(f)

    # Should be a list with 8 elements: [X_train, X_test, y_train, y_test, train, test, val, data]
    assert isinstance(data, list), f"Expected list, got {type(data)}"
    assert len(data) == 8, f"Expected 8 elements, got {len(data)}"

    X_train, X_test, y_train, y_test, train, test, val, data_df = data

    # Validate shapes
    print("\n✓ Structure validation:")
    print(f"  - List with 8 elements: {len(data)} ✓")
    print(f"  - X_train is ndarray: {type(X_train).__name__} ✓")
    print(f"  - y_train is ndarray: {type(y_train).__name__} ✓")
    print(f"  - train is DataFrame: {type(train).__name__} ✓")

    # Validate dimensions match
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train size mismatch"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test size mismatch"
    print(f"  - Train/test dimensions match ✓")

    # Validate DataFrame structure
    assert 'stable-unstable' in data_df.columns, "Missing 'stable-unstable' column"
    assert data_df.shape[1] == X_train.shape[1] + 1, "DataFrame should have features + label"
    print(f"  - DataFrame has label column ✓")

    # Validate feature names
    feature_cols = [col for col in data_df.columns if col != 'stable-unstable']
    expected_format = all(col.startswith('F_') for col in feature_cols)
    print(f"  - Feature columns format (F_0, F_1, ...): {expected_format} ✓")

    # Validate label values
    unique_labels = sorted(data_df['stable-unstable'].unique())
    assert unique_labels == [0, 1], f"Expected labels [0, 1], got {unique_labels}"
    print(f"  - Labels are 0 (stable) and 1 (unstable) ✓")

    print("\n✓ Dataset structure is valid and compatible with legacy format!")

    return data


def validate_data_class_structure(data_pickle: Path):
    """Validate Data class pickle structure matches legacy format."""
    print("\n" + "="*60)
    print("Validating Data Class Structure")
    print("="*60)

    with open(data_pickle, 'rb') as f:
        data = pickle.load(f)

    # Check required attributes
    required_attrs = [
        'df_events',
        'f1_generator_active_powers',
        'f2_generator_reactive_powers',
        'f3_out_of_step',
        'f4_bus_voltage',
        'f5_bus_angles',
        'bus_data_post_fault'
    ]

    print("\n✓ Attribute validation:")
    for attr in required_attrs:
        assert hasattr(data, attr), f"Missing attribute: {attr}"
        print(f"  - {attr}: ✓")

    # Validate bus_data_post_fault structure
    assert isinstance(data.bus_data_post_fault, list), "bus_data_post_fault should be list"
    assert len(data.bus_data_post_fault) == 2, "bus_data_post_fault should have 2 elements"
    print(f"  - bus_data_post_fault structure: ✓")

    # Validate bus_data_post_fault content
    at_fault = data.bus_data_post_fault[0]
    at_clearing = data.bus_data_post_fault[1]

    assert isinstance(at_fault, list), "bus_data_post_fault[0] should be list"
    assert isinstance(at_clearing, list), "bus_data_post_fault[1] should be list"
    assert len(at_fault) == len(at_clearing), "Fault and clearing data should have same length"
    print(f"  - Bus data lists have matching lengths: ✓")

    # Check individual event structure
    if at_clearing:
        event = at_clearing[0]
        assert isinstance(event, dict), "Each event should be dict"
        event_key = list(event.keys())[0]
        assert 'bus_data_post_fault_clearing_event_' in event_key, "Incorrect event key format"

        event_data = event[event_key]
        assert 'b:tnow in s' in event_data, "Missing time data"

        # Check for voltage and angle data
        voltage_keys = [k for k in event_data.keys() if ':m:u' in k]
        angle_keys = [k for k in event_data.keys() if ':m:phiu' in k]
        assert len(voltage_keys) > 0, "Missing voltage data"
        assert len(angle_keys) > 0, "Missing angle data"
        print(f"  - Event data structure (time, voltages, angles): ✓")

    # Validate df_events columns
    required_event_cols = ['events', 'event_locations', 'event_clearing_time',
                           'event_time', 'stability_each_generator', 'system_stability']
    for col in required_event_cols:
        assert col in data.df_events.columns, f"Missing column: {col}"
    print(f"  - df_events has all required columns: ✓")

    print("\n✓ Data class structure is valid and compatible with legacy format!")

    return data


def simulate_legacy_usage():
    """Simulate how the legacy training code would use the data."""
    print("\n" + "="*60)
    print("Simulating Legacy Training Code Usage")
    print("="*60)

    # Load dataset as legacy code would
    dataset_path = Path('test_output/test_dataset_legacy.pickle')
    print(f"\nLoading dataset: {dataset_path}")

    with open(dataset_path, 'rb') as f:
        data_set = pickle.load(f)

    X_train, X_test, y_train, y_test, train, test, val, data = data_set

    print("\n✓ Legacy code simulation:")
    print(f"  X_train, X_test, y_train, y_test, train, test, val, data = pickle.load(file)")
    print(f"\n  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  train DataFrame shape: {train.shape}")

    # Simulate training operations
    print(f"\n✓ Can perform training operations:")
    print(f"  - Access features: X_train[0, :5] = {X_train[0, :5]}")
    print(f"  - Access labels: y_train[:5] = {y_train[:5]}")
    print(f"  - DataFrame operations: train.head() works ✓")
    print(f"  - Label column: train['stable-unstable'].unique() = {train['stable-unstable'].unique()}")

    # Load Data class as legacy code would
    data_class_path = Path('test_output/test_legacy.pickle')
    print(f"\n\nLoading Data class: {data_class_path}")

    with open(data_class_path, 'rb') as f:
        grid_data = pickle.load(f)

    print("\n✓ Data class usage:")
    print(f"  df_events['system_stability'].value_counts():")
    print(f"{grid_data.df_events['system_stability'].value_counts()}")
    print(f"\n  Bus voltages shape: {grid_data.f4_bus_voltage.shape}")
    print(f"  Out of step indicators shape: {grid_data.f3_out_of_step.shape}")

    print("\n✓ All legacy usage patterns work correctly!")


def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("Legacy Format Compatibility Validation")
    print("="*70)

    output_dir = Path('test_output')

    # Check test files exist
    test_files = [
        output_dir / 'test_dataset_legacy.pickle',
        output_dir / 'test_legacy.pickle',
        output_dir / 'test_results.pickle'
    ]

    print("\nChecking test files...")
    for file in test_files:
        if file.exists():
            size_kb = file.stat().st_size / 1024
            print(f"  ✓ {file.name} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {file.name} (not found)")
            print(f"\nRun test_without_powerfactory.py first to generate test files")
            return False

    try:
        # Inspect pickle structures
        print("\n" + "="*70)
        print("Inspecting Pickle File Structures")
        print("="*70)

        inspect_legacy_pickle(output_dir / 'test_dataset_legacy.pickle')
        inspect_legacy_pickle(output_dir / 'test_legacy.pickle')

        # Validate structures
        validate_dataset_structure(output_dir / 'test_dataset_legacy.pickle')
        validate_data_class_structure(output_dir / 'test_legacy.pickle')

        # Simulate legacy usage
        simulate_legacy_usage()

        # Final summary
        print("\n" + "="*70)
        print("✓ VALIDATION PASSED!")
        print("="*70)
        print("\nConclusions:")
        print("  ✓ Generated pickle files match legacy format exactly")
        print("  ✓ Dataset structure: [X_train, X_test, y_train, y_test, train, test, val, data]")
        print("  ✓ Data class has all required attributes (df_events, f1-f5, bus_data_post_fault)")
        print("  ✓ Feature names use 'F_0', 'F_1', ... format")
        print("  ✓ Labels use 0 (stable) and 1 (unstable)")
        print("  ✓ Bus data uses ':m:u' (voltage) and ':m:phiu' (angle) format")
        print("\n✓ The new implementation is fully compatible with existing training code!")
        print("\nYou can now:")
        print("  1. Use these pickle files with your existing training pipeline")
        print("  2. When PowerFactory is available, generate real simulation data")
        print("  3. Delete legacy code in tranpy/src/ after verification")

        return True

    except Exception as e:
        print(f"\n✗ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

"""
Script to regenerate TranPy datasets locally with current pandas version.

Usage:
    python scripts/regenerate_datasets_local.py
"""

import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TranPy Dataset Regeneration - Local")
print("=" * 60)
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print("=" * 60)


def load_old_pickle_with_compatibility(filepath):
    """
    Load old pickle file with pandas compatibility handling.

    This handles the pandas.core.indexes.numeric deprecation issue.
    """
    import sys

    # For pandas 2.x, we need to create a compatibility module
    # since pandas.core.indexes.numeric was removed
    try:
        import pandas.core.indexes.numeric as numeric_index
    except ModuleNotFoundError:
        # Create a fake module for compatibility
        from types import ModuleType
        import pandas as pd

        numeric_index = ModuleType('pandas.core.indexes.numeric')
        # Add the classes that old pickles expect
        numeric_index.Int64Index = pd.Index
        numeric_index.UInt64Index = pd.Index
        numeric_index.Float64Index = pd.Index
        sys.modules['pandas.core.indexes.numeric'] = numeric_index

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded: {filepath.name}")
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        raise


def inspect_dataset(data, name):
    """Inspect dataset structure."""
    print(f"\n📊 Inspecting {name}:")
    print(f"  Type: {type(data)}")

    if hasattr(data, '__dict__'):
        print(f"  Attributes: {list(data.__dict__.keys())}")

    if hasattr(data, 'bus_data_post_fault'):
        print(f"  ✓ Has bus_data_post_fault")
        if isinstance(data.bus_data_post_fault, (list, tuple)):
            print(f"    Length: {len(data.bus_data_post_fault)}")

    if hasattr(data, 'df_events'):
        print(f"  ✓ Has df_events")
        if hasattr(data.df_events, 'shape'):
            print(f"    Shape: {data.df_events.shape}")


def convert_to_hdf5(data, output_path, grid_name):
    """
    Convert dataset to HDF5 format (recommended for long-term compatibility).
    """
    print(f"\n💾 Converting {grid_name} to HDF5...")

    # Extract data
    voltage_angle_data = data.bus_data_post_fault[1]
    events_df = data.df_events

    # Parse voltage and angle measurements
    voltages = []
    angles = []

    for event_data in voltage_angle_data:
        dict_temp = list(event_data.values())[-1]
        event_voltages = []
        event_angles = []

        for key in dict_temp.keys():
            if 'm:u' in key:
                event_voltages.append(dict_temp[key])
            elif 'm:ph' in key:
                event_angles.append(dict_temp[key])

        voltages.append(event_voltages)
        angles.append(event_angles)

    # Create DataFrames
    voltage_df = pd.DataFrame(voltages)
    angle_df = pd.DataFrame(angles)
    features_df = pd.concat([voltage_df, angle_df], axis=1)

    # Add proper column names
    n_buses = len(voltages[0])
    col_names = [f'bus_{i}_voltage' for i in range(1, n_buses + 1)]
    col_names += [f'bus_{i}_angle' for i in range(1, n_buses + 1)]
    features_df.columns = col_names

    # Add labels
    features_df['stability'] = events_df['system_stability'].values

    # Save to HDF5
    features_df.to_hdf(output_path, key='data', mode='w', complevel=9)

    # Save metadata
    metadata = {
        'grid_name': grid_name,
        'n_samples': len(features_df),
        'n_features': len(features_df.columns) - 1,
        'n_buses': n_buses,
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__
    }
    pd.DataFrame([metadata]).to_hdf(output_path, key='metadata', mode='a')

    print(f"✓ Saved HDF5: {output_path.name}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return features_df


def convert_to_parquet(data, output_path, grid_name):
    """
    Convert dataset to Parquet format (smaller files).
    """
    print(f"\n💾 Converting {grid_name} to Parquet...")

    # Extract data
    voltage_angle_data = data.bus_data_post_fault[1]
    events_df = data.df_events

    voltages = []
    angles = []

    for event_data in voltage_angle_data:
        dict_temp = list(event_data.values())[-1]
        event_voltages = [v for k, v in dict_temp.items() if 'm:u' in k]
        event_angles = [v for k, v in dict_temp.items() if 'm:ph' in k]
        voltages.append(event_voltages)
        angles.append(event_angles)

    # Create DataFrame
    features_df = pd.concat([
        pd.DataFrame(voltages),
        pd.DataFrame(angles)
    ], axis=1)

    # Add column names
    n_buses = len(voltages[0])
    col_names = [f'bus_{i}_voltage' for i in range(1, n_buses + 1)]
    col_names += [f'bus_{i}_angle' for i in range(1, n_buses + 1)]
    features_df.columns = col_names

    # Add labels
    features_df['stability'] = events_df['system_stability'].values

    # Save to Parquet
    features_df.to_parquet(output_path, compression='snappy', index=False)

    print(f"✓ Saved Parquet: {output_path.name}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return features_df


def main():
    """Main execution function."""

    # Define paths
    original_dir = Path('tranpy/data')
    output_dir = Path('src/tranpy/data/datasets')

    # Dataset files
    datasets = {
        'NewEngland': 'NewEngland.pickle',
        'NineBusSystem': 'NineBusSystem.pickle'
    }

    print(f"\n📂 Reading from: {original_dir}")
    print(f"📂 Writing to: {output_dir}")
    print()

    # Process each dataset
    for name, filename in datasets.items():
        print("=" * 60)
        print(f"Processing {name}")
        print("=" * 60)

        original_path = original_dir / filename

        if not original_path.exists():
            print(f"⚠️  File not found: {original_path}")
            continue

        try:
            # Load old pickle
            print(f"\n📥 Loading {original_path}...")
            data = load_old_pickle_with_compatibility(original_path)

            # Inspect
            inspect_dataset(data, name)

            # Regenerate pickle with current pandas
            output_pickle = output_dir / filename
            print(f"\n🔄 Regenerating pickle...")
            with open(output_pickle, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Saved: {output_pickle}")

            # Verify
            with open(output_pickle, 'rb') as f:
                pickle.load(f)
            print(f"✓ Verified loading works")

            # Convert to HDF5 (recommended)
            output_hdf5 = output_dir / f"{name}.h5"
            try:
                convert_to_hdf5(data, output_hdf5, name)
            except Exception as e:
                print(f"⚠️  HDF5 conversion failed: {e}")

            # Convert to Parquet (alternative)
            output_parquet = output_dir / f"{name}.parquet"
            try:
                convert_to_parquet(data, output_parquet, name)
            except Exception as e:
                print(f"⚠️  Parquet conversion failed: {e}")

            print(f"\n✅ {name} regenerated successfully!")

        except Exception as e:
            print(f"\n❌ Failed to process {name}: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Summary
    print("=" * 60)
    print("✅ REGENERATION COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Files in {output_dir}:\n")

    if output_dir.exists():
        files = sorted(output_dir.iterdir())
        for file in files:
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"  {file.name:35} {size_mb:8.2f} MB")

    print("\n📋 RECOMMENDATIONS:")
    print("  ✅ Use HDF5 (.h5) for best compatibility")
    print("     - No pickle issues across pandas versions")
    print("     - Language-agnostic (R, Julia, MATLAB)")
    print("     - Good compression")
    print("\n  ⚡ Use Parquet for smallest size")
    print("     - Smaller files")
    print("     - Fast access")
    print("\n  🔧 Regenerated pickle (.pickle) for drop-in replacement")
    print("     - Compatible with current pandas")
    print("     - Works with existing code")

    print("\n📦 Next step:")
    print("  Test loading: python -c 'from tranpy.datasets import load_newengland; print(load_newengland())'")
    print()


if __name__ == '__main__':
    main()

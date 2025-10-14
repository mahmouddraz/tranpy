"""
Script to regenerate TranPy datasets with current pandas version.

This script can be run in Google Colab to:
1. Download old pickle files from Google Drive
2. Load them with compatibility handling
3. Regenerate with current pandas version
4. Optionally convert to HDF5 format (recommended)

Usage in Google Colab:
    1. Upload this script or copy-paste the code
    2. Mount Google Drive
    3. Run the cells
    4. Download regenerated files
"""

import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TranPy Dataset Regeneration Script")
print("=" * 60)
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print("=" * 60)


# ============================================================================
# STEP 1: Download datasets from Google Drive
# ============================================================================

def download_from_google_drive(file_id, destination):
    """Download file from Google Drive using gdown."""
    import gdown

    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)
    print(f"✓ Downloaded: {destination}")


def setup_downloads():
    """Download datasets from Google Drive."""
    print("\n📥 STEP 1: Downloading datasets from Google Drive...")

    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'gdown', '-q'])
        import gdown

    # Create directories
    Path('original_datasets').mkdir(exist_ok=True)
    Path('regenerated_datasets').mkdir(exist_ok=True)

    # Dataset metadata
    datasets = {
        'NewEngland': {
            'file_id': '1eXtw44VXhYM0jQyJGGY5Eevdrg8yui0w',
            'filename': 'NewEngland.pickle'
        },
        'NineBusSystem': {
            'file_id': '1-4LrEqmDP6-EcLpL0-6tvzsNSSJOLzG1',
            'filename': 'NineBusSystem.pickle'
        }
    }

    # Download files
    for name, info in datasets.items():
        dest = Path('original_datasets') / info['filename']
        if not dest.exists():
            print(f"\nDownloading {name}...")
            try:
                download_from_google_drive(info['file_id'], str(dest))
            except Exception as e:
                print(f"❌ Error downloading {name}: {e}")
                print(f"   Please download manually from Google Drive")
        else:
            print(f"✓ {name} already downloaded")

    return datasets


# ============================================================================
# STEP 2: Load old pickle files with compatibility handling
# ============================================================================

def load_old_pickle_with_compatibility(filepath):
    """
    Load old pickle file with pandas compatibility handling.

    This handles the pandas.core.indexes.numeric deprecation issue.
    """
    import sys
    import pandas.core.indexes.numeric as numeric_index

    # Create compatibility mapping
    sys.modules['pandas.core.indexes.numeric'] = numeric_index

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded: {filepath}")
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")

        # Try alternative loading methods
        print("   Trying alternative loading method...")
        try:
            import pickle5
            with open(filepath, 'rb') as f:
                data = pickle5.load(f)
            print(f"✓ Loaded with pickle5: {filepath}")
            return data
        except:
            raise RuntimeError(f"Could not load {filepath}. Original error: {e}")


def inspect_dataset(data, name):
    """Inspect dataset structure."""
    print(f"\n📊 Inspecting {name}:")
    print(f"  Type: {type(data)}")

    if hasattr(data, '__dict__'):
        print(f"  Attributes: {list(data.__dict__.keys())}")

    # Check for common attributes
    if hasattr(data, 'bus_data_post_fault'):
        print(f"  ✓ Has bus_data_post_fault")
        print(f"    Type: {type(data.bus_data_post_fault)}")
        if isinstance(data.bus_data_post_fault, (list, tuple)):
            print(f"    Length: {len(data.bus_data_post_fault)}")

    if hasattr(data, 'df_events'):
        print(f"  ✓ Has df_events")
        print(f"    Shape: {data.df_events.shape if hasattr(data.df_events, 'shape') else 'N/A'}")
        if hasattr(data.df_events, 'columns'):
            print(f"    Columns: {list(data.df_events.columns)}")


# ============================================================================
# STEP 3: Regenerate with current pandas version
# ============================================================================

def regenerate_pickle(original_path, output_path):
    """
    Regenerate pickle file with current pandas version.
    """
    print(f"\n🔄 Regenerating: {original_path.name}")

    # Load old pickle
    data = load_old_pickle_with_compatibility(original_path)

    # Inspect
    inspect_dataset(data, original_path.stem)

    # Save with current pandas version
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Saved: {output_path}")

    # Verify we can load it
    with open(output_path, 'rb') as f:
        verified_data = pickle.load(f)
    print(f"✓ Verified loading: {output_path}")

    return data


# ============================================================================
# STEP 4: Convert to HDF5 (RECOMMENDED for long-term compatibility)
# ============================================================================

def convert_to_hdf5(data, output_path, grid_name):
    """
    Convert dataset to HDF5 format (better long-term compatibility).

    HDF5 advantages:
    - Language-agnostic (can be read by R, Julia, MATLAB, etc.)
    - No pickle compatibility issues
    - Efficient compression
    - Partial loading support
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
    col_names = []
    for i in range(1, n_buses + 1):
        col_names.append(f'bus_{i}_voltage')
    for i in range(1, n_buses + 1):
        col_names.append(f'bus_{i}_angle')
    features_df.columns = col_names

    # Combine with labels
    features_df['stability'] = events_df['system_stability'].values

    # Save to HDF5
    features_df.to_hdf(output_path, key='data', mode='w', complevel=9)

    print(f"✓ Saved HDF5: {output_path}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Save metadata
    metadata = {
        'grid_name': grid_name,
        'n_samples': len(features_df),
        'n_features': len(features_df.columns) - 1,
        'n_buses': n_buses,
        'columns': list(features_df.columns),
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_hdf(output_path, key='metadata', mode='a')

    print(f"✓ Saved metadata")

    # Verify loading
    loaded_df = pd.read_hdf(output_path, key='data')
    loaded_meta = pd.read_hdf(output_path, key='metadata')
    print(f"✓ Verified HDF5 loading")
    print(f"  Loaded shape: {loaded_df.shape}")

    return features_df


# ============================================================================
# STEP 5: Convert to Parquet (ALTERNATIVE - smaller files)
# ============================================================================

def convert_to_parquet(data, output_path, grid_name):
    """
    Convert dataset to Parquet format (modern columnar format).

    Parquet advantages:
    - Smaller file size than HDF5
    - Fast columnar access
    - Good compression
    - Standard format (Apache Arrow ecosystem)
    """
    print(f"\n💾 Converting {grid_name} to Parquet...")

    # Extract data (same as HDF5)
    voltage_angle_data = data.bus_data_post_fault[1]
    events_df = data.df_events

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

    # Create DataFrame
    voltage_df = pd.DataFrame(voltages)
    angle_df = pd.DataFrame(angles)
    features_df = pd.concat([voltage_df, angle_df], axis=1)

    # Add column names
    n_buses = len(voltages[0])
    col_names = []
    for i in range(1, n_buses + 1):
        col_names.append(f'bus_{i}_voltage')
    for i in range(1, n_buses + 1):
        col_names.append(f'bus_{i}_angle')
    features_df.columns = col_names

    # Add labels
    features_df['stability'] = events_df['system_stability'].values

    # Save to Parquet
    features_df.to_parquet(output_path, compression='snappy', index=False)

    print(f"✓ Saved Parquet: {output_path}")
    print(f"  Shape: {features_df.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify loading
    loaded_df = pd.read_parquet(output_path)
    print(f"✓ Verified Parquet loading")
    print(f"  Loaded shape: {loaded_df.shape}")

    return features_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    print("\n" + "=" * 60)
    print("STARTING DATASET REGENERATION")
    print("=" * 60)

    # Step 1: Download datasets
    datasets_info = setup_downloads()

    # Step 2 & 3: Load and regenerate
    print("\n" + "=" * 60)
    print("📥 STEP 2-3: Loading and regenerating with current pandas")
    print("=" * 60)

    for name, info in datasets_info.items():
        original_path = Path('original_datasets') / info['filename']

        if not original_path.exists():
            print(f"\n⚠️  Skipping {name} - file not found")
            continue

        # Regenerate pickle
        output_pickle = Path('regenerated_datasets') / info['filename']
        try:
            data = regenerate_pickle(original_path, output_pickle)
            print(f"✅ {name} pickle regenerated successfully")
        except Exception as e:
            print(f"❌ Failed to regenerate {name}: {e}")
            continue

        # Convert to HDF5 (RECOMMENDED)
        output_hdf5 = Path('regenerated_datasets') / f"{name}.h5"
        try:
            convert_to_hdf5(data, output_hdf5, name)
            print(f"✅ {name} HDF5 created successfully")
        except Exception as e:
            print(f"⚠️  HDF5 conversion failed for {name}: {e}")

        # Convert to Parquet (ALTERNATIVE)
        output_parquet = Path('regenerated_datasets') / f"{name}.parquet"
        try:
            convert_to_parquet(data, output_parquet, name)
            print(f"✅ {name} Parquet created successfully")
        except Exception as e:
            print(f"⚠️  Parquet conversion failed for {name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ REGENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files in 'regenerated_datasets/':")

    regen_dir = Path('regenerated_datasets')
    if regen_dir.exists():
        for file in sorted(regen_dir.iterdir()):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"  {file.name:30} {size_mb:8.2f} MB")

    print("\n📋 RECOMMENDATION:")
    print("  Use HDF5 (.h5) format for best long-term compatibility!")
    print("  - No pickle compatibility issues")
    print("  - Can be read by any language (R, Julia, MATLAB, etc.)")
    print("  - Efficient compression and partial loading")

    print("\n📦 Next steps:")
    print("  1. Download the regenerated files")
    print("  2. Replace old pickle files in src/tranpy/data/datasets/")
    print("  3. Update loaders.py to support HDF5 (optional)")
    print("  4. Test: from tranpy.datasets import load_newengland")


if __name__ == '__main__':
    main()

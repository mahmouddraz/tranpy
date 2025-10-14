"""Dataset loaders for power system stability data."""

import json
import pickle
import sys
from types import ModuleType
from pathlib import Path
from typing import Tuple, Union, Optional, List, Dict
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from .base import StabilityDataset, generate_feature_names


# Pandas compatibility for older pickle files
if 'pandas.core.indexes.numeric' not in sys.modules:
    numeric_index = ModuleType('pandas.core.indexes.numeric')
    numeric_index.Int64Index = pd.Index
    numeric_index.UInt64Index = pd.Index
    numeric_index.Float64Index = pd.Index
    sys.modules['pandas.core.indexes.numeric'] = numeric_index


def _load_registry() -> dict:
    """Load dataset registry from JSON file."""
    registry_path = Path(__file__).parent.parent / 'data' / 'datasets' / 'registry.json'
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Dataset registry not found at {registry_path}. "
            "Please ensure the package is installed correctly."
        )
    with open(registry_path, 'r') as f:
        return json.load(f)


def get_data_path(grid: str) -> Path:
    """
    Get path to dataset file using registry.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')

    Returns:
        Path to pickle file

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If grid name is not in registry
    """
    # Load registry
    registry = _load_registry()

    if grid not in registry['datasets']:
        available = list(registry['datasets'].keys())
        raise ValueError(
            f"Unknown grid '{grid}'. Available datasets: {available}"
        )

    dataset_info = registry['datasets'][grid]

    # Get path to bundled dataset
    if dataset_info.get('bundled', False):
        # Look in package data directory
        package_data_dir = Path(__file__).parent.parent / 'data'
        data_path = package_data_dir / dataset_info['local_path']

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset {grid} should be bundled but file not found at {data_path}.\n"
                f"Expected: {dataset_info['filename']} ({dataset_info['size_mb']} MB)\n"
                f"Please reinstall the package: pip install --force-reinstall -e ."
            )

        return data_path
    else:
        # Future: Download on demand from Google Drive
        raise NotImplementedError(
            f"Dataset {grid} is not bundled and download-on-demand is not yet implemented.\n"
            f"Google Drive link: {dataset_info['google_drive']['view_link']}"
        )


def list_available_datasets() -> List[Dict]:
    """
    List all available datasets from registry.

    Returns:
        List of dictionaries with dataset information

    Examples:
        >>> datasets = list_available_datasets()
        >>> for ds in datasets:
        ...     print(f"{ds['name']}: {ds['n_features']} features, {ds['size_mb']} MB")
    """
    registry = _load_registry()

    datasets = []
    for grid_key, info in registry['datasets'].items():
        datasets.append({
            'key': grid_key,
            'name': info['name'],
            'description': info['description'],
            'n_buses': info['n_buses'],
            'n_features': info['n_features'],
            'size_mb': info['size_mb'],
            'bundled': info.get('bundled', False),
            'google_drive_link': info['google_drive']['view_link']
        })

    return datasets


def load_dataset(
    grid: str,
    return_X_y: bool = False,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    as_frame: bool = False
) -> Union[StabilityDataset, Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load a power system stability dataset.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')
        return_X_y: If True, return (data, target) instead of StabilityDataset
        test_size: If provided, split into train/test sets
        random_state: Random seed for train/test split
        as_frame: If True, return data as pandas DataFrame

    Returns:
        - StabilityDataset object (default)
        - (X, y) if return_X_y=True
        - (X_train, X_test, y_train, y_test) if test_size is provided

    Examples:
        >>> # Load full dataset
        >>> dataset = load_dataset('NewEngland')
        >>>
        >>> # Get X, y arrays
        >>> X, y = load_dataset('NewEngland', return_X_y=True)
        >>>
        >>> # Get train/test split
        >>> X_train, X_test, y_train, y_test = load_dataset(
        ...     'NewEngland', test_size=0.2, random_state=42
        ... )
    """
    # Load registry to get metadata
    registry = _load_registry()

    if grid not in registry['datasets']:
        available = list(registry['datasets'].keys())
        raise ValueError(
            f"Unknown grid '{grid}'. Available: {available}"
        )

    dataset_info = registry['datasets'][grid]

    # Load pickle file
    data_path = get_data_path(grid)
    with open(data_path, 'rb') as f:
        data_raw = pickle.load(f)

    # Handle list format - use data[7] which contains the training data
    if isinstance(data_raw, list):
        data_df = data_raw[7]
    else:
        data_df = data_raw

    # Extract features and labels from DataFrame
    # The last column contains labels ('stable-unstable')
    # All other columns are features
    if isinstance(data_df, pd.DataFrame):
        # Separate features and labels
        label_column = 'stable-unstable'
        if label_column in data_df.columns:
            X = data_df.drop(columns=[label_column]).values
            y = data_df[label_column].values
        else:
            # Fallback: assume last column is labels
            X = data_df.iloc[:, :-1].values
            y = data_df.iloc[:, -1].values
    else:
        raise ValueError(f"Expected DataFrame but got {type(data_df)}")

    # Generate feature names based on actual number of features
    n_features = X.shape[1]
    feature_names = [f'F_{i}' for i in range(n_features)]
    target_names = ['stable', 'unstable']

    # Handle different return formats
    if test_size is not None:
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    if return_X_y:
        return X, y

    # Return as StabilityDataset
    frame = None
    if as_frame:
        frame = pd.DataFrame(X, columns=feature_names)
        frame['stability'] = y

    dataset = StabilityDataset(
        data=X,
        target=y,
        feature_names=feature_names,
        target_names=target_names,
        DESCR=dataset_info['description'],
        grid_name=grid,
        frame=frame
    )

    return dataset


def load_newengland(**kwargs) -> Union[StabilityDataset, Tuple]:
    """
    Load New England 39-bus power system dataset.

    This is a convenience wrapper around load_dataset('NewEngland', **kwargs).

    Args:
        **kwargs: Arguments passed to load_dataset()

    Returns:
        StabilityDataset or tuple depending on kwargs

    Examples:
        >>> # Load full dataset
        >>> dataset = load_newengland()
        >>> print(dataset)
        StabilityDataset(grid='NewEngland', n_samples=..., n_features=78)

        >>> # Get train/test split
        >>> X_train, X_test, y_train, y_test = load_newengland(test_size=0.2)
    """
    return load_dataset('NewEngland', **kwargs)


def load_ieee9bus(**kwargs) -> Union[StabilityDataset, Tuple]:
    """
    Load IEEE 9-bus power system dataset.

    This is a convenience wrapper around load_dataset('NineBusSystem', **kwargs).

    Args:
        **kwargs: Arguments passed to load_dataset()

    Returns:
        StabilityDataset or tuple depending on kwargs

    Examples:
        >>> # Load full dataset
        >>> dataset = load_ieee9bus()
        >>> print(dataset)
        StabilityDataset(grid='NineBusSystem', n_samples=..., n_features=18)

        >>> # Get train/test split
        >>> X_train, X_test, y_train, y_test = load_ieee9bus(test_size=0.2)
    """
    return load_dataset('NineBusSystem', **kwargs)

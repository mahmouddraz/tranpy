"""
Dataset generation from PowerFactory simulation results.

This module converts simulation results into training-ready datasets
compatible with the TranPy datasets API.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from ..utils.logging import get_logger
from .results import SimulationResults
from ..datasets.base import StabilityDataset, generate_feature_names

logger = get_logger(__name__)


def generate_dataset_from_simulation(
    results: SimulationResults,
    snapshot_type: str = 'clearing',
    test_size: float = 0.2,
    val_size: float = 0.2,
    shuffle: bool = False,
    random_state: Optional[int] = None
) -> Tuple[StabilityDataset, dict]:
    """
    Generate training dataset from simulation results.

    This function extracts features (bus voltage and angle at fault clearing time)
    and labels (stable/unstable) from simulation results.

    Args:
        results: SimulationResults object
        snapshot_type: 'fault' or 'clearing' - which snapshot to use
        test_size: Fraction of data for test set
        val_size: Fraction of training data for validation set
        shuffle: Whether to shuffle before splitting
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (dataset, splits_dict) where splits_dict contains
        train/test/val DataFrames with features and labels

    Examples:
        >>> results = SimulationResults.load('path/to/results.pickle')
        >>> dataset, splits = generate_dataset_from_simulation(results)
        >>> print(f"Dataset: {dataset.data.shape}")
        >>> print(f"Train: {splits['train'].shape}")
    """
    if not results.events:
        raise ValueError("No events in simulation results")

    # Extract features and labels
    feature_arrays = []
    labels = []

    for event in results.events:
        # Get the appropriate snapshot
        if snapshot_type == 'fault':
            snapshot = event.bus_snapshot_at_fault
        else:
            snapshot = event.bus_snapshot_at_clearing

        # Convert to feature array
        features = snapshot.to_feature_array()
        feature_arrays.append(features)

        # Add label (0 for stable, 1 for unstable)
        labels.append(0 if event.is_stable else 1)

    # Convert to numpy arrays
    X = np.array(feature_arrays)
    y = np.array(labels)

    # Generate feature names
    n_buses = len(feature_arrays[0]) // 2  # Each bus has voltage + angle
    feature_names = generate_feature_names(n_buses)

    # Create train/test/val splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, shuffle=shuffle, random_state=random_state
    )

    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['stable-unstable'] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['stable-unstable'] = y_test

    val_df = pd.DataFrame(X_val, columns=feature_names)
    val_df['stable-unstable'] = y_val

    all_df = pd.DataFrame(X, columns=feature_names)
    all_df['stable-unstable'] = y

    splits = {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'y_train': y_train,
        'y_test': y_test,
        'y_val': y_val,
        'train': train_df,
        'test': test_df,
        'val': val_df,
        'data': all_df
    }

    # Create StabilityDataset
    dataset = StabilityDataset(
        data=X,
        target=y,
        feature_names=feature_names,
        target_names=['stable', 'unstable'],
        DESCR=f"{results.grid_name} stability dataset generated from PowerFactory simulations",
        grid_name=results.grid_name,
        frame=all_df
    )

    return dataset, splits


def load_and_generate_dataset(
    results_path: Path,
    snapshot_type: str = 'clearing',
    test_size: float = 0.2,
    val_size: float = 0.2,
    shuffle: bool = False,
    random_state: Optional[int] = None
) -> Tuple[StabilityDataset, dict]:
    """
    Load simulation results and generate dataset.

    Args:
        results_path: Path to SimulationResults pickle file
        snapshot_type: 'fault' or 'clearing'
        test_size: Fraction for test set
        val_size: Fraction for validation set
        shuffle: Whether to shuffle data
        random_state: Random seed

    Returns:
        Tuple of (dataset, splits_dict)

    Examples:
        >>> dataset, splits = load_and_generate_dataset(
        ...     'simulation_results/NewEngland/pickles/NewEngland_results.pickle'
        ... )
    """
    results = SimulationResults.load(Path(results_path))
    return generate_dataset_from_simulation(
        results, snapshot_type, test_size, val_size, shuffle, random_state
    )


def generate_legacy_dataset(
    results: SimulationResults,
    output_path: Path
) -> dict:
    """
    Generate dataset in legacy format compatible with old training code.

    This creates the same structure as the old get_dataset() function:
    [X_train, X_test, y_train, y_test, train, test, val, data]

    Args:
        results: SimulationResults object
        output_path: Path to save pickle file

    Returns:
        Dictionary with all dataset components

    Examples:
        >>> results = SimulationResults.load('results.pickle')
        >>> data = generate_legacy_dataset(results, 'data_set/NewEngland.pickle')
    """
    dataset, splits = generate_dataset_from_simulation(results)

    # Package in legacy format
    legacy_data = [
        splits['X_train'],
        splits['X_test'],
        splits['y_train'],
        splits['y_test'],
        splits['train'],
        splits['test'],
        splits['val'],
        splits['data']
    ]

    # Save to pickle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(legacy_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved legacy dataset: {output_path}")

    return {
        'X_train': splits['X_train'],
        'X_test': splits['X_test'],
        'y_train': splits['y_train'],
        'y_test': splits['y_test'],
        'train': splits['train'],
        'test': splits['test'],
        'val': splits['val'],
        'data': splits['data']
    }


class DatasetGenerator:
    """
    Dataset generator with configuration options.

    This class provides a configurable interface for generating datasets
    from simulation results with various options.

    Args:
        results: SimulationResults object or path to pickle file
        snapshot_type: Which snapshot to use ('fault' or 'clearing')

    Examples:
        >>> gen = DatasetGenerator('results.pickle', snapshot_type='clearing')
        >>> dataset = gen.generate()
        >>> gen.save_dataset('output_dataset.pickle')
    """

    def __init__(
        self,
        results: [SimulationResults, str, Path],
        snapshot_type: str = 'clearing'
    ):
        if isinstance(results, (str, Path)):
            self.results = SimulationResults.load(Path(results))
        else:
            self.results = results

        self.snapshot_type = snapshot_type
        self.dataset = None
        self.splits = None

    def generate(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> StabilityDataset:
        """
        Generate dataset from simulation results.

        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            shuffle: Whether to shuffle data
            random_state: Random seed

        Returns:
            StabilityDataset object
        """
        self.dataset, self.splits = generate_dataset_from_simulation(
            self.results, self.snapshot_type, test_size, val_size, shuffle, random_state
        )
        return self.dataset

    def save_dataset(self, output_path: Path, format: str = 'new'):
        """
        Save dataset to file.

        Args:
            output_path: Path to save file
            format: 'new' or 'legacy' format

        Raises:
            ValueError: If dataset not yet generated
        """
        if self.dataset is None or self.splits is None:
            raise ValueError("Dataset not generated. Call generate() first.")

        output_path = Path(output_path)

        if format == 'legacy':
            generate_legacy_dataset(self.results, output_path)
        else:
            # Save dataset and splits separately
            import pickle
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                pickle.dump({
                    'dataset': self.dataset,
                    'splits': self.splits
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Saved dataset: {output_path}")

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        if self.dataset is None:
            raise ValueError("Dataset not generated. Call generate() first.")

        return {
            'grid_name': self.dataset.grid_name,
            'n_samples': self.dataset.data.shape[0],
            'n_features': self.dataset.data.shape[1],
            'n_stable': np.sum(self.dataset.target == 0),
            'n_unstable': np.sum(self.dataset.target == 1),
            'stability_ratio': np.mean(self.dataset.target == 0)
        }

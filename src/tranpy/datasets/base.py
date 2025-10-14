"""Base classes for TranPy datasets."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd


@dataclass
class StabilityDataset:
    """
    Container for power system stability datasets.

    Similar to sklearn's Bunch objects, this class holds dataset components
    with convenient attribute access.

    Attributes:
        data: Feature matrix (n_samples, n_features)
        target: Target labels (n_samples,)
        feature_names: Names of features
        target_names: Names of target classes
        DESCR: Dataset description
        grid_name: Name of the grid system
        frame: Optional pandas DataFrame with all data
    """
    data: np.ndarray
    target: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    DESCR: str
    grid_name: str
    frame: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        return (
            f"StabilityDataset(grid='{self.grid_name}', "
            f"n_samples={self.data.shape[0]}, "
            f"n_features={self.data.shape[1]})"
        )

    def to_frame(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        if self.frame is not None:
            return self.frame

        df = pd.DataFrame(self.data, columns=self.feature_names)
        df['stability'] = self.target
        return df


def generate_feature_names(n_buses: int) -> List[str]:
    """
    Generate feature names for voltage and phase angle measurements.

    Args:
        n_buses: Number of buses in the system

    Returns:
        List of feature names in format ['bus_1:u', 'bus_1:phi', ...]
    """
    feature_names = []
    for bus in range(1, n_buses + 1):
        feature_names.append(f'bus_{bus}:u')     # voltage magnitude
        feature_names.append(f'bus_{bus}:phi')   # phase angle
    return feature_names

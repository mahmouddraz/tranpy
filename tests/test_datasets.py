"""
Comprehensive test suite for dataset loading functionality.
"""

import numpy as np
import pandas as pd
import pytest
from tranpy.datasets import (
    load_dataset,
    load_newengland,
    load_ieee9bus,
    list_available_datasets,
    get_data_path
)


class TestDatasetRegistry:
    """Test dataset registry functionality."""

    def test_list_available_datasets(self):
        """Test listing available datasets."""
        datasets = list_available_datasets()

        assert isinstance(datasets, list)
        assert len(datasets) > 0

        # Check structure of dataset info
        for ds in datasets:
            assert 'key' in ds
            assert 'name' in ds
            assert 'description' in ds
            assert 'n_buses' in ds
            assert 'n_features' in ds
            assert 'size_mb' in ds
            assert 'bundled' in ds
            assert 'google_drive_link' in ds

    def test_get_data_path_valid(self):
        """Test getting path to valid dataset."""
        path = get_data_path('NewEngland')
        assert path.exists()
        assert path.suffix == '.pickle'
        assert 'NewEngland' in path.name

    def test_get_data_path_invalid(self):
        """Test getting path to invalid dataset raises error."""
        with pytest.raises(ValueError, match="Unknown grid"):
            get_data_path('NonExistentGrid')


class TestDatasetLoading:
    """Test dataset loading with various options."""

    def test_load_newengland_basic(self):
        """Test basic loading of New England dataset."""
        dataset = load_newengland()

        assert dataset.grid_name == 'NewEngland'
        assert dataset.data.shape[0] > 0  # Has samples
        assert dataset.data.shape[1] == 77  # 77 features
        assert dataset.target.shape[0] == dataset.data.shape[0]
        assert len(dataset.feature_names) == 77
        assert len(dataset.target_names) == 2
        assert set(dataset.target_names) == {'stable', 'unstable'}
        assert isinstance(dataset.DESCR, str)

    def test_load_ieee9bus_basic(self):
        """Test basic loading of IEEE 9-bus dataset."""
        dataset = load_ieee9bus()

        assert dataset.grid_name == 'NineBusSystem'
        assert dataset.data.shape[0] > 0
        assert dataset.data.shape[1] == 17  # 17 features
        assert dataset.target.shape[0] == dataset.data.shape[0]

    def test_load_dataset_return_X_y(self):
        """Test loading dataset as X, y arrays."""
        X, y = load_newengland(return_X_y=True)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 77
        assert set(np.unique(y)) == {0, 1}  # Binary labels

    def test_load_dataset_train_test_split(self):
        """Test loading with automatic train/test split."""
        X_train, X_test, y_train, y_test = load_newengland(
            test_size=0.2,
            random_state=42
        )

        # Check shapes
        assert X_train.shape[1] == 77
        assert X_test.shape[1] == 77
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # Check split ratio (approximately 80/20)
        total = X_train.shape[0] + X_test.shape[0]
        test_ratio = X_test.shape[0] / total
        assert 0.15 < test_ratio < 0.25  # Allow some variance

    def test_load_dataset_reproducible_split(self):
        """Test that random_state makes splits reproducible."""
        X_train1, X_test1, y_train1, y_test1 = load_newengland(
            test_size=0.2,
            random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = load_newengland(
            test_size=0.2,
            random_state=42
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_load_dataset_as_frame(self):
        """Test loading dataset with pandas DataFrame."""
        dataset = load_newengland(as_frame=True)

        assert dataset.frame is not None
        assert isinstance(dataset.frame, pd.DataFrame)
        assert dataset.frame.shape[0] == dataset.data.shape[0]
        assert 'stability' in dataset.frame.columns
        assert len(dataset.frame.columns) == 78  # 77 features + 1 target

    def test_dataset_to_frame(self):
        """Test converting dataset to DataFrame."""
        dataset = load_newengland()
        df = dataset.to_frame()

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == dataset.data.shape[0]
        assert 'stability' in df.columns


class TestDatasetProperties:
    """Test dataset data properties and quality."""

    @pytest.fixture
    def newengland_data(self):
        """Load New England dataset once for all tests."""
        return load_newengland(return_X_y=True)

    def test_data_types(self, newengland_data):
        """Test that data has correct types."""
        X, y = newengland_data

        assert X.dtype == np.float64 or X.dtype == np.float32
        assert y.dtype in [np.int32, np.int64]

    def test_no_missing_values(self, newengland_data):
        """Test that there are no NaN or infinite values."""
        X, y = newengland_data

        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
        assert not np.any(np.isnan(y))

    def test_label_distribution(self, newengland_data):
        """Test that both stability classes are present."""
        X, y = newengland_data

        unique_labels = np.unique(y)
        assert len(unique_labels) == 2
        assert 0 in unique_labels
        assert 1 in unique_labels

        # Check that classes are reasonably balanced (not too skewed)
        label_counts = np.bincount(y)
        min_count = label_counts.min()
        max_count = label_counts.max()
        imbalance_ratio = max_count / min_count
        assert imbalance_ratio < 10  # Less than 10:1 imbalance

    def test_feature_ranges(self, newengland_data):
        """Test that features have reasonable ranges."""
        X, y = newengland_data

        # Features should have finite ranges
        for i in range(X.shape[1]):
            feature = X[:, i]
            assert np.isfinite(feature).all()
            assert feature.std() > 0  # Not constant


class TestDatasetAPI:
    """Test dataset API compatibility."""

    def test_dataset_repr(self):
        """Test dataset string representation."""
        dataset = load_newengland()
        repr_str = repr(dataset)

        assert 'NewEngland' in repr_str
        assert 'n_samples' in repr_str
        assert 'n_features' in repr_str

    def test_load_dataset_generic(self):
        """Test loading via generic load_dataset function."""
        dataset1 = load_dataset('NewEngland')
        dataset2 = load_newengland()

        np.testing.assert_array_equal(dataset1.data, dataset2.data)
        np.testing.assert_array_equal(dataset1.target, dataset2.target)

    def test_invalid_grid_name(self):
        """Test that invalid grid name raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown grid"):
            load_dataset('InvalidGrid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

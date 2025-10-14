"""
Basic functionality tests for TranPy.

These tests verify that the main API works correctly.
"""

import numpy as np
import pytest


class TestDatasets:
    """Test dataset loading functionality."""

    def test_load_newengland_as_dataset(self):
        """Test loading New England dataset as object."""
        from tranpy.datasets import load_newengland

        dataset = load_newengland()

        assert dataset.grid_name == 'NewEngland'
        assert dataset.data.shape[1] == 77  # 77 feature columns
        assert len(dataset.feature_names) == 77
        assert len(dataset.target_names) == 2
        assert dataset.target_names == ['stable', 'unstable']

    def test_load_newengland_train_test_split(self):
        """Test loading with train/test split."""
        from tranpy.datasets import load_newengland

        X_train, X_test, y_train, y_test = load_newengland(
            test_size=0.2,
            random_state=42
        )

        assert X_train.shape[1] == 77
        assert X_test.shape[1] == 77
        assert len(X_train) + len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

    def test_load_newengland_return_X_y(self):
        """Test loading as X, y arrays."""
        from tranpy.datasets import load_newengland

        X, y = load_newengland(return_X_y=True)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1] == 77
        assert len(X) == len(y)

    def test_load_ieee9bus(self):
        """Test loading IEEE 9-bus dataset."""
        from tranpy.datasets import load_ieee9bus

        dataset = load_ieee9bus()

        assert dataset.grid_name == 'NineBusSystem'
        assert dataset.data.shape[1] == 17  # 17 feature columns


class TestModels:
    """Test model functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        from tranpy.datasets import load_newengland

        X_train, X_test, y_train, y_test = load_newengland(
            test_size=0.2,
            random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_svm_classifier(self, sample_data):
        """Test SVM classifier."""
        from tranpy.models import SVMClassifier

        X_train, X_test, y_train, y_test = sample_data

        # Create and train model
        model = SVMClassifier(kernel='rbf')
        model.fit(X_train[:100], y_train[:100])  # Use subset for speed

        # Make predictions
        predictions = model.predict(X_test[:10])

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
        assert model.is_fitted_

    def test_mlp_classifier(self, sample_data):
        """Test MLP classifier."""
        from tranpy.models import MLPClassifier

        X_train, X_test, y_train, y_test = sample_data

        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10)
        model.fit(X_train[:100], y_train[:100])

        predictions = model.predict(X_test[:10])

        assert len(predictions) == 10
        assert model.is_fitted_

    def test_decision_tree_classifier(self, sample_data):
        """Test Decision Tree classifier."""
        from tranpy.models import DecisionTreeClassifier

        X_train, X_test, y_train, y_test = sample_data

        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X_train[:100], y_train[:100])

        predictions = model.predict(X_test[:10])

        assert len(predictions) == 10
        assert model.is_fitted_

    def test_model_evaluate(self, sample_data):
        """Test model evaluation method."""
        from tranpy.models import DecisionTreeClassifier

        X_train, X_test, y_train, y_test = sample_data

        model = DecisionTreeClassifier()
        model.fit(X_train[:100], y_train[:100])

        results = model.evaluate(X_test[:50], y_test[:50], verbose=False)

        assert 'accuracy' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results
        assert 0 <= results['accuracy'] <= 1


class TestUtils:
    """Test utility functions."""

    def test_config_management(self, tmp_path):
        """Test configuration save/load."""
        from tranpy.utils import save_config, load_config

        config = {
            'grid': 'NewEngland',
            'epochs': 10,
            'batch_size': 32
        }

        config_file = tmp_path / "config.yaml"
        save_config(config, str(config_file))

        loaded_config = load_config(str(config_file))

        assert loaded_config['grid'] == 'NewEngland'
        assert loaded_config['epochs'] == 10

    def test_results_io(self, tmp_path):
        """Test results save/load."""
        from tranpy.utils import save_results, load_results

        results = {
            'accuracy': 0.95,
            'predictions': np.array([0, 1, 0, 1])
        }

        results_file = tmp_path / "results.pkl"
        save_results(results, str(results_file), format='pickle')

        loaded_results = load_results(str(results_file), format='pickle')

        assert loaded_results['accuracy'] == 0.95
        assert np.array_equal(loaded_results['predictions'], results['predictions'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

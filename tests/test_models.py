"""
Comprehensive test suite for machine learning models.
"""

import numpy as np
import pytest
from pathlib import Path
from tranpy.datasets import load_newengland
from tranpy.models import (
    SVMClassifier,
    MLPClassifier,
    DecisionTreeClassifier
)


@pytest.fixture
def sample_data():
    """Create sample training data for all tests."""
    X_train, X_test, y_train, y_test = load_newengland(
        test_size=0.2,
        random_state=42
    )
    # Use subset for faster tests
    return X_train[:200], X_test[:50], y_train[:200], y_test[:50]


@pytest.fixture
def tiny_data():
    """Create tiny dataset for quick tests."""
    X_train, X_test, y_train, y_test = load_newengland(
        test_size=0.2,
        random_state=42
    )
    return X_train[:50], X_test[:10], y_train[:50], y_test[:10]


class TestSVMClassifier:
    """Test SVM classifier functionality."""

    def test_svm_init(self):
        """Test SVM initialization."""
        model = SVMClassifier(kernel='rbf', C=1.0)
        assert model.kernel == 'rbf'
        assert model.C == 1.0
        assert not model.is_fitted_

    def test_svm_fit(self, tiny_data):
        """Test SVM training."""
        X_train, X_test, y_train, y_test = tiny_data
        model = SVMClassifier(kernel='linear')
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert hasattr(model, 'model_')

    def test_svm_predict(self, tiny_data):
        """Test SVM prediction."""
        X_train, X_test, y_train, y_test = tiny_data
        model = SVMClassifier(kernel='linear')
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_svm_predict_proba(self, tiny_data):
        """Test SVM probability predictions."""
        X_train, X_test, y_train, y_test = tiny_data
        model = SVMClassifier(kernel='linear', probability=True)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)

        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_svm_different_kernels(self, tiny_data):
        """Test SVM with different kernel types."""
        X_train, X_test, y_train, y_test = tiny_data

        for kernel in ['linear', 'rbf', 'poly']:
            model = SVMClassifier(kernel=kernel)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert len(predictions) == len(X_test)


class TestMLPClassifier:
    """Test MLP (Neural Network) classifier functionality."""

    def test_mlp_init(self):
        """Test MLP initialization."""
        model = MLPClassifier(hidden_layer_sizes=(50, 25))
        assert model.hidden_layer_sizes == (50, 25)
        assert not model.is_fitted_

    def test_mlp_fit(self, tiny_data):
        """Test MLP training."""
        X_train, X_test, y_train, y_test = tiny_data
        model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=50)
        model.fit(X_train, y_train)

        assert model.is_fitted_

    def test_mlp_predict(self, tiny_data):
        """Test MLP prediction."""
        X_train, X_test, y_train, y_test = tiny_data
        model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=50)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_mlp_predict_proba(self, tiny_data):
        """Test MLP probability predictions."""
        X_train, X_test, y_train, y_test = tiny_data
        model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=50)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)

        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestDecisionTreeClassifier:
    """Test Decision Tree classifier functionality."""

    def test_dt_init(self):
        """Test Decision Tree initialization."""
        model = DecisionTreeClassifier(max_depth=5)
        assert model.max_depth == 5
        assert not model.is_fitted_

    def test_dt_fit(self, tiny_data):
        """Test Decision Tree training."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)

        assert model.is_fitted_

    def test_dt_predict(self, tiny_data):
        """Test Decision Tree prediction."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_dt_feature_importance(self, tiny_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)

        importance = model.model_.feature_importances_

        assert len(importance) == X_train.shape[1]
        assert np.all(importance >= 0)
        assert np.isclose(importance.sum(), 1.0)


class TestModelEvaluation:
    """Test model evaluation functionality."""

    def test_evaluate_returns_metrics(self, tiny_data):
        """Test that evaluate returns expected metrics."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)

        results = model.evaluate(X_test, y_test, verbose=False)

        assert 'accuracy' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results

    def test_evaluate_accuracy_range(self, tiny_data):
        """Test that accuracy is in valid range."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X_train, y_train)

        results = model.evaluate(X_test, y_test, verbose=False)

        assert 0 <= results['accuracy'] <= 1

    def test_evaluate_confusion_matrix_shape(self, tiny_data):
        """Test confusion matrix shape."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        results = model.evaluate(X_test, y_test, verbose=False)

        cm = results['confusion_matrix']
        assert cm.shape == (2, 2)  # Binary classification


class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_model(self, tiny_data, tmp_path):
        """Test saving trained model."""
        from tranpy.models import save_model

        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)

        model_path = tmp_path / "test_model.pkl"
        save_model(model, str(model_path))

        assert model_path.exists()

    def test_load_model(self, tiny_data, tmp_path):
        """Test loading saved model."""
        from tranpy.models import save_model, load_model

        X_train, X_test, y_train, y_test = tiny_data

        # Train and save model
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)
        predictions_original = model.predict(X_test)

        model_path = tmp_path / "test_model.pkl"
        save_model(model, str(model_path))

        # Load and compare
        loaded_model = load_model(str(model_path))
        predictions_loaded = loaded_model.predict(X_test)

        np.testing.assert_array_equal(predictions_original, predictions_loaded)


class TestModelValidation:
    """Test model validation and error handling."""

    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises error."""
        model = DecisionTreeClassifier()
        X_test = np.random.rand(10, 77)

        with pytest.raises(Exception, match="not fitted|must be fitted"):
            model.predict(X_test)

    def test_fit_with_mismatched_shapes(self, tiny_data):
        """Test fitting with mismatched X and y shapes."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier()

        with pytest.raises(ValueError):
            model.fit(X_train, y_train[:-5])  # Mismatched length

    def test_predict_with_wrong_features(self, tiny_data):
        """Test prediction with wrong number of features."""
        X_train, X_test, y_train, y_test = tiny_data
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        X_wrong = np.random.rand(10, 50)  # Wrong feature count

        with pytest.raises(ValueError):
            model.predict(X_wrong)


class TestModelComparison:
    """Test comparing different models."""

    def test_models_on_same_data(self, sample_data):
        """Test multiple models on same dataset."""
        X_train, X_test, y_train, y_test = sample_data

        models = {
            'SVM': SVMClassifier(kernel='linear'),
            'DecisionTree': DecisionTreeClassifier(max_depth=5),
            'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=50)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            eval_results = model.evaluate(X_test, y_test, verbose=False)
            results[name] = eval_results['accuracy']

        # All models should achieve reasonable accuracy
        for name, accuracy in results.items():
            assert accuracy > 0.4, f"{name} accuracy too low: {accuracy}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

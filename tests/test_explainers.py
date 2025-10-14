"""
Comprehensive tests for explainability module.
"""

import pytest
import numpy as np
from tranpy.datasets import load_newengland
from tranpy.models import DecisionTreeClassifier, RandomForestClassifier


@pytest.fixture(scope='module')
def setup_model_and_data():
    """Setup trained model and data for explainer tests."""
    X_train, X_test, y_train, y_test = load_newengland(
        test_size=0.2,
        random_state=42
    )

    # Use smaller subset for speed
    X_train_small = X_train[:500]
    y_train_small = y_train[:500]
    X_test_small = X_test[:50]
    y_test_small = y_test[:50]

    # Train simple model
    model = DecisionTreeClassifier(max_depth=10)
    model.fit(X_train_small, y_train_small)

    return model, X_train_small, X_test_small, y_test_small


class TestExplainersImport:
    """Test that explainers can be imported."""

    def test_import_explainers_module(self):
        """Test importing explainers module."""
        import tranpy.explainers
        assert tranpy.explainers is not None

    def test_lime_explainer_available(self):
        """Test LIME explainer import."""
        try:
            from tranpy.explainers import LIMEExplainer
            assert LIMEExplainer is not None
        except ImportError:
            pytest.skip("LIME not installed")

    def test_shap_explainer_available(self):
        """Test SHAP explainer import."""
        try:
            from tranpy.explainers import SHAPExplainer
            assert SHAPExplainer is not None
        except ImportError:
            pytest.skip("SHAP not installed")

    def test_dalex_explainer_available(self):
        """Test DALEX explainers import."""
        try:
            from tranpy.explainers import BreakdownExplainer, SurrogateExplainer
            assert BreakdownExplainer is not None
            assert SurrogateExplainer is not None
        except ImportError:
            pytest.skip("DALEX not installed")


class TestLIMEExplainer:
    """Test LIME explainer functionality."""

    def test_lime_explainer_init(self, setup_model_and_data):
        """Test LIME explainer initialization."""
        try:
            from tranpy.explainers import LIMEExplainer
        except ImportError:
            pytest.skip("LIME not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        explainer = LIMEExplainer(
            model=model,
            X_train=X_train,
            X_test=X_test
        )

        assert explainer is not None
        assert explainer.model is not None
        assert explainer.X_train.shape[0] > 0

    def test_lime_explain_instance(self, setup_model_and_data):
        """Test LIME instance explanation."""
        try:
            from tranpy.explainers import LIMEExplainer
        except ImportError:
            pytest.skip("LIME not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        explainer = LIMEExplainer(
            model=model,
            X_train=X_train,
            X_test=X_test
        )

        # Explain single instance
        explanation = explainer.explain_instance(X_test[0])

        assert explanation is not None

    def test_lime_with_feature_names(self, setup_model_and_data):
        """Test LIME with custom feature names."""
        try:
            from tranpy.explainers import LIMEExplainer
        except ImportError:
            pytest.skip("LIME not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

        explainer = LIMEExplainer(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names
        )

        assert len(explainer.feature_names) == X_train.shape[1]


class TestSHAPExplainer:
    """Test SHAP explainer functionality."""

    def test_shap_explainer_init(self, setup_model_and_data):
        """Test SHAP explainer initialization."""
        try:
            from tranpy.explainers import SHAPExplainer
        except ImportError:
            pytest.skip("SHAP not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        try:
            explainer = SHAPExplainer(
                model=model,
                X_train=X_train[:100],  # Use smaller subset
                X_test=X_test[:10]
            )
            assert explainer is not None
            assert explainer.model is not None
        except ImportError:
            pytest.skip("SHAP not installed")

    def test_shap_explain_instance(self, setup_model_and_data):
        """Test SHAP instance explanation."""
        try:
            from tranpy.explainers import SHAPExplainer
        except ImportError:
            pytest.skip("SHAP not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        try:
            explainer = SHAPExplainer(
                model=model,
                X_train=X_train[:100],
                X_test=X_test[:10]
            )
            # Explain single instance
            explanation = explainer.explain_instance(X_test[0])
            assert explanation is not None
        except ImportError:
            pytest.skip("SHAP not installed")


class TestDALEXExplainer:
    """Test DALEX explainer functionality."""

    def test_breakdown_explainer_init(self, setup_model_and_data):
        """Test DALEX Breakdown explainer initialization."""
        try:
            from tranpy.explainers import BreakdownExplainer
        except ImportError:
            pytest.skip("DALEX not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        try:
            explainer = BreakdownExplainer(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test
            )
            assert explainer is not None
        except ImportError:
            pytest.skip("DALEX not installed")

    def test_surrogate_explainer_init(self, setup_model_and_data):
        """Test DALEX Surrogate explainer initialization."""
        try:
            from tranpy.explainers import SurrogateExplainer
        except ImportError:
            pytest.skip("DALEX not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        explainer = SurrogateExplainer(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test
        )

        assert explainer is not None


class TestExplainerCompatibility:
    """Test explainer compatibility with different models."""

    def test_explainer_with_random_forest(self):
        """Test explainers work with RandomForest."""
        try:
            from tranpy.explainers import LIMEExplainer
        except ImportError:
            pytest.skip("LIME not installed")

        X_train, X_test, y_train, y_test = load_newengland(
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, max_depth=5)
        model.fit(X_train[:200], y_train[:200])

        explainer = LIMEExplainer(
            model=model,
            X_train=X_train[:200],
            X_test=X_test[:20]
        )

        assert explainer is not None

    def test_explainer_requires_predict_proba(self, setup_model_and_data):
        """Test that explainer works with models having predict_proba."""
        try:
            from tranpy.explainers import LIMEExplainer
        except ImportError:
            pytest.skip("LIME not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        # Verify model has predict_proba
        assert hasattr(model, 'predict_proba')

        explainer = LIMEExplainer(
            model=model,
            X_train=X_train,
            X_test=X_test
        )

        assert explainer is not None


class TestExplainerBase:
    """Test base explainer functionality."""

    def test_base_explainer_cannot_instantiate(self):
        """Test that BaseExplainer cannot be instantiated directly."""
        from tranpy.explainers.base import BaseExplainer

        X_train = np.random.rand(10, 5)

        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            BaseExplainer(model=None, X_train=X_train)

    def test_feature_name_generation(self, setup_model_and_data):
        """Test automatic feature name generation."""
        try:
            from tranpy.explainers import LIMEExplainer
        except ImportError:
            pytest.skip("LIME not installed")

        model, X_train, X_test, y_test = setup_model_and_data

        # Without providing feature names
        explainer = LIMEExplainer(
            model=model,
            X_train=X_train,
            X_test=X_test
        )

        # Should auto-generate feature names
        assert len(explainer.feature_names) == X_train.shape[1]
        assert all('F_' in name for name in explainer.feature_names)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

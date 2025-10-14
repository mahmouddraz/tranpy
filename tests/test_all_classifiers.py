"""
Comprehensive tests for all 18 classifiers.

Tests that all classifiers from the registry can be instantiated,
trained, and used for prediction.
"""

import pytest
import numpy as np
from tranpy.datasets import load_newengland

# All classifiers to test
ALL_CLASSIFIERS = [
    ('AdaBoostClassifier', {}),
    ('DecisionTreeClassifier', {'max_depth': 5}),
    ('DummyClassifier', {}),
    ('ExtraTreesClassifier', {'n_estimators': 10, 'max_depth': 5}),
    ('GaussianNB', {}),
    ('GaussianProcessClassifier', {}),
    ('GradientBoostingClassifier', {'n_estimators': 10, 'max_depth': 3}),
    ('KNeighborsClassifier', {'n_neighbors': 3}),
    ('LinearDiscriminantAnalysis', {}),
    ('LogisticRegression', {'max_iter': 200}),
    ('MLPClassifier', {'hidden_layer_sizes': (20,), 'max_iter': 50}),
    ('QuadraticDiscriminantAnalysis', {}),
    ('RandomForestClassifier', {'n_estimators': 10, 'max_depth': 5}),
    ('RidgeClassifier', {}),
    ('SGDClassifier', {'max_iter': 100}),
    ('SVMClassifier', {'kernel': 'linear'}),
]

# Optional classifiers
OPTIONAL_CLASSIFIERS = [
    ('XGBClassifier', {'n_estimators': 10}),
    ('LGBMClassifier', {'n_estimators': 10, 'verbose': -1}),
]


@pytest.fixture(scope='module')
def tiny_data():
    """Load small dataset for quick testing."""
    X_train, X_test, y_train, y_test = load_newengland(
        test_size=0.2,
        random_state=42
    )
    return X_train[:30], X_test[:10], y_train[:30], y_test[:10]


class TestAllClassifiersBasic:
    """Test basic functionality for all classifiers."""

    @pytest.mark.parametrize("classifier_name,params", ALL_CLASSIFIERS)
    def test_classifier_instantiation(self, classifier_name, params):
        """Test that classifier can be instantiated."""
        from tranpy import models

        cls = getattr(models, classifier_name)
        model = cls(**params)

        assert model is not None
        assert not model.is_fitted_

    @pytest.mark.parametrize("classifier_name,params", ALL_CLASSIFIERS)
    def test_classifier_fit(self, classifier_name, params, tiny_data):
        """Test that classifier can be trained."""
        from tranpy import models

        X_train, X_test, y_train, y_test = tiny_data
        cls = getattr(models, classifier_name)
        model = cls(**params)

        model.fit(X_train, y_train)

        assert model.is_fitted_

    @pytest.mark.parametrize("classifier_name,params", ALL_CLASSIFIERS)
    def test_classifier_predict(self, classifier_name, params, tiny_data):
        """Test that classifier can make predictions."""
        from tranpy import models

        X_train, X_test, y_train, y_test = tiny_data
        cls = getattr(models, classifier_name)
        model = cls(**params)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    @pytest.mark.parametrize("classifier_name,params", ALL_CLASSIFIERS)
    def test_classifier_evaluate(self, classifier_name, params, tiny_data):
        """Test that classifier can be evaluated."""
        from tranpy import models

        X_train, X_test, y_train, y_test = tiny_data
        cls = getattr(models, classifier_name)
        model = cls(**params)

        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test, verbose=False)

        assert 'accuracy' in results
        assert 0 <= results['accuracy'] <= 1


class TestOptionalClassifiers:
    """Test optional classifiers (XGBoost, LightGBM)."""

    @pytest.mark.parametrize("classifier_name,params", OPTIONAL_CLASSIFIERS)
    def test_optional_classifier(self, classifier_name, params, tiny_data):
        """Test optional classifiers if available."""
        from tranpy import models

        # Skip if not available
        if not hasattr(models, classifier_name):
            pytest.skip(f"{classifier_name} not installed")

        X_train, X_test, y_train, y_test = tiny_data
        cls = getattr(models, classifier_name)
        model = cls(**params)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert model.is_fitted_


class TestClassifierRegistry:
    """Test that all classifiers match the pretrained model registry."""

    def test_registry_has_all_grids(self):
        """Test that registry has both grid systems."""
        from tranpy.models import list_pretrained_models

        models = list_pretrained_models()
        grids = set(m['grid'] for m in models)

        assert 'NewEngland' in grids
        assert 'NineBusSystem' in grids

    def test_registry_has_36_models(self):
        """Test that registry has all 36 models (18 per grid)."""
        from tranpy.models import list_pretrained_models

        all_models = list_pretrained_models()
        ne_models = [m for m in all_models if m['grid'] == 'NewEngland']
        nb_models = [m for m in all_models if m['grid'] == 'NineBusSystem']

        assert len(all_models) == 36, f"Expected 36 models, got {len(all_models)}"
        assert len(ne_models) == 18, f"Expected 18 NewEngland models, got {len(ne_models)}"
        assert len(nb_models) == 18, f"Expected 18 NineBusSystem models, got {len(nb_models)}"


class TestClassifierCompatibility:
    """Test sklearn-compatible interface for all classifiers."""

    @pytest.mark.parametrize("classifier_name,params", ALL_CLASSIFIERS[:5])  # Test subset
    def test_sklearn_clone_compatible(self, classifier_name, params):
        """Test that classifiers can be cloned (sklearn compatible)."""
        from sklearn.base import clone
        from tranpy import models

        cls = getattr(models, classifier_name)
        model = cls(**params)

        # Should be clonable
        cloned = clone(model)
        assert cloned is not model
        assert not cloned.is_fitted_


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

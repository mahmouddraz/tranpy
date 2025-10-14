"""Base classes for TranPy models."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ..utils.logging import get_logger

logger = get_logger(__name__)


class StabilityClassifier(BaseEstimator, ClassifierMixin, ABC):
    """
    Abstract base class for power system stability classifiers.

    This class provides a common interface for all TranPy models,
    following sklearn's estimator conventions.

    All subclasses must implement:
        - fit(X, y): Train the model
        - predict(X): Make predictions
        - predict_proba(X): Predict class probabilities
    """

    def __init__(self):
        """Initialize the classifier."""
        self.classes_ = np.array([0, 1])  # stable=0, unstable=1
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StabilityClassifier':
        """
        Train the classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            self: Fitted classifier
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the model.

        Args:
            X: Test features
            y: True labels
            verbose: If True, print results

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy
                - confusion_matrix: Confusion matrix
                - classification_report: Detailed classification metrics
        """
        predictions = self.predict(X)

        results = {
            'accuracy': accuracy_score(y, predictions),
            'confusion_matrix': confusion_matrix(y, predictions),
            'classification_report': classification_report(
                y, predictions,
                target_names=['stable', 'unstable'],
                output_dict=True
            )
        }

        if verbose:
            logger.info(f"Accuracy: {results['accuracy']:.4f}")
            logger.info("\nConfusion Matrix:")
            logger.info(results['confusion_matrix'])
            logger.info("\nClassification Report:")
            logger.info(classification_report(
                y, predictions,
                target_names=['stable', 'unstable']
            ))

        return results

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate input features.

        Args:
            X: Input features

        Returns:
            Validated features

        Raises:
            ValueError: If input shape is invalid
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        if self.is_fitted_ and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        return X

    def _set_fitted_attributes(self, X: np.ndarray):
        """Set attributes after fitting."""
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

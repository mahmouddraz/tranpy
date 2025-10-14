"""Ensemble models for stability prediction."""

from typing import List, Tuple, Literal
import numpy as np
from sklearn.ensemble import VotingClassifier
from timeit import default_timer as timer

from .base import StabilityClassifier
from .classical import SVMClassifier, MLPClassifier, DecisionTreeClassifier

try:
    from .neural import DNNClassifier
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False


class EnsembleClassifier(StabilityClassifier):
    """
    Ensemble classifier combining multiple base models.

    Combines predictions from multiple classifiers using voting strategies
    to improve overall accuracy and robustness.

    Args:
        estimators: List of (name, estimator) tuples. If None, uses default ensemble
                    of SVM, MLP, and DecisionTree (plus DNN if TensorFlow available)
        voting: Voting strategy ('hard' for majority vote, 'soft' for weighted average)
        weights: Optional list of weights for each estimator in soft voting
        flatten_transform: For soft voting, flatten the predictions

    Examples:
        >>> from tranpy.models import EnsembleClassifier, SVMClassifier, MLPClassifier
        >>>
        >>> # Use default ensemble
        >>> model = EnsembleClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Custom ensemble
        >>> estimators = [
        ...     ('svm', SVMClassifier()),
        ...     ('mlp', MLPClassifier())
        ... ]
        >>> model = EnsembleClassifier(estimators=estimators, voting='soft')
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        estimators: List[Tuple[str, StabilityClassifier]] = None,
        voting: Literal['hard', 'soft'] = 'soft',
        weights: List[float] = None,
        flatten_transform: bool = True
    ):
        super().__init__()

        if estimators is None:
            # Default ensemble: SVM + MLP + DecisionTree
            estimators = [
                ('svm', SVMClassifier(kernel='rbf')),
                ('mlp', MLPClassifier()),
                ('dt', DecisionTreeClassifier())
            ]

            # Add DNN if available
            if NEURAL_AVAILABLE:
                estimators.append(('dnn', DNNClassifier()))

        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.flatten_transform = flatten_transform

        # Extract underlying sklearn models for VotingClassifier
        sklearn_estimators = []
        for name, estimator in estimators:
            if hasattr(estimator, 'model_'):
                # If already fitted, use the underlying model
                sklearn_estimators.append((name, estimator))
            else:
                # Use the estimator directly
                sklearn_estimators.append((name, estimator))

        self.ensemble_ = VotingClassifier(
            estimators=sklearn_estimators,
            voting=voting,
            weights=weights,
            flatten_transform=flatten_transform
        )

        self.training_time_ = None
        self.prediction_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """
        Train all base models in the ensemble.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            self: Fitted ensemble classifier
        """
        X = self._validate_input(X)
        y = np.asarray(y)

        start = timer()

        # For TranPy models, fit them first to get underlying sklearn models
        fitted_estimators = []
        for name, estimator in self.estimators:
            if isinstance(estimator, StabilityClassifier):
                estimator.fit(X, y)
                # Use the underlying sklearn model if available
                if hasattr(estimator, 'model_'):
                    fitted_estimators.append((name, estimator.model_))
                else:
                    fitted_estimators.append((name, estimator))
            else:
                fitted_estimators.append((name, estimator))

        # Create new VotingClassifier with fitted models
        self.ensemble_ = VotingClassifier(
            estimators=fitted_estimators,
            voting=self.voting,
            weights=self.weights,
            flatten_transform=self.flatten_transform
        )

        # Fit the ensemble (this is fast since base models are already fitted)
        self.ensemble_.fit(X, y)

        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using ensemble voting.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        X = self._validate_input(X)

        start = timer()
        predictions = self.ensemble_.predict(X)
        self.prediction_time_ = timer() - start

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble.

        Args:
            X: Test features, shape (n_samples, n_features)

        Returns:
            Class probabilities, shape (n_samples, n_classes)

        Raises:
            ValueError: If voting strategy is not 'soft'
        """
        X = self._validate_input(X)

        if self.voting != 'soft':
            raise ValueError(
                "Probability estimates only available with voting='soft'"
            )

        return self.ensemble_.predict_proba(X)

    def get_estimator_predictions(self, X: np.ndarray) -> dict:
        """
        Get predictions from each base estimator.

        Args:
            X: Test features

        Returns:
            Dictionary mapping estimator names to their predictions
        """
        X = self._validate_input(X)

        predictions = {}
        for name, estimator in self.ensemble_.estimators_:
            predictions[name] = estimator.predict(X)

        return predictions

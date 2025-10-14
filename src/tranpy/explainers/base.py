"""Base classes for explainers."""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np


class BaseExplainer(ABC):
    """
    Abstract base class for model explainers.

    All explainers should inherit from this class and implement
    the required methods.
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the explainer.

        Args:
            model: Trained model to explain
            X_train: Training data for background/reference
            X_test: Test data to explain (optional)
            feature_names: Names of features
            class_names: Names of target classes
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.X_test = np.asarray(X_test) if X_test is not None else None
        self.feature_names = feature_names or [f'F_{i}' for i in range(X_train.shape[1])]
        self.class_names = class_names or ['stable', 'unstable']

    @abstractmethod
    def explain_instance(self, instance: np.ndarray, **kwargs):
        """
        Explain a single prediction.

        Args:
            instance: Single data instance to explain
            **kwargs: Additional arguments

        Returns:
            Explanation object
        """
        pass

    @abstractmethod
    def explain_global(self, **kwargs):
        """
        Generate global explanation for the model.

        Args:
            **kwargs: Additional arguments

        Returns:
            Global explanation object
        """
        pass

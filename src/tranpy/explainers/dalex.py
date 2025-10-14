"""DALEX-based explainers (Break Down, Surrogate)."""

from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

try:
    import dalex as dx
    DALEX_AVAILABLE = True
except ImportError:
    DALEX_AVAILABLE = False

from .base import BaseExplainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


def check_dalex():
    """Check if DALEX is available."""
    if not DALEX_AVAILABLE:
        raise ImportError(
            "DALEX is required for this explainer. "
            "Install with: pip install dalex"
        )


class BreakdownExplainer(BaseExplainer):
    """
    Break Down explainer using DALEX.

    Provides instance-level explanations showing how each feature
    contributes to the final prediction.

    Args:
        model: Trained model
        X_train: Training data (not used but kept for compatibility)
        X_test: Test data to explain
        y_test: Test labels
        feature_names: Names of features
        class_names: Names of target classes
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        check_dalex()
        super().__init__(model, X_train, X_test, feature_names, class_names)

        self.y_test = y_test
        self.explainer_ = None

    def _create_explainer(self):
        """Create DALEX explainer object."""
        # Convert to DataFrame with feature names
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        y_test_series = pd.Series(self.y_test)

        model_name = type(self.model).__name__

        self.explainer_ = dx.Explainer(
            self.model,
            X_test_df,
            y_test_series,
            label=f"Power System Stability - {model_name}"
        )

    def explain_instance(
        self,
        instance_idx: int = 0,
        max_vars: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Generate Break Down explanation for a single instance.

        Args:
            instance_idx: Index of instance to explain
            max_vars: Maximum features to show
            save_path: Path to save plot (SVG format)

        Returns:
            DALEX prediction parts object

        Examples:
            >>> explainer = BreakdownExplainer(model, X_train, X_test, y_test)
            >>> breakdown = explainer.explain_instance(0)
        """
        if self.explainer_ is None:
            self._create_explainer()

        # Get instance
        instance = self.X_test[instance_idx]

        # Generate break down
        breakdown = self.explainer_.predict_parts(
            instance,
            type='break_down'
        )

        # Plot
        fig = breakdown.plot(max_vars=max_vars, show=False)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(save_path)
            logger.info(f"Break down plot saved to {save_path}")

        return breakdown

    def explain_global(self, **kwargs):
        """Not implemented for Break Down (instance-level only)."""
        raise NotImplementedError(
            "Break Down is an instance-level explainer. "
            "Use explain_instance() instead."
        )


class SurrogateExplainer(BaseExplainer):
    """
    Surrogate model explainer.

    Creates an interpretable surrogate model (Logistic Regression)
    that approximates the complex model's behavior.

    Args:
        model: Trained model
        X_train: Training data
        X_test: Test data
        y_test: Test labels
        feature_names: Names of features
        class_names: Names of target classes
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        super().__init__(model, X_train, X_test, feature_names, class_names)

        self.y_test = y_test
        self.surrogate_model_ = None
        self.fidelity_score_ = None

    def explain_global(
        self,
        solver: str = 'lbfgs',
        max_iter: int = 5000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Train a surrogate logistic regression model.

        The surrogate model learns to mimic the complex model's predictions,
        providing interpretable feature coefficients.

        Args:
            solver: Solver for logistic regression
            max_iter: Maximum iterations
            **kwargs: Additional arguments for LogisticRegression

        Returns:
            DataFrame with feature importances from surrogate model

        Examples:
            >>> explainer = SurrogateExplainer(model, X_train, X_test, y_test)
            >>> importances = explainer.explain_global()
            >>> print(importances.head(10))
        """
        # Get predictions from complex model on training data
        logger.info("Generating predictions from complex model...")
        train_predictions = self.model.predict(self.X_train)

        # Train surrogate model
        logger.info("Training surrogate logistic regression model...")
        self.surrogate_model_ = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            **kwargs
        )
        self.surrogate_model_.fit(self.X_train, train_predictions)

        # Evaluate fidelity (how well surrogate approximates complex model)
        surrogate_preds = self.surrogate_model_.predict(self.X_test)
        complex_preds = self.model.predict(self.X_test)
        self.fidelity_score_ = r2_score(surrogate_preds, complex_preds)

        # Also check accuracy on true labels
        accuracy = self.surrogate_model_.score(self.X_test, self.y_test)

        logger.info(f"Surrogate model fidelity (R²): {self.fidelity_score_:.4f}")
        logger.info(f"Surrogate model accuracy: {accuracy:.4f}")

        # Extract feature importances (absolute coefficients)
        coefficients = np.abs(self.surrogate_model_.coef_[0])

        # Sort by importance
        sorted_indices = np.argsort(coefficients)[::-1]

        # Create DataFrame
        df = pd.DataFrame({
            'feature_name': [self.feature_names[i] for i in sorted_indices],
            'importance': coefficients[sorted_indices]
        })

        return df

    def explain_instance(self, instance: np.ndarray, **kwargs):
        """
        For surrogate models, use the model coefficients.

        This method is not typically used for surrogate explainers.
        """
        raise NotImplementedError(
            "Surrogate explainer is a global method. "
            "Use explain_global() instead."
        )

    def save_feature_importances(self, filepath: str):
        """
        Save feature importances to Excel file.

        Args:
            filepath: Path to save Excel file
        """
        if self.surrogate_model_ is None:
            raise ValueError("Surrogate model not trained. Run explain_global() first.")

        df = self.explain_global()

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(filepath, index=False)
        logger.info(f"Feature importances saved to {filepath}")

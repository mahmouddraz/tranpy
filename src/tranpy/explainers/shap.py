"""SHAP (SHapley Additive exPlanations) explainer."""

import os
import warnings
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from timeit import default_timer as timer

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..utils.logging import get_logger
from .base import BaseExplainer

logger = get_logger(__name__)


def check_shap():
    """Check if SHAP is available."""
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP is required for this explainer. "
            "Install with: pip install shap"
        )


class SHAPExplainer(BaseExplainer):
    """
    SHAP explainer for power system stability models.

    Provides SHAP value computation and visualization.

    Args:
        model: Trained model with predict or predict_proba method
        X_train: Training data for background
        X_test: Test data to explain
        feature_names: Names of features
        class_names: Names of target classes
        n_background: Number of background samples for KernelExplainer
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        n_background: int = 100
    ):
        check_shap()
        super().__init__(model, X_train, X_test, feature_names, class_names)

        self.n_background = n_background
        self.explainer_ = None
        self.shap_values_ = None

    def _get_predict_fn(self):
        """Get the appropriate prediction function for classification."""
        if hasattr(self.model, 'predict_proba'):
            # For binary classification, return probability of positive class
            return lambda x: self.model.predict_proba(x)[:, 1]
        elif hasattr(self.model, 'predict'):
            return self.model.predict
        else:
            raise ValueError("Model must have predict_proba or predict method")

    def _create_explainer(self):
        """Create SHAP explainer with background data."""
        # Use masker for background dataset (modern API)
        background = shap.sample(self.X_train, self.n_background)
        predict_fn = self._get_predict_fn()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use modern shap.Explainer API which returns Explanation objects
            self.explainer_ = shap.Explainer(predict_fn, background)

    def explain_instance(
        self,
        instance: np.ndarray,
        **kwargs
    ):
        """
        Compute SHAP values for a single instance.

        Args:
            instance: Single data instance (1D array)
            **kwargs: Additional arguments for explainer()

        Returns:
            SHAP Explanation object

        Examples:
            >>> explainer = SHAPExplainer(model, X_train, X_test)
            >>> shap_vals = explainer.explain_instance(X_test[0])
        """
        if self.explainer_ is None:
            self._create_explainer()

        instance = np.asarray(instance).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Modern API: call explainer directly, returns Explanation object
            shap_values = self.explainer_(instance, **kwargs)

        return shap_values

    def explain_global(
        self,
        X: Optional[np.ndarray] = None,
        use_cache: bool = True,
        cache_dir: str = 'explainer_outputs/shap_cache'
    ):
        """
        Compute SHAP values for all test instances.

        Args:
            X: Data to explain (uses X_test if None)
            use_cache: Whether to use cached results
            cache_dir: Directory for caching

        Returns:
            SHAP values array

        Examples:
            >>> explainer = SHAPExplainer(model, X_train, X_test)
            >>> shap_values = explainer.explain_global()
            >>> explainer.plot_summary(shap_values)
        """
        if X is None:
            if self.X_test is None:
                raise ValueError("X_test required for global explanations")
            X = self.X_test

        X = np.asarray(X)

        # Check cache
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        model_name = type(self.model).__name__
        cache_file = cache_path / f"{model_name}_shap.pkl"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached SHAP values from {cache_file}")
            import pickle
            with open(cache_file, 'rb') as f:
                self.shap_values_ = pickle.load(f)
            return self.shap_values_

        # Compute SHAP values
        if self.explainer_ is None:
            self._create_explainer()

        logger.info("Computing SHAP values...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = timer()
            # Modern API: call explainer directly, returns Explanation object
            self.shap_values_ = self.explainer_(X)
            elapsed = timer() - start
            logger.info(f"SHAP values computed in {elapsed:.3f}s")

        # Cache results
        if use_cache:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(self.shap_values_, f)
            logger.info(f"Cached SHAP values to {cache_file}")

        return self.shap_values_

    def plot_summary(
        self,
        shap_values=None,
        X: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        plot_type: str = 'dot',
        max_display: int = 20
    ):
        """
        Plot SHAP summary.

        Args:
            shap_values: SHAP values (uses cached if None)
            X: Data instances (uses X_test if None)
            save_path: Path to save plot
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum features to display
        """
        if shap_values is None:
            shap_values = self.shap_values_

        if shap_values is None:
            raise ValueError("No SHAP values available. Run explain_global() first.")

        # Modern API: shap_values is an Explanation object
        # Use shap.plots.beeswarm for modern API (equivalent to summary_plot with dots)
        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == 'dot':
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        elif plot_type == 'bar':
            shap.plots.bar(shap_values, max_display=max_display, show=False)
        else:
            # Fallback to old API for violin plots
            if isinstance(shap_values, list):
                shap_values_array = shap_values[1]
            else:
                shap_values_array = shap_values.values if hasattr(shap_values, 'values') else shap_values
            X = self.X_test if X is None else X
            shap.summary_plot(
                shap_values_array,
                X,
                feature_names=self.feature_names,
                plot_type=plot_type,
                max_display=max_display,
                show=False
            )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"SHAP summary plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_waterfall(
        self,
        instance_idx: int = 0,
        shap_values=None,
        X: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP waterfall for a single instance.

        Args:
            instance_idx: Index of instance to explain
            shap_values: SHAP values (uses cached if None)
            X: Data instances (uses X_test if None)
            save_path: Path to save plot
        """
        if shap_values is None:
            shap_values = self.shap_values_

        if shap_values is None:
            raise ValueError("No SHAP values available. Run explain_global() first.")

        if X is None:
            X = self.X_test

        # Modern API: shap_values is an Explanation object
        # Simply pass the instance to the waterfall plot
        shap.plots.waterfall(shap_values[instance_idx], max_display=15, show=False)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"SHAP waterfall plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def get_top_features(
        self,
        shap_values=None,
        n_features: int = 10
    ) -> pd.DataFrame:
        """
        Get top important features based on mean absolute SHAP values.

        Args:
            shap_values: SHAP values (uses cached if None)
            n_features: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if shap_values is None:
            shap_values = self.shap_values_

        if shap_values is None:
            raise ValueError("No SHAP values available. Run explain_global() first.")

        # Modern API: extract values array from Explanation object
        if hasattr(shap_values, 'values'):
            shap_values_array = shap_values.values
        elif isinstance(shap_values, list):
            shap_values_array = shap_values[1]
        else:
            shap_values_array = shap_values

        # Compute mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)

        # Sort by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:n_features]

        # Create DataFrame
        df = pd.DataFrame({
            'feature_name': [self.feature_names[i] for i in sorted_indices],
            'importance': mean_abs_shap[sorted_indices]
        })

        return df

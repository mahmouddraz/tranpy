"""LIME (Local Interpretable Model-agnostic Explanations) explainer."""

import os
from pathlib import Path
from typing import Optional, List, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from timeit import default_timer as timer

try:
    from lime import lime_tabular
    from lime import submodular_pick
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from .base import BaseExplainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


def check_lime():
    """Check if LIME is available."""
    if not LIME_AVAILABLE:
        raise ImportError(
            "LIME is required for this explainer. "
            "Install with: pip install lime"
        )


class LIMEExplainer(BaseExplainer):
    """
    LIME explainer for power system stability models.

    Provides local and global explanations using LIME.

    Args:
        model: Trained model with predict_proba method
        X_train: Training data for background
        X_test: Test data to explain
        feature_names: Names of features
        class_names: Names of target classes
        mode: Explanation mode ('classification' or 'regression')
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = 'classification'
    ):
        check_lime()
        super().__init__(model, X_train, X_test, feature_names, class_names)

        self.mode = mode
        self.explainer_ = lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.mode
        )
        self.global_explainer_ = None

    def _get_predict_fn(self) -> Callable:
        """Get the appropriate prediction function."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        elif hasattr(self.model, 'predict'):
            # Wrap predict to return probabilities
            def predict_fn(X):
                preds = self.model.predict(X)
                # Convert to probability format
                probs = np.column_stack([1 - preds, preds])
                return probs
            return predict_fn
        else:
            raise ValueError("Model must have predict_proba or predict method")

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        **kwargs
    ):
        """
        Explain a single prediction.

        Args:
            instance: Single data instance (1D array)
            num_features: Number of features to include in explanation
            **kwargs: Additional arguments for LIME

        Returns:
            LIME explanation object

        Examples:
            >>> explainer = LIMEExplainer(model, X_train, X_test)
            >>> explanation = explainer.explain_instance(X_test[0])
            >>> explanation.show_in_notebook()
        """
        instance = np.asarray(instance).flatten()
        predict_fn = self._get_predict_fn()

        start = timer()
        explanation = self.explainer_.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            **kwargs
        )
        elapsed = timer() - start

        logger.info(f"LIME explanation computed in {elapsed:.3f}s")
        return explanation

    def explain_global(
        self,
        num_features: Optional[int] = None,
        num_exps_desired: int = 5,
        use_cache: bool = True,
        cache_dir: str = 'explainer_outputs/lime_cache'
    ):
        """
        Generate global explanations using Submodular Pick.

        Args:
            num_features: Number of features to explain (default: all)
            num_exps_desired: Number of representative instances
            use_cache: Whether to use cached results
            cache_dir: Directory for caching

        Returns:
            SubmodularPick object with global explanations

        Examples:
            >>> explainer = LIMEExplainer(model, X_train, X_test)
            >>> sp_obj = explainer.explain_global()
            >>> explainer.plot_global_explanations(sp_obj)
        """
        if self.X_test is None:
            raise ValueError("X_test required for global explanations")

        if num_features is None:
            num_features = self.X_train.shape[1]

        # Check cache
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        model_name = type(self.model).__name__
        cache_file = cache_path / f"{model_name}_lime_sp.pkl"

        if use_cache and cache_file.exists():
            logger.info(f"Loading cached LIME explanations from {cache_file}")
            import pickle
            with open(cache_file, 'rb') as f:
                self.global_explainer_ = pickle.load(f)
            return self.global_explainer_

        # Generate new explanations
        logger.info("Generating LIME global explanations...")
        predict_fn = self._get_predict_fn()

        start = timer()
        self.global_explainer_ = submodular_pick.SubmodularPick(
            self.explainer_,
            self.X_train,
            predict_fn,
            num_features=num_features,
            num_exps_desired=num_exps_desired
        )
        elapsed = timer() - start
        logger.info(f"Global LIME explanations computed in {elapsed:.3f}s")

        # Cache results
        if use_cache:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(self.global_explainer_, f)
            logger.info(f"Cached explanations to {cache_file}")

        return self.global_explainer_

    def plot_global_explanations(
        self,
        sp_obj=None,
        save_path: Optional[str] = None
    ):
        """
        Plot global LIME explanations.

        Args:
            sp_obj: SubmodularPick object (uses cached if None)
            save_path: Path to save PDF (default: explainer_outputs/lime_global.pdf)
        """
        if sp_obj is None:
            sp_obj = self.global_explainer_

        if sp_obj is None:
            raise ValueError("No global explanations available. Run explain_global() first.")

        if save_path is None:
            save_path = 'explainer_outputs/lime_global.pdf'

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(save_path) as pdf:
            for exp in sp_obj.sp_explanations:
                fig = exp.as_pyplot_figure(label=exp.available_labels()[0])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        logger.info(f"Global explanations saved to {save_path}")

    def get_top_features(
        self,
        sp_obj=None,
        n_features: int = 10
    ) -> pd.DataFrame:
        """
        Get top important features from global explanations.

        Args:
            sp_obj: SubmodularPick object (uses cached if None)
            n_features: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if sp_obj is None:
            sp_obj = self.global_explainer_

        if sp_obj is None:
            raise ValueError("No global explanations available. Run explain_global() first.")

        # Aggregate feature importances across all explanations
        feature_importances = {}

        for exp in sp_obj.sp_explanations:
            for label_exp in exp.local_exp.values():
                for feature_idx, importance in label_exp:
                    if feature_idx not in feature_importances:
                        feature_importances[feature_idx] = []
                    feature_importances[feature_idx].append(abs(importance))

        # Average importances
        avg_importances = {
            idx: np.mean(values)
            for idx, values in feature_importances.items()
        }

        # Sort by importance
        sorted_features = sorted(
            avg_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_features]

        # Create DataFrame
        df = pd.DataFrame(sorted_features, columns=['feature_idx', 'importance'])
        df['feature_name'] = df['feature_idx'].apply(lambda x: self.feature_names[x])

        return df[['feature_name', 'importance']]

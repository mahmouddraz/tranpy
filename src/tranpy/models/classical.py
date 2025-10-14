"""Classical machine learning models for stability prediction."""

from typing import Optional, Literal, Dict, Any
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.ensemble import (
    AdaBoostClassifier as SklearnAdaBoost,
    RandomForestClassifier as SklearnRF,
    ExtraTreesClassifier as SklearnET,
    GradientBoostingClassifier as SklearnGBC
)
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as SklearnLDA,
    QuadraticDiscriminantAnalysis as SklearnQDA
)
from sklearn.linear_model import (
    LogisticRegression as SklearnLR,
    RidgeClassifier as SklearnRidge,
    SGDClassifier as SklearnSGD
)
from sklearn.dummy import DummyClassifier as SklearnDummy
from sklearn.gaussian_process import GaussianProcessClassifier as SklearnGPC
from timeit import default_timer as timer

from .base import StabilityClassifier

# Try to import optional models
try:
    from xgboost import XGBClassifier as SklearnXGB
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier as SklearnLGBM
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


class SVMClassifier(StabilityClassifier):
    """
    Support Vector Machine classifier for power system stability.

    Args:
        kernel: Kernel type ('rbf', 'linear', 'poly')
        C: Regularization parameter
        gamma: Kernel coefficient (for 'rbf', 'poly')
        probability: Enable probability estimates
        **kwargs: Additional arguments passed to sklearn.svm.SVC
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        probability: bool = True,
        **kwargs
    ):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.kwargs = kwargs

        self.model_ = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            **kwargs
        )
        self.training_time_ = None
        self.prediction_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Train the SVM classifier."""
        X = self._validate_input(X)
        y = np.asarray(y)

        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start

        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = self._validate_input(X)

        start = timer()
        predictions = self.model_.predict(X)
        self.prediction_time_ = timer() - start

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._validate_input(X)

        if not self.probability:
            raise ValueError(
                "Probability estimates not available. "
                "Set probability=True when creating the classifier."
            )

        return self.model_.predict_proba(X)

    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        return self.model_.support_vectors_


class MLPClassifier(StabilityClassifier):
    """
    Multi-Layer Perceptron classifier for power system stability.

    Args:
        hidden_layer_sizes: Tuple of hidden layer sizes
        activation: Activation function ('relu', 'tanh', 'logistic')
        solver: Solver for weight optimization ('adam', 'sgd', 'lbfgs')
        alpha: L2 regularization parameter
        max_iter: Maximum number of iterations
        **kwargs: Additional arguments passed to sklearn.neural_network.MLPClassifier
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100,),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.0001,
        max_iter: int = 200,
        **kwargs
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.kwargs = kwargs

        self.model_ = SklearnMLP(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            **kwargs
        )
        self.training_time_ = None
        self.prediction_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifier':
        """Train the MLP classifier."""
        X = self._validate_input(X)
        y = np.asarray(y)

        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start

        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = self._validate_input(X)

        start = timer()
        predictions = self.model_.predict(X)
        self.prediction_time_ = timer() - start

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._validate_input(X)
        return self.model_.predict_proba(X)

    @property
    def loss_curve_(self) -> np.ndarray:
        """Training loss curve."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        return self.model_.loss_curve_


class DecisionTreeClassifier(StabilityClassifier):
    """
    Decision Tree classifier for power system stability.

    Args:
        criterion: Split quality measure ('gini', 'entropy')
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples in a leaf
        **kwargs: Additional arguments passed to sklearn.tree.DecisionTreeClassifier
    """

    def __init__(
        self,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        **kwargs
    ):
        super().__init__()
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.kwargs = kwargs

        self.model_ = SklearnDT(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            **kwargs
        )
        self.training_time_ = None
        self.prediction_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """Train the decision tree classifier."""
        X = self._validate_input(X)
        y = np.asarray(y)

        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start

        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = self._validate_input(X)

        start = timer()
        predictions = self.model_.predict(X)
        self.prediction_time_ = timer() - start

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._validate_input(X)
        return self.model_.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        return self.model_.feature_importances_

    @property
    def tree_(self):
        """Access the underlying tree structure."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        return self.model_.tree_


# =============================================================================
# Ensemble Classifiers
# =============================================================================

class RandomForestClassifier(StabilityClassifier):
    """
    Random Forest classifier for power system stability.

    Random Forest is an ensemble method that constructs multiple decision trees
    and outputs the mode of their predictions for classification tasks.

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree (None for unlimited)
        **kwargs: Additional arguments passed to sklearn.ensemble.RandomForestClassifier

    Examples:
        >>> from tranpy.models import RandomForestClassifier
        >>> from tranpy.datasets import load_newengland
        >>>
        >>> X_train, X_test, y_train, y_test = load_newengland(test_size=0.2)
        >>> model = RandomForestClassifier(n_estimators=100, max_depth=10)
        >>> model.fit(X_train, y_train)
        >>> accuracy = model.score(X_test, y_test)
    """

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.model_ = SklearnRF(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """Train the Random Forest classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


class AdaBoostClassifier(StabilityClassifier):
    """
    AdaBoost ensemble classifier for power system stability.

    AdaBoost (Adaptive Boosting) combines multiple weak classifiers
    to create a strong classifier by iteratively focusing on misclassified samples.

    Args:
        n_estimators: Number of boosting iterations
        learning_rate: Learning rate shrinks contribution of each classifier
        **kwargs: Additional arguments passed to sklearn.ensemble.AdaBoostClassifier

    Examples:
        >>> from tranpy.models import AdaBoostClassifier
        >>> model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self.model_ = SklearnAdaBoost(n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostClassifier':
        """Train the AdaBoost classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


class ExtraTreesClassifier(StabilityClassifier):
    """
    Extra Trees (Extremely Randomized Trees) classifier for power system stability.

    Extra Trees builds an ensemble of unpruned decision trees with random splits,
    providing faster training than Random Forests while maintaining accuracy.

    Args:
        n_estimators: Number of trees in the ensemble
        max_depth: Maximum depth of each tree (None for unlimited)
        **kwargs: Additional arguments passed to sklearn.ensemble.ExtraTreesClassifier

    Examples:
        >>> from tranpy.models import ExtraTreesClassifier
        >>> model = ExtraTreesClassifier(n_estimators=100)
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.model_ = SklearnET(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExtraTreesClassifier':
        """Train the Extra Trees classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


class GradientBoostingClassifier(StabilityClassifier):
    """
    Gradient Boosting classifier for power system stability.

    Gradient Boosting builds an additive model in a forward stage-wise fashion,
    optimizing an arbitrary differentiable loss function.

    Args:
        n_estimators: Number of boosting stages
        learning_rate: Learning rate shrinks contribution of each tree
        max_depth: Maximum depth of individual trees
        **kwargs: Additional arguments passed to sklearn.ensemble.GradientBoostingClassifier

    Examples:
        >>> from tranpy.models import GradientBoostingClassifier
        >>> model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.model_ = SklearnGBC(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        """Train the Gradient Boosting classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


# =============================================================================
# Naive Bayes and Neighbors
# =============================================================================

class GaussianNB(StabilityClassifier):
    """
    Gaussian Naive Bayes classifier for power system stability.

    Implements the Gaussian Naive Bayes algorithm, assuming features
    follow a normal distribution within each class.

    Args:
        **kwargs: Additional arguments passed to sklearn.naive_bayes.GaussianNB

    Examples:
        >>> from tranpy.models import GaussianNB
        >>> model = GaussianNB()
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model_ = SklearnGNB(**kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        """Train the Gaussian Naive Bayes classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


class KNeighborsClassifier(StabilityClassifier):
    """
    K-Nearest Neighbors classifier for power system stability.

    Classifies samples based on the majority class among k nearest neighbors
    in the feature space using distance metrics.

    Args:
        n_neighbors: Number of neighbors to consider
        **kwargs: Additional arguments passed to sklearn.neighbors.KNeighborsClassifier

    Examples:
        >>> from tranpy.models import KNeighborsClassifier
        >>> model = KNeighborsClassifier(n_neighbors=5)
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.model_ = SklearnKNN(n_neighbors=n_neighbors, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNeighborsClassifier':
        """Train the K-Nearest Neighbors classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


class GaussianProcessClassifier(StabilityClassifier):
    """
    Gaussian Process classifier for power system stability.

    A non-parametric Bayesian approach to classification that places a
    Gaussian Process prior on the latent function.

    Args:
        **kwargs: Additional arguments passed to sklearn.gaussian_process.GaussianProcessClassifier

    Examples:
        >>> from tranpy.models import GaussianProcessClassifier
        >>> model = GaussianProcessClassifier()
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model_ = SklearnGPC(**kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessClassifier':
        """Train the Gaussian Process classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


# =============================================================================
# Linear Models
# =============================================================================

class LogisticRegression(StabilityClassifier):
    """
    Logistic Regression classifier for power system stability.

    A linear model for binary classification that estimates probabilities
    using the logistic function.

    Args:
        C: Inverse of regularization strength (smaller values = stronger regularization)
        max_iter: Maximum number of iterations for solver convergence
        **kwargs: Additional arguments passed to sklearn.linear_model.LogisticRegression

    Examples:
        >>> from tranpy.models import LogisticRegression
        >>> model = LogisticRegression(C=1.0, max_iter=100)
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, C: float = 1.0, max_iter: int = 100, **kwargs):
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.model_ = SklearnLR(C=C, max_iter=max_iter, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """Train the Logistic Regression classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


class RidgeClassifier(StabilityClassifier):
    """
    Ridge classifier for power system stability.

    Uses Ridge regression (L2 regularization) with a decision threshold
    for classification tasks.

    Args:
        alpha: Regularization strength (larger values = stronger regularization)
        **kwargs: Additional arguments passed to sklearn.linear_model.RidgeClassifier

    Examples:
        >>> from tranpy.models import RidgeClassifier
        >>> model = RidgeClassifier(alpha=1.0)
        >>> model.fit(X_train, y_train)

    Note:
        RidgeClassifier does not support probability predictions (predict_proba).
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.kwargs = kwargs
        self.model_ = SklearnRidge(alpha=alpha, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeClassifier':
        """Train the Ridge classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ridge doesn't support predict_proba natively."""
        raise NotImplementedError("RidgeClassifier does not support probability predictions")


class SGDClassifier(StabilityClassifier):
    """
    Stochastic Gradient Descent classifier for power system stability.

    Linear classifier with SGD optimization, suitable for large-scale learning
    with various loss functions.

    Args:
        loss: Loss function ('log_loss', 'hinge', 'modified_huber', etc.)
        max_iter: Maximum number of passes over training data
        **kwargs: Additional arguments passed to sklearn.linear_model.SGDClassifier

    Examples:
        >>> from tranpy.models import SGDClassifier
        >>> model = SGDClassifier(loss='log_loss', max_iter=1000)
        >>> model.fit(X_train, y_train)

    Note:
        Probability predictions only available with loss='log_loss' or 'modified_huber'.
    """

    def __init__(self, loss: str = 'log_loss', max_iter: int = 1000, **kwargs):
        super().__init__()
        self.loss = loss
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.model_ = SklearnSGD(loss=loss, max_iter=max_iter, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SGDClassifier':
        """Train the SGD classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities if loss supports it."""
        if self.loss in ['log_loss', 'modified_huber']:
            return self.model_.predict_proba(self._validate_input(X))
        raise NotImplementedError(f"SGDClassifier with loss='{self.loss}' does not support probability predictions")


# =============================================================================
# Discriminant Analysis
# =============================================================================

class LinearDiscriminantAnalysis(StabilityClassifier):
    """
    Linear Discriminant Analysis (LDA) for power system stability.

    Finds a linear combination of features that characterizes or separates
    two or more classes, assuming Gaussian distributions with equal covariance.

    Args:
        **kwargs: Additional arguments passed to sklearn.discriminant_analysis.LinearDiscriminantAnalysis

    Examples:
        >>> from tranpy.models import LinearDiscriminantAnalysis
        >>> model = LinearDiscriminantAnalysis()
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model_ = SklearnLDA(**kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearDiscriminantAnalysis':
        """Train the LDA classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


class QuadraticDiscriminantAnalysis(StabilityClassifier):
    """
    Quadratic Discriminant Analysis (QDA) for power system stability.

    Similar to LDA but assumes each class has its own covariance matrix,
    resulting in quadratic decision boundaries.

    Args:
        **kwargs: Additional arguments passed to sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis

    Examples:
        >>> from tranpy.models import QuadraticDiscriminantAnalysis
        >>> model = QuadraticDiscriminantAnalysis()
        >>> model.fit(X_train, y_train)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model_ = SklearnQDA(**kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuadraticDiscriminantAnalysis':
        """Train the QDA classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


# =============================================================================
# Baseline and Boosting Libraries
# =============================================================================

class DummyClassifier(StabilityClassifier):
    """
    Dummy classifier (baseline) for power system stability.

    Provides simple baseline predictions using strategies like predicting
    the most frequent class. Useful for establishing baseline performance.

    Args:
        strategy: Strategy for prediction ('most_frequent', 'stratified', 'uniform', etc.)
        **kwargs: Additional arguments passed to sklearn.dummy.DummyClassifier

    Examples:
        >>> from tranpy.models import DummyClassifier
        >>> baseline = DummyClassifier(strategy='most_frequent')
        >>> baseline.fit(X_train, y_train)
        >>> baseline_accuracy = baseline.score(X_test, y_test)
    """

    def __init__(self, strategy: str = 'most_frequent', **kwargs):
        super().__init__()
        self.strategy = strategy
        self.kwargs = kwargs
        self.model_ = SklearnDummy(strategy=strategy, **kwargs)
        self.training_time_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DummyClassifier':
        """Train the Dummy classifier."""
        X, y = self._validate_input(X), np.asarray(y)
        start = timer()
        self.model_.fit(X, y)
        self.training_time_ = timer() - start
        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model_.predict(self._validate_input(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model_.predict_proba(self._validate_input(X))


# XGBoost and LightGBM (optional dependencies)
if _XGB_AVAILABLE:
    class XGBClassifier(StabilityClassifier):
        """
        XGBoost (Extreme Gradient Boosting) classifier for power system stability.

        High-performance implementation of gradient boosting with advanced
        regularization and parallel processing capabilities.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage to prevent overfitting
            **kwargs: Additional arguments passed to xgboost.XGBClassifier

        Examples:
            >>> from tranpy.models import XGBClassifier
            >>> model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3)
            >>> model.fit(X_train, y_train)

        Note:
            Requires xgboost to be installed: pip install xgboost
        """

        def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3, **kwargs):
            super().__init__()
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.kwargs = kwargs
            self.model_ = SklearnXGB(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, **kwargs)
            self.training_time_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBClassifier':
            """Train the XGBoost classifier."""
            X, y = self._validate_input(X), np.asarray(y)
            start = timer()
            self.model_.fit(X, y)
            self.training_time_ = timer() - start
            self._set_fitted_attributes(X)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            return self.model_.predict(self._validate_input(X))

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities."""
            return self.model_.predict_proba(self._validate_input(X))


if _LGBM_AVAILABLE:
    class LGBMClassifier(StabilityClassifier):
        """
        LightGBM (Light Gradient Boosting Machine) classifier for power system stability.

        Fast, distributed gradient boosting framework that uses tree-based learning
        with histogram-based algorithms for improved efficiency.

        Args:
            n_estimators: Number of boosting iterations
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Boosting learning rate
            **kwargs: Additional arguments passed to lightgbm.LGBMClassifier

        Examples:
            >>> from tranpy.models import LGBMClassifier
            >>> model = LGBMClassifier(n_estimators=100, learning_rate=0.1)
            >>> model.fit(X_train, y_train)

        Note:
            Requires lightgbm to be installed: pip install lightgbm
        """

        def __init__(self, n_estimators: int = 100, max_depth: int = -1, learning_rate: float = 0.1, **kwargs):
            super().__init__()
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.kwargs = kwargs
            self.model_ = SklearnLGBM(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, verbose=-1, **kwargs)
            self.training_time_ = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> 'LGBMClassifier':
            """Train the LightGBM classifier."""
            X, y = self._validate_input(X), np.asarray(y)
            start = timer()
            self.model_.fit(X, y)
            self.training_time_ = timer() - start
            self._set_fitted_attributes(X)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            return self.model_.predict(self._validate_input(X))

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities."""
            return self.model_.predict_proba(self._validate_input(X))

"""Deep learning models for stability prediction."""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from timeit import default_timer as timer

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .base import StabilityClassifier


def check_tensorflow():
    """Check if TensorFlow is available."""
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for neural network models. "
            "Install with: pip install tensorflow"
        )


class DNNClassifier(StabilityClassifier):
    """
    Deep Neural Network classifier for power system stability.

    Args:
        hidden_layers: List of hidden layer sizes
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        optimizer: Optimizer name or instance
        loss: Loss function
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data for validation
        **kwargs: Additional arguments passed to model.fit()
    """

    def __init__(
        self,
        hidden_layers: List[int] = [10],
        activation: str = 'relu',
        output_activation: str = 'sigmoid',
        optimizer: str = 'adam',
        loss: str = 'binary_crossentropy',
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        **kwargs
    ):
        check_tensorflow()
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.kwargs = kwargs

        self.model_ = None
        self.history_ = None
        self.training_time_ = None
        self.prediction_time_ = None

    def _build_model(self, n_features: int):
        """Build the neural network architecture."""
        model = keras.Sequential()

        # Input layer
        model.add(keras.layers.InputLayer(input_shape=(n_features,)))

        # Hidden layers
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(units, activation=self.activation))

        # Output layer
        model.add(keras.layers.Dense(1, activation=self.output_activation))

        # Compile
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )

        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DNNClassifier':
        """Train the DNN classifier."""
        X = self._validate_input(X)
        y = np.asarray(y)

        # Build model
        self.model_ = self._build_model(X.shape[1])

        # Train
        start = timer()
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=0,
            **self.kwargs
        )
        self.training_time_ = timer() - start

        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = self._validate_input(X)

        start = timer()
        probabilities = self.model_.predict(X, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()
        self.prediction_time_ = timer() - start

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._validate_input(X)

        probabilities = self.model_.predict(X, verbose=0)
        # Return [P(class=0), P(class=1)]
        return np.column_stack([1 - probabilities, probabilities])

    def get_training_history(self) -> dict:
        """Get training history (loss, accuracy per epoch)."""
        if self.history_ is None:
            raise ValueError("Model not trained yet.")
        return self.history_.history


class RNNClassifier(StabilityClassifier):
    """
    Recurrent Neural Network (LSTM) classifier for power system stability.

    This model treats the feature vector as a time series and uses LSTM layers.

    Args:
        lstm_units: List of LSTM layer sizes
        lstm_activation: Activation function for LSTM layers
        dropout: Dropout rate
        output_activation: Activation function for output layer
        optimizer: Optimizer name or instance
        loss: Loss function
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data for validation
        n_steps: Number of time steps (for reshaping input)
        **kwargs: Additional arguments passed to model.fit()
    """

    def __init__(
        self,
        lstm_units: List[int] = [150, 150, 150],
        lstm_activation: str = 'tanh',
        dropout: float = 0.0,
        output_activation: str = 'sigmoid',
        optimizer: str = 'adam',
        loss: str = 'binary_crossentropy',
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        n_steps: int = 1,
        **kwargs
    ):
        check_tensorflow()
        super().__init__()

        self.lstm_units = lstm_units
        self.lstm_activation = lstm_activation
        self.dropout = dropout
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.n_steps = n_steps
        self.kwargs = kwargs

        self.model_ = None
        self.history_ = None
        self.training_time_ = None
        self.prediction_time_ = None

    def _build_model(self, n_features: int):
        """Build the RNN architecture."""
        model = keras.Sequential()

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)

            if i == 0:
                model.add(keras.layers.LSTM(
                    units,
                    activation=self.lstm_activation,
                    return_sequences=return_sequences,
                    input_shape=(self.n_steps, n_features)
                ))
            else:
                model.add(keras.layers.LSTM(
                    units,
                    activation=self.lstm_activation,
                    return_sequences=return_sequences
                ))

            if self.dropout > 0:
                model.add(keras.layers.Dropout(self.dropout))

        # Output layer
        model.add(keras.layers.Dense(1, activation=self.output_activation))

        # Compile
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )

        return model

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape input for RNN (add time dimension)."""
        n_samples, n_features = X.shape
        return X.reshape(n_samples, self.n_steps, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RNNClassifier':
        """Train the RNN classifier."""
        X = self._validate_input(X)
        y = np.asarray(y)

        # Reshape for RNN
        n_features_per_step = X.shape[1]
        X_reshaped = self._reshape_input(X)

        # Build model
        self.model_ = self._build_model(n_features_per_step)

        # Train
        start = timer()
        self.history_ = self.model_.fit(
            X_reshaped, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=0,
            **self.kwargs
        )
        self.training_time_ = timer() - start

        self._set_fitted_attributes(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = self._validate_input(X)
        X_reshaped = self._reshape_input(X)

        start = timer()
        probabilities = self.model_.predict(X_reshaped, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()
        self.prediction_time_ = timer() - start

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._validate_input(X)
        X_reshaped = self._reshape_input(X)

        probabilities = self.model_.predict(X_reshaped, verbose=0)
        # Return [P(class=0), P(class=1)]
        return np.column_stack([1 - probabilities, probabilities])

    def get_training_history(self) -> dict:
        """Get training history (loss, accuracy per epoch)."""
        if self.history_ is None:
            raise ValueError("Model not trained yet.")
        return self.history_.history

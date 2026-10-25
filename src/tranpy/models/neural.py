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

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'optimizer': self.optimizer,
            'loss': self.loss,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
        }


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


class LSTMClassifier(StabilityClassifier):
    """LSTM classifier for temporal stability prediction.

    Takes raw time series trajectories as input (batch, timesteps, features)
    and uses stacked LSTM layers with optional attention for classification.

    Args:
        lstm_units: List of LSTM layer sizes
        dropout: Dropout rate between LSTM layers
        use_attention: If True, apply attention after the last LSTM layer
        dense_units: Hidden dense layer size before output
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Fraction of data for validation
    """

    def __init__(
        self,
        lstm_units: List[int] = [64, 32],
        dropout: float = 0.2,
        use_attention: bool = True,
        dense_units: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        check_tensorflow()
        super().__init__()
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.use_attention = use_attention
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model_ = None
        self.history_ = None
        self.training_time_ = None

    def _build_model(self, input_shape: Tuple[int, int]):
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1) or self.use_attention
            x = keras.layers.LSTM(units, return_sequences=return_sequences)(x)
            if self.dropout > 0:
                x = keras.layers.Dropout(self.dropout)(x)

        if self.use_attention:
            attention = keras.layers.Dense(1, activation='tanh')(x)
            attention = keras.layers.Flatten()(attention)
            attention = keras.layers.Activation('softmax')(attention)
            attention = keras.layers.RepeatVector(self.lstm_units[-1])(attention)
            attention = keras.layers.Permute([2, 1])(attention)
            x = keras.layers.Multiply()([x, attention])
            x = keras.layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

        x = keras.layers.Dense(self.dense_units, activation='relu')(x)
        x = keras.layers.Dropout(self.dropout)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs)
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMClassifier':
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        self.model_ = self._build_model(X.shape[1:])
        start = timer()
        self.history_ = self.model_.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.validation_split, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        self.training_time_ = timer() - start
        self.n_features_in_ = X.shape[-1]
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        proba = self.model_.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        proba = self.model_.predict(X, verbose=0)
        return np.column_stack([1 - proba, proba])


class TCNClassifier(StabilityClassifier):
    """Temporal Convolutional Network classifier for stability prediction.

    Uses dilated causal 1D convolutions to capture temporal patterns
    at multiple scales without recurrence.

    Args:
        n_filters: Number of filters per convolutional layer
        kernel_size: Kernel size for 1D convolutions
        dilation_rates: List of dilation rates
        dropout: Dropout rate
        dense_units: Hidden dense layer size before output
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Fraction of data for validation
    """

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 3,
        dilation_rates: List[int] = [1, 2, 4, 8, 16],
        dropout: float = 0.2,
        dense_units: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        check_tensorflow()
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model_ = None
        self.history_ = None
        self.training_time_ = None

    def _build_model(self, input_shape: Tuple[int, int]):
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        for dilation in self.dilation_rates:
            conv = keras.layers.Conv1D(
                self.n_filters, self.kernel_size,
                padding='causal', dilation_rate=dilation, activation='relu',
            )(x)
            conv = keras.layers.Dropout(self.dropout)(conv)
            conv = keras.layers.Conv1D(
                self.n_filters, self.kernel_size,
                padding='causal', dilation_rate=dilation, activation='relu',
            )(conv)
            conv = keras.layers.Dropout(self.dropout)(conv)

            if x.shape[-1] != self.n_filters:
                x = keras.layers.Conv1D(self.n_filters, 1)(x)
            x = keras.layers.Add()([x, conv])

        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(self.dense_units, activation='relu')(x)
        x = keras.layers.Dropout(self.dropout)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs)
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TCNClassifier':
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        self.model_ = self._build_model(X.shape[1:])
        start = timer()
        self.history_ = self.model_.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.validation_split, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        self.training_time_ = timer() - start
        self.n_features_in_ = X.shape[-1]
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        proba = self.model_.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        proba = self.model_.predict(X, verbose=0)
        return np.column_stack([1 - proba, proba])


class TransformerClassifier(StabilityClassifier):
    """Transformer-based classifier for stability prediction.

    Uses multi-head self-attention with positional encoding to capture
    temporal dependencies in trajectory data.

    Args:
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        n_layers: Number of transformer encoder layers
        ff_dim: Feed-forward network dimension
        dropout: Dropout rate
        dense_units: Hidden dense layer size before output
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Fraction of data for validation
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        dense_units: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        check_tensorflow()
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model_ = None
        self.history_ = None
        self.training_time_ = None

    def _build_model(self, input_shape: Tuple[int, int]):
        inputs = keras.layers.Input(shape=input_shape)

        # Project to d_model dimension
        x = keras.layers.Dense(self.d_model)(inputs)

        # Learned positional encoding
        seq_len = input_shape[0]
        positions = keras.layers.Embedding(seq_len, self.d_model)(
            tf.range(seq_len)
        )
        x = x + positions

        # Transformer encoder blocks
        for _ in range(self.n_layers):
            attn_output = keras.layers.MultiHeadAttention(
                num_heads=self.n_heads, key_dim=self.d_model // self.n_heads,
            )(x, x)
            attn_output = keras.layers.Dropout(self.dropout)(attn_output)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

            ff_output = keras.layers.Dense(self.ff_dim, activation='relu')(x)
            ff_output = keras.layers.Dense(self.d_model)(ff_output)
            ff_output = keras.layers.Dropout(self.dropout)(ff_output)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(self.dense_units, activation='relu')(x)
        x = keras.layers.Dropout(self.dropout)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs, outputs)
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TransformerClassifier':
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        self.model_ = self._build_model(X.shape[1:])
        start = timer()
        self.history_ = self.model_.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.validation_split, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
        )
        self.training_time_ = timer() - start
        self.n_features_in_ = X.shape[-1]
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        proba = self.model_.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        proba = self.model_.predict(X, verbose=0)
        return np.column_stack([1 - proba, proba])

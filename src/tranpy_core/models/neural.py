"""Keras networks as scikit-learn estimators (optional ``[neural]`` extra).

All hyperparameters are constructor arguments stored unchanged, the network is built in
``fit``, and ``random_state`` seeds TensorFlow/Keras for every fit. These classes are
therefore clonable and usable in pipelines, cross-validation and grid search.

Sequence architectures (LSTM, TCN, Transformer) take 2-D input, as everything in a
scikit-learn pipeline does, and reshape each row into ``n_steps`` time steps of
``n_features / n_steps`` values; the column order must therefore be time-major
(all features of step 0, then step 1, ...). 3-D input is accepted directly.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from tranpy_core._optional import require
from tranpy_core.errors import ConfigurationError, ModelError

__all__ = [
    "DenseNetworkClassifier",
    "LSTMNetworkClassifier",
    "TCNNetworkClassifier",
    "TransformerNetworkClassifier",
]


class _KerasClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Shared fit/predict logic for binary Keras classifiers with a sigmoid output."""

    epochs: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    early_stopping_patience: int | None
    shuffle: bool
    deterministic: bool
    random_state: int
    n_steps: int

    def _build(self, keras: Any, input_shape: tuple[int, ...]) -> Any:
        raise NotImplementedError

    def _as_input(self, X: npt.ArrayLike, *, fitting: bool) -> npt.NDArray[np.float32]:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 3:
            return arr
        if arr.ndim != 2:
            raise ModelError(f"expected 2-D or 3-D input, got shape {arr.shape}")
        if fitting:
            self.n_features_in_ = arr.shape[1]
        elif arr.shape[1] != self.n_features_in_:
            raise ModelError(f"expected {self.n_features_in_} features, got {arr.shape[1]}")
        if self.n_steps == 0:
            return arr
        if arr.shape[1] % self.n_steps:
            raise ConfigurationError(
                f"{arr.shape[1]} features cannot be split into "
                f"n_steps={self.n_steps} equal time steps"
            )
        return arr.reshape(arr.shape[0], self.n_steps, arr.shape[1] // self.n_steps)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> _KerasClassifier:
        tf = require("tensorflow", "neural")
        keras = tf.keras
        y_arr = np.asarray(y)
        self.classes_ = unique_labels(y_arr)
        if len(self.classes_) != 2:
            raise ModelError(f"binary classification only, got classes {self.classes_.tolist()}")
        target = (y_arr == self.classes_[1]).astype(np.float32)
        inputs = self._as_input(X, fitting=True)
        # keras.utils.set_random_seed reseeds the global NumPy/Python RNGs; restore them afterwards
        np_state, py_state = np.random.get_state(), random.getstate()  # noqa: NPY002
        try:
            keras.utils.set_random_seed(int(self.random_state))
            if self.deterministic:
                tf.config.experimental.enable_op_determinism()
            self.model_ = self._build(keras, tuple(inputs.shape[1:]))
            self.model_.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            callbacks = []
            if self.early_stopping_patience is not None:
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        patience=self.early_stopping_patience, restore_best_weights=True
                    )
                )
            history = self.model_.fit(
                inputs,
                target,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_fraction,
                shuffle=self.shuffle,
                callbacks=callbacks,
                verbose=0,
            )
        finally:
            np.random.set_state(np_state)  # noqa: NPY002
            random.setstate(py_state)
        self.history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        check_is_fitted(self, "model_")
        p = np.asarray(
            self.model_.predict(self._as_input(X, fitting=False), verbose=0), dtype=np.float64
        )
        p = p.reshape(-1)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        proba = self.predict_proba(X)[:, 1]
        return np.asarray(self.classes_[(proba > 0.5).astype(int)])

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.non_deterministic = not self.deterministic
        return tags


class DenseNetworkClassifier(_KerasClassifier):
    """Fully connected network with a sigmoid output."""

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (10,),
        activation: str = "relu",
        dropout: float = 0.0,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        early_stopping_patience: int | None = None,
        shuffle: bool = True,
        deterministic: bool = True,
        random_state: int = 0,
    ) -> None:
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.random_state = random_state

    n_steps = 0

    def _build(self, keras: Any, input_shape: tuple[int, ...]) -> Any:
        model = keras.Sequential([keras.layers.Input(shape=input_shape)])
        for units in self.hidden_units:
            model.add(keras.layers.Dense(units, activation=self.activation))
            if self.dropout:
                model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        return model


class LSTMNetworkClassifier(_KerasClassifier):
    """Stacked LSTM with optional additive attention pooling."""

    def __init__(
        self,
        n_steps: int = 1,
        lstm_units: tuple[int, ...] = (64, 32),
        use_attention: bool = True,
        dense_units: int = 32,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        early_stopping_patience: int | None = 10,
        shuffle: bool = True,
        deterministic: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.lstm_units = lstm_units
        self.use_attention = use_attention
        self.dense_units = dense_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.random_state = random_state

    def _build(self, keras: Any, input_shape: tuple[int, ...]) -> Any:
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs
        for i, units in enumerate(self.lstm_units):
            last = i == len(self.lstm_units) - 1
            x = keras.layers.LSTM(units, return_sequences=(not last) or self.use_attention)(x)
            if self.dropout:
                x = keras.layers.Dropout(self.dropout)(x)
        if self.use_attention:
            scores = keras.layers.Dense(1, activation="tanh")(x)
            weights = keras.layers.Softmax(axis=1)(scores)
            x = keras.layers.Dot(axes=1)([weights, x])
            x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(self.dense_units, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        return keras.Model(inputs, keras.layers.Dense(1, activation="sigmoid")(x))


class TCNNetworkClassifier(_KerasClassifier):
    """Temporal convolutional network: dilated causal residual blocks."""

    def __init__(
        self,
        n_steps: int = 1,
        n_filters: int = 64,
        kernel_size: int = 3,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8, 16),
        dense_units: int = 32,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        early_stopping_patience: int | None = 10,
        shuffle: bool = True,
        deterministic: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.dense_units = dense_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.random_state = random_state

    def _build(self, keras: Any, input_shape: tuple[int, ...]) -> Any:
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs
        for rate in self.dilation_rates:
            conv = x
            for _ in range(2):
                conv = keras.layers.Conv1D(
                    self.n_filters,
                    self.kernel_size,
                    dilation_rate=rate,
                    padding="causal",
                    activation="relu",
                )(conv)
                conv = keras.layers.Dropout(self.dropout)(conv)
            if x.shape[-1] != self.n_filters:
                x = keras.layers.Conv1D(self.n_filters, 1)(x)
            x = keras.layers.Add()([x, conv])
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(self.dense_units, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        return keras.Model(inputs, keras.layers.Dense(1, activation="sigmoid")(x))


class TransformerNetworkClassifier(_KerasClassifier):
    """Transformer encoder with learned positional embeddings and average pooling."""

    def __init__(
        self,
        n_steps: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dense_units: int = 32,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.2,
        early_stopping_patience: int | None = 10,
        shuffle: bool = True,
        deterministic: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.dense_units = dense_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.shuffle = shuffle
        self.deterministic = deterministic
        self.random_state = random_state

    def _build(self, keras: Any, input_shape: tuple[int, ...]) -> Any:
        if self.d_model % self.n_heads:
            raise ConfigurationError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        seq_len = int(input_shape[0])
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Dense(self.d_model)(inputs)
        x = _PositionalEmbedding(keras)(seq_len, self.d_model)(x)
        for _ in range(self.n_layers):
            attn = keras.layers.MultiHeadAttention(
                num_heads=self.n_heads, key_dim=self.d_model // self.n_heads
            )(x, x)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(
                x + keras.layers.Dropout(self.dropout)(attn)
            )
            ff = keras.layers.Dense(self.ff_dim, activation="relu")(x)
            ff = keras.layers.Dropout(self.dropout)(keras.layers.Dense(self.d_model)(ff))
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(self.dense_units, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        return keras.Model(inputs, keras.layers.Dense(1, activation="sigmoid")(x))


def _PositionalEmbedding(keras: Any) -> Any:
    class PositionalEmbedding(keras.layers.Layer):  # type: ignore[misc]
        def __init__(self, seq_len: int, d_model: int) -> None:
            super().__init__()
            self.embedding = keras.layers.Embedding(seq_len, d_model)
            self.seq_len = seq_len

        def call(self, x: Any) -> Any:
            return x + self.embedding(keras.ops.arange(self.seq_len))

    return PositionalEmbedding

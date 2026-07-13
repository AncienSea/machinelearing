from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="assignment")
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=d_model
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        encoded_positions = self.position_embedding(positions)
        return inputs + encoded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {"sequence_length": self.sequence_length, "d_model": self.d_model}
        )
        return config


def transformer_block(x, d_model: int = 32, num_heads: int = 2, dropout: float = 0.1):
    attention = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention)
    feed_forward = keras.Sequential(
        [
            layers.Dense(d_model * 2, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ]
    )(x)
    return layers.LayerNormalization(epsilon=1e-6)(x + feed_forward)


def build_lstm(input_shape: tuple[int, int], horizon: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(32, dropout=0.1, recurrent_dropout=0.0)(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(horizon, name="forecast")(x)
    model = keras.Model(inputs, outputs, name=f"lstm_h{horizon}")
    return model


def build_transformer(input_shape: tuple[int, int], horizon: int) -> keras.Model:
    sequence_length, _ = input_shape
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(32)(inputs)
    x = PositionalEmbedding(sequence_length, 32)(x)
    x = transformer_block(x, d_model=32, num_heads=2, dropout=0.1)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(horizon, name="forecast")(x)
    model = keras.Model(inputs, outputs, name=f"transformer_h{horizon}")
    return model


def build_cnn_transformer(input_shape: tuple[int, int], horizon: int) -> keras.Model:
    sequence_length, _ = input_shape
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=5, padding="causal", activation="relu")(inputs)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = PositionalEmbedding(sequence_length, 32)(x)
    x = transformer_block(x, d_model=32, num_heads=2, dropout=0.1)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    gate = layers.Dense(32, activation="sigmoid")(avg_pool)
    gated_avg = layers.Multiply()([avg_pool, gate])
    x = layers.Concatenate()([gated_avg, max_pool])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(horizon, name="forecast")(x)
    model = keras.Model(inputs, outputs, name=f"cnn_transformer_h{horizon}")
    return model


MODEL_BUILDERS = {
    "LSTM": build_lstm,
    "Transformer": build_transformer,
    "CNN-Transformer": build_cnn_transformer,
}

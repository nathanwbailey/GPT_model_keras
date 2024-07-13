from typing import Callable

import tensorflow as tf
from tensorflow import keras


def casual_attention_mask(
    batch_size: int,
    key_length: int,
    query_length: int,
    dtype: tf.dtypes.DType,
) -> tf.Tensor:
    """
    Create a Casual Mask for
    the multi head attention layer.
    """
    i = tf.range(query_length)[:, None]
    j = tf.range(key_length)
    # Create a mask of size (query_length, key_length)
    # (i, j) is true if i >= j - key_length + query_length
    mask = i >= j - key_length + query_length
    # Cast the mask to the dtype
    mask = tf.cast(mask, dtype)
    # Reshape the mask to add a dimension
    mask = tf.reshape(mask, [1, query_length, key_length])
    # Gives a tensor = [batch_size, 1, 1]
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
        0,
    )
    # Tile the mask to produce the size (batch_size, n_dest, n_src)
    # Replicates mask mult times
    # Repeats the mask batch_size number of times across the first axis
    # E.g [mask, mask, mask ...]
    return tf.tile(mask, mult)


class TransformerBlock(keras.layers.Layer):
    """Transformer Block Layer."""

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        embed_dim: int,
        ff_dim: int,
        mask_function: Callable[
            [int, int, int, tf.dtypes.DType], tf.Tensor
        ] = casual_attention_mask,
        dropout_rate: float = 0.1,
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        # Output shape must be equal to the embedding dim
        # to allow skip connection
        self.attn = keras.layers.MultiHeadAttention(
            num_heads, key_dim, output_shape=embed_dim
        )
        self.dropout_1 = keras.layers.Dropout(self.dropout_rate)
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = keras.layers.Dense(self.ff_dim, activation="relu")
        self.ffn_2 = keras.layers.Dense(self.embed_dim)
        self.dropout_2 = keras.layers.Dropout(self.dropout_rate)
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.mask_function = mask_function

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        mask = self.mask_function(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.attn(
            query=inputs, value=inputs, key=inputs, attention_mask=mask
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.layer_norm_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        output = self.layer_norm_2(out1 + ffn_output)
        return output

    def get_config(self) -> dict:
        """Update config for saving model."""
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TokenAndPositionEmbedding(keras.layers.Layer):
    """Token and positioning embedding layer for a sequence."""

    def __init__(
        self, max_len_input: int, vocab_size: int, embed_dim: int
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.max_len = max_len_input
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(
            input_dim=max_len_input, output_dim=embed_dim
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward Pass."""
        len_input = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=len_input, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self) -> dict:
        """Update config for saving model."""
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len_input,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

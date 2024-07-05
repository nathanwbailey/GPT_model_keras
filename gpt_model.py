import tensorflow as tf
from tensorflow import keras

from model_building_blocks import (TokenAndPositionEmbedding, TransformerBlock,
                                   casual_attention_mask)


def create_gpt_model(
    max_len_input: int,
    vocab_size: int,
    embed_dim: int,
    feed_forward_dim: int,
    num_heads: int,
    key_dim: int,
) -> keras.models.Model:
    inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(max_len_input, vocab_size, embed_dim)(
        inputs
    )
    x = TransformerBlock(
        num_heads,
        key_dim,
        embed_dim,
        feed_forward_dim,
        mask_function=casual_attention_mask,
    )(x)
    outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)

    gpt_model = keras.models.Model(inputs=inputs, outputs=outputs)
    return gpt_model

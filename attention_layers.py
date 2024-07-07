import tensorflow as tf
from tensorflow import keras

from model_building_blocks import casual_attention_mask

# Adapted from:
# https://machinelearningmastery.com/how-to-implement-scaled-dot-product-attention-from-scratch-in-tensorflow-and-keras/
# https://machinelearningmastery.com/how-to-implement-multi-head-attention-from-scratch-in-tensorflow-and-keras/


class DotProductAttention(keras.layers.Layer):
    """
    Custom Dot Product Layer.
    Takes the Dot Product between queries and keys.
    """

    def call(
        self,
        queries: tf.Tensor,
        keys: tf.Tensor,
        values: tf.Tensor,
        d_k: int,
        mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Forward Pass."""
        scores = tf.matmul(queries, keys, transpose_b=True) / tf.sqrt(
            tf.cast(d_k, dtype=tf.float32)
        )
        if mask is not None:
            scores += -1e9 * tf.cast(
                tf.where(tf.cast(mask, dtype=tf.uint8) == 0, 1, 0), tf.float32
            )
        attention_weights = keras.backend.softmax(scores)
        attention_output = tf.matmul(attention_weights, values)
        return attention_output


class MultiHeadAttention(keras.layers.Layer):
    """Multi Head Attention Layer Class."""

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        output_dim: int | None = None,
        value_dim: int | None = None,
    ) -> None:
        """Init Variables and Layers."""
        super().__init__()
        self.attention = DotProductAttention()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim

        if self.value_dim is None:
            self.value_dim = key_dim

        if self.output_dim is None:
            self.output_dim = key_dim
        """
        We can use single dense layers here.
        Essentially, we take our queries, keys, values pass them through a layer.
        Then split up the output into num_heads sections of size output_shape[-1]/num_heads.
        So that we end up with a single weight matrix operating on the input.
        And then split up, which is like having num_heads matrices of weights of size output_shape[-1]/num_heads operating on the input.
        """
        self.w_q = keras.layers.Dense(self.key_dim)
        self.w_k = keras.layers.Dense(self.key_dim)
        self.w_v = keras.layers.Dense(self.value_dim)
        self.w_o = keras.layers.Dense(self.output_dim)

    def reshape_tensor(self, x: tf.Tensor, flag: bool = True) -> tf.Tensor:
        """Reshape inputs to allow them to be processed by multiple heads."""
        if flag:
            input_shape = tf.shape(x)
            x = tf.reshape(
                x, shape=(input_shape[0], input_shape[1], self.num_heads, -1)
            )
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            return x

        x = tf.transpose(x, perm=(0, 2, 1, 3))
        input_shape = tf.shape(x)
        x = tf.reshape(x, shape=(input_shape[0], input_shape[1], -1))
        return x

    def call(
        self,
        query: tf.Tensor,
        value: tf.Tensor,
        key: tf.Tensor,
        mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Forward Pass."""
        print(query.get_shape())
        query = self.reshape_tensor(self.w_q(query))
        key = self.reshape_tensor(self.w_k(key))
        value = self.reshape_tensor(self.w_v(value))
        print(query.get_shape())
        output = self.attention(query, key, value, self.key_dim, mask=mask)
        print(output.get_shape())
        output = self.reshape_tensor(output, flag=False)
        output = self.w_o(output)
        return output


# d_k = 64
# d_v = 64
# batch_size = 1
# input_seq_length = 10
# output_dim = 512
# num_heads = 8

# queries = tf.random.uniform(shape=(batch_size, input_seq_length, d_k))
# keys = tf.random.uniform(shape=(batch_size, input_seq_length, d_k))
# values = tf.random.uniform(shape=(batch_size, input_seq_length, d_v))
# mask = casual_attention_mask(
#     batch_size, input_seq_length, input_seq_length, dtype=tf.bool
# )

# multihead_attention = MultiHeadAttention(num_heads, d_k, output_dim, d_v)
# output = multihead_attention(queries, values, keys, mask)
# print(output)

# attention = DotProductAttention()
# output = attention(queries, keys, values, d_k, mask=mask)

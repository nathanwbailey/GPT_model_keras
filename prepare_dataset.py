import re
import string

import tensorflow as tf
from tensorflow import keras


def prepare_dataset(
    json_data: dict, vocab_size: int, maxlen: int, batch_size: int
) -> tuple[tf.data.Dataset, list]:
    """Prepare the wine dataset for the model."""

    vectorize_layer = keras.layers.TextVectorization(
        standardize="lower",
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=maxlen + 1,
    )

    def pad_punctuation(text: str) -> str:
        """Prepare a string."""
        # Replace punctuation characters and new lines
        # with itself surronded by spaces
        text = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", text)
        # Replace a sequence of spaces with a space
        text = re.sub(" +", " ", text)
        return text

    def prepare_inputs(text: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        text = tf.expand_dims(text, -1)
        tokenized_sequences = vectorize_layer(text)
        return tokenized_sequences[:, :-1], tokenized_sequences[:, 1:]

    filtered_data = [
        f"wine review : {country} : {province} : {variety} : {description}"
        for x in json_data
        if all(
            (
                country := x.get("country"),
                province := x.get("province"),
                variety := x.get("variety"),
                description := x.get("description"),
            )
        )
    ]

    text_dataset = [pad_punctuation(x) for x in filtered_data]
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(text_dataset)
        .batch(batch_size)
        .shuffle(1000)
    )
    vectorize_layer.adapt(train_dataset)
    vocab = vectorize_layer.get_vocabulary()
    train_dataset = train_dataset.map(prepare_inputs)
    return train_dataset, vocab

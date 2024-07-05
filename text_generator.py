import numpy as np
from tensorflow import keras


class TextGenerator(keras.callbacks.Callback):
    def __init__(self, index_to_word: list[str]) -> None:
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs: np.ndarray, temp: float) -> int:
        probs = probs ** (1 / temp)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs)

    def generate(
        self, start_prompt: str, max_tokens: int, temp: float
    ) -> None:
        # If not found return the unknown token (1)
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][-1], temp)
            start_tokens.append(sample_token)
            start_prompt += f" {self.index_to_word[sample_token]}"
        print(f"\ngenerated text:\n{start_prompt}\n")

    def on_epoch_end(self, epoch: int, logs: None = None):
        self.generate(start_prompt="wine review", max_tokens=80, temp=1.0)

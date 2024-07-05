import json

from tensorflow import keras

from gpt_model import create_gpt_model
from prepare_dataset import prepare_dataset
from text_generator import TextGenerator

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = False
BATCH_SIZE = 32
EPOCHS = 20


with open(
    "data/wine-reviews/winemag-data-130k-v2.json", encoding="utf-8"
) as json_data:
    wine_data = json.load(json_data)

train_dataset, vocab = prepare_dataset(
    wine_data, VOCAB_SIZE, MAX_LEN, BATCH_SIZE
)

gpt_model = create_gpt_model(
    MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM, FEED_FORWARD_DIM, N_HEADS, KEY_DIM
)

gpt_model.compile(
    "adam", loss=[keras.losses.SparseCategoricalCrossentropy(), None]
)
gpt_model.summary()

text_generator = TextGenerator(vocab)
gpt_model.fit(
    train_dataset, epochs=EPOCHS, callbacks=[text_generator], verbose=2
)

text_generator.generate("wine review : us", max_tokens=80, temp=1.0)
text_generator.generate("wine review : italy", max_tokens=80, temp=0.5)
text_generator.generate("wine review : germany", max_tokens=80, temp=0.5)

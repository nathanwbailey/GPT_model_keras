# GPT Model

Implements a GPT Model in Keras on the Wine Reviews Dataset: https://www.kaggle.com/datasets/zynicide/wine-reviews

Further information can be found in the following blog post:

https://nathanbaileyw.medium.com/transformers-explained-the-holy-grail-of-genai-9d4a46408418

### Code:
The main code is located in the following files:
* main.py - Main entry file for training the network
* gpt_model.py - Implements the GPT model
* model_building_blocks.py - Embedding block, transformer block and causal attention mask implementations to use in the model
* attention_layers.py - Implements a MultiHeadAttention layer from scratch
* prepare_dataset.py - Pre-processes the dataset
* text_generator.py - Keras callback to generate new text during training
* lint.sh - runs linters on the code

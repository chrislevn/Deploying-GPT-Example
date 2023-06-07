from flask import Flask, render_template
import os
from keras.models import load_model
from keras.layers import Layer
import model_prep
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
from flask import request
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)

# Register the custom layer
custom_objects = {"TokenAndPositionEmbedding": model_prep.TokenAndPositionEmbedding, "TransformerBlock": model_prep.TransformerBlock}

# Load the model with the custom layer
model = load_model('model.h5', custom_objects=custom_objects)

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        return txt


vocab = np.load('array.npy')
    
def predict(input_text):
    """
    Predicts the next 40 words based on the input text
    
    Args:
        input_text (str): The input text to predict from
        
    Returns:
        str: The predicted text
    """
    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index

    start_prompt = input_text
    start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
    num_tokens_generated = 40

    text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab).on_epoch_end(25)
    return text_gen_callback

@app.route('/', methods=['GET', 'POST'])
def home():
    result_text = ''
    if request.method == 'POST':
        input_text = request.form['input_text']
        if len(input_text) > 0:
            result_text = predict(input_text).replace("[UNK]", "")
            wordcloud = WordCloud().generate(result_text)
            wordcloud.to_file("static/images/wordcloud.png")
            return render_template('index.html', result_text=result_text)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002, use_reloader=False)
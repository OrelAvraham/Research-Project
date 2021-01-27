# Imports
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io

# Data Downloading and Processing

# -- Downloading and reading
path = keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")

with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # removing newlines for easing the learning
print("Corpus length:", len(text))

# -- Collecting the characters of the text and indexing them
chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))  # dictionary of char->idx
indices_char = dict((i, c) for i, c in enumerate(chars))  # dictionary of idx->char

# -- Dividing the data into sentences and the wanted optputs for them
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)  # defining the shape of X
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)  # defining the shape of Y
for i, sentence in enumerate(sentences):  # one-hot encoding
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Building the model

model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        layers.LSTM(128),
        # TODO: add dense layer here
        #  after running it on a faster computer of a friend it appears to work well
        layers.Dense(len(chars), activation="softmax"),
    ]
)

optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


# Training and testing

def sample(preds, temperature=1.0):
    # a function to sample the index of the letter to use from the prediction
    """
    :param preds: an array-like object that represents the probability each letter to be the next
    :param temperature: the degree of freedom for the decision
        (temperature > 1.0 -> more freedom, temperature < 1.0 less freedom)
    :return: the next letter of the sentence according to the statistical distribution
    """

    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # e^(ln(x)) = x (temperature = 1)
    # the role of the temperature is to tweak the values to lower and raise the probabilities

    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


epochs = 40
batch_size = 128

# -- Training the model for {epochs} epochs
for epoch in range(epochs):

    # --- Fitting it once
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print()
    print("Generating text after epoch: %d" % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)

    # -- Going over some diversities(temperatures)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("...Diversity:", diversity)

        generated = ""
        sentence = text[start_index: start_index + maxlen]
        seed = sentence # to keep track of the first sentence
        print('...Sentence ' + sentence)

        # -- Generating 400 characters
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print("...Generated: ", generated)

        # -- Saving the generations of the model
        path = rf'generations/epoch{epoch}.txt'
        with open(path, 'a') as file:
            content = f'================================\nDiversity: {diversity}\n'
            content += f'S  eed: {seed}\n'
            content += f'Generated:\n' + generated[:100] + '\n' + generated[100:200] + '\n' + generated[200:300] + '\n' + generated[300:] + '\n'
            content += '\n\n\n'
            # print(f'========\n{text}\n========')
            file.write(content)

        sentence = 'BUG BUG BUG'
        print()

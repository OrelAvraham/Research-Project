from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io
import os
import sys
from datetime import datetime

"""
This is the early second one between the runs 
"""

print('Coirolanus maxlen RNN units 16 nodes in hidden layer 0.2 dropout ')

# Data downloading and processing
path = keras.utils.get_file('shakespeare.txt',
                            'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

with io.open(path, encoding="utf-8") as f:
    text = f.read()
print(f'Amount of Characters: {len(text)}')

chars = sorted(list(set(text)))  # saving all the unique chars of the text
print(f'Amount of Unique Characters: {len(chars)}')

char_indices = dict((c, i) for i, c in enumerate(chars))  # dictionary of char->idx
indices_char = dict((i, c) for i, c in enumerate(chars))  # dictionary of idx->char

maxlen = 40  # a length of a sentence
step = 3  # the distance (amount of chars) between 1 sentence and the next sentence we are taking
sentences = []  # the list of the input sentences
next_chars = []  # the list of the chars coming after those sentences

# filling the lists with the data
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
        layers.LSTM(maxlen),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(chars), activation="softmax"),
    ]
)

optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


# TODO: read more about recommended optimizers & loss functions for RNN text gen,
#       these ones I saw was used in the internet.


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


input('\nPress Enter to start training\n')

epochs = 40  # the amount of epochs to train
batch_size = 128  # the batch size

time = datetime.now().strftime("%d.%m.%Y.%H.%M.%S")
curr_path = rf'{time}/'
models_path = curr_path + 'models/'
texts_path = curr_path + 'texts/'

if os.path.exists(curr_path) or os.path.exists(models_path) or os.path.exists(texts_path):
    sys.exit('---------------------------------\n' +
             'on of the paths is not good\n' +
             '---------------------------------')

os.mkdir(Path(curr_path))
os.mkdir(Path(models_path))
os.mkdir(Path(texts_path))

# The training loop: training the model for {epochs} epochs
for epoch in range(epochs):

    # fitting it once
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print()
    print("Generating text after epoch: %d" % epoch)

    model_file_path = models_path + f'model_generation{epoch}'
    text_file_path = texts_path + f'text_generation{epoch}.txt'

    model.save(model_file_path)  # saving the model

    for num in range(3):
        start_index = random.randint(0, len(text) - maxlen - 1)
        sentence = text[start_index: start_index + maxlen]
        seed = sentence  # to keep track of the first sentence

        with open(text_file_path, 'a') as file:
            file.write(f'SEED:{seed}\n\n\n')

        # using different kinds of divercities
        for diversity in [0.2, 0.5, 0.7, 1.0, 1.2]:
            # print("...Diversity:", diversity)

            print(f'E{epoch} S{num} Div{diversity}')

            generated = ""
            sentence = seed # for each diversity we want the starting sentence to be the seed
            # print('...Seed:\n' + sentence)

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

            # print("\n...Generated:\n", generated)

            with open(text_file_path, 'a') as file:
                # I am using the append method because for each diversity i am appending the generated text
                content = f'Div {diversity}\n'
                content += f'Generated:\n' + seed + generated + '\n\n\n'
                file.write(content)

            # print()

        with open(text_file_path, 'a') as file:
            file.write(
                f'\n------------------------------------------------------------------------------------------------\n')

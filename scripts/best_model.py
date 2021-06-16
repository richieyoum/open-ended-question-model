import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
import numpy as np
import kerastuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, Embedding, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from scripts.util import use_pkl
from scripts.load_data import load_glove, load_amazon_data


# load best epochs from hp search
with open('../models/amzn_model_kerastuner/best_epoch.txt', 'r') as f:
    epochs = int(f.read())

# load data
df = load_amazon_data(data_dir='../data/amazon_data')

# split into open ended and close ended
open_ended = df[df['questionType'] == 'open-ended']
close_ended = df[df['questionType'] == 'yes/no']

# remove open ended questions shorter than 15 characters in length (ref: notebooks/EDA.ipynb)
open_ended = open_ended[open_ended.question.apply(len) > 15]

# combine the two dfs back
df = pd.concat([open_ended, close_ended])

# set true class as 1, false class as 0
x = df.question.values
y = np.where(df.questionType == 'open-ended', 1, 0)

# split into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, stratify=y)

# define and train tokenizer on train set
tokenizer = Tokenizer(num_words=None, lower=True, oov_token="<UNK>")
tokenizer.fit_on_texts(xtrain)
word_index = tokenizer.word_index

# sequentialize & pad sentences
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

maxlen = 100
xtrain = pad_sequences(sequences=xtrain, padding='post', truncating='post', maxlen=maxlen)
xtest = pad_sequences(sequences=xtest, padding='post', truncating='post', maxlen=maxlen)

# populate embeddings
glove_w2v = load_glove(f'../data/glove/glove.6B.100d.txt')
# populate embedding matrix using glove w2v
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = glove_w2v.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

def model_builder(hp):
    # hyperparameters to tune
    dense_units = hp.Int('dense_units', min_value=6, max_value=60, step=6)
    recurrent_units = hp.Int('recurrent_units', min_value=4, max_value=64, step=6)
    recurrent_dropout = hp.Float('dropout_val', min_value=0.1, max_value=.5, step=0.1)
    dropout_val = hp.Float('dropout_val', min_value=0.1, max_value=.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = Sequential([
        Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False),
        Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=recurrent_dropout)),
        GlobalMaxPool1D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_val),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=[Precision(), Recall()])
    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=10,
                     factor=2,
                     hyperband_iterations=1,
                     project_name='../models/amzn_model_kerastuner')

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(xtrain, ytrain, epochs=epochs, validation_data=(xtest, ytest))

model.save('../models/amzn_model')
use_pkl('../models/amzn_model/tokenizer.pkl', 'wb', tokenizer)

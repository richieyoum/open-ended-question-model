import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, Embedding, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from load_data import load_glove, load_amazon_data
from util import use_pkl


# load model configs
with open('../config/train_config.json','r') as f:
    configs = json.load(f)
model_config = configs['bilstm_100d']

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
tokenizer = Tokenizer(num_words=model_config["num_words"], lower=True, oov_token="<UNK>")
tokenizer.fit_on_texts(xtrain)
word_index = tokenizer.word_index

# sequentialize & pad sentences
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = pad_sequences(sequences=xtrain, padding='post', truncating='post', maxlen=model_config['max_seq_len'])
xtest = pad_sequences(sequences=xtest, padding='post', truncating='post', maxlen=model_config['max_seq_len'])

# populate embeddings
glove_w2v = load_glove(f'../data/glove/glove.6B.{model_config["emb_dim"]}d.txt')
# populate embedding matrix using glove w2v
embedding_matrix = np.zeros((len(word_index) + 1, model_config["emb_dim"]))
for word, i in word_index.items():
    embedding_vector = glove_w2v.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# define model
model = Sequential([
    Embedding(len(word_index) + 1, model_config['emb_dim'], weights=[embedding_matrix], input_length=model_config['max_seq_len'], trainable=model_config['emb_trainable']),
    Bidirectional(LSTM(model_config["lstm_units"], return_sequences=True, dropout=model_config["recurrent_dropout"])),
    GlobalMaxPool1D(),
    Dense(model_config['dense_units'], activation='relu'),
    Dropout(model_config['dropout']),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=model_config['learning_rate']), metrics=[Precision(), Recall(), AUC()], loss='binary_crossentropy')

model.fit(xtrain, ytrain, batch_size=1024, epochs=model_config['epochs'], validation_data=(xtest, ytest))
model.save('../models/bilstm_100d')
use_pkl('../models/bilstm_100d/tokenizer.pkl', 'wb', tokenizer)

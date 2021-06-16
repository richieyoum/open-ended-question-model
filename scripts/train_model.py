import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout, Embedding, GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from scripts.util import use_pkl, load_glove, load_amazon_data, get_embedding_matrix
from scripts.preprocess import preprocess
import kerastuner as kt


# load best epochs from hp search
with open('../models/amzn_model_kerastuner/best_epoch.txt', 'r') as f:
    epochs = int(f.read())

# load data
df = load_amazon_data(data_dir='../data/amazon_data')

tokenizer, xtrain, xtest, ytrain, ytest = preprocess(df, maxlen=100)
word_index = tokenizer.word_index

# populate embeddings
glove_w2v = load_glove(f'../data/glove/glove.6B.100d.txt')
embedding_matrix = get_embedding_matrix(glove_w2v, word_index, emb_dim=100)


def model_builder(hp):
    # hyperparameters to tune
    dense_units = hp.Int('dense_units', min_value=6, max_value=60, step=6)
    recurrent_units = hp.Int('recurrent_units', min_value=4, max_value=64, step=6)
    recurrent_dropout = hp.Float('recurrent_dropout', min_value=0.1, max_value=.5, step=0.1)
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

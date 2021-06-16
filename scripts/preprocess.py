import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def preprocess(df, maxlen, tokenizer=None, random_state=None):
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
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2, stratify=y, random_state=random_state)

    # define and train tokenizer on train set if no tokenizer provided
    if not tokenizer:
        tokenizer = Tokenizer(num_words=None, lower=True, oov_token="<UNK>")
        tokenizer.fit_on_texts(xtrain)

    # sequentialize & pad sentences
    xtrain = tokenizer.texts_to_sequences(xtrain)
    xtest = tokenizer.texts_to_sequences(xtest)

    xtrain = pad_sequences(sequences=xtrain, padding='post', truncating='post', maxlen=maxlen)
    xtest = pad_sequences(sequences=xtest, padding='post', truncating='post', maxlen=maxlen)

    return tokenizer, xtrain, xtest, ytrain, ytest

import os
import sys
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from tensorflow.keras.models import load_model
from scripts.util import use_pkl, load_amazon_data
from scripts.preprocess import preprocess
from sklearn.metrics import classification_report


# load data
df = load_amazon_data(data_dir='../data/amazon_data')

# load model
model = load_model('../models/amzn_model')

# load tokenizer
tokenizer = use_pkl('../models/amzn_model/tokenizer.pkl', 'rb')

# split, tokenized, and padded
tokenizer, xtrain, xtest, ytrain, ytest = preprocess(df, maxlen=100, tokenizer=tokenizer, random_state=1234)

pred = np.where(np.squeeze(model.predict(xtest))>=.5, 1, 0)
print(classification_report(ytest, pred))

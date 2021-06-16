import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from scripts.util import use_pkl
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# load model
model = load_model('../models/amzn_model')

# load tokenizer
tokenizer = use_pkl('../models/amzn_model/tokenizer.pkl', 'rb')

# sample sentences to run inference on
sample_text = ["What kind of TV shows do you like to watch?", "What do you use your phone for?", "Do you work from home?", "Whale is a big animal!", "Do you like video games?", "What's your phone number please?"]
transformed_text = tokenizer.texts_to_sequences(sample_text)
transformed_text = pad_sequences(sequences=transformed_text, padding='post', truncating='post', maxlen=100)

pred = model.predict(transformed_text)

for i in zip(sample_text, pred):
    print(i)

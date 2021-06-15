from util import use_pkl
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# load model configs
with open('../config/train_config.json','r') as f:
    configs = json.load(f)
model_config = configs['bilstm_50d']

# load model
model = load_model('../models/bilstm_50d')

# load tokenizer
tokenizer = use_pkl('../models/bilstm_50d/tokenizer.pkl','rb')

# sample sentences to run inference on
sample_text = ["What kind of TV shows do you like to watch?", "What do you use your phone for?", "Do you work from home?", "Whale is a big animal!", "Do you like video games?", "What's your phone number please?"]
transformed_text = tokenizer.texts_to_sequences(sample_text)
transformed_text = pad_sequences(sequences=transformed_text, padding='post', truncating='post', maxlen=model_config['max_seq_len'])

pred = model.predict(transformed_text)

for i in zip(sample_text, pred):
    print(i)
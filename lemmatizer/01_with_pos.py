from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import json
import pandas as pd
import numpy as np
import keras
from keras.callbacks.callbacks import ModelCheckpoint

np.random.seed(1)

#path_train = '/Users/technician/Documents/lt4hala/myscrips/train.json'
#path_dev = '/Users/technician/Documents/lt4hala/myscrips/dev.json'
#path_test = '/Users/technician/Documents/lt4hala/myscrips/test.json'
path_train = '/home/ubuntu/train.json'
path_dev = '/home/ubuntu/dev.json'
path_test = '/home/ubuntu/test.json'

batch_size = 64 #64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 2500 #256  # Latent dimensionality of the encoding space.

with open(path_train, "r") as f:
    train = json.load(f)

with open(path_dev, "r") as f:
    dev = json.load(f)

with open(path_test, "r") as f:
    test = json.load(f)

input_texts = []
for t in train:
     if t["postag"] == "PROPN":
       input_texts.append(t['postag'] + " " + t['form'])
     else:
     	 input_texts.append(t['postag'] + " " + t['form'].lower())

# \t and \n needed by the template
target_texts = []
for t in train:
      target_texts.append("\t" + t['lemma'] + "\n")


for t in dev:
     if t["postag"] == "PROPN":
       input_texts.append(t['postag'] + " " + t['form'])
     else:
       input_texts.append(t['postag'] + " " + t['form'].lower())
     
for t in dev:
      target_texts.append("\t" + t['lemma'] + "\n")


for t in test:
     if t["postag"] == "PROPN":
       input_texts.append(t['postag'] + " " + t['form'])
     else:
       input_texts.append(t['postag'] + " " + t['form'].lower())
     
for t in test:
      target_texts.append("\t" + t['lemma'] + "\n")

input_characters =set()
target_characters = set()
for t in input_texts:
        for c in t:
            if c not in input_characters:
                input_characters.add(c)
for t in target_texts:
        for c in t:
            if c not in target_characters:
                target_characters.add(c)

input_characters.add(" ")
target_characters.add(" ")

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
    

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.


decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)


decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
opt = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
mcp_save = ModelCheckpoint('model_seq2seq_best', save_best_only=True, monitor='val_acc', mode='max')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=[mcp_save])
          
          
             

### predict

input_textsT = []
for t in test:
     if t["postag"] == "PROPN":
       input_textsT.append(t['postag'] + " " + t['form'])
     else:
       input_textsT.append(t['postag'] + " " + t['form'].lower())
     

true_labels = []
for t in test:
      true_labels.append(t['lemma'])

encoder_input_dataT = np.zeros(
    (len(input_textsT), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
    

for i, input_textT in enumerate(input_textsT):
    for t, char in enumerate(input_textT):
        encoder_input_dataT[i, t, input_token_index[char]] = 1.
    encoder_input_dataT[i, t + 1:, input_token_index[' ']] = 1.
    
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]



decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


lemmas_predicted = []
for seq_index in range(0, encoder_input_dataT.shape[0] ):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_dataT[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    lemmas_predicted.append(decoded_sentence)
    print('-')
    print('Input sentence:', input_textsT[seq_index])
    print('Decoded sentence:', decoded_sentence)


#lemmas_predicted = []
#for t in encoder_input_dataT:
#     e = decode_sequence(np.array([t]))
#     lemmas_predicted.append(e)
  
from sklearn.metrics import accuracy_score

lemmas_predicted2 = list(map(lambda x: x.replace("\n", ""), lemmas_predicted))
#0.9763409371146733
accuracy_score(true_labels, lemmas_predicted2)

'''
Using TensorFlow backend.
Train on 233680 samples, validate on 25965 samples
Epoch 1/10
233680/233680 [==============================] - 910s 4ms/step - loss: 0.3428 - accuracy: 0.9014 - val_loss: 0.0371 - val_accuracy: 0.9896
/home/ubuntu/anaconda3/envs/tensorflow2_p36/lib/python3.6/site-packages/keras/callbacks/callbacks.py:707: RuntimeWarning: Can save best model only with val_acc available, skipping.
  'skipping.' % (self.monitor), RuntimeWarning)
Epoch 2/10
233680/233680 [==============================] - 946s 4ms/step - loss: 0.0146 - accuracy: 0.9960 - val_loss: 0.0150 - val_accuracy: 0.9962
Epoch 3/10
233680/233680 [==============================] - 947s 4ms/step - loss: 0.0065 - accuracy: 0.9983 - val_loss: 0.0102 - val_accuracy: 0.9974
Epoch 4/10
233680/233680 [==============================] - 938s 4ms/step - loss: 0.0043 - accuracy: 0.9989 - val_loss: 0.0102 - val_accuracy: 0.9977
Epoch 5/10
233680/233680 [==============================] - 951s 4ms/step - loss: 0.0033 - accuracy: 0.9991 - val_loss: 0.0095 - val_accuracy: 0.9980
Epoch 6/10
233680/233680 [==============================] - 946s 4ms/step - loss: 0.0029 - accuracy: 0.9993 - val_loss: 0.0119 - val_accuracy: 0.9979
Epoch 7/10
233680/233680 [==============================] - 954s 4ms/step - loss: 0.0026 - accuracy: 0.9993 - val_loss: 0.0105 - val_accuracy: 0.9981
Epoch 8/10
233680/233680 [==============================] - 953s 4ms/step - loss: 0.0024 - accuracy: 0.9994 - val_loss: 0.0104 - val_accuracy: 0.9981
Epoch 9/10
233680/233680 [==============================] - 948s 4ms/step - loss: 0.0022 - accuracy: 0.9994 - val_loss: 0.0110 - val_accuracy: 0.9981
Epoch 10/10
233680/233680 [==============================] - 942s 4ms/step - loss: 0.0021 - accuracy: 0.9995 - val_loss: 0.0115 - val_accuracy: 0.9982
'''


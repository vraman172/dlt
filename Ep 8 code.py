import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============ SAMPLE DATA ============
input_texts = ['hello', 'how are you', 'what is your name', 'bye']
target_texts = ['hi', 'i am fine', 'i am a bot', 'goodbye']

# Add start and end tokens
target_texts = ['<start> ' + t + ' <end>' for t in target_texts]

# ============ TOKENIZATION ============
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)
input_seq = tokenizer.texts_to_sequences(input_texts)
target_seq = tokenizer.texts_to_sequences(target_texts)

max_input_len = max(len(s) for s in input_seq)
max_target_len = max(len(s) for s in target_seq)
vocab_size = len(tokenizer.word_index) + 1

input_seq = pad_sequences(input_seq, maxlen=max_input_len, padding='post')
target_seq = pad_sequences(target_seq, maxlen=max_target_len, padding='post')

# ============ HYPERPARAMETERS ============
embedding_dim = 64
lstm_units = 128

# ============ ENCODER ============
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(enc_emb)

# ============ DECODER ============
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(dec_emb, initial_state=[state_h, state_c])

# Attention
attn = Attention()([decoder_lstm, encoder_lstm])
decoder_concat = Concatenate(axis=-1)([decoder_lstm, attn])

decoder_outputs = Dense(vocab_size, activation='softmax')(decoder_concat)

# ============ MODEL ============
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare target data
decoder_target_data = np.expand_dims(target_seq, -1)

# ============ TRAIN ============
model.fit([input_seq, target_seq], decoder_target_data, batch_size=2, epochs=200, verbose=0)

# ============ INFERENCE SETUP ============
# Encoder
encoder_model = Model(encoder_inputs, [encoder_lstm, state_h, state_c])

# Decoder inputs for inference
dec_state_h = Input(shape=(lstm_units,))
dec_state_c = Input(shape=(lstm_units,))
enc_out_input = Input(shape=(max_input_len, lstm_units))
dec_input_inf = Input(shape=(1,))

# Reuse decoder embedding
dec_emb_layer = Embedding(vocab_size, embedding_dim)
dec_emb2 = dec_emb_layer(dec_input_inf)

dec_lstm2, state_h2, state_c2 = LSTM(lstm_units, return_sequences=True, return_state=True)(
    dec_emb2, initial_state=[dec_state_h, dec_state_c]
)

attn2 = Attention()([dec_lstm2, enc_out_input])
decoder_concat2 = Concatenate(axis=-1)([dec_lstm2, attn2])
dec_outputs2 = Dense(vocab_size, activation='softmax')(decoder_concat2)

decoder_model = Model(
    [dec_input_inf, enc_out_input, dec_state_h, dec_state_c],
    [dec_outputs2, state_h2, state_c2]
)

# ============ RESPONSE GENERATOR ============
def generate_response(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_input_len, padding='post')
    enc_out, h, c = encoder_model.predict(seq, verbose=0)
    
    target = np.array([[tokenizer.word_index['<start>']]])
    stop = False
    decoded = ''
    
    while not stop and len(decoded.split()) < 2 * max_target_len:
        output, h, c = decoder_model.predict([target, enc_out, h, c], verbose=0)
        sampled_idx = np.argmax(output[0, -1, :])
        word = tokenizer.index_word.get(sampled_idx, '')
        
        if word == '<end>' or word == '':
            stop = True
        else:
            decoded += word + ' '
            target = np.array([[sampled_idx]])
            
    return decoded.strip()

# ============ TEST ============
test_input = "how are you"
print("Input:", test_input)
print("Bot  :", generate_response(test_input))

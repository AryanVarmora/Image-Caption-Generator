#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add, RepeatVector
from tensorflow.keras.models import Model

# Constants

EMBEDDING_DIM = 256
LSTM_UNITS = 512
VOCAB_SIZE = 8781   # Must match tokenizer size
MAX_CAPTION_LENGTH = 35


# 1. Define the Encoder (Image Feature Extractor)
def build_encoder():
    # Input shape matches the feature vector dimensions (e.g., 2048 from InceptionV3)
    image_input = Input(shape=(2048,), name="image_input")
    dense = Dense(LSTM_UNITS, activation="relu", name="encoder_dense")(image_input)  
    dropout = Dropout(0.5, name="encoder_dropout")(dense)
    return Model(inputs=image_input, outputs=dropout, name="Encoder")

# 2. Define the Decoder (Caption Generator)
def build_decoder():
    # Inputs for the decoder
    caption_input = Input(shape=(MAX_CAPTION_LENGTH,), name="caption_input")
    
    # Embedding layer for captions
    embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True, name="embedding_layer")(caption_input)

    # LSTM for sequence processing
    lstm_output = LSTM(LSTM_UNITS, return_sequences=True, name="lstm_layer")(embedding)

    return Model(inputs=caption_input, outputs=lstm_output, name="Decoder")

# 3. Combine Encoder and Decoder into a Full Model
def build_image_captioning_model():
    # Encoder
    encoder = build_encoder()

    # Decoder
    decoder = build_decoder()

    # Inputs
    image_features_input = encoder.input
    caption_input = decoder.input

    # Combine encoder and decoder outputs
    encoder_output = encoder.output  # (batch_size, LSTM_UNITS)
    print(f"Encoder output shape: {encoder_output.shape}")

    decoder_output = decoder.output  # (batch_size, MAX_CAPTION_LENGTH, LSTM_UNITS)
    print(f"Decoder output shape: {decoder_output.shape}")

    # Add the encoded image features to each time step of the decoder output
    image_features_expanded = RepeatVector(MAX_CAPTION_LENGTH, name="repeat_features")(encoder_output)  # (batch_size, MAX_CAPTION_LENGTH, LSTM_UNITS)
    print(f"Expanded image features shape: {image_features_expanded.shape}")

    combined_features = Add(name="combine_features")([image_features_expanded, decoder_output])

    # Final dense layer to generate predictions
    outputs = Dense(VOCAB_SIZE, activation="softmax", name="output_layer")(combined_features)

    # Full model
    model = Model(inputs=[image_features_input, caption_input], outputs=outputs, name="ImageCaptioningModel")

    return model

# Build the model
if __name__ == "__main__":
    print("Building the encoder-decoder model...")
    model = build_image_captioning_model()
    model.summary()


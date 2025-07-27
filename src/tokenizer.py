#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
captions_file = "/Users/aryan/Desktop/Image-Caption-Generator/data/captions.txt"
processed_captions_file = "/Users/aryan/Desktop/Image-Caption-Generator/data/processed_captions.npy"
tokenizer_file = "/Users/aryan/Desktop/Image-Caption-Generator/data/tokenizer.pkl"

# Load captions
def load_captions(file_path):
    captions = {}
    with open(file_path, "r") as file:
        for line in file:
            try:
                image, caption = line.strip().split(",", 1)  # Split image and caption
                if image == "image" or caption == "caption":  # Skip header or invalid lines
                    continue
                if image not in captions:
                    captions[image] = []
                captions[image].append(caption)
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")
    return captions

# Clean captions
def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z ]", "", caption)  # Remove special characters and numbers
    caption = re.sub(r"\s+", " ", caption).strip()  # Remove extra spaces
    return caption

# Preprocess all captions
def preprocess_captions(captions):
    cleaned_captions = {}
    for image, caps in captions.items():
        cleaned_captions[image] = [clean_caption(cap) for cap in caps]
    return cleaned_captions

# Tokenize captions
def tokenize_captions(captions, max_vocab_size=10000):
    all_captions = [cap for caps in captions.values() for cap in caps]
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

# Encode and pad captions
def encode_and_pad_captions(captions, tokenizer, max_length):
    encoded_captions = {}
    for image, caps in captions.items():
        encoded_captions[image] = pad_sequences(
            tokenizer.texts_to_sequences(caps), maxlen=max_length, padding="post"
        )
    return encoded_captions

# Main script
if __name__ == "__main__":
    print("Loading captions...")
    captions = load_captions(captions_file)

    print("Cleaning captions...")
    cleaned_captions = preprocess_captions(captions)

    print("Tokenizing captions...")
    tokenizer = tokenize_captions(cleaned_captions)
    vocab_size = len(tokenizer.word_index) + 1  # Include padding token
    print(f"Vocabulary size: {vocab_size}")

    max_caption_length = max(len(cap.split()) for caps in cleaned_captions.values() for cap in caps)
    print(f"Maximum caption length: {max_caption_length}")

    print("Encoding and padding captions...")
    encoded_captions = encode_and_pad_captions(cleaned_captions, tokenizer, max_caption_length)

    print("Saving tokenizer and processed captions...")
    with open(tokenizer_file, "wb") as f:
        pickle.dump(tokenizer, f)
    np.save(processed_captions_file, encoded_captions, allow_pickle=True)

    print(f"Tokenizer saved to {tokenizer_file}")
    print(f"Processed captions saved to {processed_captions_file}")


# In[ ]:


import numpy as np
import pickle

# Load tokenizer
tokenizer_file = "/Users/aryan/Desktop/Image-Caption-Generator/data/tokenizer.pkl"
with open(tokenizer_file, "rb") as f:
    tokenizer = pickle.load(f)

# Load processed captions
processed_captions_file = "/Users/aryan/Desktop/Image-Caption-Generator/data/processed_captions.npy"
processed_captions = np.load(processed_captions_file, allow_pickle=True).item()

# Sample Image and Caption
sample_image = list(processed_captions.keys())[0]
sample_encoded_caption = processed_captions[sample_image][0]

# Decode the caption
decoded_caption = " ".join(
    [tokenizer.index_word.get(index, "<unk>") for index in sample_encoded_caption if index > 0]
)

print(f"Sample Image: {sample_image}")
print(f"Encoded Caption: {sample_encoded_caption}")
print(f"Decoded Caption: {decoded_caption}")


# In[ ]:





# In[ ]:


# Iterate through a few more images and their captions
for i, image in enumerate(processed_captions.keys()):
    if i == 5:  # Inspect the first 5 images
        break
    sample_encoded_caption = processed_captions[image][0]
    decoded_caption = " ".join(
        [tokenizer.index_word.get(index, "<unk>") for index in sample_encoded_caption if index > 0]
    )
    print(f"Image: {image}")
    print(f"Decoded Caption: {decoded_caption}")
    print("-" * 50)


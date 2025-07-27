#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/AryanVarmora/Image-Caption-Generator/blob/main/src/extract_image_features.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import os
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


# Data set
# 

# In[ ]:


# Paths
image_folder = "/Users/aryan/Desktop/Image-Caption-Generator/data"  # Replace with your image folder path
output_features_file = "/Users/aryan/Desktop/Image-Caption-Generator/data/image_features.npy"  # File to save features


# Pre-Processing
# 

# In[ ]:


# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Remove top layer


# In[ ]:


# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Resize to InceptionV3 input size
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Normalize for InceptionV3

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Resize to InceptionV3 input size
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Normalize for InceptionV3


# Features Extraction
# 

# In[ ]:


# Extract features for all images
def extract_features(image_folder):
    features = {}
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_preprocessed = preprocess_image(img_path)
            features[img_name] = model.predict(img_preprocessed).flatten()  # Extract and flatten features
    return features


# Saving the Features

# In[ ]:


# Extract and save features
if __name__ == "__main__":
    print("Extracting image features...")
    image_features = extract_features(image_folder)
    np.save(output_features_file, image_features)  # Save features as a NumPy file
    print(f"Features saved to {output_features_file}")


# In[ ]:


import os
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Paths
image_folder = "/Users/aryan/Desktop/Image-Caption-Generator/data/images"  # Replace with your image folder path
output_features_file = "/Users/aryan/Desktop/Image-Caption-Generator/data/image_features.npy"  # File to save features

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)  # Remove top layer

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Resize to InceptionV3 input size
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Normalize for InceptionV3

# Extract features for all images
def extract_features(image_folder):
    features = {}
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"The directory {image_folder} does not exist.")
    if not os.listdir(image_folder):
        raise FileNotFoundError(f"The directory {image_folder} is empty. Please add image files.")

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                img_preprocessed = preprocess_image(img_path)
                features[img_name] = model.predict(img_preprocessed).flatten()  # Extract and flatten features
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    return features

# Extract and save features
if __name__ == "__main__":
    print("Extracting image features...")
    image_features = extract_features(image_folder)
    np.save(output_features_file, image_features, allow_pickle=True)  # Save features as a NumPy file
    print(f"Features saved to {output_features_file}")


# In[ ]:


import numpy as np

# Load the saved features
features = np.load("/Users/aryan/Desktop/Image-Caption-Generator/data/image_features.npy", allow_pickle=True).item()

# Check keys and a sample feature vector
print(f"Number of images: {len(features)}")
print(f"Sample keys (image filenames): {list(features.keys())[:5]}")
print(f"Sample feature vector shape: {features[next(iter(features))].shape}")
 


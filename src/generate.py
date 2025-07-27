import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from image_caption_model import build_image_captioning_model

# ✅ Determine base path
if "__file__" in globals():
    base_path = os.path.dirname(__file__)
else:
    base_path = os.getcwd()

data_dir = os.path.abspath(os.path.join(base_path, "../data"))
model_path = os.path.join(base_path, "model_checkpoint.keras")
tokenizer_path = os.path.join(data_dir, "tokenizer.pkl")

MAX_CAPTION_LENGTH = 35

# ✅ Load tokenizer
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load trained model
print("📥 Loading trained model...")
model = build_image_captioning_model()
model.load_weights(model_path)

# ✅ Function to preprocess an image and extract features
def extract_features(img_path):
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    from tensorflow.keras.models import Model

    model_incep = InceptionV3(weights="imagenet")
    model_incep = Model(model_incep.input, model_incep.layers[-2].output)

    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    feature = model_incep.predict(img_array, verbose=0)
    return feature.squeeze()

# ✅ Generate caption
def generate_caption(photo_feature):
    inv_map = {v: k for k, v in tokenizer.word_index.items()}
    in_text = "<start>"

    for _ in range(MAX_CAPTION_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_CAPTION_LENGTH, padding="post")

        yhat = model.predict([photo_feature.reshape((1, 2048)), sequence], verbose=0)
        yhat = np.argmax(yhat[0, -1, :])

        word = inv_map.get(yhat)
        if word is None or word == "end":
            break

        in_text += " " + word

    return in_text.replace("<start>", "").strip()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    features = extract_features(args.image)
    caption = generate_caption(features)
    print(f"🖼️ Caption: {caption}")

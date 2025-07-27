import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from image_caption_model import build_image_captioning_model

# ✅ Handle both script & notebook execution
if "__file__" in globals():
    base_path = os.path.dirname(__file__)
else:
    base_path = os.getcwd()

data_dir = os.path.abspath(os.path.join(base_path, "../data"))
image_features_path = os.path.join(data_dir, "image_features.npy")
captions_path = os.path.join(data_dir, "processed_captions.npy")
tokenizer_path = os.path.join(data_dir, "tokenizer.pkl")

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# Load data
def load_data():
    print("📂 Loading data...")
    image_features = np.load(image_features_path, allow_pickle=True).item()
    captions = np.load(captions_path, allow_pickle=True).item()
    return image_features, captions

# Data generator
def data_generator(image_features, captions, tokenizer, max_caption_length, batch_size):
    while True:
        image_inputs, caption_inputs, targets = [], [], []
        for image_id, caption_set in captions.items():
            for caption in caption_set:
                input_seq, target_seq = caption[:-1], caption[1:]

                input_seq_padded = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_caption_length, padding="post")[0]
                target_seq_padded = tf.keras.preprocessing.sequence.pad_sequences([target_seq], maxlen=max_caption_length, padding="post")[0]

                image_inputs.append(image_features[image_id])
                caption_inputs.append(input_seq_padded)
                targets.append(target_seq_padded)

                if len(image_inputs) == batch_size:
                    yield ((np.array(image_inputs), np.array(caption_inputs)), np.array(targets))
                    image_inputs, caption_inputs, targets = [], [], []

# Build & compile model
print("⚙️ Building model...")
model = build_image_captioning_model()
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Load tokenizer
print("🔤 Loading tokenizer...")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

max_caption_length = 35
image_features, captions = load_data()

# Create dataset
dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(image_features, captions, tokenizer, max_caption_length, BATCH_SIZE),
    output_signature=(
        (
            tf.TensorSpec(shape=(BATCH_SIZE, 2048), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, max_caption_length), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(BATCH_SIZE, max_caption_length), dtype=tf.int32),
    ),
)

steps_per_epoch = sum(len(captions[img]) for img in captions) // BATCH_SIZE

# Callbacks
checkpoint = ModelCheckpoint("model_checkpoint.keras", save_best_only=True, monitor="loss", mode="min")
early_stopping = EarlyStopping(monitor="loss", patience=3, mode="min")

print("🚀 Starting training...")
model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[checkpoint, early_stopping])
print("✅ Training complete! Model saved as 'model_checkpoint.keras'")

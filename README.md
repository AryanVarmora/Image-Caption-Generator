##Image Caption Generator

This project generates descriptive captions for images using a neural network that combines Convolutional Neural Networks (CNNs) for image feature extraction and Long Short-Term Memory (LSTM) networks for natural language processing.

##Project Overview

Image captioning is a challenging task that bridges computer vision and natural language processing. This project leverages pre-trained CNNs for visual feature extraction and LSTMs for generating meaningful captions for images.

##Features

Preprocesses and tokenizes textual captions.

Extracts image features using pre-trained models like InceptionV3.

Generates captions for images through a CNN-LSTM architecture.

Includes evaluation metrics like BLEU score to measure caption quality.



#How to Run

### 1. Clone the Repository

git clone https://github.com/AryanVarmora/Image-Caption-Generator.git
cd Image-Caption-Generator

### 2. Install Dependencies

Ensure you have Python 3.8 or higher installed, then run:

pip install -r requirements.txt

### 3. Preprocess Data

Run the preprocessing script to tokenize captions and prepare the dataset:

python src/tokenizer.py

###4. Train the Model

Train the CNN-LSTM model using:

python src/train.py

5. Generate Captions

After training, use the model to generate captions for new images.

Dataset

Captions: Derived from captions.txt, where each image has multiple captions for training.

Images: Include sample images in the data/images/ folder. Additional images can be added for testing.

Technologies Used

Deep Learning Frameworks: TensorFlow or PyTorch

Python Libraries: NumPy, Pandas, Matplotlib, OpenCV

Data Preprocessing: Tokenization, Padding, Vocabulary Building

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contributions

Contributions are welcome! If you want to improve this project, feel free to fork the repository and submit a pull request.

Contact

For questions or suggestions, please reach out through GitHub Issues or email aryanvarmora8@gmail.com


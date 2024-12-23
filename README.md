# Image Caption Generator

This project generates captions for images using a neural network that combines CNNs and LSTMs.

## How to Run
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run preprocessing: `python src/tokenizer.py`
4. Train the model: `python src/train.py`

## Dataset
- **Captions:** Derived from `captions.txt`, each image has 5 captions.
- **Images:** Sample images included in `data/images/`.

## Technologies Used
- TensorFlow
- Python
- NumPy, Matplotlib

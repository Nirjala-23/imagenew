"""Flask web app: upload an image -> get a generated caption.

Pipeline: VGG16 (fc2, 4096-d) image features -> CNN+LSTM decoder -> greedy
word-by-word decoding until 'endseq' or max_length.
"""
import os
import pickle
import uuid

import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
MAX_LENGTH = 39
ALLOWED_EXT = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- load everything ONCE at startup ----------
def load_caption_model():
    """Prefer best.keras (written by train.py); fall back to the original best.pkl."""
    keras_path = os.path.join(MODEL_DIR, "best.keras")
    pkl_path = os.path.join(MODEL_DIR, "best.pkl")
    if os.path.exists(keras_path):
        return load_model(keras_path)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        "No trained model found. Put best.pkl (or best.keras) in models/, "
        "or run `python train.py` to create models/best.keras."
    )


with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# Reverse lookup (index -> word) built once: O(1) instead of scanning the dict per step.
index_to_word = {i: w for w, i in tokenizer.word_index.items()}

model_caption = load_caption_model()

# VGG16 without the final softmax layer -> 4096-d feature vector from 'fc2'.
_vgg = VGG16()
model_img = Model(inputs=_vgg.inputs, outputs=_vgg.layers[-2].output)


# ---------- inference ----------
def extract_features(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img).reshape((1, 224, 224, 3))
    img = preprocess_input(img)
    return model_img.predict(img, verbose=0)


def predict_caption(feature, max_length=MAX_LENGTH):
    """Greedy autoregressive decoding."""
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding="post")
        yhat = np.argmax(model_caption.predict([feature, seq], verbose=0))
        word = index_to_word.get(int(yhat))
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text


def clean_caption(text):
    words = text.split()
    if words and words[0] == "startseq":
        words = words[1:]
    if words and words[-1] == "endseq":
        words = words[:-1]
    return " ".join(words)


# ---------- web app ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload cap


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file part", 400
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400
    if not allowed(file.filename):
        return "Only .jpg, .jpeg, .png files are allowed", 400

    # unique, sanitised name -> no path traversal, no overwriting
    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    img_path = os.path.join(UPLOAD_DIR, filename)
    file.save(img_path)

    caption = clean_caption(predict_caption(extract_features(img_path)))
    return render_template("after.html", data=caption, image_file=f"uploads/{filename}")


if __name__ == "__main__":
    app.run(debug=True)

"""Train the CNN + LSTM image-captioning model on Flickr8k.

Usage (from the project root):
    python train.py                 # extract features -> train -> save models/best.keras
    python train.py --epochs 30 --batch-size 32
    python train.py --eval-only     # BLEU score of models/best.keras on the test split

Expected data layout (see data/README.md):
    data/Images/*.jpg
    data/Flickr8k.token.txt
    data/Flickr_8k.trainImages.txt
    data/Flickr_8k.devImages.txt
    data/Flickr_8k.testImages.txt
"""
import argparse
import math
import os
import pickle
import random
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "Images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "best.keras")


# ------------------------------------------------------------------ features
def extract_all_features():
    """Run every image through VGG16 (up to fc2) once and cache the 4096-d vectors."""
    if os.path.exists(FEATURES_PATH):
        print("Loading cached features:", FEATURES_PATH)
        with open(FEATURES_PATH, "rb") as f:
            return pickle.load(f)

    vgg = VGG16()
    extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    features = {}
    files = sorted(os.listdir(IMG_DIR))
    for n, name in enumerate(files, 1):
        if not name.lower().endswith(".jpg"):
            continue
        img = load_img(os.path.join(IMG_DIR, name), target_size=(224, 224))
        img = preprocess_input(img_to_array(img).reshape((1, 224, 224, 3)))
        features[name.split(".")[0]] = extractor.predict(img, verbose=0)[0]
        if n % 500 == 0:
            print(f"  features: {n}/{len(files)}")
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(features, f)
    return features


# ------------------------------------------------------------------ captions
def clean(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z ]", " ", caption)  # drop punctuation / digits
    caption = re.sub(r"\s+", " ", caption).strip()
    return "startseq " + caption + " endseq"


def load_captions():
    """image_id -> list of 5 cleaned captions."""
    mapping = {}
    with open(os.path.join(DATA_DIR, "Flickr8k.token.txt"), encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            img, cap = line.strip().split("\t", 1)
            img_id = img.split(".")[0]
            mapping.setdefault(img_id, []).append(clean(cap))
    return mapping


def load_split(filename):
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
        return [l.strip().split(".")[0] for l in f if l.strip()]


def get_tokenizer(mapping, refit):
    if os.path.exists(TOKENIZER_PATH) and not refit:
        print("Re-using existing tokenizer:", TOKENIZER_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    tok = Tokenizer()
    tok.fit_on_texts([c for caps in mapping.values() for c in caps])
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tok, f)
    return tok


# ------------------------------------------------------------------ data
def make_dataset(keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    """Each caption of N tokens becomes N-1 (image, prefix) -> next_word samples."""

    def gen():
        keys_ = list(keys)
        random.shuffle(keys_)
        X1, X2, y = [], [], []
        n = 0
        for k in keys_:
            for cap in mapping[k]:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    X1.append(features[k])
                    X2.append(pad_sequences([seq[:i]], maxlen=max_length, padding="post")[0])
                    y.append(seq[i])
            n += 1
            if n == batch_size:  # batch_size = number of IMAGES per step
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y, n = [], [], [], 0
        if n:
            yield (np.array(X1), np.array(X2)), np.array(y)

    sig = (
        (tf.TensorSpec((None, 4096), tf.float32), tf.TensorSpec((None, max_length), tf.int32)),
        tf.TensorSpec((None,), tf.int32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(2)


# ------------------------------------------------------------------ model
def build_model(vocab_size, max_length):
    # image branch: 4096-d VGG16 feature -> 256
    in_img = Input(shape=(4096,))
    x1 = Dropout(0.4)(in_img)
    x1 = Dense(256, activation="relu")(x1)

    # text branch: word ids -> embedding -> LSTM
    in_txt = Input(shape=(max_length,))
    x2 = Embedding(vocab_size, 256, mask_zero=True)(in_txt)
    x2 = Dropout(0.4)(x2)
    x2 = LSTM(256)(x2)

    # merge + predict next word
    d = add([x1, x2])
    d = Dense(256, activation="relu")(d)
    out = Dense(vocab_size, activation="softmax")(d)

    model = Model(inputs=[in_img, in_txt], outputs=out)
    # sparse loss: labels are word indices, so no giant one-hot matrix is needed
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


# ------------------------------------------------------------------ eval
def generate(model, tokenizer, feature, max_length):
    idx2word = {i: w for w, i in tokenizer.word_index.items()}
    text = "startseq"
    for _ in range(max_length):
        seq = pad_sequences([tokenizer.texts_to_sequences([text])[0]], maxlen=max_length, padding="post")
        word = idx2word.get(int(np.argmax(model.predict([feature.reshape(1, -1), seq], verbose=0))))
        if word is None:
            break
        text += " " + word
        if word == "endseq":
            break
    return text


def evaluate(model, tokenizer, test_keys, mapping, features, max_length):
    from nltk.translate.bleu_score import corpus_bleu  # pip install nltk

    actual, predicted = [], []
    for k in test_keys:
        pred = generate(model, tokenizer, features[k], max_length).split()
        predicted.append(pred)
        actual.append([c.split() for c in mapping[k]])
    for n, w in enumerate([(1, 0, 0, 0), (0.5, 0.5, 0, 0)], 1):
        print(f"BLEU-{n}: {corpus_bleu(actual, predicted, weights=w):.4f}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32, help="images per step")
    ap.add_argument("--refit-tokenizer", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    features = extract_all_features()
    mapping = load_captions()
    tokenizer = get_tokenizer(mapping, args.refit_tokenizer)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for caps in mapping.values() for c in caps)
    print(f"vocab_size={vocab_size}  max_length={max_length}")

    train_keys = load_split("Flickr_8k.trainImages.txt")
    val_keys = load_split("Flickr_8k.devImages.txt")
    test_keys = load_split("Flickr_8k.testImages.txt")

    if args.eval_only:
        model = tf.keras.models.load_model(MODEL_PATH)
        evaluate(model, tokenizer, test_keys, mapping, features, max_length)
        return

    model = build_model(vocab_size, max_length)
    model.summary()

    train_ds = make_dataset(train_keys, mapping, features, tokenizer, max_length, vocab_size, args.batch_size)
    val_ds = make_dataset(val_keys, mapping, features, tokenizer, max_length, vocab_size, args.batch_size)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=math.ceil(len(train_keys) / args.batch_size),
        validation_steps=math.ceil(len(val_keys) / args.batch_size),
        callbacks=[
            ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ],
    )

    evaluate(model, tokenizer, test_keys, mapping, features, max_length)


if __name__ == "__main__":
    main()

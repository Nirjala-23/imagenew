# Image Captioning (Flickr8k) — CNN + LSTM

Upload an image in the browser and get a generated English caption.

**Pipeline:** VGG16 (pretrained, `fc2` layer → 4096-d image vector) + Embedding/LSTM text encoder → merged → softmax over the vocabulary → greedy word-by-word decoding from `startseq` to `endseq`.

```
image ──VGG16──► 4096-d ──Dense(256)──────────┐
                                              ├─ add ─► Dense(256) ─► softmax(vocab) ─► next word
"startseq a dog" ─Embedding─► LSTM(256) ──────┘
```

## Project structure

```
image-captioning/
├── main.py              # Flask app (inference)
├── train.py             # feature extraction + training + BLEU evaluation
├── requirements.txt
├── templates/
│   ├── home.html        # upload page
│   └── after.html       # result page
├── static/
│   ├── css/styles.css   # upload page styles
│   ├── css/second.css   # result page styles
│   └── uploads/         # uploaded images (git-ignored)
├── models/
│   ├── tokenizer.pkl    # fitted Keras Tokenizer
│   └── best.pkl         # trained model (add yourself, see models/README.md)
└── data/                # Flickr8k dataset (see data/README.md)
```

## Run the web app (Windows, PowerShell / CMD)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Then open http://127.0.0.1:5000. (Note: the correct command is `python main.py`, not `python -c main.py`.)

Make sure `models/best.pkl` (or `models/best.keras`) exists first.

## Train your own model

1. Put Flickr8k in `data/` as described in `data/README.md`.
2. Run:

```bash
python train.py
```

This caches VGG16 features, trains with early stopping, saves `models/best.keras`, and prints BLEU-1 / BLEU-2 on the test split. Evaluate an existing model with `python train.py --eval-only`.

## Model file

`best.pkl` is ~72 MB. GitHub accepts files under 100 MB (with a warning above 50 MB), so you can commit it, or use Git LFS, or attach it to a GitHub Release / Google Drive link and tell users to place it in `models/`.

## Limitations / future work

- Greedy decoding → add beam search.
- Frozen VGG16 features, no attention → try attention (Show-Attend-Tell) or a Transformer decoder.
- Small dataset (8k images) → try Flickr30k / MS-COCO, GloVe embeddings.

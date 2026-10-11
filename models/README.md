# Model files

| File | In git? | What it is |
|------|---------|-----------|
| `tokenizer.pkl` | yes (~327 KB) | Keras `Tokenizer` fitted on the Flickr8k captions (~8.5k words incl. `startseq`/`endseq`) |
| `best.pkl` | optional (~72 MB, under GitHub 100 MB limit) | Your original trained caption model |
| `best.keras` | no | Model produced by `python train.py` (alternative to `best.pkl`) |
| `features.pkl` | no | Cached VGG16 features, created by `train.py` |

`main.py` loads `best.keras` if present, otherwise `best.pkl`.
Put whichever file you have in this folder.

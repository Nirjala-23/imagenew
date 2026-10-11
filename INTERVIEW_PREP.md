# Interview Prep — Image Captioning (Sep'24 – Nov'24)

## 1. 30-second pitch

"I built an image-captioning system on Flickr8k, about 8,000 images with 5 human captions each. A pretrained VGG16 turns each image into a 4096-number feature vector. A second branch embeds the words generated so far and runs them through an LSTM. I merge the two and a softmax layer predicts the next word. At inference I start with `startseq` and feed the model its own predictions one word at a time until it outputs `endseq`. I wrapped it in a small Flask app where you upload an image and get a caption."

## 2. Your resume bullets — what each one really means in the code

**"Built an image-captioning pipeline using Flickr8K, pairing 8K images with multiple captions"**
8,091 images × 5 captions ≈ 40k (image, caption) pairs. Each caption of N tokens becomes N−1 training samples: (image, first i words) → word i+1.

**"Applied NLP preprocessing such as removing stop words, tokenization and count vectorization"** ⚠️ *check this one before the interview*
Your `tokenizer.pkl` contains `a`, `in`, `is`, `the`, `of`, … So **stop words were not removed** in the tokenizer you shipped. And the pipeline uses the Keras `Tokenizer` (word → integer id) feeding an `Embedding` layer, **not** count vectorization (bag-of-words). If you say "I removed stop words and used count vectorization", a good interviewer will ask why a bag-of-words works with an LSTM (it doesn't — it throws away word order), and the code will contradict you.
Safer, accurate wording:
- *Cleaning:* lowercase, strip punctuation/digits, add `startseq`/`endseq` markers.
- *Tokenization:* Keras `Tokenizer` builds a ~8.5k-word vocabulary and maps words to integer ids; sequences are padded to length 39.
- *Why I kept stop words:* captions need "a", "is", "on" to be grammatical; removing them would make the output sound like "dog running grass".
Suggested resume line: *"Preprocessed captions (cleaning, start/end tokens, Keras tokenization, padding) …"*. If you did try stop-word removal / count vectorization in an experiment, say that honestly as "I tried it in early experiments, but dropped it because…".

**"Built a model combining image embeddings using CNN layers and text embeddings using Embedding and LSTM layers"**
Precise version: VGG16 (a CNN) is used **pretrained and frozen** as a feature extractor (transfer learning), taking the `fc2` layer output (4096-d). Only the decoder (Dense, Embedding, LSTM, Dense) is trained. Say "pretrained CNN", not "I built CNN layers", unless you really trained one.

**"Trained for next-word prediction and generated captions using autoregressive decoding"**
Training uses teacher forcing (true previous words as input). Inference is greedy: `argmax` at each step, append, repeat until `endseq` or 39 words.

## 3. Architecture (the "merge" model)

| Part | Layer | Shape |
|------|-------|-------|
| Image input | VGG16 `fc2` features | 4096 |
| | Dropout(0.4) → Dense(256, relu) | 256 |
| Text input | padded word ids | 39 |
| | Embedding(vocab≈8.5k, 256, mask_zero) → Dropout → LSTM(256) | 256 |
| Merge | add (element-wise) → Dense(256, relu) | 256 |
| Output | Dense(vocab, softmax) | ~8.5k |

> `train.py` is a standard reconstruction of this architecture. Your original `best.pkl` may differ slightly (layer sizes, dropout). Print yours to be certain, from the project folder:
> ```bash
> python -c "import pickle; m=pickle.load(open('models/best.pkl','rb')); m.summary()"
> ```
> and be ready to quote *that* in the interview.

## 4. Questions you should expect (with short answers)

1. **Why VGG16 and not train a CNN?** 8k images is too few to train a CNN from scratch; ImageNet features transfer well. Frozen weights also let me cache features once, making training much faster.
2. **Why remove the last layer?** The final softmax predicts 1000 ImageNet classes; I want the 4096-d representation before it.
3. **Why an LSTM?** Caption words depend on previous words; LSTM keeps a memory of the sequence and avoids the vanishing gradient of a plain RNN.
4. **What is `startseq`/`endseq`?** Markers added to every caption: `startseq` seeds generation, `endseq` tells the decoder when to stop.
5. **Why pad and what is `mask_zero`?** Batches need equal length (39 = longest caption). `mask_zero=True` makes the LSTM ignore the padding zeros.
6. **What is teacher forcing?** During training the input is the ground-truth prefix, not the model's own previous prediction. At test time it uses its own predictions, which is why errors can compound (exposure bias).
7. **Greedy vs beam search?** Greedy takes the single best word each step and can get stuck in a poor sentence. Beam search keeps the top-k partial sentences and usually gives better captions. (Listed as future work.)
8. **How did you evaluate?** BLEU-1/BLEU-2 against the 5 reference captions (`python train.py --eval-only`). BLEU measures n-gram overlap; it's imperfect because many different captions are valid. Mention CIDEr/METEOR as better alternatives. *Run the evaluation and note your own numbers before the interview; typical VGG16+LSTM on Flickr8k gets BLEU-1 around 0.5–0.6.*
9. **Overfitting?** Dropout (0.4), early stopping on validation loss, `ModelCheckpoint` saves the best epoch (that's why the file is called `best`).
10. **Why `sparse_categorical_crossentropy`?** Labels are word indices; sparse loss avoids creating a (samples × 8.5k) one-hot matrix.
11. **Why is the dataset small an issue?** Vocabulary is limited and captions are generic ("a dog is running through the grass"). Bigger data (Flickr30k, COCO) or attention/Transformers would help.
12. **How would you improve it?** Beam search, attention over spatial CNN features, ResNet/InceptionV3/EfficientNet or CLIP encoder, GloVe embeddings, Transformer decoder.
13. **Deployment questions (about `main.py`):** models are loaded once at startup (not per request); VGG16 + caption model are separate; uploads are saved with a sanitized unique filename; file-type and size are validated; reverse word index is built once for O(1) lookup.

## 5. Code walkthrough cheat-sheet

- `extract_features()` → image → 224×224 → `preprocess_input` → VGG16 → 4096-d vector.
- `predict_caption()` → loop: tokenize text so far → pad to 39 → `model.predict([feature, seq])` → `argmax` → word → append; stop at `endseq`.
- `clean_caption()` → strips `startseq` / `endseq` for display.
- Flask routes: `/` renders the upload form; `/upload` saves the file, runs the pipeline, renders `after.html`.

## 6. Things worth fixing/knowing about the original `main.py` (already fixed in this version)

- `generate_caption(..., model, ...)` used the global `model_caption` inside, which is confusing; now cleaner.
- The tokenizer was reloaded from disk on every request; now loaded once.
- `idx_to_word` looped over the whole vocabulary every step; now a dictionary.
- Uploaded filenames were used raw (overwrite / path-traversal risk); now sanitized + UUID.
- README said `python -c main.py`; the right command is `python main.py`.

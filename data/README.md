# Dataset (not stored in git)

Download Flickr8k (images + text) from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k
or the official Flickr8k release, then arrange it like this:

```
data/
├── Images/                      # 8,091 .jpg files
├── Flickr8k.token.txt           # captions (5 per image): "<img>.jpg#0<TAB>caption"
├── Flickr_8k.trainImages.txt    # 6,000 train image names
├── Flickr_8k.devImages.txt      # 1,000 validation image names
└── Flickr_8k.testImages.txt     # 1,000 test image names
```

If your download names the caption file `captions.txt` (Kaggle version with a CSV header),
the official `Flickr8k.token.txt` + split files are available in the "Flickr8k_text" archive.

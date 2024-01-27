# VALL-E-X Multilanguage

This is an attempt to train Vall-E-X on a set of multiple languages, after finding out that the 
mozilla commonvoice dataset does not provide enough data for a good convergence rate, neither in default VALL-E,
nor in VALL-E-X using language embeddings. Therefore, we'll mix different datasets in this collection,
to significantly increase the total amount of speakers and audio duration.

The goal is to achieve ~10k hours of audio for each language that's part of the training, so we exceed the
baseline of VALL-E-X's 70k audio hours (https://arxiv.org/pdf/2303.03926.pdf)

### Legend
- Each dataset listed below is categorized by Size

### Included Languages and Datasets
```
# Stats:
- Total hours: 186.610 audio hours
- English: 33.726 audio hours
- German: 26.542 audio hours
- French: 31.637 audio hours
- Italian: 22.524 audio hours
- Arabic: ~1.300 audio hours
- Spanish: 24.492 audio hours
- Russian: 1592 audio hours
- Simplified Chinese: 25.571 audio hours
- Japanese: 19.226 audio hours

# Engish
- Mozilla CommonVoice 14.0 (77,82GB - 3.279 audio hours - 88.154 speakers - Transcribed)
- Librilight Small (35GB - 577 audio hours - 489 speakers - No Transcript)
- Librilight Medium (321GB - 5770 audio hours - 1742 speakers - No Transcript)
- VoxPopuli (321GB - 24100 audio hours - 1313+ speakers - No Transcript)

# German
- Mozilla CommonVoice 14.0 (32,25GB - 1.376 audio hours - 18.187 speakers - Transcribed)
- Multilingual LibriSpeech (SLR94) (115GB - 1966 audio hours - 176 speakers - Transcribed)
- VoxPopuli (379GB - 23200 audio hours - 531+ speakers - No Transcript)

# French
- Mozilla CommonVoice 14.0 (25,6GB - 1.079 audio hours - 17.761 speakers - Transcribed)
- Multilingual LibriSpeech (SLR94) (61GB - 1076 audio hours - 142 speakers - Transcribed)
- Audiocite.net (SLR139) (329,4GB - 6682 audio hours - 130 speakers - No Transcript)
- VoxPopuli (376GB - 22800 audio hours - 534+ speakers - No Transcript)

# Italian
- Mozilla CommonVoice 14.0 (8,65GB - 377 audio hours - 6.930 speakers - Transcribed)
- Multilingual LibriSpeech (SLR94) (15GB - 247 audio hours - 65 speakers - Transcribed)
- VoxPopuli (361GB - 21900 audio hours - 306+ speakers - No Transcript)

# Arabic
- Mozilla CommonVoice 14.0 (3,05GB - 154 audio hours - 1.466 speakers - Transcribed)
- Mohammed (SLR132) (24GB - XX audio hours - 40 speakers)
- [Unable to access - QASR (SLR132) (XXGB - 2.041 audio hours - 27977 speakers - Transcribed)]
- MASC (https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus) (169GB - 1.000 audio hours - XX speakers - Transcribed)

# Spanish
- Mozilla CommonVoice 14.0 (45,98GB - 2.175 audio hours - 25.261 speakers - Transcribed)
- Multilingual LibriSpeech (SLR94) (50GB - 917 audio hours - 86 speakers - Transcribed)
- VoxPopuli (362GB - 21400 audio hours - 305+ speakers - No Transcript)

# Russian
- Mozilla CommonVoice 14.0 (5,81GB - 254 audio hours - 3.001 speakers - Transcribed)
- Russian LibriSpeech (SLR96) (9,1GB - 98 audio hours - XX speakers - Transcribed)
- Golos (SLR114) (18GB - 1240 audio hours - XX speakers - Transcribed)

# Simplified Chinese
- Mozilla CommonVoice 14.0 (20,87GB - 1.053 audio hours - 6.764 speakers - Transcribed)
- MAGICDATA (SLR68) (52GB - 755 audio hours - 1080 speakers - Transcribed)
- CN-Celeb 1 (SLR82) (22GB - 273 audio hours - 1.000 speakers - No Transcript)
- CN-Celeb 2 (SLR82) (75GB - 1090 audio hours - 2.000 speakers - No Transcript)
- WenetSpeech (SLR121) (500GB - 22.400 audio hours - XX speakers - Partly Transcribed)

# Japanese
- Mozilla CommonVoice 14.0 (4,47GB - 226 audio hours - 1.707 speakers - Transcribed)
- ReazonSpeech (https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/A5-3.pdf) (~1300GB - 19.000 audio hours - XX speakers - Transcribed)

```

## Install deps
```
pip install librosa==0.8.1 matplotlib h5py 
```

## Prepare Dataset
```
cd egs/commonvoice
```

# Those stages are very time-consuming
```
bash prepare.sh --stage -1 --stop-stage 3
```




## Training & Inference
refer to [Training](../../README.md##Training&Inference)

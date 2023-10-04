#!/bin/bash

# Downloads and extracts all required datasets
# Further preparation happens in python scripts

# Basic setup
dl_dir=$PWD/datasets

# Corpus passwords
wenet_speech_pass=""

### Preparations
sudo apt update
sudo apt install unzip git huggingface-cli
# Huggingface
huggingface-cli login
# WenetSpeech
git clone https://github.com/wenet-e2e/WenetSpeech.git repos/WenetSpeech
echo $wenet_speech_pass > repos/WenetSpeech/SAFEBOX/password
# VoxPopuli
git clone https://github.com/facebookresearch/voxpopuli.git repos/VoxPopuli
pip install -r repos/VoxPopuli/requirements.txt

### Dataset Downloads

## English datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/en/clips ]; then
  wget -c -O $dl_dir/commonvoice_en.tar.gz  https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-en.tar.gz
  tar xvf $dl_dir/commonvoice_en.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_en.tar.gz
fi
# Librilight Small
if [ ! -d $dl_dir/librilight_small ]; then
  wget -c -O $dl_dir/librilight_small.tar https://dl.fbaipublicfiles.com/librilight/data/small.tar
  mkdir -p  -C $dl_dir/librilight
  tar xvf $dl_dir/librilight_small.tar -C $dl_dir/librilight
  mv small librilight_small
  rm $dl_dir/librilight_small.tar
fi
# Librilight Medium
if [ ! -d $dl_dir/librilight_medium ]; then
  wget -c -O $dl_dir/librilight_medium.tar https://dl.fbaipublicfiles.com/librilight/data/medium.tar
  mkdir -p  -C $dl_dir/librilight
  tar xvf $dl_dir/librilight_medium.tar -C $dl_dir/librilight
  mv medium librilight_medium
  rm $dl_dir/librilight_medium.tar
fi
# Librilight Large
if [ ! -d $dl_dir/librilight_large ]; then
  wget -c -O $dl_dir/librilight_large.tar https://dl.fbaipublicfiles.com/librilight/data/large.tar
  mkdir -p  -C $dl_dir/librilight
  tar xvf $dl_dir/librilight_large.tar -C $dl_dir/librilight
  mv medium librilight_large
  rm $dl_dir/librilight_large.tar
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/en ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset en_v2
fi

## German datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/de/clips ]; then
  wget -c -O $dl_dir/commonvoice_de.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-de.tar.gz
  tar xvf $dl_dir/commonvoice_de.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_de.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_german ]; then
  wget -c -O $dl_dir/mls_german.tar.gz https://dl.fbaipublicfiles.com/mls/mls_german.tar.gz
  tar xvf $dl_dir/mls_german.tar.gz -C $dl_dir/
  rm $dl_dir/mls_german.tar.gz
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/de ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset de_v2
fi

## French datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/fr/clips ]; then
  wget -c -O $dl_dir/commonvoice_fr.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-fr.tar.gz
  tar xvf $dl_dir/commonvoice_fr.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_fr.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_french ]; then
  wget -c -O $dl_dir/mls_french.tar.gz https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz
  tar xvf $dl_dir/mls_french.tar.gz -C $dl_dir/
  rm $dl_dir/mls_french.tar.gz
fi
# Audiocite.net (SLR139)
if [ ! -d $dl_dir/audiocite.net ]; then
  wget -c -O $dl_dir/audiocite.net_0.zip https://www.openslr.org/resources/139/audiocite.net_0.zip
  wget -c -O $dl_dir/audiocite.net_1.zip https://www.openslr.org/resources/139/audiocite.net_1.zip
  wget -c -O $dl_dir/audiocite.net_2.zip https://www.openslr.org/resources/139/audiocite.net_2.zip
  wget -c -O $dl_dir/audiocite.net_3.zip https://www.openslr.org/resources/139/audiocite.net_3.zip
  wget -c -O $dl_dir/audiocite.net_4.zip https://www.openslr.org/resources/139/audiocite.net_4.zip
  wget -c -O $dl_dir/audiocite.net_5.zip https://www.openslr.org/resources/139/audiocite.net_5.zip
  wget -c -O $dl_dir/audiocite.net_6.zip https://www.openslr.org/resources/139/audiocite.net_6.zip
  wget -c -O $dl_dir/audiocite.net_7.zip https://www.openslr.org/resources/139/audiocite.net_7.zip
  wget -c -O $dl_dir/audiocite.net_8.zip https://www.openslr.org/resources/139/audiocite.net_8.zip
  wget -c -O $dl_dir/audiocite.net_9.zip https://www.openslr.org/resources/139/audiocite.net_9.zip
  wget -c -O $dl_dir/audiocite.net_10.zip https://www.openslr.org/resources/139/audiocite.net_10.zip
  wget -c -O $dl_dir/audiocite.net_11.zip https://www.openslr.org/resources/139/audiocite.net_11.zip
  wget -c -O $dl_dir/audiocite.net_12.zip https://www.openslr.org/resources/139/audiocite.net_12.zip
  wget -c -O $dl_dir/audiocite.net_13.zip https://www.openslr.org/resources/139/audiocite.net_13.zip
  wget -c -O $dl_dir/audiocite.net_14.zip https://www.openslr.org/resources/139/audiocite.net_14.zip
  wget -c -O $dl_dir/audiocite.net_15.zip https://www.openslr.org/resources/139/audiocite.net_15.zip
  wget -c -O $dl_dir/audiocite.net_16.zip https://www.openslr.org/resources/139/audiocite.net_16.zip
  wget -c -O $dl_dir/audiocite.net_17.zip https://www.openslr.org/resources/139/audiocite.net_17.zip
  wget -c -O $dl_dir/audiocite.net_18.zip https://www.openslr.org/resources/139/audiocite.net_18.zip
  wget -c -O $dl_dir/audiocite.net_19.zip https://www.openslr.org/resources/139/audiocite.net_19.zip
  wget -c -O $dl_dir/audiocite.net_20.zip https://www.openslr.org/resources/139/audiocite.net_20.zip
  wget -c -O $dl_dir/audiocite.net_21.zip https://www.openslr.org/resources/139/audiocite.net_21.zip
  wget -c -O $dl_dir/audiocite.net_22.zip https://www.openslr.org/resources/139/audiocite.net_22.zip
  wget -c -O $dl_dir/audiocite.net_23.zip https://www.openslr.org/resources/139/audiocite.net_23.zip
  wget -c -O $dl_dir/audiocite.net_24.zip https://www.openslr.org/resources/139/audiocite.net_24.zip
  wget -c -O $dl_dir/audiocite.net_25.zip https://www.openslr.org/resources/139/audiocite.net_25.zip
  wget -c -O $dl_dir/audiocite.net_26.zip https://www.openslr.org/resources/139/audiocite.net_26.zip
  wget -c -O $dl_dir/audiocite.net_27.zip https://www.openslr.org/resources/139/audiocite.net_27.zip
  wget -c -O $dl_dir/audiocite.net_28.zip https://www.openslr.org/resources/139/audiocite.net_28.zip
  wget -c -O $dl_dir/audiocite.net_29.zip https://www.openslr.org/resources/139/audiocite.net_29.zip
  wget -c -O $dl_dir/audiocite.net_30.zip https://www.openslr.org/resources/139/audiocite.net_30.zip
  wget -c -O $dl_dir/audiocite.net_31.zip https://www.openslr.org/resources/139/audiocite.net_31.zip
  wget -c -O $dl_dir/audiocite.net_32.zip https://www.openslr.org/resources/139/audiocite.net_32.zip
  wget -c -O $dl_dir/audiocite.net_33.zip https://www.openslr.org/resources/139/audiocite.net_33.zip
  wget -c -O $dl_dir/audiocite.net_34.zip https://www.openslr.org/resources/139/audiocite.net_34.zip
  unzip $dl_dir/audiocite.net_0.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_1.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_2.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_3.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_4.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_5.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_6.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_7.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_8.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_9.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_10.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_11.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_12.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_13.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_14.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_15.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_16.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_17.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_18.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_19.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_20.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_21.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_22.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_23.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_24.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_25.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_26.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_27.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_28.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_29.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_30.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_31.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_32.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_33.zip -d $dl_dir/audiocite.net
  unzip $dl_dir/audiocite.net_34.zip -d $dl_dir/audiocite.net
  rm $dl_dir/audiocite.net_*.zip
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/fr ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset fr_v2
fi

## Italian datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/it/clips ]; then
  wget -c -O $dl_dir/commonvoice_it.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-it.tar.gz
  tar xvf $dl_dir/commonvoice_it.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_it.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_italian ]; then
  wget -c -O $dl_dir/mls_italian.tar.gz https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz
  tar xvf $dl_dir/mls_italian.tar.gz -C $dl_dir/
  rm $dl_dir/mls_italian.tar.gz
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/it ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset it_v2
fi

## Arabic datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/ar/clips ]; then
  wget -c -O $dl_dir/commonvoice_ar.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-ar.tar.gz
  tar xvf $dl_dir/commonvoice_ar.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_ar.tar.gz
fi
# Mohammed (SLR132)
if [ ! -d $dl_dir/mohammed ]; then
  wget -c -O $dl_dir/Quran_Speech_Dataset.tar.xz https://www.openslr.org/resources/132/Quran_Speech_Dataset.tar.xz
  mkdir -p $dl_dir/Quran_Speech_Dataset
  tar xvf $dl_dir/Quran_Speech_Dataset.tar.xz -C $dl_dir/Quran_Speech_Dataset
  rm $dl_dir/mohammed.tar.gx
fi

## Spanish datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/es/clips ]; then
  wget -c -O $dl_dir/commonvoice_es.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-es.tar.gz
  tar xvf $dl_dir/commonvoice_es.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_es.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_spanish ]; then
  wget -c -O $dl_dir/mls_spanish.tar.gz https://dl.fbaipublicfiles.com/mls/mls_spanish.tar.gz
  tar xvf $dl_dir/mls_spanish.tar.gz -C $dl_dir/
  rm $dl_dir/mls_spanish.tar.gz
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/es ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset es_v2
fi

## Russian datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/ru/clips ]; then
  wget -c -O $dl_dir/commonvoice_ru.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-ru.tar.gz
  tar xvf $dl_dir/commonvoice_ru.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_ru.tar.gz
fi
# Russian LibriSpeech (SLR96)
if [ ! -d $dl_dir/ruls_data ]; then
  wget -c -O $dl_dir/ruls_data.tar.gz https://openslr.elda.org/resources/96/ruls_data.tar.gz
  mkdir -p $dl_dir/ruls
  tar xvf $dl_dir/ruls_data.tar.gz -C $dl_dir/ruls
  rm $dl_dir/ruls_data.tar.gz
fi
# Golos (SLR114)
if [ ! -d $dl_dir/golos ]; then
  wget -c -O $dl_dir/golos.tar.gz https://www.openslr.org/resources/114/golos_opus.tar.gz
  mkdir -p $dl_dir/golos
  tar xvf $dl_dir/golos.tar.gz -C $dl_dir/golos
  rm $dl_dir/golos.tar.gz
fi

## Simplified Chinese datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/zh-CN/clips ]; then
  wget -c -O $dl_dir/commonvoice_zh-CN.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-zh-CN.tar.gz
  tar xvf $dl_dir/commonvoice_zh-CN.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_zh-CN.tar.gz
fi
# CN-Celeb 1 (SLR82)
if [ ! -d $dl_dir/cn_celeb ]; then
  wget -c -O $dl_dir/cn_celeb.tar.gz https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz
  mkdir -p $dl_dir/cn_celeb
  tar xvf $dl_dir/cn_celeb.tar.gz -C $dl_dir/cn_celeb
  rm $dl_dir/cn_celeb.tar.gz
fi
# CN-Celeb 2 (SLR82)
if [ ! -d $dl_dir/cn_celeb_2 ]; then
  wget -c -O $dl_dir/cn_celeb_2.tar.gzaa https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzaa
  wget -c -O $dl_dir/cn_celeb_2.tar.gzab https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzab
  wget -c -O $dl_dir/cn_celeb_2.tar.gzac https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzac
  cat $dl_dir/cn_celeb_2.tar.gzab >> $dl_dir/cn_celeb_2.tar.gzaa
  cat $dl_dir/cn_celeb_2.tar.gzac >> $dl_dir/cn_celeb_2.tar.gzaa
  mv $dl_dir/cn_celeb_2.tar.gzaa $dl_dir/cn_celeb_2.tar.gz
  mkdir -p $dl_dir/cn_celeb_2
  tar xvf $dl_dir/cn_celeb_2.tar.gz -C $dl_dir/cn_celeb_2
  rm $dl_dir/cn_celeb_2.tar.gz*
fi
# MAGICDATA (SLR68)
if [ ! -d $dl_dir/magicdata ]; then
  wget -c -O $dl_dir/magicdata_dev.tar.gz https://www.openslr.org/resources/68/dev_set.tar.gz
  wget -c -O $dl_dir/magicdata_test.tar.gz https://www.openslr.org/resources/68/test_set.tar.gz
  wget -c -O $dl_dir/magicdata_train.tar.gz https://www.openslr.org/resources/68/train_set.tar.gz
  mkdir -p $dl_dir/magicdata
  tar xvf $dl_dir/magicdata_dev.tar.gz -C $dl_dir/magicdata
  tar xvf $dl_dir/magicdata_test.tar.gz -C $dl_dir/magicdata
  tar xvf $dl_dir/magicdata_train.tar.gz -C $dl_dir/magicdata
  rm $dl_dir/magicdata_train.tar.gz
  rm $dl_dir/magicdata_dev.tar.gz
  rm $dl_dir/magicdata_test.tar.gz
fi
# WenetSpeech (SLR121)
if [ ! -d $dl_dir/wenet_speech ]; then
  bash repos/WenetSpeech/utils/download_wenetspeech.sh $dl_dir $dl_dir/wenet_speech
  # rm $dl_dir/cn_celeb.tar.gz
fi

## Japanese datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/ja/clips ]; then
  wget -c -O $dl_dir/commonvoice_ja.tar.gz https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-ja.tar.gz
  tar xvf $dl_dir/commonvoice_ja.tar.gz -C $dl_dir/
  rm $dl_dir/commonvoice_ja.tar.gz
fi
# ReazonSpeech
python scripts/reazonspeech.py $dl_dir
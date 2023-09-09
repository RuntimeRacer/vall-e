#!/bin/bash

# Downloads and extracts all required datasets
# Further preparation happens in python scripts

# Basic setup
dl_dir=$PWD/datasets

# Corpus passwords
wenet_speech_pass=""

### Preparations
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
  wget -c -O commonvoice_en.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-en.tar.gz
  tar xvf $dl_dir/commonvoice_en.tar.gz
#  rm $dl_dir/commonvoice_en.tar.gz
fi
# Librilight Small
if [ ! -d $dl_dir/librilight_small ]; then
  wget -c -O librilight_small.tar -P $dl_dir/ https://dl.fbaipublicfiles.com/librilight/data/small.tar
  tar xvf $dl_dir/librilight_small.tar
  mv small librilight_small
#  rm $dl_dir/librilight_small.tar
fi
# Librilight Medium
if [ ! -d $dl_dir/librilight_medium ]; then
  wget -c -O librilight_medium.tar -P $dl_dir/ https://dl.fbaipublicfiles.com/librilight/data/medium.tar
  tar xvf $dl_dir/librilight_medium.tar
  mv medium librilight_medium
#  rm $dl_dir/librilight_medium.tar
fi
# Librilight Large
#wget -c -O librilight_small.tar -P $dl_dir/ https://dl.fbaipublicfiles.com/librilight/data/large.tar
#tar xvf  $dl_dir/librilight_large.tar
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/en ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset en_v2
fi

## German datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/de/clips ]; then
  wget -c -O commonvoice_de.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-de.tar.gz
  tar xvf $dl_dir/commonvoice_de.tar.gz
#  rm $dl_dir/commonvoice_de.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_german ]; then
  wget -c -O mls_german.tar.gz -P $dl_dir/ https://dl.fbaipublicfiles.com/mls/mls_german.tar.gz
  tar xvf $dl_dir/mls_german.tar.gz
#  rm $dl_dir/mls_german.tar.gz
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/de ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset de_v2
fi

## French datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/fr/clips ]; then
  wget -c -O commonvoice_fr.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-fr.tar.gz
  tar xvf $dl_dir/commonvoice_fr.tar.gz
#  rm $dl_dir/commonvoice_fr.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_french ]; then
  wget -c -O mls_french.tar.gz -P $dl_dir/ https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz
  tar xvf $dl_dir/mls_french.tar.gz
#  rm $dl_dir/mls_french.tar.gz
fi
# Audiocite.net (SLR139)
if [ ! -d $dl_dir/audiocite.net ]; then
  wget -c -O audiocite.net_0.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_0.zip
  wget -c -O audiocite.net_1.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_1.zip
  wget -c -O audiocite.net_2.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_2.zip
  wget -c -O audiocite.net_3.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_3.zip
  wget -c -O audiocite.net_4.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_4.zip
  wget -c -O audiocite.net_5.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_5.zip
  wget -c -O audiocite.net_6.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_6.zip
  wget -c -O audiocite.net_7.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_7.zip
  wget -c -O audiocite.net_8.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_8.zip
  wget -c -O audiocite.net_9.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_9.zip
  wget -c -O audiocite.net_10.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_10.zip
  wget -c -O audiocite.net_11.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_11.zip
  wget -c -O audiocite.net_12.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_12.zip
  wget -c -O audiocite.net_13.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_13.zip
  wget -c -O audiocite.net_14.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_14.zip
  wget -c -O audiocite.net_15.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_15.zip
  wget -c -O audiocite.net_16.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_16.zip
  wget -c -O audiocite.net_17.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_17.zip
  wget -c -O audiocite.net_18.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_18.zip
  wget -c -O audiocite.net_19.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_19.zip
  wget -c -O audiocite.net_20.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_20.zip
  wget -c -O audiocite.net_21.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_21.zip
  wget -c -O audiocite.net_22.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_22.zip
  wget -c -O audiocite.net_23.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_23.zip
  wget -c -O audiocite.net_24.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_24.zip
  wget -c -O audiocite.net_25.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_25.zip
  wget -c -O audiocite.net_26.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_26.zip
  wget -c -O audiocite.net_27.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_27.zip
  wget -c -O audiocite.net_28.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_28.zip
  wget -c -O audiocite.net_29.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_29.zip
  wget -c -O audiocite.net_30.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_30.zip
  wget -c -O audiocite.net_31.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_31.zip
  wget -c -O audiocite.net_32.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_32.zip
  wget -c -O audiocite.net_33.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_33.zip
  wget -c -O audiocite.net_34.zip -P $dl_dir/ https://www.openslr.org/resources/139/audiocite.net_34.zip
  unzip $dl_dir/audiocite.net_*.zip $dl_dir/audiocite.net
#  rm $dl_dir/audiocite.net_*.zip
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/fr ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset fr_v2
fi

## Italian datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/it/clips ]; then
  wget -c -O commonvoice_it.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-it.tar.gz
  tar xvf $dl_dir/commonvoice_it.tar.gz
#  rm $dl_dir/commonvoice_it.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_italian ]; then
  wget -c -O mls_italian.tar.gz -P $dl_dir/ https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz
  tar xvf $dl_dir/mls_italian.tar.gz
#  rm $dl_dir/mls_italian.tar.gz
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/it ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset it_v2
fi

## Arabic datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/ar/clips ]; then
  wget -c -O commonvoice_ar.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-ar.tar.gz
  tar xvf $dl_dir/commonvoice_ar.tar.gz
#  rm $dl_dir/commonvoice_ar.tar.gz
fi
# Mohammed (SLR132)
if [ ! -d $dl_dir/mohammed ]; then
  wget -c -O mohammed.tar.gx -P $dl_dir/ https://www.openslr.org/resources/132/Quran_Speech_Dataset.tar.xz
  tar xvf $dl_dir/mohammed.tar.xz
#  rm $dl_dir/mohammed.tar.gx
fi

## Spanish datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/es/clips ]; then
  wget -c -O commonvoice_es.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-es.tar.gz
  tar xvf $dl_dir/commonvoice_es.tar.gz
#  rm $dl_dir/commonvoice_es.tar.gz
fi
# Multilingual LibriSpeech (SLR94)
if [ ! -d $dl_dir/mls_spanish ]; then
  wget -c -O mls_spanish.tar.gz -P $dl_dir/ https://dl.fbaipublicfiles.com/mls/mls_spanish.tar.gz
  tar xvf $dl_dir/mls_spanish.tar.gz
#  rm $dl_dir/mls_spanish.tar.gz
fi
# VoxPopuli
if [ ! -d $dl_dir/voxpopuli/raw_audios/es ]; then
  python -m repos.VoxPopuli.voxpopuli.download_audios --root $dl_dir/voxpopuli --subset es_v2
fi

## Russian datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/ru/clips ]; then
  wget -c -O commonvoice_ru.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-ru.tar.gz
  tar xvf $dl_dir/commonvoice_ru.tar.gz
#  rm $dl_dir/commonvoice_ru.tar.gz
fi
# Russian LibriSpeech (SLR96)
if [ ! -d $dl_dir/ruls_data ]; then
  wget -c -O ruls_data.tar.gz -P $dl_dir/ https://openslr.elda.org/resources/96/ruls_data.tar.gz
  tar xvf $dl_dir/ruls_data.tar.gz
#  rm $dl_dir/ruls_data.tar.gz
fi
# Golos (SLR114)
if [ ! -d $dl_dir/golos ]; then
  wget -c -O golos.tar.gz -P $dl_dir/ https://www.openslr.org/resources/114/golos_opus.tar.gz
  tar xvf $dl_dir/golos.tar.gz
#  rm $dl_dir/golos.tar.gz
fi

## Simplified Chinese datasets
# Commonvoice 14
if [ ! -d $dl_dir/cv-corpus-14.0-2023-06-23/zh-CN/clips ]; then
  wget -c -O commonvoice_zh-CN.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-zh-CN.tar.gz
  tar xvf $dl_dir/commonvoice_zh-CN.tar.gz
#  rm $dl_dir/commonvoice_zh-CN.tar.gz
fi
# CN-Celeb 1 (SLR82)
if [ ! -d $dl_dir/cn_celeb ]; then
  wget -c -O cn_celeb.tar.gz -P $dl_dir/ https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz
  mkdir -p $dl_dir/cn_celeb
  tar xvf $dl_dir/cn_celeb.tar.gz -C $dl_dir/cn_celeb
  rm $dl_dir/cn_celeb.tar.gz
fi
# CN-Celeb 2 (SLR82)
if [ ! -d $dl_dir/cn_celeb_2 ]; then
  wget -c -O cn_celeb_2.tar.gzaa -P $dl_dir/ https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzaa
  wget -c -O cn_celeb_2.tar.gzab -P $dl_dir/ https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzab
  wget -c -O cn_celeb_2.tar.gzac -P $dl_dir/ https://www.openslr.org/resources/82/cn-celeb2_v2.tar.gzac
  cat $dl_dir/cn_celeb_2.tar.gzab >> $dl_dir/cn_celeb_2.tar.gzaa
  cat $dl_dir/cn_celeb_2.tar.gzac >> $dl_dir/cn_celeb_2.tar.gzaa
  mv $dl_dir/cn_celeb_2.tar.gzaa $dl_dir/cn_celeb_2.tar.gz
  mkdir -p $dl_dir/cn_celeb_2
  tar xvf $dl_dir/cn_celeb_2.tar.gz -C $dl_dir/cn_celeb_2
  rm $dl_dir/cn_celeb_2.tar.gz
fi
# MAGICDATA (SLR68)
if [ ! -d $dl_dir/magicdata ]; then
  wget -c -O magicdata_dev.tar.gz -P $dl_dir/ https://www.openslr.org/resources/68/dev_set.tar.gz
  wget -c -O magicdata_test.tar.gz -P $dl_dir/ https://www.openslr.org/resources/68/test_set.tar.gz
  wget -c -O magicdata_train.tar.gz -P $dl_dir/ https://www.openslr.org/resources/68/train_set.tar.gz
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
  wget -c -O commonvoice_ja.tar.gz -P $dl_dir/ https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-ja.tar.gz
  tar xvf $dl_dir/commonvoice_ja.tar.gz
#  rm $dl_dir/commonvoice_ja.tar.gz
fi
# ReazonSpeech
python scripts/reazonspeech.py $dl_dir
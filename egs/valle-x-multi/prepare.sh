#!/bin/bash

set -eou pipefail

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

stage=-1
stop_stage=4

# languages="en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk ja"
# Use simplified corpus for POC first
languages="ar en de fr it es ru zh-CN ja"

dataset_parts="train dev test"

audio_extractor="Encodec"  # or Fbank
audio_feats_dir=data/tokenized

. shared/parse_options.sh || exit 1

# Stage 0: Organize Datasets
# 1. Use download.sh and manual downloading to collect source datasets
#
# 2. Transcribe source datasets:
#   - Use Whisper or preferred ASR Pipeline to create transcriptions for datasets which have no transcripts
#   - Make sure to cut audios to proper lenght before or during transcription processing.
#   - Use Scripts in ./scripts folder to process various transcription formats.
#   - Make sure for each audio file there is a '*_transcript.txt' with transcription
#   - Run ./scripts/lhotse_manifests_create_from_data.py to create recording + supervision-sets

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Tokenize/Fbank for vall-e-x"

  # Build all Parts for all the datasets
  languagePartList=""
  for lang in $languages
  do
    languagePartList+="${lang}"
  done
  # echo "${languagePartList}" # debug

  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.vall-e-x.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "${languagePartList}" \
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --prefix "vall-e-x" \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}" \
        --convert-to-ascii true \
        --threads-per-device 8
  fi
  touch ${audio_feats_dir}/.vall-e-x.tokenize.done
fi
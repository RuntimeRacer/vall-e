#!/usr/bin/env bash

set -eou pipefail

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=3

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LJSpeech-1.1

dl_dir=$PWD/download

audio_extractor="Encodec"  # or Fbank
audio_feats_dir=data/tokenized


. shared/parse_options.sh || exit 1


# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LJSpeech,
  # you can create a symlink
  #
  #   ln -sfv /path/to/LJSpeech $dl_dir/LJSpeech
  #
  if [ ! -d $dl_dir/LJSpeech-1.1 ];then
    lhotse download ljspeech $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare LJSpeech manifest"
  # We assume that you have downloaded the LJSpeech corpus
  # to $dl_dir/LJSpeech
  mkdir -p data/manifests
  if [ ! -e data/manifests/.ljspeech.done ]; then
    lhotse prepare ljspeech $dl_dir/LJSpeech-1.1 data/manifests
    touch data/manifests/.ljspeech.done
  fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Split LJSpeech"

  # 13100 = dev/test/train = 200/200/12500
  if [ ! -e data/manifests/ljspeech_recordings_test.jsonl.gz ]; then
    for manifest in "recordings" "supervisions";do
      lhotse subset --last 600 data/manifests/ljspeech_${manifest}_all.jsonl.gz \
        data/manifests/ljspeech_${manifest}_dev_test.jsonl.gz || exit 1
      lhotse subset --last 400 data/manifests/ljspeech_${manifest}_dev_test.jsonl.gz \
        data/manifests/ljspeech_${manifest}_test.jsonl.gz || exit 1
      lhotse subset --first 200 data/manifests/ljspeech_${manifest}_dev_test.jsonl.gz \
        data/manifests/ljspeech_${manifest}_dev.jsonl.gz || exit 1

      lhotse subset --first 12500 data/manifests/ljspeech_${manifest}_all.jsonl.gz \
        data/manifests/ljspeech_${manifest}_train.jsonl.gz || exit 1

      rm -f data/manifests/ljspeech_${manifest}_dev_test.jsonl.gz
    done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: ${audio_extractor} LJSpeech"

  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.ljspeech.done ]; then
    python3 bin/tokenizer.py --dataset-parts "train test dev" --prefix "ljspeech" \
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}"
  fi
  touch ${audio_feats_dir}/.ljspeech.done

  cd ${audio_feats_dir}
  ln -sf ljspeech_cuts_train.jsonl.gz cuts_train.jsonl.gz
  ln -sf ljspeech_cuts_dev.jsonl.gz cuts_dev.jsonl.gz
  ln -sf ljspeech_cuts_test.jsonl.gz cuts_test.jsonl.gz
  cd -
fi
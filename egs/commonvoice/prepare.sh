#!/usr/bin/env bash

set -eou pipefail

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

nj=16
stage=-1
stop_stage=4

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/aishell
#      You can download aishell from https://www.openslr.org/33/
#

dl_dir=$PWD/download
release=cv-corpus-13.0-2023-03-09
lang="en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk ja"

dataset_parts="-p train -p dev -p test"  # debug

text_extractor="pypinyin_initials_finals"
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

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "dl_dir: $dl_dir"
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/commonvoice,
  # you can create a symlink
  #
  #   ln -sfv /path/to/aishell $dl_dir/aishell
  #
  if [ ! -d $dl_dir/$release/$lang/clips ]; then
    lhotse download commonvoice --languages $lang --release $release $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare commonvoice manifest"
  # We assume that you have downloaded the commonvoice corpus
  # to $dl_dir/commonvoice
  mkdir -p data/manifests
  if [ ! -e data/manifests/.commonvoice.done ]; then
    lhotse prepare commonvoice --languages $lang -j $nj $dl_dir/$release data/manifests
    touch data/manifests/.commonvoice.done
  fi
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Tokenize/Fbank commonvoice"
  mkdir -p ${audio_feats_dir}
  if [ ! -e ${audio_feats_dir}/.commonvoice.tokenize.done ]; then
    python3 bin/tokenizer.py --dataset-parts "${dataset_parts}" \
        --text-extractor ${text_extractor} \
        --audio-extractor ${audio_extractor} \
        --batch-duration 400 \
        --src-dir "data/manifests" \
        --output-dir "${audio_feats_dir}"
  fi
  touch ${audio_feats_dir}/.commonvoice.tokenize.done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare commonvoice train/dev/test"
  if [ ! -e ${audio_feats_dir}/.commonvoice.train.done ]; then
    # dev 14326
    lhotse subset --first 400 \
        ${audio_feats_dir}/commonvoice_cuts_dev.jsonl.gz \
        ${audio_feats_dir}/cuts_dev.jsonl.gz

    lhotse subset --last 13926 \
        ${audio_feats_dir}/commonvoice_cuts_dev.jsonl.gz \
        ${audio_feats_dir}/cuts_dev_others.jsonl.gz

    # train
    lhotse combine \
        ${audio_feats_dir}/cuts_dev_others.jsonl.gz \
        ${audio_feats_dir}/commonvoice_cuts_train.jsonl.gz \
        ${audio_feats_dir}/cuts_train.jsonl.gz

    # test
    lhotse copy \
      ${audio_feats_dir}/commonvoice_cuts_test.jsonl.gz \
      ${audio_feats_dir}/cuts_test.jsonl.gz

    touch ${audio_feats_dir}/.commonvoice.train.done
  fi
fi

python3 ./bin/display_manifest_statistics.py --manifest-dir ${audio_feats_dir}

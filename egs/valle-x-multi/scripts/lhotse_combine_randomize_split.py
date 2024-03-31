# This script finds all transcriptions and related audio files in a dataset directory
# Requires Lhotse
# Transcription Files need to be named exactly like their audio counterparts


import argparse
import glob
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from logging import Formatter
from pathlib import Path

import lhotse
from audioread.ffdec import ReadTimeoutError
from lhotse import Recording, SupervisionSegment, RecordingSet, SupervisionSet, fix_manifests, \
    validate_recordings_and_supervisions, CutSet
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def cuts_combine_randomize_split(src_dir, out_dir, dataset_parts, prefix, suffix, validation_size, randomize_cuts):
    dataset_parts = dataset_parts.strip().split(" ")
    assert len(dataset_parts) >= 1

    # Lhotse is a b**ch - Even if combined it only shuffles per cut set level, not global
    # to overcome this, we load lazy here, save a combined set, then
    # load the combined set again and perform shuffle and splitting on THAT

    logging.info(f"Loading manifests for dataset parts {dataset_parts}")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
        types=["cuts"],
        lazy=True
    )
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")
    assert len(manifests) >= 1

    all_cuts = [c["cuts"] for c in manifests.values()]

    logging.info("Combining cuts...")
    from lhotse.manipulation import combine
    combined_cut_set = combine(all_cuts)
    logging.info("Combining cuts... done")

    logging.info("Saving combined cut set...")
    combined_cut_set.to_file(f"{out_dir}/cuts_complete.jsonl.gz")

    # Load complete CutSet into memory
    combined_cut_set = lhotse.load_manifest(f"{out_dir}/cuts_complete.jsonl")

    # Shuffle if defined
    if randomize_cuts:
        logging.info("Shuffling...")
        combined_cut_set = combined_cut_set.shuffle()
        logging.info("Shuffling... done")

    # Split into Train and Validation set
    total_cuts = len(combined_cut_set)
    split_idx = total_cuts - validation_size
    train_set = combined_cut_set.subset(first=split_idx)
    validation_set = combined_cut_set.subset(last=validation_size)

    # Save
    logging.info("Saving Train set...")
    train_set.to_file(f"{out_dir}/cuts_train.jsonl.gz")
    logging.info("Saving Validation set...")
    validation_set.to_file(f"{out_dir}/cuts_validation.jsonl.gz")


if __name__ == "__main__":
    # Init logger
    Formatter.converter = time.gmtime
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z'
    )

    # Parse from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to store the output files",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="vall-e-x",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=10000,
        help="size of the validation dataset",
    )
    parser.add_argument(
        "--randomize-cuts",
        type=bool,
        default=True,
        help="Randomizes the complete training data before each epoch. Needs a lot of RAM.",
    )

    # Run
    args = parser.parse_args()
    cuts_combine_randomize_split(
        args.src_dir,
        args.out_dir,
        args.dataset_parts,
        args.prefix,
        args.suffix,
        args.validation_size,
        args.randomize_cuts
    )

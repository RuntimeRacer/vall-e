#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/tokenizer.py \
        --src_dir ./data/manifests --output_dir ./data/tokenized

"""
import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from anyascii import anyascii
import torch
import torch.multiprocessing
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from valle.data import (
    AudioTokenConfig,
    AudioTokenExtractor,
    TextTokenizer,
    tokenize_text,
)
from valle.data.fbank import get_fbank_extractor
from valle.utils import SymbolTable

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--audio-extractor",
        type=str,
        default="Encodec",
        help="Encodec or Fbank",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="dev-clean test-clean",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="libritts",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=400.0,
        help="The maximum number of audio seconds in a batch."
             "Determines batch size dynamically.",
    )
    parser.add_argument(
        "--convert-to-ascii",
        type=bool,
        default=False,
        help="Internally transcribe all texts into ascii using anyascii.",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default="unique_text_tokens.k2symbols",
        help="Path to a token file",
    )
    parser.add_argument(
        "--tokenizers-per-device",
        type=int,
        default=1,
        help="Tokenizers to use per device for processing",
    )
    parser.add_argument(
        "--threads-per-tokenizer",
        type=int,
        default=8,
        help="Threads to use per tokenizer for processing",
    )

    return parser.parse_args()


def tokenize_cut_set(cut_set, index, working_dir, args):
    # Setup extractor
    text_tokenizer = None
    if args.text_extractor:
        text_tokenizer = TextTokenizer(backend=args.text_extractor)
    # Symbol set for this instance
    unique_symbols = set()

    prefix = args.prefix
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    with logging_redirect_tqdm():
        if args.text_extractor:
            logging.info(f"Extracting CutSet phonemes for partition {partition}")
    
            if args.prefix == "baker" and args.text_extractor == "labeled_pinyin":
                for c in tqdm(cut_set):
                    phonemes = c.supervisions[0].custom["tokens"]["text"]
                    unique_symbols.update(phonemes)
            else:
                for c in tqdm(cut_set):
                    if args.prefix == "ljspeech":
                        text = c.supervisions[0].custom["normalized_text"]
                        text = text.replace("”", '"').replace("“", '"')
                        if args.convert_to_ascii:
                            text = anyascii(text)
                        phonemes = tokenize_text(
                            text_tokenizer, text=text
                        )
                    elif args.prefix == "aishell":
                        text = c.supervisions[0].text
                        if args.convert_to_ascii:
                            text = anyascii(text)
                        phonemes = tokenize_text(
                            text_tokenizer, text=text
                        )
                    else:  # libritts, commonvoice, custom
                        text = c.supervisions[0].text
                        if args.convert_to_ascii:
                            text = anyascii(text)
                        text = text.lower()
                        phonemes = tokenize_text(
                            text_tokenizer, text=text
                        )

                    # ensure there's a map for custom data in the supervision
                    if not c.supervisions[0].custom:
                        c.supervisions[0].custom = {}

                    # Add phonemes for text
                    c.supervisions[0].custom["tokens"] = {"text": phonemes}
                    unique_symbols.update(phonemes)

        # Save CutSet with Index
        cuts_filename = f"{prefix}cuts_{partition}_{index}.{args.suffix}"
        cut_set.to_file(f"{working_dir}/{cuts_filename}")

        if args.text_extractor:
            return index, cut_set, unique_symbols
        else:
            return None

# def process_manifests(args, accelerator, manifests_to_process):
#
#     audio_extractor = None
#     if args.audio_extractor:
#         if args.audio_extractor == "Encodec":
#             audio_extractor = AudioTokenExtractor(AudioTokenConfig(), device=accelerator.device)
#         else:
#             assert args.audio_extractor == "Fbank"
#             audio_extractor = get_fbank_extractor()
#
#     logging.info(f"Process using Device: {accelerator.device}")
#
#     # AudioTokenizer
#     if args.audio_extractor:
#         if args.audio_extractor == "Encodec":
#             storage_path = (
#                 f"{args.output_dir}/{args.prefix}_encodec_{partition}"
#             )
#         else:
#             storage_path = (
#                 f"{args.output_dir}/{args.prefix}_fbank_{partition}"
#             )
#
#         if args.prefix.lower() in ["ljspeech", "aishell", "baker", "commonvoice", "vall-e-x"]:
#             # resample
#             logging.info(f"resampling CutSet audio for partition {partition}")
#             cut_set = cut_set.resample(24000)
#             # https://github.com/lifeiteng/vall-e/issues/90
#             # if args.prefix == "aishell":
#             #     # NOTE: the loudness of aishell audio files is around -33
#             #     # The best way is datamodule --on-the-fly-feats --enable-audio-aug
#             #     cut_set = cut_set.normalize_loudness(
#             #         target=-20.0, affix_id=True
#             #     )
#
#         with torch.no_grad():
#             logging.info(f"Extracting CutSet features for partition {partition}")
#             if (
#                 torch.cuda.is_available()
#                 and args.audio_extractor == "Encodec"
#             ):
#                 cut_set = cut_set.compute_and_store_features_batch(
#                     extractor=audio_extractor,
#                     storage_path=storage_path,
#                     num_workers=args.threads_per_device,
#                     batch_duration=args.batch_duration,
#                     collate=False,
#                     overwrite=True,
#                     storage_type=NumpyHdf5Writer,
#                 )
#             else:
#                 cut_set = cut_set.compute_and_store_features(
#                     extractor=audio_extractor,
#                     storage_path=storage_path,
#                     num_jobs=args.threads_per_device,
#                     executor=None,
#                     storage_type=NumpyHdf5Writer,
#                 )


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    # Get args
    args = get_args()

    # Read Manifests
    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip()
    if dataset_parts == "all":  # LibriTTS
        dataset_parts = [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]
    else:
        dataset_parts = dataset_parts.replace("-p", "").strip().split(" ")

    assert len(dataset_parts) >= 1

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
        lazy=True
    )
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    assert len(manifests) >= 1

    # determine total task capacity
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    device_count = 1  # default 1 CPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
    task_capacity = device_count * args.tokenizers_per_device * args.threads_per_tokenizer
    logging.info(f"{device_count} available devices for processing")
    logging.info(f"total task capacity: {task_capacity}")

    # Setup working directory
    working_dir = Path(f"{args.output_dir}/work")
    if working_dir.exists():
        shutil.rmtree(working_dir)  # clear existing dir
    os.makedirs(working_dir, exist_ok=True)

    # Setup Symbol Table - reuse symbols file in case we want to extend existing training data with a new language
    if args.text_extractor:
        phonemes = SymbolTable()
        phonemes_file = f"{args.output_dir}/{args.symbols_file}"
        if Path(phonemes_file).is_file():
            phonemes.from_file(phonemes_file)

    # Get CutSets and split them according to task count
    for partition, m in manifests.items():
        logging.info(
            f"Pre-processing partition: {partition}"
        )
        # Ensure Manifest is not lazy
        try:
            logging.info(f"creating CutSet for partition {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"].to_eager(),
                supervisions=m["supervisions"].to_eager(),
            )
        except Exception:
            cut_set = m["cuts"].to_eager()

        # filter
        logging.info(
            f"removing entries of partition {partition} which are longer than batchsize duration or have empty text")
        cut_set = cut_set.filter(
            lambda x:
            x.duration < args.batch_duration and
            x.supervisions[0].text and
            len(x.supervisions[0].text.strip()) > 0
        )

        # Split the CutSet according to processing threads
        split_cut_sets = cut_set.split(num_splits=task_capacity)
        # clean up memory
        cut_set = cut_set.truncate(max_duration=0.000001, offset_type="start")
        del cut_set
        # Perform tokenization across all threads
        with ProcessPoolExecutor(max_workers=task_capacity) as ex:
            futures = []
            for subset_id, subset in enumerate(split_cut_sets):
                futures.append(
                    ex.submit(tokenize_cut_set, subset, subset_id, working_dir, args)
                )

            # Wait for processing to be done
            # for future in tqdm(futures, desc="Processing", leave=False):
            for future in futures:
                result = future.result()
                subset_id, subset, symbol_list = result
                # Append Phonemes to list if applicable
                if symbol_list is not None:
                    if phonemes is None:
                        raise RuntimeError("Symbol table not defined but phonemes returned by subprocess")
                    else:
                        for s in sorted(list(symbol_list)):
                            phonemes.add(s)
                # Update CutSet references in list
                split_cut_sets[subset_id] = subset
                with logging_redirect_tqdm():
                    logging.info(f"Finished tokenizing Cut-SubSet with ID {subset_id}.")

            with logging_redirect_tqdm():
                logging.info("All Cut-Subsets have been tokenized")

    # TODO: Parallel Feature extraction

    # Save Phonemes
    if phonemes_file:
        phonemes.to_file(phonemes_file)


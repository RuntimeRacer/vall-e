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
import time
from pathlib import Path

from accelerate import Accelerator

from anyascii import anyascii
import torch
import torch.multiprocessing
from icefall.utils import get_executor
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

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
        "--threads-per-device",
        type=int,
        default=8,
        help="Threads to use per device for processing",
    )

    return parser.parse_args()


def process_manifests(args, accelerator, manifests_to_process):

    text_tokenizer = None
    if args.text_extractor:
        text_tokenizer = TextTokenizer(backend=args.text_extractor)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    unique_symbols = set()

    prefix = args.prefix
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"

    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    audio_extractor = None
    if args.audio_extractor:
        if args.audio_extractor == "Encodec":
            audio_extractor = AudioTokenExtractor(AudioTokenConfig(), device=accelerator.device)
        else:
            assert args.audio_extractor == "Fbank"
            audio_extractor = get_fbank_extractor()

    logging.info(f"Process using Device: {accelerator.device}")

    with get_executor() as ex:
        for partition, m in manifests_to_process.items():
            logging.info(
                f"Processing partition: {partition}"
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

            # AudioTokenizer
            if args.audio_extractor:
                if args.audio_extractor == "Encodec":
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_encodec_{partition}"
                    )
                else:
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_fbank_{partition}"
                    )

                if args.prefix.lower() in ["ljspeech", "aishell", "baker", "commonvoice", "vall-e-x"]:
                    # filter
                    logging.info(f"removing entries of partition {partition} which are longer than batchsize duration or have empty text")
                    cut_set = cut_set.filter(
                        lambda x: x.duration < args.batch_duration and x.supervisions[0].text and len(x.supervisions[0].text) > 0
                    )

                    # resample
                    logging.info(f"resampling CutSet audio for partition {partition}")
                    cut_set = cut_set.resample(24000)
                    # https://github.com/lifeiteng/vall-e/issues/90
                    # if args.prefix == "aishell":
                    #     # NOTE: the loudness of aishell audio files is around -33
                    #     # The best way is datamodule --on-the-fly-feats --enable-audio-aug
                    #     cut_set = cut_set.normalize_loudness(
                    #         target=-20.0, affix_id=True
                    #     )

                with torch.no_grad():
                    logging.info(f"Extracting CutSet features for partition {partition}")
                    if (
                        torch.cuda.is_available()
                        and args.audio_extractor == "Encodec"
                    ):
                        cut_set = cut_set.compute_and_store_features_batch(
                            extractor=audio_extractor,
                            storage_path=storage_path,
                            num_workers=args.threads_per_device,
                            batch_duration=args.batch_duration,
                            collate=False,
                            overwrite=True,
                            storage_type=NumpyHdf5Writer,
                        )
                    else:
                        cut_set = cut_set.compute_and_store_features(
                            extractor=audio_extractor,
                            storage_path=storage_path,
                            num_jobs=args.threads_per_device if ex is None else 64,
                            executor=ex,
                            storage_type=NumpyHdf5Writer,
                        )

            # TextTokenizer
            if args.text_extractor:
                logging.info(f"Extracting CutSet phonemes for partition {partition}")
                if (
                    args.prefix == "baker"
                    and args.text_extractor == "labeled_pinyin"
                ):
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

            cuts_filename = f"{prefix}cuts_{partition}.{args.suffix}"
            cut_set.to_file(f"{args.output_dir}/{cuts_filename}")

    if args.text_extractor:
        process_phonemes = SymbolTable()
        process_phonemes_file = f"{args.output_dir}/{args.symbols_file}_{accelerator.local_process_index}"

        # reuse symbols file in case we want to extend existing training data with a new language
        if Path(process_phonemes_file).is_file():
            process_phonemes.from_file(process_phonemes_file)

        for s in sorted(list(unique_symbols)):
            process_phonemes.add(s)

        logging.info(f"Process Group {accelerator.local_process_index}: {len(unique_symbols)} unique phonemes.")
        process_phonemes.to_file(process_phonemes_file)


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

    # Use Accelerator for speedup of this pre-processing
    accelerator = Accelerator()

    # Distribute Manifest Elements across processes
    manifests_to_process = {}
    manifest_idx = accelerator.process_index
    manifest_keys = list(manifests.keys())
    while manifest_idx < len(manifests):
        manifests_to_process[manifest_keys[manifest_idx]] = manifests[manifest_keys[manifest_idx]]
        manifest_idx += accelerator.num_processes

    # Process Manifests
    process_manifests(args, accelerator, manifests_to_process)

    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Only do this on the main process
    with accelerator.local_main_process_first():
        if accelerator.is_local_main_process:
            logging.info("Combining results of different processing threads...")
            # Wait for 5 seconds to incorporate potential IO delays
            time.sleep(5)

            # Get global Symbols file
            unique_phonemes = SymbolTable()
            unique_phonemes_file = f"{args.output_dir}/{args.symbols_file}"

            # reuse symbols file in case we want to extend existing training data with a new language
            if Path(unique_phonemes_file).is_file():
                unique_phonemes.from_file(args.symbols_file)

            # Combine all Symbol files in main process
            for proc_idx in range(accelerator.num_processes):
                logging.info(f"Adding Phonemes from process group {proc_idx} ...")
                subprocess_phonemes = SymbolTable()
                subprocess_phonemes_file = f"{args.output_dir}/{args.symbols_file}_{proc_idx}"
                subprocess_phonemes.from_file(subprocess_phonemes_file)
                unique_phonemes.merge(subprocess_phonemes)

            # Save Combined Phonemes File
            unique_phonemes.to_file(unique_phonemes_file)


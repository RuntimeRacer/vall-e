import os
from pathlib import Path
import argparse

from datasets import load_dataset

# Parser for Arguments
parser = argparse.ArgumentParser(description='Download and process ReazonSpeech for VALL-E-X')
parser.add_argument("datasets_root", type=Path, help="Path to the directory to download the dataset to.")

# parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
#     "Path to the ouput directory for this preprocessing script")
# parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# Download dataset from Huggingface
reazon_speech = load_dataset(
    path="reazon-research/reazonspeech",
    name="all",
)
# This loads the dataset into huggingface cache. Processing will happen in a later step.
print(reazon_speech)

# reazon_speech = reazon_speech['train']
# reazon_speech.save_to_disk(args.datasets_root)


# ds = load_dataset(
#     path="reazon-research/reazonspeech",
#     name="small",
# )

# base_dir = args.datasets_root
# dest_dir = base_dir.joinpath('small')
# dest_json_file = dest_dir.joinpath('small.json')
# print(dest_json_file)
#
# os.makedirs(dest_dir, exist_ok=True)
# ds["train"].to_json(dest_json_file)

# Preprocess data after download

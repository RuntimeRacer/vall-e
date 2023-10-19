import os
import shutil
from pathlib import Path
import argparse

from tqdm import tqdm

from huggingface_hub import login
from datasets import load_dataset, DownloadConfig, Audio

# Parser for Arguments
parser = argparse.ArgumentParser(description='Download and process ReazonSpeech for VALL-E-X')
parser.add_argument("datasets_root", type=Path, help="Path to the directory to download the dataset to.")

# parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
#     "Path to the ouput directory for this preprocessing script")
# parser.add_argument("-t", "--threads", type=int, default=8)
args = parser.parse_args()

# Login to Hugginface
login(token=os.environ['HUGGINGFACE_TOKEN'])

# Download dataset from Huggingface
reazon_speech = load_dataset(
    path="reazon-research/reazonspeech",
    name="all", # name="small", # => use this for testing
    download_config=DownloadConfig(resume_download=True, force_download=False, use_etag=True, max_retries=10, num_proc=2),
    download_mode="reuse_cache_if_exists",
    #verification_mode="all_checks",
).cast_column("audio", Audio(decode=False))
# This loads the dataset into huggingface cache. Actual write to target location will happen in a later step.
# Since it's a huge dataset of ~1.2TB, make sure to have at least 4TB available when running this script.
# It will be stored 3 times effectively:
# - Downloaded archives in cache
# - Extracted data + metadata in cache
# - Processed data in provided dataset_root

base_dir = os.path.join(args.datasets_root, "reazonspeech")
for key, subset in reazon_speech.items():
    print('Processing subset: {0}'.format(key))
    subset_dir = os.path.join(base_dir, key)
    os.makedirs(subset_dir, exist_ok=True)

    # Iterate over items and save to disk
    for data_item in tqdm(enumerate(subset), total=len(subset), desc="Saving {0} subset".format(key)):
        idx, item = data_item

        # Store samples as provided by the dataset
        audio_file_path = os.path.join(subset_dir, item['name'])
        transcript_file_path = audio_file_path.replace(".flac", "_transcript.txt")

        # Ensure dir exists
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)

        # Store Audio data and transcript
        shutil.copy(item['audio']['path'], audio_file_path)

        # Save transcription
        with open(transcript_file_path, 'wb') as f:
            f.write(item['transcription'].encode('utf8'))


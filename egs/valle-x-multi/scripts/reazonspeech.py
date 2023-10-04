import os
from pathlib import Path
import argparse

from tqdm import tqdm
import soundfile as sf

from huggingface_hub import login
from datasets import load_dataset, disable_caching

disable_caching()

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
    name="all",
)
# This loads the dataset into huggingface cache. Processing will happen in a later step.
#print(reazon_speech)

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
        sf.write(audio_file_path, item['audio']['array'], item['audio']['sampling_rate'])

        # Save transcription
        with open(transcript_file_path, 'wb') as f:
            f.write(item['transcription'].encode('utf8'))


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

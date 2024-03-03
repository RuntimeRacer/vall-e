# This script reads an existing manifest file for an audio dataset and extracts the text for each item
# into a simple text file containing the transcription

import argparse
import json
import logging
import os
import re
import sys
import time
from logging import Formatter


def extract_transcripts(manifest_file, audiofile_key, text_key):
    # open manifest and extend with language property
    with open(manifest_file, 'r', encoding='utf-8') as file:
        manifest_data = [json.loads(line) for line in file]

    # Iterate manifest
    for line in manifest_data:
        # Get the important data from the manifest
        audio_path = line[audiofile_key]
        transcription = line[text_key]

        # Extract the file name without the extension and language code
        base_path_name, manifest_filename = os.path.split(os.path.realpath(manifest_file))

        # Extract the transcript file name without the extension and language code
        audio_file_name = audio_path.rsplit('.', 2)[0]
        transcript_file = os.path.join(base_path_name, f"{audio_file_name}_transcript.txt")

        with open(transcript_file, 'w', encoding='utf-8') as out_file:
            out_file.write(transcription)


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
    parser.add_argument("-m", "--manifest", type=str, help="manifest file path")
    parser.add_argument("-a", "--audiofile_key", type=str, default="audio_filepath", help="key in manifest for the path to audio")
    parser.add_argument("-t", "--text_key", type=str, default="text", help="key in manifest for the transcription")

    # Run
    args = parser.parse_args()
    extract_transcripts(args.manifest, args.audiofile_key, args.text_key)
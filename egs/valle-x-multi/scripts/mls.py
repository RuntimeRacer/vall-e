# This script converts MLS transcript files into simple text files containing the full transcript for all audio files inside

import argparse
import logging
import os
import csv
import re
import sys
import time
from logging import Formatter


def extract_transcripts(library_file, target_dir=''):
    # Open and read the file
    with open(library_file, 'r', encoding='utf-8') as file:
        for line in file:
            elements = line.strip().split('\t')
            key = elements[0]
            transcript = elements[1]

            # Get Path elements
            path_elements = key.split('_')
            transcript_file_path = os.path.join(target_dir, path_elements[0], path_elements[1], f"{key}_transcript.txt")

            # Construct the transcript file & write the content
            os.makedirs(os.path.dirname(transcript_file_path), exist_ok=True)
            with open(transcript_file_path, 'w', encoding='utf-8') as transcript_file:
                transcript_file.write(transcript)

            logging.debug(f"Transcript file created: {transcript_file_path}")


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
    parser.add_argument("-f", "--library_file", type=str, help="library file")
    parser.add_argument("-t", "--target_dir", type=str, default="", help="dir to store target files")

    # Run
    args = parser.parse_args()
    extract_transcripts(args.library_file, args.target_dir)
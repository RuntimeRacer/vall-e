# This script converts Quran Speech Dataset's .tsv files into simple text files containing the full transcript for all audio files inside

import argparse
import logging
import os
import csv
import re
import sys
import time
from logging import Formatter


def extract_transcripts(directory, target_dir=''):
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            # Construct the path to the current file
            file_path = os.path.join(directory, filename)
            # Open and read the .tsv file
            with open(file_path, 'r', encoding='utf-8') as tsvfile:
                reader = csv.DictReader(tsvfile, delimiter='\t')
                first_line = True
                for row in reader:
                    # Construct the transcript file name based on the 'path' column
                    # Assuming the 'path' column contains the filename of the audio file, adjust as necessary
                    if first_line:
                        transcript_filename = os.path.join(target_dir, f"{os.path.splitext(row['PATH'])[0]}_transcript.txt")
                        first_line = False
                        continue

                    # Write the sentence to a new file in the clips directory
                    with open(transcript_filename, 'w', encoding='utf-8') as transcript_file:
                        transcript_file.write(row['DURATION'])
                    first_line = True

                    logging.debug(f"Transcript file created: {transcript_filename}")

            logging.debug(f"Transcript files have been created in the clips directory for {filename}")


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
    parser.add_argument("-d", "--dir", type=str, help="dir with .tsv files")
    parser.add_argument("-t", "--target_dir", type=str, default="", help="dir to store target files")

    # Run
    args = parser.parse_args()
    extract_transcripts(args.dir, args.target_dir)
# This script converts .vtt files into simple text files containing the full transcript for an audio file

import argparse
import logging
import os
import re
import sys
import time
from logging import Formatter


def extract_transcripts(directory, target_dir=''):
    for filename in os.listdir(directory):
        if filename.endswith(".vtt"):
            # Construct the path to the current file
            file_path = os.path.join(directory, filename)
            # Open and read the .vtt file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()

            # Extract the file name without the extension and language code
            base_name = filename.rsplit('.', 2)[0]
            # Construct the output file name
            if len(target_dir) > 0:
                transcript_filename = os.path.join(target_dir, f"{base_name}_transcript.txt")
            else:
                transcript_filename = os.path.join(directory, f"{base_name}_transcript.txt")

            # Process the content to remove timestamps and other .vtt specific texts
            transcript_content = []
            i = 0
            for line in content:
                i += 1
                # Skip unnecessary lines (like those containing timestamps)
                if i < 4:
                    continue
                if (re.match(r'^\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}', line) or
                        re.match(r'^\d{2}:\d{2}:\d{2}.\d{3} --&gt; \d{2}:\d{2}:\d{2}.\d{3}', line)):
                    continue
                if line.strip() and not line.strip().isdigit():
                    transcript_content.append(line.strip())



            # Write the processed content to a new file
            with open(transcript_filename, 'w', encoding='utf-8') as transcript_file:
                transcript_file.write(' '.join(transcript_content))

            logging.debug(f"Transcript file created: {transcript_filename}")


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
    parser.add_argument("-d", "--dir", type=str, help="dir with .vtt files")
    parser.add_argument("-t", "--target_dir", type=str, default="", help="dir to store target files")

    # Run
    args = parser.parse_args()
    extract_transcripts(args.dir, args.target_dir)
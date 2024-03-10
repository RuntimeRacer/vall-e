import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from logging import Formatter
from pathlib import Path

from tqdm import tqdm


def convert_to_flac(file_path, sample_rate):
    """
    Convert an audio file to FLAC format at 24kHz sample rate and delete the original file.
    """
    # Construct the new filename with .flac extension
    new_file_path = file_path.with_suffix('.flac')

    # Command to convert the file using ffmpeg
    command = [
        'ffmpeg',
        "-y",
        "-loglevel",
        "fatal",
        '-i',
        str(file_path),
        '-ar',
        str(sample_rate),
        "-threads",
        str(1),
        str(new_file_path)
    ]

    try:
        # Execute the ffmpeg command
        result = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # If conversion was successful, delete the original file
        if result == 0:
            # os.remove(file_path)
            logging.debug(f"Converted and deleted {file_path}")
        else:
            logging.error(f"Failed to convert {file_path}")
    except Exception as e:
        logging.error(f"Error converting {file_path}: {e}")


def find_files(root_dir):
    """
    Generate a list of file paths for supported audio formats within the given directory tree.
    """
    supported_extensions = ['.opus', '.ogg', '.mp3', '.wav', '.m4a', '.flac']
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix in supported_extensions:
                yield Path(root) / file


def convert_files(root_dir, sample_rate, threads):
    """
    Find all supported audio files in the directory tree and convert them to FLAC using multiple processes.
    """
    files_to_convert = list(find_files(root_dir))
    with ProcessPoolExecutor(threads) as executor:
        list(tqdm(executor.map(convert_to_flac, files_to_convert, sample_rate), total=len(files_to_convert)))


if __name__ == '__main__':
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
    parser.add_argument("-d", "--dir", type=str, help="dir with audio files and transcripts")
    parser.add_argument("-sr", "--sample_rate", type=int, default=24000, help="target sample rate")
    parser.add_argument("-t", "--threads", type=int, default=16, help="processing threads to use")

    # Run
    args = parser.parse_args()
    convert_files(args.dir, args.sample_rate, args.threads)

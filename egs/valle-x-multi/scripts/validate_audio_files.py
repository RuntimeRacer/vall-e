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
from tqdm.contrib.logging import logging_redirect_tqdm


def check_files_and_invalidate_broken(file_path):
    """
    Check for errors in decoding an audio file, and remove the file if it cannot be decoded
    """

    # Command to convert the file using ffmpeg
    command = [
        'ffmpeg',
        "-y",
        "-xerror",
        '-i',
        str(file_path),
        "-f",
        "null",
        "-"
    ]

    try:
        # Execute the ffmpeg command
        result = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # If conversion was successful, delete the original file
        if result == 0:
            with logging_redirect_tqdm():
                logging.debug(f"File {file_path} ik OK.")
        else:
            with logging_redirect_tqdm():
                logging.error(f"File {file_path} is broken. Removing...")
            # os.remove(file_path)
    except Exception as e:
        with logging_redirect_tqdm():
            logging.error(f"Error validating {file_path}: {e}")


def find_files(root_dir):
    """
    Generate a list of file paths for supported audio formats within the given directory tree.
    """
    supported_extensions = ['.opus', '.ogg', '.mp3', '.wav', '.m4a', '.flac']
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix in supported_extensions:
                yield Path(root) / file


def convert_files(root_dir, threads):
    """
    Find all supported audio files in the directory tree and convert them to FLAC using multiple processes.
    """
    files_to_convert = list(find_files(root_dir))
    with ProcessPoolExecutor(threads) as executor:
        list(tqdm(executor.map(check_files_and_invalidate_broken, files_to_convert), total=len(files_to_convert)))


if __name__ == '__main__':
    # Init logger
    Formatter.converter = time.gmtime
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z'
    )

    # Parse from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="dir with audio files and transcripts")
    parser.add_argument("-t", "--threads", type=int, default=16, help="processing threads to use")

    # Run
    args = parser.parse_args()
    convert_files(args.dir, args.threads)

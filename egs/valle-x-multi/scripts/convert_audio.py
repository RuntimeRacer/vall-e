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


def convert_to_target_format(file_path, target_format='.opus'):
    """
    Convert an audio file to OPUS format at 24kHz sample rate and Mono channel. delete the original file.
    """
    # Construct the new filename with .opus extension
    suffix_ext = ''
    if target_format in str(file_path):
        logging.debug(f"Input file and output file have same format, creating secondary file")
        suffix_ext = '.new'
        new_suffix = suffix_ext + target_format
        new_file_path = file_path.with_suffix(new_suffix)
    else:
        new_file_path = file_path.with_suffix(target_format)

    # Command to convert the file using ffmpeg
    command = [
        'ffmpeg',
        "-y",
        "-loglevel",
        "fatal",
        '-i',
        str(file_path),
        '-ar',
        '24000',
        '-ac',
        '1',
        "-threads",
        str(1),
        str(new_file_path)
    ]

    # logging.info(command)

    try:
        # Execute the ffmpeg command
        result = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # If conversion was successful, delete the original file
        if result == 0:
            if suffix_ext != '':
                os.remove(file_path)
                os.rename(new_file_path, file_path)
                with logging_redirect_tqdm():
                    logging.debug(f"Resampled {file_path}")
            else:
                os.remove(file_path)
                with logging_redirect_tqdm():
                    logging.debug(f"Converted and deleted {file_path}")
            return True
        else:
            with logging_redirect_tqdm():
                logging.error(f"Failed to convert {file_path}")
            return False
    except Exception as e:
        with logging_redirect_tqdm():
            logging.error(f"Error converting {file_path}: {e}")
        return False


def find_files(root_dir):
    """
    Generate a list of file paths for supported audio formats within the given directory tree.
    """
    supported_extensions = ['.opus', '.ogg', '.mp3', '.wav', '.m4a', '.flac']
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if Path(file).suffix in supported_extensions:
                yield Path(root) / file


def convert_files(root_dir, target_format, threads):
    """
    Find all supported audio files in the directory tree and convert them to FLAC using multiple processes.
    """
    files_to_convert = list(find_files(root_dir))
    with ProcessPoolExecutor(threads) as ex:
        futures = []

        for file_path in tqdm(files_to_convert, desc="Creating Tasks", leave=False):
            # Submit to processing
            futures.append(
                ex.submit(convert_to_target_format, file_path, target_format)
            )

        # Just wait for processing to be done here
        for future in tqdm(futures, desc="Processing", leave=False):
            _ = future.result()

    logging.info(f"Finished {len(futures)} conversion jobs.")


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
    parser.add_argument("-f", "--target-format", type=str, default='.opus', help="target format")
    parser.add_argument("-t", "--threads", type=int, default=16, help="processing threads to use")

    # Run
    args = parser.parse_args()
    convert_files(args.dir, args.target_format, args.threads)

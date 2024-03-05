import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from logging import Formatter
from pathlib import Path
import argparse

from tqdm import tqdm


def prepare_dataset(dataset_root, output_dir, threads=16):
    # Get Manifest file
    manifest_file = os.path.join(dataset_root, 'WenetSpeech.json')
    # open manifest and read data into memory
    with open(manifest_file, 'r', encoding='utf-8') as file:
        manifest_data = json.load(file)

    # Get list of audio files
    audio_list = manifest_data['audios']

    with ThreadPoolExecutor(threads) as ex:
        # Prepare processing
        futures = []
        processed = 0
        skipped = 0
        failed = 0

        for audio_file in tqdm(audio_list, "Distributing tasks", leave=False):
            # Split each Segment using FFMPEG in second step
            # Create futures for each segment creation task
            audio_file_path = os.path.join(dataset_root, audio_file['path'])
            for segment in audio_file['segments']:
                futures.append(
                    ex.submit(
                        process_audio_file_segment,
                        audio_file_path,
                        dataset_root,
                        output_dir,
                        segment['sid'],
                        segment['begin_time'],
                        segment['end_time'],
                        segment['text'],
                    )
                )

        # Wait for all futures to return
        for future in tqdm(futures, desc="Processing", leave=False):
            result = future.result()
            if result is None:
                skipped += 1
            elif result is True:
                processed += 1
            else:
                failed += 1

        logging.info(f"finished iterating through {len(futures)} segments. Skipped: {skipped} | Processed: {processed} | Failed: {failed}")


def process_audio_file_segment(audio_file_path, dataset_root, output_dir, segment_id, start, end, text):
    # get base file path and extension for output file
    file_ext = os.path.splitext(audio_file_path)[1]
    file_dir = os.path.dirname(audio_file_path)
    file_dir = file_dir.replace(dataset_root, output_dir)
    segment_basename = os.path.join(str(file_dir), segment_id)
    segment_audio_file = segment_basename + file_ext
    segment_transcript_file = segment_basename + '_transcript.txt'

    # Convert start and end time strings to float and miliseconds
    start = float(start) * 1000
    end = float(end) * 1000

    # Create dirs and Files if they don't exist
    dest_transcript_path = Path(segment_transcript_file)
    os.makedirs(os.path.dirname(dest_transcript_path), exist_ok=True)
    if not dest_transcript_path.is_file():
        with open(dest_transcript_path, "w", encoding="utf-8-sig") as out:
            out.write(text.strip())

    # if the file already exists, skip conversion
    dest_audio_path = Path(segment_audio_file)
    if dest_audio_path.is_file():
        return None

    # Process using FFMPEG
    convert_args = [
        "/usr/bin/ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-i",
        str(audio_file_path),
        "-ss",
        format_timestamp(start, always_include_hours=True),
        "-to",
        format_timestamp(end, always_include_hours=True),
        "-c:a",
        "copy",
        "-threads",
        str(1),
        str(segment_audio_file)
    ]
    subprocess.call(convert_args)
    return True


def format_timestamp(
    milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert milliseconds >= 0, "non-negative timestamp expected"

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds)

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


if __name__ == "__main__":
    # Init logger
    Formatter.converter = time.gmtime
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z'
    )

    # Parser for Arguments
    parser = argparse.ArgumentParser(description='Download and process WenetSpeech for VALL-E-X')
    parser.add_argument("dataset_root", type=str, help="Path to the directory with the dataset")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the outputs and transcripts will be placed")
    parser.add_argument("-t", "--threads", type=int, default=16, help="processing threads to use")

    # Run
    args = parser.parse_args()
    prepare_dataset(args.dataset_root, args.output_dir, args.threads)
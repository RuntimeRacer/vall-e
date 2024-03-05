# This script finds all transcriptions and related audio files in a dataset directory
# Requires Lhotse
# Transcription Files need to be named exactly like their audio counterparts


import argparse
import glob
import json
import logging
import os.path
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from logging import Formatter
from pathlib import Path

from lhotse import Recording, SupervisionSegment, RecordingSet, SupervisionSet, fix_manifests, \
    validate_recordings_and_supervisions
from tqdm import tqdm

# Make sure this matches the one in valle.data.dataset
LANG_ID_DICT = {
    'en': 0,  # English
    'de': 1,  # German
    'fr': 2,  # French
    'it': 3,  # Italian
    'ar': 4,  # Arabic
    'es': 5,  # Spanish
    'ru': 6,  # Russian
    'zh-CN': 7,  # Chinese
    'ja': 8  # Japanese
}


def build_audio_dataset_manifest(directory, output_file_name=None, language='', threads=16):
    # Check for valid language key
    if language not in LANG_ID_DICT:
        raise RuntimeError(f"provided language {language} is not a member of allowed languages")

    with ThreadPoolExecutor(threads) as ex:
        # Setup
        recordings = []
        supervisions = []
        futures = []

        # find all transcripts
        directory_path = Path(directory) # convert to path object

        for transcript_path in tqdm(
                directory_path.rglob("*_transcript.txt"), desc="Distributing tasks", leave=False
        ):
            # We will create a separate Recording and SupervisionSegment for each file.
            # get base path of the transcript file to search for corresponding audio file
            transcript_path_str = str(transcript_path)
            base_name = transcript_path_str.rsplit('_transcript.txt', 1)[0]
            # Use glob to find matching audio files with any extension
            audio_files = glob.glob(f"{base_name}.*")
            if len(audio_files) == 0:
                logging.warning(f"No matching audio file found for transcript file {transcript_path}.")
                continue
            if len(audio_files) > 1:
                logging.warning(f"more than one possible audio files for transcript file {transcript_path}. Only first one is picked.")
            # Take first match
            audio_file_path = Path(audio_files[0])  # Take the first match

            # Submit to processing
            futures.append(
                ex.submit(process_transcript, transcript_path, audio_file_path, language)
            )

        for future in tqdm(futures, desc="Processing", leave=False):
            result = future.result()
            if result is None:
                continue
            recording, segment = result
            recordings.append(recording)
            supervisions.append(segment)

        # Convert to Lhotse sets
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Basic final validation
        recording_set, supervision_set = fix_manifests(
            recording_set, supervision_set
        )
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_file_name is not None:
            supervision_set.to_file(
                f"{output_file_name}_supervisions.jsonl.gz"
            )
            recording_set.to_file(
                f"{output_file_name}_recordings.jsonl.gz"
            )


def process_transcript(transcript_path, audio_file_path, language):
    if not transcript_path.is_file():
        logging.warning(f"No such file: {transcript_path}")
        return None
    if not audio_file_path.is_file():
        logging.warning(f"No such file: {audio_file_path}")
        return None

    # Read transcript file content
    with open(transcript_path) as f:
        # get transcript text
        transcript_text = f.readline().strip()

    # Create random UUID for this recording
    recording_id = str(uuid.uuid4())
    # Use Lhotse recording backend to analyse audio
    recording = Recording.from_file(audio_file_path.stem, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=language,
        text=transcript_text,
    )
    return recording, segment


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
    parser.add_argument("-d", "--dir", type=str, help="dir with audio files and transcripts")
    parser.add_argument("-o", "--output_file", type=str, default=None, help="path and name to store recording and supervision sets")
    parser.add_argument("-l", "--language", type=str, default="", help="language value to add in supervisions")
    parser.add_argument("-t", "--threads", type=int, default=16, help="processing threads to use")

    # Run
    args = parser.parse_args()
    build_audio_dataset_manifest(args.dir, args.output_file, args.language, args.threads)

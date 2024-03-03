# This script takes an existing manifest from a dataset and extends it by a language parameter

import argparse
import json
import logging
import os.path
import sys
import time
from logging import Formatter


def extend_manifest_language(manifest_file, language_id='en'):
    # open manifest and extend with language property
    with open(manifest_file, 'r', encoding='utf-8') as file:
        manifest_data = [json.loads(line) | {"language": language_id} for line in file]

    # Write to output file
    file_path, extension = os.path.splitext(manifest_file)

    with open('{0}-annotated.jsonl'.format(file_path), 'w', encoding='utf-8') as out_file:
        json.dump(manifest_data, out_file, indent=4)


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
    parser.add_argument("-l", "--language", type=str, help="language to add")

    # Run
    args = parser.parse_args()
    extend_manifest_language(args.manifest, args.language)
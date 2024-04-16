import argparse
import logging
import sys
import time
from logging import Formatter

import torch


def analyse_batch(batch_file):
    logging.info(f"loading batch from file {batch_file}")
    batch = torch.load(batch_file)

    # Remove entry from batch
    audio_features = batch['audio_features'].tolist()
    audio_features_lens = batch['audio_features_lens'].tolist()
    text_tokens = batch['text_tokens'].tolist()
    text_tokens_lens = batch['text_tokens_lens'].tolist()
    batch_size = len(text_tokens_lens)
    indexes_to_remove = []
    for idx, audio_len in enumerate(audio_features_lens):
        text_len = text_tokens_lens[idx]
        if text_len > audio_len * 2:
            indexes_to_remove.append(idx)

    # Reverse and remove
    indexes_to_remove.reverse()
    for idx in indexes_to_remove:
        del batch['utt_id'][idx]
        del batch['text'][idx]
        if idx < (batch_size - 1):
            batch['audio_features'] = torch.cat((batch['audio_features'][:idx], batch['audio_features'][idx + 1:]))
            batch['audio_features_lens'] = torch.cat((batch['audio_features_lens'][:idx], batch['audio_features_lens'][idx + 1:]))
            batch['text_tokens'] = torch.cat((batch['text_tokens'][:idx], batch['text_tokens'][idx + 1:]))
            batch['text_tokens_lens'] = torch.cat((batch['text_tokens_lens'][:idx], batch['text_tokens_lens'][idx + 1:]))
            batch['languages'] = torch.cat((batch['languages'][:idx], batch['languages'][idx + 1:]))
        else:
            batch['audio_features'] = batch['audio_features'][:idx]
            batch['audio_features_lens'] = batch['audio_features_lens'][:idx]
            batch['text_tokens'] = batch['text_tokens'][:idx]
            batch['text_tokens_lens'] = batch['text_tokens_lens'][:idx]
            batch['languages'] = batch['languages'][:idx]

    # Reshape tensors
    new_lengths_audio = batch['audio_features_lens'].tolist()
    new_lengths_text = torch.count_nonzero(batch['text_tokens'], dim=1)
    new_max_length_audio = max(new_lengths_audio)
    new_max_length_text = torch.max(new_lengths_text)
    batch['audio_features'] = batch['audio_features'][:, :new_max_length_audio]
    batch['text_tokens'] = batch['text_tokens'][:, :new_max_length_text]

    logging.info(batch)


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
    parser.add_argument("-f", "--file", type=str, help="batch file path")

    # Run
    args = parser.parse_args()
    analyse_batch(args.file)
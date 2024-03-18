import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from lhotse import CutSet, NumpyHdf5Writer

from valle.data import AudioTokenExtractor, AudioTokenConfig, get_fbank_extractor


class FeatureExtractionWorker:
    def __init__(
            self,
            worker_id: str,
            cuts_file_name: str,
            work_dir: str,
            device: str = 'cuda:0',
            audio_extractor: str = 'Encodec',
            batch_duration: int = 400,
            sample_rate: int = None,
            worker_threads: int = 1
    ):
        if len(work_dir) == 0:
            raise RuntimeError("work_dir not set")

        # Setup Params
        self.worker_id = worker_id
        self.cuts_file_name = cuts_file_name
        self.work_dir = work_dir
        self.device = device
        self.audio_extractor = audio_extractor
        self.batch_duration = batch_duration
        self.sample_rate = sample_rate
        self.worker_threads = worker_threads

        # Cuts file, Tokenizer and features storage
        self.audio_extractor_instance = None
        self.cuts_file = None
        self.storage_path = None

    def run(self):
        logging.info(f"Worker-{self.worker_id}: Initializing...")

        # Get CutSet from workdir
        self.cuts_file = Path(f"{self.work_dir}/{self.cuts_file_name}")
        if not self.cuts_file.exists():
            raise RuntimeError(f"Worker-{self.worker_id}: Cuts file {self.cuts_file} does not exist")
        cut_set = CutSet.from_file(self.cuts_file)
        logging.info(f"Worker-{self.worker_id}: processing '{self.cuts_file}' using Device: {self.device}")

        # Init extractor
        self.initialize_extractor()
        if not self.audio_extractor_instance:
            raise RuntimeError("Extractor not initialized because improperly configured")

        # Resample if necessary
        if self.sample_rate:
            logging.info(f"Worker-{self.worker_id}: resampling CutSet audio to a sample rate of {self.sample_rate} kHz")
            cut_set = cut_set.resample(24000)
            # https://github.com/lifeiteng/vall-e/issues/90
            # if args.prefix == "aishell":
            #     # NOTE: the loudness of aishell audio files is around -33
            #     # The best way is datamodule --on-the-fly-feats --enable-audio-aug
            #     cut_set = cut_set.normalize_loudness(
            #         target=-20.0, affix_id=True
            #     )

        # Extract Features
        with torch.no_grad():
            logging.info(f"Worker-{self.worker_id}: extracting CutSet features...")
            if (
                torch.cuda.is_available()
                and self.audio_extractor == "Encodec"
            ):
                cut_set = cut_set.compute_and_store_features_batch(
                    extractor=self.audio_extractor_instance,
                    storage_path=self.storage_path,
                    manifest_path=f"{self.storage_path}_manifest.jsonl.gz",
                    num_workers=self.worker_threads,
                    batch_duration=self.batch_duration,
                    collate=False,
                    overwrite=True,
                    storage_type=NumpyHdf5Writer,
                )
            else:
                cut_set = cut_set.compute_and_store_features(
                    extractor=self.audio_extractor_instance,
                    storage_path=self.storage_path,
                    manifest_path=f"{self.storage_path}_manifest.jsonl.gz",
                    num_jobs=self.worker_threads,
                    executor=None,
                    storage_type=NumpyHdf5Writer,
                )

        logging.info(f"Worker-{self.worker_id}: feature extraction done. Saving updated Manifest...")
        # Update Cuts file
        cut_set.to_file(self.cuts_file.with_name(f"{self.cuts_file.stem}_encodec_processed{self.cuts_file.suffix}"))
        logging.info(f"Worker-{self.worker_id}: saving done.")

    def initialize_extractor(self):
        if not self.audio_extractor:
            return None

        if self.audio_extractor == "Encodec":
            self.audio_extractor_instance = AudioTokenExtractor(AudioTokenConfig(), device=self.device)
            self.storage_path = self.cuts_file.with_name(f"{self.cuts_file.stem}_encodec")
        else:
            self.audio_extractor_instance = get_fbank_extractor(device=self.device)
            self.storage_path = self.cuts_file.with_name(f"{self.cuts_file.stem}_fbank")


    def shutdown(self):
        logging.info("Shutting down worker...")


if __name__ == "__main__":
    # Init logger
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z'
    )

    # Parse from arguments
    parser = argparse.ArgumentParser()

    # Worker parameters
    parser.add_argument("--worker-id", dest="worker_id", type=str, help="ID of the worker process in tokenization context")
    parser.add_argument("--cuts-file-name", dest="cuts_file_name", type=str, help="Name of the cuts file to process")
    parser.add_argument("--work-dir", dest="work_dir", type=str, help="directory with CutSets to store intermediate Feature archives inside")
    parser.add_argument("--device", dest="device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="specifies device to execute this worker on")
    parser.add_argument("--audio-extractor", dest="audio_extractor", default="Encodec", help="Encodec or Fbank")
    parser.add_argument("--batch-duration", dest="batch_duration", default=400, help="The maximum number of audio seconds in a batch. Determines batch size dynamically.")
    parser.add_argument("--sample-rate", dest="sample_rate", type=int, default=None, help="If defined, resample cuts to this sample rate")
    parser.add_argument("--threads", dest="worker_threads", type=int, default=1, help="Support threads for this worker")

    # Run
    args = parser.parse_args()
    worker = FeatureExtractionWorker(**vars(args))
    try:
        worker.run()
    except KeyboardInterrupt:
        logging.info("Manual Interrupt!")
        # Try clean shutdown
        worker.shutdown()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

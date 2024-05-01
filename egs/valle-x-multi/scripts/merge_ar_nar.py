import argparse
import copy
import logging
import sys
import time
from logging import Formatter

import torch
from icefall import load_checkpoint, save_checkpoint

from torch.nn.parallel import DistributedDataParallel as DDP

from valle.bin.trainer import load_checkpoint_if_available
from valle.models import get_model, add_model_arguments
from valle.modules.optim import ScaledAdam, Eve
from valle.modules.scheduler import get_scheduler


def get_optimizer(params, model, model_parameters):
    if params.optimizer_name == "ScaledAdam":
        parameters_names = []
        if params.train_stage:  # != 0
            _model = model.module if isinstance(model, DDP) else model
            parameters_names.append(
                [
                    name_param_pair[0]
                    for name_param_pair in _model.stage_named_parameters(
                        params.train_stage
                    )
                ]
            )
        else:
            parameters_names.append(
                [
                    name_param_pair[0]
                    for name_param_pair in model.named_parameters()
                ]
            )

        optimizer = ScaledAdam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )
    elif params.optimizer_name == "Eve":
        optimizer = Eve(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.98),
            target_rms=0.1,
        )
    elif params.optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
            eps=1e-8,
        )
    elif params.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=params.base_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        raise NotImplementedError()

    return optimizer


def merge_model_parts(params, ar_path, nar_path, out_file):
    device = torch.device("cpu")

    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    logging.info(f"loading AR model from file {ar_path}")
    ar_model = get_model(params)
    ar_checkpoints = load_checkpoint(filename=ar_path, model=ar_model)
    _ar_model = ar_model.module if isinstance(ar_model, DDP) else ar_model
    ar_model_parameters = _ar_model.stage_parameters(1)
    ar_optimizer = get_optimizer(params, ar_model, ar_model_parameters)
    ar_scheduler = get_scheduler(params, ar_optimizer)
    ar_optimizer.zero_grad()

    if ar_checkpoints and "optimizer" in ar_checkpoints:
        ar_optimizer.load_state_dict(ar_checkpoints["optimizer"])
    if ar_checkpoints and "scheduler" in ar_checkpoints:
        ar_scheduler.load_state_dict(ar_checkpoints["scheduler"])
    if ar_checkpoints and "sampler" in ar_checkpoints:
        ar_sampler_state_dict = ar_checkpoints["sampler"]

    logging.info(f"loading NAR model from file {nar_path}")
    nar_model = get_model(params)
    nar_checkpoints = load_checkpoint(filename=nar_path, model=nar_model)
    _nar_model = nar_model.module if isinstance(nar_model, DDP) else nar_model
    nar_model_parameters = _nar_model.stage_parameters(2)
    nar_optimizer = get_optimizer(params, nar_model, nar_model_parameters)
    nar_scheduler = get_scheduler(params, nar_optimizer)
    nar_optimizer.zero_grad()

    if nar_checkpoints and "optimizer" in nar_checkpoints:
        nar_optimizer.load_state_dict(nar_checkpoints["optimizer"])
    if nar_checkpoints and "scheduler" in nar_checkpoints:
        nar_scheduler.load_state_dict(nar_checkpoints["scheduler"])
    if nar_checkpoints and "sampler" in nar_checkpoints:
        nar_sampler_state_dict = nar_checkpoints["sampler"]

    logging.info("Both models loaded. Merging AR and NAR model...")

    _ar_model.nar_accuracy_metric = copy.deepcopy(_nar_model.nar_accuracy_metric)
    _ar_model.nar_audio_embeddings = copy.deepcopy(_nar_model.nar_audio_embeddings)
    _ar_model.nar_audio_position = copy.deepcopy(_nar_model.nar_audio_position)
    _ar_model.nar_audio_prenet = copy.deepcopy(_nar_model.nar_audio_prenet)
    _ar_model.nar_decoder = copy.deepcopy(_nar_model.nar_decoder)
    _ar_model.nar_language_embedding = copy.deepcopy(_nar_model.nar_language_embedding)
    _ar_model.nar_predict_layers = copy.deepcopy(_nar_model.nar_predict_layers)
    _ar_model.nar_stage_embeddings = copy.deepcopy(_nar_model.nar_stage_embeddings)
    _ar_model.nar_text_embedding = copy.deepcopy(_nar_model.nar_text_embedding)
    _ar_model.nar_text_position = copy.deepcopy(_nar_model.nar_text_position)
    _ar_model.nar_text_prenet = copy.deepcopy(_nar_model.nar_text_prenet)

    logging.info("Saving merged model...")
    save_checkpoint(
        filename=out_file,
        model=_ar_model
    )
    logging.info("done! Merged model saved successfully.")


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
    parser.add_argument("-ap", "--ar-path", type=str, help="path of the file to take the AR model from")
    parser.add_argument("-np", "--nar-path", type=str, help="path of the file to take the NAR model from")
    parser.add_argument("-o", "--out-file", type=str, help="path of the output file to store the combined data")

    # Model Setup Parameters
    parser.add_argument("--optimizer-name", type=str, default="ScaledAdam", help="The optimizer.")
    parser.add_argument("--scheduler-name", type=str, default="Eden", help="The scheduler.")
    parser.add_argument("--dtype", type=str, default="float32", help="Training dtype: float32 bfloat16 float16.")
    add_model_arguments(parser)

    # irrelevant for merge, but required by some modules
    parser.add_argument("--base-lr", type=float, default=0.05, help="The base learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=200, help="""Number of steps that affects how rapidly the learning rate decreases. We suggest not to change this.""",)

    # Run
    args = parser.parse_args()
    merge_model_parts(args, args.ar_path, args.nar_path, args.out_file)

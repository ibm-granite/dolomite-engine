import json
import logging
import os
import random
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .arguments import ExportArgs, InferenceArgs, TrainingArgs
from .data import ResumableDataLoader
from .enums import DistributedBackend, Mode, TuningMethod
from .hf_models.models.gpt_dolomite_TP import unshard
from .model_wrapper import ModelWrapper, get_model
from .utils import ExperimentsTracker, ProcessGroupManager, load_yaml, log_rank_0, run_rank_n, string_to_torch_dtype


_TRAINING_CONFIG_PREFIX = "training_config"
_INFERENCE_CONFIG_PREFIX = "inference_config"


def save_checkpoint(
    args: TrainingArgs,
    model: ModelWrapper,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: ResumableDataLoader,
    experiments_tracker: ExperimentsTracker,
    iteration: int,
    metadata: dict = None,
) -> None:
    """save checkpoint during training

    Args:
        args (TrainingArgs): arguments for training
        model (ModelWrapper): model to save
        optimizer (Optimizer): optimizer to save
        lr_scheduler (LambdaLR): learning rate scheduler to save
        train_dataloader (DataLoader): train dataloader to save
        experiments_tracker (ExperimentsTracker): experiment tracker to save
        iteration (int): current iteration
        metadata (dict): extra stuff to store

    Raises:
        ValueError: if unexpected distributed backend is found
    """

    distributed_backend = args.distributed_args.distributed_backend
    save_optimizer = args.save_args.save_optimizer

    save_path = _get_base_path(args.save_args.save_path, iteration)
    os.makedirs(save_path, exist_ok=True)

    if distributed_backend == DistributedBackend.deepspeed:
        assert save_optimizer
        model.save_checkpoint(args.save_args.save_path, tag=_get_checkpoint_tag(iteration))
    elif distributed_backend == DistributedBackend.torch:
        if ProcessGroupManager.get_tensor_parallel_world_size() > 1:
            state_dict = {name: param.full_tensor() for name, param in model.named_parameters()}

            if ProcessGroupManager.get_data_parallel_rank() == 0:
                torch.save(state_dict, _get_model_path(save_path))
        else:
            dcp.save(get_model_state_dict(model), checkpoint_id=_get_model_path(save_path))

        if save_optimizer:
            # TODO add options=StateDictOptions(flatten_optimizer_state_dict=True))
            dcp.save(get_optimizer_state_dict(model, optimizer), checkpoint_id=_get_optimizer_path(save_path))

        run_rank_n(torch.save)(lr_scheduler.state_dict(), _get_lr_scheduler_path(save_path))
    else:
        raise ValueError(f"unexpected distributed_backend ({distributed_backend})")

    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    rng_state_path = _get_rng_state_path(save_path)
    os.makedirs(os.path.dirname(rng_state_path), exist_ok=True)
    torch.save(rng_state, rng_state_path)

    if train_dataloader is not None:
        dataloader_path = _get_dataloader_path(save_path)
        os.makedirs(os.path.dirname(dataloader_path), exist_ok=True)
        torch.save(train_dataloader.state_dict(), dataloader_path)

    if experiments_tracker is not None:
        run_rank_n(json.dump)(
            experiments_tracker.state_dict(), run_rank_n(open)(_get_experiments_tracker_path(save_path), "w"), indent=4
        )

    if metadata is not None:
        run_rank_n(json.dump)(metadata, run_rank_n(open)(_get_metadata_path(save_path), "w"), indent=4)

    save_args(args, save_path, mode=Mode.training)

    torch.distributed.barrier()

    run_rank_n(json.dump)(
        {"latest_checkpointed_iteration": iteration},
        run_rank_n(open)(_get_latest_checkpointed_iterations_path(args.save_args.save_path), "w"),
        indent=4,
    )


def load_checkpoint_for_training(
    args: TrainingArgs,
    model: ModelWrapper,
    optimizer: Optimizer,
    lr_scheduler: LambdaLR,
    train_dataloader: ResumableDataLoader,
) -> Tuple[int, dict]:
    """load checkpoint for training

    Args:
        args (TrainingArgs): arguments for training
        model (ModelWrapper): model to load
        optimizer (Optimizer): optimizer to save
        lr_scheduler (LambdaLR): learning rate scheduler to load
        train_dataloader (ResumableDataLoader): train dataloader to load

    Raises:
        ValueError: if unexpected distributed backend is found

    Returns:
        Tuple[int, dict, dict]: checkpointed iteration, metadata, experiments_tracker state dict
    """

    if args.load_args is None or args.load_args.load_path is None:
        return

    distributed_backend = args.distributed_args.distributed_backend
    load_optimizer = args.load_args.load_optimizer
    load_lr_scheduler = args.load_args.load_lr_scheduler
    load_rng_state = args.load_args.load_rng_state
    load_dataloader_state = args.load_args.load_dataloader_state
    load_experiments_tracker_state = args.load_args.load_experiments_tracker_state
    load_starting_iteration = args.load_args.load_starting_iteration

    iteration = args.load_args.iteration
    if iteration is None:
        iteration = json.load(open(_get_latest_checkpointed_iterations_path(args.load_args.load_path), "r"))[
            "latest_checkpointed_iteration"
        ]

    load_path = _get_base_path(args.load_args.load_path, iteration)

    if distributed_backend == DistributedBackend.deepspeed:
        from deepspeed import DeepSpeedEngine

        assert isinstance(model, DeepSpeedEngine)

        model.load_checkpoint(
            args.load_args.load_path,
            tag=_get_checkpoint_tag(iteration),
            load_optimizer_states=load_optimizer,
            load_lr_scheduler_states=load_lr_scheduler,
        )
    elif distributed_backend == DistributedBackend.torch:
        model_state_dict = get_model_state_dict(model)
        dcp.load(model_state_dict, checkpoint_id=_get_model_path(load_path))
        set_model_state_dict(model, model_state_dict)
        del model_state_dict

        if load_optimizer:
            # TODO add options=StateDictOptions(flatten_optimizer_state_dict=True))
            optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
            dcp.load(optimizer_state_dict, checkpoint_id=_get_optimizer_path(load_path))
            set_optimizer_state_dict(model, optimizer, optim_state_dict=optimizer_state_dict)
            del optimizer_state_dict

        if load_lr_scheduler:
            lr_scheduler.load_state_dict(torch.load(_get_lr_scheduler_path(load_path)))
    else:
        raise ValueError(f"unexpected distributed_backend ({distributed_backend})")

    if load_rng_state:
        rng_state = torch.load(_get_rng_state_path(load_path))
        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])
        torch.cuda.set_rng_state(rng_state["cuda_rng_state"])

    metadata = None
    if os.path.isfile(_get_metadata_path(load_path)):
        metadata = json.load(open(_get_metadata_path(load_path), "r"))

    if load_dataloader_state and train_dataloader is not None:
        train_dataloader.load_state_dict(torch.load(_get_dataloader_path(load_path)))

    experiments_tracker_json = None
    if load_experiments_tracker_state and os.path.exists(_get_experiments_tracker_path(load_path)):
        experiments_tracker_json = json.load(open(_get_experiments_tracker_path(load_path), "r"))

    if not load_starting_iteration:
        iteration = 0

    return iteration, metadata, experiments_tracker_json


def load_checkpoint_for_inference(
    args: Union[InferenceArgs, ExportArgs], mode: Mode, use_meta: bool = False
) -> Tuple[ModelWrapper, TrainingArgs]:
    """load deepspeed checkpoint for inference

    Args:
        args (Union[InferenceArgs, ExportArgs]): arguments
        mode (Mode): training/inference mode
        use_meta (bool): whether to use meta device
    """

    load_path = args.load_args.load_path
    iteration = args.load_args.iteration

    args_file = os.path.join(_get_base_path(load_path, iteration), f"{_TRAINING_CONFIG_PREFIX}.yml")
    args_from_checkpoint = load_yaml(args_file)

    args_from_checkpoint: TrainingArgs = TrainingArgs(**args_from_checkpoint)

    if args.mixed_precision_args is not None:
        log_rank_0(logging.INFO, "overriding mixed precision args")
        args_from_checkpoint.mixed_precision_args = args.mixed_precision_args

    distributed_backend = args_from_checkpoint.distributed_args.distributed_backend
    checkpoint_tp_world_size = args_from_checkpoint.distributed_args.tensor_parallel_size

    with (
        torch.device("meta") if use_meta else torch.device(torch.cuda.current_device()),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
    ):
        model = get_model(args_from_checkpoint, mode)

    if distributed_backend == DistributedBackend.deepspeed:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        state = get_fp32_state_dict_from_zero_checkpoint(load_path, _get_checkpoint_tag(iteration))

        if use_meta:
            model = model.to_empty(device="cpu")

        if model.tuning_method == TuningMethod.prompt_tuning:
            model.load_state_dict(state, strict=False)
        elif model.tuning_method in [TuningMethod.pretraining, TuningMethod.full_finetuning]:
            dtype = string_to_torch_dtype(model.dtype)
            for key in list(state.keys()):
                state[key] = state[key].to(dtype)
                # fix for gradient checkpointing
                state[key.replace("._checkpoint_wrapped_module", "")] = state.pop(key)

            model.load_state_dict(state)
    elif distributed_backend == DistributedBackend.torch:
        if checkpoint_tp_world_size > 1:
            tp_state_dicts = []
            with ProcessGroupManager.set_dummy_tensor_parallel_world_size(checkpoint_tp_world_size):
                for rank in range(checkpoint_tp_world_size):
                    with ProcessGroupManager.set_dummy_tensor_parallel_rank(rank):
                        tp_state_dicts.append(
                            torch.load(_get_model_path(_get_base_path(load_path, iteration)), map_location="cpu")
                        )

            state = unshard(
                config=model.config,
                tensor_parallel_state_dicts=tp_state_dicts,
                tensor_parallel_embeddings=args_from_checkpoint.distributed_args.tensor_parallel_embeddings,
                prefix="model.",
                # with bf16 nn.Embedding backward pass is non-deteministic
                check_correctness=args_from_checkpoint.distributed_args.tensor_parallel_embeddings,
            )
            del tp_state_dicts
        else:
            state = {}
            _load_state_dict(
                state,
                storage_reader=FileSystemReader(_get_model_path(_get_base_path(load_path, iteration))),
                planner=_EmptyStateDictLoadPlanner(),
                no_dist=True,
            )

        if use_meta:
            model = model.to_empty(device="cpu")

        model.load_state_dict(state)
    else:
        raise ValueError(f"unexpected distributed_backend ({args['distributed_args']['distributed_backend']})")

    return model, args_from_checkpoint


@run_rank_n
def save_args(args: Union[TrainingArgs, InferenceArgs], save_path: str, mode: Mode) -> None:
    """saves training args as a json

    Args:
        args (Union[TrainingArgs, InferenceArgs]): arguments for training or inference
        save_path (str): save location on disk
    """

    file_prefix = _TRAINING_CONFIG_PREFIX if mode == Mode.training else _INFERENCE_CONFIG_PREFIX
    save_path = os.path.join(save_path, f"{file_prefix}.yml")
    yaml.dump(args.to_dict(), open(save_path, "w"), indent=2)


def _get_checkpoint_tag(iteration: int) -> str:
    return f"global_step{iteration}"


def _get_base_path(path: str, iteration: int) -> str:
    return os.path.join(path, _get_checkpoint_tag(iteration))


def _get_model_path(path: str) -> str:
    suffix = "model"
    if ProcessGroupManager.get_tensor_parallel_world_size() > 1:
        suffix += f"-{ProcessGroupManager.get_tensor_parallel_rank()}.pt"

    return os.path.join(path, suffix)


def _get_optimizer_path(path: str) -> str:
    suffix = "optimizer"
    if ProcessGroupManager.get_tensor_parallel_world_size() > 1:
        suffix += f"-{ProcessGroupManager.get_tensor_parallel_rank()}.pt"

    return os.path.join(path, suffix)


def _get_lr_scheduler_path(path: str) -> str:
    return os.path.join(path, "lr_scheduler.pt")


def _get_dataloader_path(path: str) -> str:
    return os.path.join(path, "dataloader", f"dataloader-{ProcessGroupManager.get_data_parallel_rank()}.pt")


def _get_rng_state_path(path: str) -> str:
    return os.path.join(path, "rng_state", f"rng_state-{ProcessGroupManager.get_global_rank()}.pt")


def _get_latest_checkpointed_iterations_path(path: str) -> str:
    return os.path.join(path, "latest_checkpointed_iteration.json")


def _get_experiments_tracker_path(path: str) -> str:
    return os.path.join(path, "experiments_tracker.json")


def _get_metadata_path(path: str) -> str:
    return os.path.join(path, "metadata.json")

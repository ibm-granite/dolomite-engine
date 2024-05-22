import logging
from functools import partial
from typing import List, Tuple, Union

import torch
from transformers import AutoTokenizer

from ...arguments import InferenceArgs, TrainingArgs
from ...enums import DatasetSplit, Mode, TuningMethod
from ...utils import get_global_rank, get_world_size, log_rank_0
from ..dataloader import DispatchingDataLoader, ResumableDataLoader
from ..sampler import BlendedDistributedSampler
from ..utils import collate_fn
from .base import BaseDataset, BlendedDatasets
from .debug import DebugDataset
from .instruction_tuning import AlpacaDataset, DollyDataset, SlimOrcaDataset
from .jsonlines import JSONLinesDataset
from .sst2 import SST2Dataset


_DATASETS_LIST = {
    "AlpacaDataset": AlpacaDataset,
    "DebugDataset": DebugDataset,
    "DollyDataset": DollyDataset,
    "JSONLinesDataset": JSONLinesDataset,
    "SlimOrcaDataset": SlimOrcaDataset,
    "SST2Dataset": SST2Dataset,
}


def get_datasets_list(
    args: Union[TrainingArgs, InferenceArgs],
    split: DatasetSplit,
    mode: Mode,
    tokenizer: AutoTokenizer,
    is_encoder_decoder: bool,
) -> Tuple[List[BaseDataset], List[int]]:
    """get the list of datasets from their configs

    Args:
        args (Union[TrainingArgs, InferenceArgs]): arguments based on training / inference mode
        split (DatasetSplit): train / val / test split
        mode (Mode): training / inference mode for running the program
        tokenizer (AutoTokenizer): tokenizer
        is_encoder_decoder (bool): whether the model is an encoder-decoder or a decoder-only model

    Raises:
        ValueError: if invalid class_name for dataset is found

    Returns:
        Tuple[List[BaseDataset], List[int]]: tuple of list of datasets and the respective dataset sampling ratios
    """

    dataset_args_list = args.datasets
    tuning_method = args.tuning_args.tuning_method
    num_virtual_tokens = (
        args.tuning_args.prompt_tuning_args.num_virtual_tokens
        if args.tuning_args.tuning_method == TuningMethod.prompt_tuning
        else None
    )

    datasets_list = []
    data_sampling_ratios = []
    for data_args in dataset_args_list:
        if data_args.class_name not in _DATASETS_LIST:
            raise ValueError(f"invalid class_name ({data_args.class_name}) for dataset")

        dataset = _DATASETS_LIST[data_args.class_name](
            class_args=data_args.class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            tuning_method=tuning_method,
            data_name=data_args.data_name,
            input_format=data_args.input_format,
            output_format=data_args.output_format,
            max_input_tokens=data_args.max_input_tokens,
            max_output_tokens=data_args.max_output_tokens,
            num_virtual_tokens=num_virtual_tokens if tuning_method == TuningMethod.prompt_tuning else None,
        )

        if len(dataset) > 0:
            datasets_list.append(dataset)
            data_sampling_ratios.append(data_args.data_sampling_ratio)

            log_rank_0(
                logging.INFO, f"examples in {dataset.__class__.__name__} ({data_args.data_name}) = {len(dataset)}"
            )

    assert all([i is not None for i in data_sampling_ratios]) or all(
        [i is None for i in data_sampling_ratios]
    ), "either all data_sampling_ratios should be specified or all should be None"
    if all([i is None for i in data_sampling_ratios]):
        data_sampling_ratios = [len(i) for i in datasets_list]

    return datasets_list, data_sampling_ratios


def get_dataloader(
    args: Union[TrainingArgs, InferenceArgs],
    split: DatasetSplit,
    mode: Mode,
    tokenizer: AutoTokenizer,
    is_encoder_decoder: bool,
) -> Tuple[ResumableDataLoader]:
    """prepares datasets and sampler

    Args:
        args (Union[TrainingArgs, InferenceArgs]): arguments based on training / inference mode
        split (DatasetSplit): train / val / test split
        mode (Mode): training / inference mode
        tokenizer (AutoTokenizer): tokenizer
        is_encoder_decoder (bool): whether the model is an encoder-decoder or a decoder-only model

    Returns:
        Tuple[ResumableDataLoader]: dataloader for a blended dataset
    """

    assert mode == Mode.training, "blended dataset is only supported in training mode"

    if args.distributed_args.dispatching_dataloader:
        dataloader = _get_dispatching_dataloader(
            args, split=split, mode=mode, tokenizer=tokenizer, is_encoder_decoder=is_encoder_decoder
        )
    else:
        dataloader = _get_non_dispatching_dataloader(
            args, split=split, mode=mode, tokenizer=tokenizer, is_encoder_decoder=is_encoder_decoder
        )

    return dataloader


def _get_dispatching_dataloader(
    args: Union[TrainingArgs, InferenceArgs],
    split: DatasetSplit,
    mode: Mode,
    tokenizer: AutoTokenizer,
    is_encoder_decoder: bool,
) -> Tuple[ResumableDataLoader]:
    micro_batch_size = args.training_parameters.micro_batch_size

    source_rank = get_global_rank() // torch.cuda.device_count()
    broadcast_ranks = list(range(source_rank, torch.cuda.device_count()))
    num_nodes = get_world_size() // torch.cuda.device_count()

    if get_global_rank() == source_rank:
        datasets_list, data_sampling_ratios = get_datasets_list(
            args=args,
            split=split,
            mode=Mode.training,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
        )

        if len(datasets_list) == 0:
            return None

        blended_dataset = BlendedDatasets(datasets=datasets_list, split=split)
    else:
        blended_dataset = None

    # each node is given a data sampler
    # TODO modify this when we add model parallelism

    # sampler routes to the dispatching parent worker
    sampler = BlendedDistributedSampler(
        dataset=blended_dataset,
        data_sampling_ratios=[1] if len(datasets_list) == 1 else data_sampling_ratios,
        ignore_sampling_proportion_for_validation=args.training_parameters.ignore_sampling_proportion_for_validation,
        num_replicas=num_nodes,
        rank=source_rank,
        shuffle=split == DatasetSplit.train,
        seed=args.random_args.seed,
        drop_last=False,
    )

    # dataloader does local dispatching and thus needs source_rank and broadcast_ranks
    dataloader = DispatchingDataLoader(
        blended_dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            mode=mode,
            loss_mask=args.training_parameters.loss_mask,
            eos_token_id=tokenizer.eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            use_padding_free_transformer=args.model_args.use_padding_free_transformer,
        ),
        source_rank=source_rank,
        broadcast_ranks=broadcast_ranks,
    )

    _log_dataset(
        blended_dataset=blended_dataset,
        sampler=sampler,
        split=split,
        num_training_steps=args.training_parameters.num_training_steps,
        gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
    )

    return dataloader


def _get_non_dispatching_dataloader(
    args: Union[TrainingArgs, InferenceArgs],
    split: DatasetSplit,
    mode: Mode,
    tokenizer: AutoTokenizer,
    is_encoder_decoder: bool,
) -> Tuple[ResumableDataLoader]:
    micro_batch_size = args.training_parameters.micro_batch_size

    datasets_list, data_sampling_ratios = get_datasets_list(
        args=args,
        split=split,
        mode=Mode.training,
        tokenizer=tokenizer,
        is_encoder_decoder=is_encoder_decoder,
    )

    if len(datasets_list) == 0:
        return None

    blended_dataset = BlendedDatasets(datasets=datasets_list, split=split)

    # routing to data parallel worker is done by sampler
    sampler = BlendedDistributedSampler(
        dataset=blended_dataset,
        data_sampling_ratios=[1] if len(datasets_list) == 1 else data_sampling_ratios,
        ignore_sampling_proportion_for_validation=args.training_parameters.ignore_sampling_proportion_for_validation,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=split == DatasetSplit.train,
        seed=args.random_args.seed,
        drop_last=False,
    )

    # dataloader is unaware of data parallel routing
    dataloader = ResumableDataLoader(
        blended_dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            mode=mode,
            loss_mask=args.training_parameters.loss_mask,
            eos_token_id=tokenizer.eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            use_padding_free_transformer=args.model_args.use_padding_free_transformer,
        ),
    )

    _log_dataset(
        blended_dataset=blended_dataset,
        sampler=sampler,
        split=split,
        num_training_steps=args.training_parameters.num_training_steps,
        gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
    )

    return dataloader


def _log_dataset(
    blended_dataset: BlendedDatasets,
    sampler: BlendedDistributedSampler,
    split: DatasetSplit,
    num_training_steps: int,
    gradient_accumulation_steps: int,
    micro_batch_size: int,
) -> None:
    log_rank_0(logging.INFO, f"{'-' * 25} {split.value} {'-' * 25}")
    log_rank_0(logging.INFO, blended_dataset)

    if split == DatasetSplit.train:
        total_samples_seen = num_training_steps * gradient_accumulation_steps * micro_batch_size * get_world_size()
    else:
        if len(blended_dataset) % (micro_batch_size * get_world_size()) == 0:
            num_steps = len(blended_dataset) // (micro_batch_size * get_world_size())
        else:
            num_steps = (len(blended_dataset) // (micro_batch_size * get_world_size())) + 1

        total_samples_seen = num_steps * micro_batch_size * get_world_size()

    log_rank_0(logging.INFO, "*" * 57)
    log_rank_0(logging.INFO, f"total samples seen = {total_samples_seen}")
    log_rank_0(logging.INFO, f"total epochs for the dataset mixture = {total_samples_seen / len(blended_dataset)}")
    log_rank_0(logging.INFO, sampler)
    log_rank_0(logging.INFO, "-" * 57)

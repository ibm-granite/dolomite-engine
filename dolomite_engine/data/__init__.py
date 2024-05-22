import logging
from functools import partial
from typing import Iterable, Tuple, Union

import torch
from transformers import AutoTokenizer

from ..arguments import InferenceArgs, TrainingArgs
from ..enums import DatasetSplit, Mode
from ..utils import get_global_rank, get_world_size, log_rank_0
from .dataloader import DispatchingDataLoader, ResumableDataLoader
from .datasets import BaseDataset, BlendedDatasets, get_datasets_list
from .megatron import get_megatron_gpt_dataloaders
from .sampler import BlendedDistributedSampler
from .utils import collate_fn


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

    micro_batch_size = args.training_parameters.micro_batch_size
    dispatching_dataloader = args.distributed_args.dispatching_dataloader

    if dispatching_dataloader:
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
    else:
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

    log_dataset(
        blended_dataset,
        split=split,
        num_training_steps=args.training_parameters.num_training_steps,
        gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
        micro_batch_size=args.training_parameters.micro_batch_size,
    )

    return dataloader


def log_dataset(
    blended_dataset: BlendedDatasets,
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


def infinite_iterator(x: Iterable) -> Iterable:
    """converts and iterable into a non-ending infinite iterable

    Args:
        x (Iterable): the iterable to convert

    Returns:
        Iterable: the converted iterable

    Yields:
        Iterator[Iterable]: an element from the original iterator
    """

    while True:
        for i in x:
            yield i

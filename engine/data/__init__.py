import logging
from typing import Iterable, List, Tuple, Union

from transformers import AutoTokenizer

from ..arguments import InferenceArgs, TrainingArgs
from ..enums import DatasetSplit, Mode, TuningMethod
from ..utils import get_world_size, log_rank_0
from .base import BaseDataset, BlendedDatasets, collate
from .dataloader import DataLoader
from .debug import DebugDataset
from .instruction_tuning import AlpacaDataset, DollyDataset
from .jsonlines import JSONLinesDataset
from .sampler import BlendedDistributedSampler
from .sst2 import SST2Dataset


_DATASETS_LIST = {
    "AlpacaDataset": AlpacaDataset,
    "DebugDataset": DebugDataset,
    "DollyDataset": DollyDataset,
    "JSONLinesDataset": JSONLinesDataset,
    "SST2Dataset": SST2Dataset,
}


def get_dataloader(
    args: Union[TrainingArgs, InferenceArgs],
    split: DatasetSplit,
    mode: Mode,
    tokenizer: AutoTokenizer,
    is_encoder_decoder: bool,
) -> Tuple[DataLoader]:
    """prepares datasets and sampler

    Args:
        args (Union[TrainingArgs, InferenceArgs]): arguments based on training / inference mode
        split (DatasetSplit): train / val / test split
        mode (Mode): training / inference mode
        tokenizer (AutoTokenizer): tokenizer
        is_encoder_decoder (bool): whether the model is an encoder-decoder or a decoder-only model

    Returns:
        Tuple[DataLoader]: dataloader for a blended dataset
    """

    assert mode == Mode.training, "blended dataset is only supported in training mode"

    datasets_list, data_sampling_ratios = get_datasets_list(
        args=args,
        split=split,
        mode=Mode.training,
        tokenizer=tokenizer,
        is_encoder_decoder=is_encoder_decoder,
    )

    blended_dataset = BlendedDatasets(datasets=datasets_list, split=split)

    log_rank_0(logging.INFO, f"{'-' * 25} {split.value} {'-' * 25}")
    log_rank_0(logging.INFO, blended_dataset)

    sampler = BlendedDistributedSampler(
        dataset=blended_dataset,
        data_sampling_ratios=data_sampling_ratios if len(datasets_list) == 1 else [1],
        ignore_sampling_proportion_for_validation=args.training_parameters.ignore_sampling_proportion_for_validation,
        shuffle=split == DatasetSplit.train,
        seed=args.random_args.seed,
        drop_last=False,
    )

    batch_size_per_gpu = args.training_parameters.batch_size_per_gpu

    dataloader = DataLoader(
        blended_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        collate_fn=collate,
    )

    if split == DatasetSplit.train:
        total_samples_seen = (
            args.training_parameters.num_training_steps
            * args.training_parameters.gradient_accumulation_steps
            * batch_size_per_gpu
            * get_world_size()
        )
    else:
        if len(blended_dataset) % (batch_size_per_gpu * get_world_size()) == 0:
            num_steps = len(blended_dataset) // (batch_size_per_gpu * get_world_size())
        else:
            num_steps = (len(blended_dataset) // (batch_size_per_gpu * get_world_size())) + 1

        total_samples_seen = num_steps * batch_size_per_gpu * get_world_size()

    log_rank_0(logging.INFO, "*" * 57)
    log_rank_0(logging.INFO, f"total samples seen = {total_samples_seen}")
    log_rank_0(logging.INFO, f"total epochs for the dataset mixture = {total_samples_seen / len(blended_dataset)}")
    log_rank_0(logging.INFO, sampler)
    log_rank_0(logging.INFO, "-" * 57)

    return dataloader


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
        else None,
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

    return datasets_list, data_sampling_ratios


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

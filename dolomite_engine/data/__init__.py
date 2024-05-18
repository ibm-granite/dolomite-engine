import logging
from functools import partial
from typing import Iterable, List, Tuple, Union

from transformers import AutoTokenizer

from ..arguments import InferenceArgs, TrainingArgs
from ..enums import DatasetSplit, Mode, PaddingSide, TuningMethod
from ..utils import get_world_size, log_rank_0
from .base import BaseDataset, BlendedDatasets
from .dataloader import DataLoader
from .debug import DebugDataset
from .instruction_tuning import AlpacaDataset, DollyDataset, SlimOrcaDataset
from .jsonlines import JSONLinesDataset
from .megatron import get_megatron_gpt_dataloaders
from .sampler import BlendedDistributedSampler
from .sst2 import SST2Dataset
from .utils import collate_fn


_DATASETS_LIST = {
    "AlpacaDataset": AlpacaDataset,
    "DebugDataset": DebugDataset,
    "DollyDataset": DollyDataset,
    "JSONLinesDataset": JSONLinesDataset,
    "SlimOrcaDataset": SlimOrcaDataset,
    "SST2Dataset": SST2Dataset,
}


def get_dataloader(
    args: Union[TrainingArgs, InferenceArgs],
    split: DatasetSplit,
    mode: Mode,
    tokenizer: AutoTokenizer,
    is_encoder_decoder: bool,
    padding_side: PaddingSide,
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

    if len(datasets_list) == 0:
        return None

    blended_dataset = BlendedDatasets(datasets=datasets_list, split=split)

    log_rank_0(logging.INFO, f"{'-' * 25} {split.value} {'-' * 25}")
    log_rank_0(logging.INFO, blended_dataset)

    sampler = BlendedDistributedSampler(
        dataset=blended_dataset,
        data_sampling_ratios=[1] if len(datasets_list) == 1 else data_sampling_ratios,
        ignore_sampling_proportion_for_validation=args.training_parameters.ignore_sampling_proportion_for_validation,
        shuffle=split == DatasetSplit.train,
        seed=args.random_args.seed,
        drop_last=False,
    )

    micro_batch_size = args.training_parameters.micro_batch_size

    dataloader = DataLoader(
        blended_dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            mode=mode,
            padding_side=padding_side,
            loss_mask=args.training_parameters.loss_mask,
            eos_token_id=tokenizer.eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            use_padding_free_transformer=args.model_args.use_padding_free_transformer,
        ),
    )

    if split == DatasetSplit.train:
        total_samples_seen = (
            args.training_parameters.num_training_steps
            * args.training_parameters.gradient_accumulation_steps
            * micro_batch_size
            * get_world_size()
        )
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

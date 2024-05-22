from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer

from ..defaults import INPUT_FORMAT, OUTPUT_FORMAT
from ..enums import DatasetSplit, Mode, TuningMethod


class BaseDataset(torch.utils.data.Dataset):
    """BaseDataset class to be implemented by all the datasets"""

    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        tuning_method: TuningMethod,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
        num_virtual_tokens: int = None,
    ) -> None:
        super().__init__()

        self.split = split
        self.mode = mode

        self.class_args = class_args

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        self.tuning_method = tuning_method
        # used for prompt tuning
        self.num_virtual_tokens = num_virtual_tokens if self.tuning_method == TuningMethod.prompt_tuning else None

        self.data_name = data_name
        self.input_format = input_format
        self.output_format = output_format

        # if format is __input__ or __output__ formatting is a no-op
        self.do_format_input = self.input_format != INPUT_FORMAT
        self.do_format_output = self.output_format != OUTPUT_FORMAT

        # length to use for trimming (excludes eos)
        self.max_input_tokens = get_max_input_length(
            max_input_tokens,
            self.tuning_method,
            self.num_virtual_tokens,
            self.is_encoder_decoder,
        )
        self.max_output_tokens = get_max_output_length(
            max_output_tokens,
            self.tuning_method,
            self.num_virtual_tokens,
            self.is_encoder_decoder,
        )

        self.examples = []

    def construct_input_from_format(self, input: str) -> str:
        """construct input with the specified input_format

        Args:
            input (str): input text

        Returns:
            str: formatted text
        """

        if self.do_format_input:
            return self.input_format.replace(INPUT_FORMAT, input, 1)
        return input

    def construct_output_from_format(self, output: str) -> str:
        """construct output with the specified output_format

        Args:
            output (str): output text

        Returns:
            str: formatted text
        """

        if self.do_format_output:
            return self.output_format.replace("__output__", output, 1)
        return output

    def get_input_output_token_ids(self, input: str, output: str) -> dict:
        """tokenizes the input and output text

        Args:
            input (str): input text
            output (str): output text

        Returns:
            dict: an example
        """

        eos_token_id: int = self.tokenizer.eos_token_id

        input: List[int] = self.tokenizer(input, add_special_tokens=False)["input_ids"]

        if self.is_encoder_decoder:
            if self.max_input_tokens is not None:
                input = input[: self.max_input_tokens - 1]
            input.append(eos_token_id)
        else:
            if self.max_input_tokens is not None:
                input = input[: self.max_input_tokens]

        if self.mode == Mode.training:
            output: List[int] = self.tokenizer(output, add_special_tokens=False)["input_ids"]

            if self.max_output_tokens is not None:
                output = output[: self.max_output_tokens - 1]
            output.append(eos_token_id)

            if not self.is_encoder_decoder:
                input.extend(output)

            result = {"input": input, "output": output}
        else:
            result = {"input": input}

        return result

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        return

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)


class BlendedDatasets(torch.utils.data.Dataset):
    """Concatenated list of datasets for training or inference"""

    def __init__(self, datasets: List[BaseDataset], split: DatasetSplit) -> None:
        super().__init__()

        self.split = split
        self.datasets = datasets

        num_examples_in_each_dataset = self.get_num_examples_in_each_dataset()
        self.num_examples = sum(num_examples_in_each_dataset)
        self.start_indices = np.cumsum([0] + num_examples_in_each_dataset[:-1]).tolist()

    def get_num_datasets(self) -> int:
        """returns the number of datasets in the mixture

        Returns:
            int: number of datasets in the mixture
        """

        return len(self.datasets)

    def get_num_examples_in_each_dataset(self) -> List[int]:
        """returns the number of examples in each dataset component

        Returns:
            List[int]: the number of examples in each dataset component
        """

        return [len(dataset) for dataset in self.datasets]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        return

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict:
        num_datasets = self.get_num_datasets()

        # get the dataset the example belongs to
        dataset_index = num_datasets - 1
        for i in range(num_datasets):
            if index < self.start_indices[i]:
                dataset_index = i - 1
                break

        # get the position of the example in the specific dataset
        index -= self.start_indices[dataset_index]

        # get the example
        example = self.datasets[dataset_index][index]

        return example

    def __repr__(self) -> str:
        x = f"number of datasets = {self.get_num_datasets()}\n"
        x += f"total examples in the entire dataset mixture = {len(self)}"

        for dataset in self.datasets:
            x += f"\nexamples in {dataset.__class__.__name__} ({dataset.data_name}) = {len(dataset)}"

        return x


def get_max_input_length(
    max_input_tokens_specified: int,
    tuning_method: TuningMethod,
    num_virtual_tokens: int,
    is_encoder_decoder: bool,
) -> int:
    """max input length for the model, depends on the training / inference type and whether the model is decoder-only or encoder-decoder

    Args:
        max_input_tokens_specified (int): maximum number of specified input tokens
        tuning_method (TuningMethod): full finetuning / prompt tuning
        num_virtual_tokens (int): virtual tokens for prompt tuning
        is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

    Returns:
        int: max input length
    """

    if max_input_tokens_specified is None:
        return None

    max_input_tokens = max_input_tokens_specified

    if tuning_method == TuningMethod.prompt_tuning:
        max_input_tokens -= num_virtual_tokens

    if is_encoder_decoder:
        max_input_tokens -= 1

    return max_input_tokens


def get_max_output_length(
    max_output_tokens_specified: int,
    tuning_method: TuningMethod,
    num_virtual_tokens: int,
    is_encoder_decoder: bool,
) -> int:
    """max output length for the model, depends on the training / inference type and whether the model is decoder-only or encoder-decoder

    Args:
        max_output_tokens_specified (int): maximum number of specified output tokens
        tuning_method (TuningMethod): full finetuning / prompt tuning
        num_virtual_tokens (int): virtual tokens for prompt tuning
        is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

    Returns:
        int: max output length
    """

    if max_output_tokens_specified is None:
        return None

    max_output_tokens = max_output_tokens_specified - 1

    if is_encoder_decoder:
        if tuning_method == TuningMethod.prompt_tuning:
            max_output_tokens -= num_virtual_tokens

    return max_output_tokens


# def _binary_search_index(x: List[int], i: int) -> int:
#     middle = len(x) // 2

#     if x[middle] <= i < x[middle + 1]:
#         result = middle
#     elif x[middle] <= i:
#         result = middle + _binary_search_index(x[middle + 1:], i)
#     elif x[middle] > i:
#         result = _binary_search_index(x[:middle], i)

#     return result

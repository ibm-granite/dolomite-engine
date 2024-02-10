import json
import os
from typing import List

from transformers import AutoTokenizer

from ...enums import DatasetSplit, Mode, TuningMethod
from ..utils import train_val_test_split
from .base import BaseInstructionDataset


class AlpacaDataset(BaseInstructionDataset):
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
        super().__init__(
            class_args=class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            tuning_method=tuning_method,
            data_name=data_name,
            input_format=input_format,
            output_format=output_format,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            num_virtual_tokens=num_virtual_tokens,
        )

        self.examples = self.prepare_examples()

    def prepare_examples(self) -> List[dict]:
        data = json.load(open(os.path.join(self.data_path, "alpaca_data.json"), "r"))
        data = train_val_test_split(
            data,
            self.split,
            self.class_args.get("seed", 42),
            self.class_args.get("val_samples", 2600),
            self.class_args.get("test_samples", 2600),
        )

        examples = []
        for raw_example in data:
            input = self.construct_input_from_format(raw_example["instruction"], raw_example.get("input", ""))
            output = self.construct_output_from_format(raw_example["output"].strip())

            example = self.get_input_output_token_ids(input, output)
            examples.append(example)

        return examples

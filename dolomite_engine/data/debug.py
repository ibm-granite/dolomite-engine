from transformers import AutoTokenizer

from ..enums import DatasetSplit, Mode, TuningMethod
from .base import BaseDataset


class DebugDataset(BaseDataset):
    """A dummy dataset for profiling and timing the code"""

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

        if self.do_format_input:
            raise ValueError("DebugDataset does not support input formatting")
        if self.do_format_output:
            raise ValueError("DebugDataset does not support output formatting")

        self._length = class_args.get("num_examples")
        assert isinstance(self._length, int) and self._length > 0

        self._token_id = class_args.get("token_id", self.tokenizer.eos_token_id)

        if mode == Mode.training:
            self._example = {
                "input": [self._token_id] * self.max_input_tokens,
                "output": [self._token_id] * self.max_output_tokens,
            }
        else:
            self._example = {"output": [self._token_id] * self.max_input_tokens}

    def __getitem__(self, index: int) -> dict:
        return self._example

    def __len__(self) -> int:
        return self._length

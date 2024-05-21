from unittest import TestCase

import torch.distributed
from transformers import AutoConfig, AutoTokenizer

from dolomite_engine.arguments import TrainingArgs
from dolomite_engine.data import get_dataloader
from dolomite_engine.enums import DatasetSplit, Mode, PaddingSide
from dolomite_engine.utils import load_yaml


class InstructionTuningTest(TestCase):
    def test_alpaca_dataloader(self) -> None:
        args = self.load_training_args_for_unit_tests()

        config = AutoConfig.from_pretrained(args.model_args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_args.model_name)

        dataloader = get_dataloader(
            args, DatasetSplit.train, Mode.training, tokenizer, config.is_encoder_decoder, PaddingSide.left
        )

    def load_training_args_for_unit_tests(self) -> TrainingArgs:
        return TrainingArgs(**load_yaml("tests/data/test_config.yml"))

from functools import partial

from transformers import AutoTokenizer

from dolomite_engine.data import (
    BlendedDatasets,
    BlendedDistributedSampler,
    ResumableDataLoader,
    collate_fn,
    get_dataloader,
    get_datasets_list,
)
from dolomite_engine.enums import DatasetSplit, Mode

from .test_commons import TestCommons


class MegatronDatasetTest(TestCommons):
    def test_dataloader_returns_correct_batch_size(self) -> None:
        args = TestCommons.load_training_args_for_unit_tests()
        split = DatasetSplit.train
        mode = Mode.training

        tokenizer = AutoTokenizer.from_pretrained(args.model_args.model_name)
        dataloader = get_dataloader(args, split=split, mode=mode, tokenizer=tokenizer, is_encoder_decoder=False)

        for example in dataloader:
            assert example["input_ids"].shape[0] == args.training_parameters.micro_batch_size
            break

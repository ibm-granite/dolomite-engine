import json
import os
from copy import deepcopy
from typing import List, Tuple, Union

import torch
from deepspeed import DeepSpeedEngine
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig

from src.arguments import InferenceArgs, TrainingArgs
from src.constants import DatasetConfigKeys, Mode, OptimizerKeys, TrainingInferenceType
from src.utils import get_deepspeed_config, get_local_rank, register_profiler, register_timer, run_rank_n, warn_rank_0


def pad(arrays: list, padding: int, max_length: int = None) -> Tuple[List[int], List[int]]:
    """pads the arrays with the specified padding value

    Args:
        arrays (list): token ids
        padding (int): token id to pad with
        max_length (int, optional): length to pad to. Defaults to None. If None, pads to the longest sequence

    Returns:
        Tuple[List[int], List[int]]: token ids and the corresponding attention masks
    """

    if max_length is None:
        max_length = max(list(map(len, arrays)))

    inputs = [[padding] * (max_length - len(array)) + array for array in arrays]
    masks = [[0] * (max_length - len(array)) + [1] * len(array) for array in arrays]

    return inputs, masks


class Model(torch.nn.Module):
    """Model class which wraps any HuggingFace model"""

    @register_profiler("initialize_model")
    @register_timer("initialize_model")
    def __init__(self, args: Union[TrainingArgs, InferenceArgs], mode: Mode):
        """initializes a Model wrapper for a HuggingFace model

        Args:
            args (Union[TrainingArgs, InferenceArgs]): arguments based on training / inference mode
            mode (Mode): training / inference mode for running the program
        """

        super().__init__()

        self.mode = mode
        self.model_name = args.model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.is_encoder_decoder = self.config.is_encoder_decoder
        self.training_inference_type = args.training_inference_type
        self.dtype = args.dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.original_vocab_size = len(self.tokenizer)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.training_inference_type == TrainingInferenceType.full_finetuning:
            if mode == Mode.training:
                # this tells from_pretrained to instantiate directly on gpus
                # this only instantiates a single instance of the model across the ranks
                self.deepspeed_config = HfDeepSpeedConfig(get_deepspeed_config(args))
                self.model = args.model_class.from_pretrained(self.model_name)
            else:
                self.model = args.model_class.from_pretrained(self.model_name, torch_dtype=self.dtype)
        elif args.training_inference_type == TrainingInferenceType.prompt_tuning:
            self.peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                prompt_tuning_init=args.prompt_tuning_init,
                num_virtual_tokens=args.num_virtual_tokens,
                prompt_tuning_init_text=args.prompt_tuning_init_text,
                tokenizer_name_or_path=args.model_name,
            )

            self.model = args.model_class.from_pretrained(self.model_name, torch_dtype=self.dtype)
            self.model = get_peft_model(self.model, self.peft_config)

        if mode == Mode.training:
            # if using deepspeed
            self.input_device = get_local_rank()
        else:
            self.input_device = 0
            if not torch.cuda.is_available():
                warn_rank_0("no CUDA device found, running on CPU")
                self.input_device = "cpu"

            self.to(self.input_device)

    def post_init(self) -> None:
        """a post init method for expanding word embeddings"""

        if len(self.tokenizer) != self.original_vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    @register_profiler("forward_pass")
    @register_timer("forward_pass")
    def forward(self, batch: Tuple[List[int]]) -> torch.Tensor:
        """forward function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch

        Returns:
            torch.Tensor: loss tensor
        """

        batch = self.prepare_batch(batch)

        for i in batch:
            batch[i] = batch[i].to(self.input_device)

        model_outputs = self.model(**batch)

        return model_outputs.loss

    @register_profiler("generate")
    @register_timer("generate")
    def generate(self, batch: Tuple[List[int]], generate_kwargs: dict) -> List[str]:
        """generate function for a batch

        Args:
            batch (dict): a dict of key, value pairs for a batch
            generate_kwargs (dict): generate kwargs for the batch

        Returns:
            List[str]: list of generated text. input is trimmed from the generated text
        """

        batch = self.prepare_batch(batch)

        for i in batch:
            batch[i] = batch[i].to(self.input_device)

        generated = self.model.generate(**batch, **generate_kwargs)

        if not self.is_encoder_decoder:
            generated = generated[:, batch["input_ids"].shape[1] :]

        num_generated_tokens = (generated != self.tokenizer.pad_token_id).sum(dim=-1).tolist()
        generated_text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        return generated_text, num_generated_tokens

    @register_profiler("prepare_batch")
    @register_timer("prepare_batch")
    def prepare_batch(self, batch: Tuple[List[int]]) -> dict:
        """prepares the batch with padding to pass into the forward function of the HuggingFace model

        Args:
            inputs (List[int]): input tokens
            outputs (List[int], optional): output tokens, optional when running generation but required for training. Defaults to None.

        Returns:
            dict: dict containing input_ids, attention_mask and labels if outputs is specified
        """

        if self.mode == Mode.training:
            inputs, outputs = batch
        else:
            inputs = batch

        result = {}

        if self.mode == Mode.training:
            assert outputs is not None, "outputs can't be None during training"

            max_length = None
            if not self.is_encoder_decoder:
                max_length = max(list(map(len, inputs)))

            input_ids, attention_mask = pad(inputs, padding=self.tokenizer.pad_token_id, max_length=max_length)
            labels, _ = pad(outputs, padding=-100, max_length=max_length)

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            labels = torch.tensor(labels)

            result["labels"] = labels
        elif self.mode == Mode.inference:
            input_ids, attention_mask = pad(inputs, padding=self.tokenizer.pad_token_id)

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask
        return result


class ModelCheckpointer:
    """class for loading and saving models"""

    @classmethod
    @register_timer("load_checkpoint_for_training")
    def load_checkpoint_for_training(cls, model: DeepSpeedEngine, load_path: str) -> None:
        """loads the deepspeed checkpoint saved for training

        Args:
            model (DeepSpeedEngine): loaded checkpoint is filled into this model
            load_path (str): path to load the deepspeed checkpoint from
        """

        model.load_checkpoint(load_path)

    @classmethod
    @register_timer("save_deepspeed_checkpoint")
    def save_deepspeed_checkpoint(cls, model: DeepSpeedEngine, args: TrainingArgs, save_path: str) -> None:
        """save deepspeed checkpoint during training

        Args:
            model (DeepSpeedEngine): model to save
            args (InferenceArgs): arguments for training
            save_path (str): save location on disk
        """

        model.save_checkpoint(save_path)
        cls.save_training_args(
            args, os.path.join(save_path, f"global_step{model.global_steps}", "training_config.json")
        )

    @classmethod
    @register_timer("load_checkpoint_for_inference")
    def load_checkpoint_for_inference(cls, model: Model, load_path: str) -> None:
        """load deepspeed checkpoint for inference

        Args:
            model (Model): model to save
            load_path (str): path to load the deepspeed checkpoint from
        """

        checkpoint_dir = os.path.dirname(load_path)
        tag = os.path.basename(load_path)
        state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

        if model.training_inference_type == TrainingInferenceType.prompt_tuning:
            model.load_state_dict(state, strict=False)
        elif model.training_inference_type == TrainingInferenceType.full_finetuning:
            for key in state:
                state[key] = state[key].to(model.dtype)

            model.load_state_dict(state)

    @classmethod
    @register_timer("convert_deepspeed_to_huggingface_checkpoint")
    def convert_deepspeed_to_huggingface_checkpoint(cls, model: Model, load_path: str, save_path: str) -> None:
        """load the model as a deepspeed checkpoint, converts to huggingface and saves it

        Args:
            model (Model): model to save
            load_path (str): path to load the deepspeed checkpoint from
            save_path (str): save location on disk for huggingface checkpoint
        """

        cls.load_checkpoint_for_inference(model, load_path)

        model.tokenizer.save_pretrained(save_path)
        model.model.save_pretrained(save_path)

        args = json.load(open(os.path.join(load_path, "training_config.json"), "r"))
        json.dump(args, open(os.path.join(save_path, "training_config.json"), "w"), indent=4)

    @classmethod
    @run_rank_n
    def save_training_args(cls, args: TrainingArgs, save_path: str) -> None:
        """saves training args as a json

        Args:
            args (TrainingArgs): arguments for training
            save_path (str): save location on disk
        """

        args = deepcopy(args)

        # model_class
        args.model_class = args.model_class.__name__
        # dtype
        args.dtype = str(args.dtype).split(".")[1]

        # training_inference_type
        args.training_inference_type = args.training_inference_type.value
        # prompt_tuning_init
        if args.prompt_tuning_init is not None:
            args.training_inference_type = args.prompt_tuning_init.value

        # datasets
        for data_config in args.datasets:
            data_config[DatasetConfigKeys.data_class.value] = data_config[DatasetConfigKeys.data_class.value].__name__

        # optimizer
        args.optimizer[OptimizerKeys.optimizer_class.value] = args.optimizer[
            OptimizerKeys.optimizer_class.value
        ].__name__

        json.dump(vars(args), open(save_path, "w"), indent=4)

    @classmethod
    @run_rank_n
    def save_inference_args(cls, args: InferenceArgs, save_path: str) -> None:
        """saves inference args as a json

        Args:
            args (InferenceArgs): arguments for inference
            save_path (str): save location on disk
        """

        args = deepcopy(args)

        # model_class
        args.model_class = args.model_class.__name__
        # dtype
        args.dtype = str(args.dtype).split(".")[1]

        # training_inference_type
        args.training_inference_type = args.training_inference_type.value
        # prompt_tuning_init
        if args.prompt_tuning_init is not None:
            args.training_inference_type = args.prompt_tuning_init.value

        # datasets
        for data_config in args.datasets:
            data_config[DatasetConfigKeys.data_class.value] = data_config[DatasetConfigKeys.data_class.value].__name__

        json.dump(vars(args), open(save_path, "w"), indent=4)

import torch
import torch.distributed
from peft import LoraConfig, PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from dolomite_engine.enums import AttentionImplementation, DistributedBackend

from ..arguments import InferenceArgs, TrainingArgs, UnshardingArgs
from ..enums import Mode, TuningMethod
from ..utils import string_to_torch_dtype
from .finetuning import ModelWrapperForFinetuning


class ModelWrapperForPEFT(ModelWrapperForFinetuning):
    def __init__(
        self,
        mode: Mode,
        model_name: str | None,
        pretrained_config: dict | None,
        model_class: AutoModelForCausalLM | AutoModelForSeq2SeqLM,
        dtype: torch.dtype,
        efficient_initialization: bool,
        attention_implementation: AttentionImplementation,
        use_padding_free_transformer: bool,
        tensor_parallel_word_embeddings: bool,
        sequence_parallel: bool,
        distributed_backend: DistributedBackend,
        random_seed: int,
        neft_alpha: float | None = None,
        trust_remote_code: bool = False,
        tokenizer_name: str | None = None,
        additional_special_tokens: list[str] = None,
        reset_attention_mask: bool = False,
        reset_position_ids: bool = False,
    ) -> None:
        super().__init__(
            mode=mode,
            model_name=model_name,
            pretrained_config=pretrained_config,
            model_class=model_class,
            dtype=dtype,
            efficient_initialization=efficient_initialization,
            attention_implementation=attention_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            tensor_parallel_word_embeddings=tensor_parallel_word_embeddings,
            sequence_parallel=sequence_parallel,
            distributed_backend=distributed_backend,
            random_seed=random_seed,
            neft_alpha=neft_alpha,
            trust_remote_code=trust_remote_code,
            tokenizer_name=tokenizer_name,
            additional_special_tokens=additional_special_tokens,
            reset_attention_mask=reset_attention_mask,
            reset_position_ids=reset_position_ids,
        )

        assert not self.reset_attention_mask, "reset_attention_mask is only supported with pretraining"
        assert not self.reset_position_ids, "reset_position_ids is only supported with pretraining"

    def _setup_model(self, args: TrainingArgs | InferenceArgs | UnshardingArgs) -> None:
        if self.model_name is None:
            model_kwargs = {"config": self.config}
        else:
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "trust_remote_code": args.model_args.trust_remote_code,
            }

        if self.attention_implementation is not None:
            model_kwargs["attn_implementation"] = self.attention_implementation.value

        assert not self.use_padding_free_transformer

        if args.tuning_args.tuning_method == TuningMethod.prompt_tuning:
            self.peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                prompt_tuning_init=args.tuning_args.prompt_tuning_args.prompt_tuning_init,
                num_virtual_tokens=args.tuning_args.prompt_tuning_args.num_virtual_tokens,
                prompt_tuning_init_text=args.tuning_args.prompt_tuning_args.prompt_tuning_init_text,
                tokenizer_name_or_path=args.model_args.model_name,
            )
        elif args.tuning_args.tuning_method == TuningMethod.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.is_encoder_decoder else TaskType.CAUSAL_LM,
                inference_mode=self.mode != Mode.training,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

        self.model = args.model_args.model_class.from_pretrained(
            **model_kwargs, torch_dtype=string_to_torch_dtype(self.dtype)
        )
        self.model = get_peft_model(self.model, self.peft_config)

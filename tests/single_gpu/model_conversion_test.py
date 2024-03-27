import torch
from parameterized import parameterized

from dolomite_engine.hf_models import (
    AttentionHeadType,
    PositionEmbeddingType,
    export_to_huggingface_bigcode,
    export_to_huggingface_llama,
    export_to_huggingface_mixtral,
    import_from_huggingface_bigcode,
    import_from_huggingface_llama,
    import_from_huggingface_mixtral,
)

from ..test_common import TestCommons


class ModelConversionTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), [AttentionHeadType.mha, AttentionHeadType.mqa])
    )
    def test_bigcode_model_conversion(self, device: torch.device, attention_head_type: AttentionHeadType) -> None:
        megatron_config = self.get_dense_test_config(attention_head_type, PositionEmbeddingType.learned_absolute)

        self.model_conversion_test(
            megatron_config=megatron_config,
            export_to_huggingface_function=export_to_huggingface_bigcode,
            import_from_huggingface_function=import_from_huggingface_bigcode,
            device=device,
            exact_match=False,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_attention_head_types())
    )
    def test_llama_model_conversion(self, device: torch.device, attention_head_type: AttentionHeadType) -> None:
        megatron_config = self.get_dense_test_config(
            attention_head_type,
            PositionEmbeddingType.rope,
            add_bias=False,
            activation_function="swiglu",
            normalization_function="rmsnorm",
        )

        self.model_conversion_test(
            megatron_config=megatron_config,
            export_to_huggingface_function=export_to_huggingface_llama,
            import_from_huggingface_function=import_from_huggingface_llama,
            device=device,
            exact_match=False,
        )

    @parameterized.expand(
        TestCommons.make_args_matrix(TestCommons.get_all_devices(), TestCommons.get_attention_head_types())
    )
    def test_mixtral_model_conversion(self, device: torch.device, attention_head_type: AttentionHeadType) -> None:
        megatron_config = self.get_moe_test_config(
            attention_head_type,
            PositionEmbeddingType.rope,
            add_bias=False,
            activation_function="swiglu",
            normalization_function="rmsnorm",
        )

        self.model_conversion_test(
            megatron_config=megatron_config,
            export_to_huggingface_function=export_to_huggingface_mixtral,
            import_from_huggingface_function=import_from_huggingface_mixtral,
            device=device,
            exact_match=False,
        )

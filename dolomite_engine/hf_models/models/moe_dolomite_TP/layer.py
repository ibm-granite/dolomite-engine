import torch.nn as nn
from moe_TP.scatter import ScatterMoE_TP

from dolomite_engine.hf_models.enums import AttentionHeadType
from dolomite_engine.hf_models.modeling_utils_TP import get_attention_module_TP, get_normalization_function_TP
from dolomite_engine.hf_models.models.moe_dolomite import MoEDolomiteConfig
from dolomite_engine.hf_models.models.moe_dolomite.layer import SparseMoEBlock
from dolomite_engine.utils import SafeTensorsWeightsManager

# from .moe_TP import get_moe
from ..moe_dolomite.moe import get_moe


class SparseMoEBlock_TP(SparseMoEBlock):
    def __init__(
        self,
        config: MoEDolomiteConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int | None = None,
        sequence_parallel: bool = False,
    ) -> None:
        nn.Module.__init__(self)

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )
        self.attn = get_attention_module_TP(
            config,
            True,
            attention_implementation=attention_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
            sequence_parallel=sequence_parallel,
        )
        self.ln_2 = get_normalization_function_TP(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
        )
        self.mlp = ScatterMoE_TP(config, use_padding_free_transformer, layer_idx)

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "ln_1.weight")}
        if hasattr(self.ln_1, "bias"):
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "ln_1.bias")
        self.ln_1.load_state_dict(state_dict)

        state_dict = {"weight": safetensors_weight_manager.get_tensor(prefix + "ln_2.weight")}
        if hasattr(self.ln_2, "bias"):
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "ln_2.bias")
        self.ln_2.load_state_dict(state_dict)

        self.attn.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "attn.")
        self.mlp.load_from_safetensors_weights_manager(safetensors_weight_manager, prefix + "mlp.")

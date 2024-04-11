from typing import Tuple, Union

import torch
import torch.nn as nn
from transformers import DynamicCache

from ...enums import AttentionHeadType
from ...modeling_utils import get_attention_module, get_normalization_function
from .config import MoEMegablocksConfig
from .moe import SparseMoE


class SparseMoEBlock(nn.Module):
    def __init__(
        self,
        config: MoEMegablocksConfig,
        normalization_implementation: str,
        attention_implementation: str,
        use_padding_free_transformer: bool,
        layer_idx: int = None,
    ) -> None:
        super().__init__()

        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner
        self.attention_head_type = AttentionHeadType(config.attention_head_type)
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.layer_idx = layer_idx
        self.m_residual = config.m_residual

        self.ln_1 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.attn = get_attention_module(
            config, True, attention_implementation, use_padding_free_transformer, layer_idx
        )
        self.ln_2 = get_normalization_function(
            config.normalization_function,
            hidden_size,
            eps=config.layer_norm_epsilon,
            normalization_implementation=normalization_implementation,
        )
        self.mlp = SparseMoE(config, use_padding_free_transformer=use_padding_free_transformer)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache = None,
        attention_mask: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
        output_router_logits: bool = False,
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if self.apply_residual_connection_post_layernorm:
            hidden_states = self.ln_1(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)

        attn_output = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            alibi_bias=alibi_bias,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.m_residual is not None:
            attn_output = attn_output * self.m_residual

        # residual connection
        hidden_states = attn_output + residual

        if self.apply_residual_connection_post_layernorm:
            hidden_states = self.ln_2(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states, router_logits = self.mlp(hidden_states)

        if self.m_residual is not None:
            feed_forward_hidden_states = feed_forward_hidden_states * self.m_residual

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

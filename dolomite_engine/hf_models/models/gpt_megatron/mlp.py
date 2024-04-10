import math
from typing import Tuple

import torch
import torch.nn as nn

from ...config import MegatronConfig
from ...enums import InitMethod
from ...modeling_utils import ParameterizedLinear, get_activation_function, is_glu


class MLP(nn.Module):
    def __init__(self, config: MegatronConfig) -> None:
        super().__init__()

        hidden_size = config.n_embd
        intermediate_size = config.n_inner
        activation_function = config.activation_function
        add_bias = config.add_bias
        residual_dropout = config.resid_pdrop

        self.init_method = config.init_method
        self.initializer_range = config.initializer_range
        self.m_width = config.m_width
        self.n_layer = config.n_layer

        std = self.initializer_range
        if self.init_method == InitMethod.mup:
            std /= math.sqrt(self.m_width)
        self.c_fc = ParameterizedLinear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            bias=add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = self.initializer_range / math.sqrt(2 * self.n_layer)
        if self.init_method == InitMethod.mup:
            std /= math.sqrt(self.m_width)
        self.c_proj = ParameterizedLinear(intermediate_size, hidden_size, bias=add_bias, std=std)

        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def interleave_up_gate_tensor_for_mlp(up_weight: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    return torch.cat([up_weight, gate_weight])


def split_up_gate_tensor_for_mlp(c_fc_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return c_fc_weight.chunk(2)

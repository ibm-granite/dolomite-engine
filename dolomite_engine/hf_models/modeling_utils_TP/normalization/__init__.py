from functools import partial
from typing import Any, Mapping

import torch.nn as nn

from ..TP import modify_state_dict_to_densor_dict
from .layernorm import get_layernorm
from .rmsnorm import get_rmsnorm


_NORMALIZATION_FUNCTIONS = {
    "layernorm": get_layernorm,
    "rmsnorm": get_rmsnorm,
}


def get_normalization_function_TP(
    name: str,
    normalized_shape: int,
    eps: float = 1e-5,
    normalization_implementation: str = "torch",
) -> nn.LayerNorm:
    if name in _NORMALIZATION_FUNCTIONS:
        normalization_function = _NORMALIZATION_FUNCTIONS[name](
            normalized_shape, eps=eps, normalization_implementation=normalization_implementation
        )

    original_load_state_dict = normalization_function.load_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_densor_dict(self, state_dict)
        return original_load_state_dict(state_dict, strict, assign)

    normalization_function.load_state_dict = partial(load_state_dict, normalization_function)

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")

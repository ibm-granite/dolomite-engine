from functools import partial

import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from ...utils import ProcessGroupManager
from ..modeling_utils import get_normalization_function
from .TP import prepare_tensor_parallel_dtensor_input, prepare_tensor_parallel_tensor_output


def get_normalization_function_TP(
    name: str,
    normalized_shape: int,
    eps: float = 1e-5,
    normalization_implementation: str = "torch",
) -> nn.LayerNorm:
    normalization_function = get_normalization_function(
        name, normalized_shape=normalized_shape, eps=eps, normalization_implementation=normalization_implementation
    )

    for name, param in normalization_function.named_parameters():
        param = DTensor.from_local(
            param, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
        )
        normalization_function.register_parameter(name, nn.Parameter(param))

    normalization_function.register_forward_pre_hook(
        partial(prepare_tensor_parallel_dtensor_input, placements=[Replicate()])
    )
    normalization_function.register_forward_hook(
        partial(prepare_tensor_parallel_tensor_output, expected_placement=Replicate)
    )

    return normalization_function

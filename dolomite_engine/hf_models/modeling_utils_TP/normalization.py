import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from ...utils import ProcessGroupManager
from ..modeling_utils import get_normalization_function


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
        setattr(normalization_function, name, param)

    normalization_function.register_forward_pre_hook(_prepare_input)
    normalization_function.register_forward_hook(_prepare_output)

    return normalization_function


def _prepare_input(input: torch.Tensor) -> DTensor:
    input = DTensor.from_local(
        input, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
    )
    return input


def _prepare_output(output: DTensor) -> torch.Tensor:
    assert isinstance(output.placements[0], Replicate)
    output = output.to_local()
    return output

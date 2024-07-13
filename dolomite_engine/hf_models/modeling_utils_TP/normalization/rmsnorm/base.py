import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from .....utils import ProcessGroupManager, is_dtensors_computation_enabled
from ....modeling_utils import RMSNorm
from ...TP import dtensor_to_tensor, tensor_to_dtensor


class RMSNorm_TP(RMSNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__(normalized_shape, eps=eps)

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        input = input.float()

        weight = self.weight

        if is_dtensors_computation_enabled():
            input = tensor_to_dtensor(input, current_placement=Replicate())
        else:
            weight = weight.to_local()

        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        input = weight * input.to(input_dtype)

        if is_dtensors_computation_enabled():
            input = dtensor_to_tensor(input, desired_placement=Replicate())

        return input

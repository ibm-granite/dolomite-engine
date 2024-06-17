import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from .....utils import ProcessGroupManager
from ....modeling_utils import RMSNorm


class DTensorRMSNorm(RMSNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__(normalized_shape, eps=eps)

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )

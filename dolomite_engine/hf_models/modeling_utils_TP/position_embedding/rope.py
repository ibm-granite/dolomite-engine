import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from ....utils import ProcessGroupManager


class RoPE(nn.Module):
    def reset_parameters(self) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer(
            "inv_freq",
            DTensor.from_local(
                inv_freq, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            ),
            persistent=False,
        )

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

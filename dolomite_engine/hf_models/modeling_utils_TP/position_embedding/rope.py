import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate

from ....utils import ProcessGroupManager
from ...modeling_utils import RoPE


class RoPE_TP(RoPE):
    @torch.no_grad()
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached",
            (
                DTensor.from_local(
                    (emb.cos() * self.mscale).to(dtype),
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    placements=[Replicate()],
                )
            ),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (
                DTensor.from_local(
                    (emb.sin() * self.mscale).to(dtype),
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    placements=[Replicate()],
                )
            ),
            persistent=False,
        )

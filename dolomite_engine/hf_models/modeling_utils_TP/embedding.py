import math
from functools import partial

import torch
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from ...utils import ProcessGroupManager, SafeTensorsWeightsManager
from ..modeling_utils import ParameterizedEmbedding
from ..utils import divide_if_divisible
from .TP import (
    prepare_tensor_parallel_dtensor_input,
    prepare_tensor_parallel_tensor_output,
    reduce_from_tensor_parallel_region,
)


class DTensorEmbedding(ParameterizedEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: torch.Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
        std=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
            std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Shard(0)]
            )
        )

        self.register_forward_pre_hook(partial(prepare_tensor_parallel_dtensor_input, placement=Replicate()))
        self.register_forward_hook(partial(prepare_tensor_parallel_tensor_output, expected_placement=Replicate()))


class Embedding_TP(DTensorEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, std: float = None) -> None:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.vocab_start_index, self.vocab_end_index, num_embeddings_per_tp_rank = get_tensor_parallel_vocab_info(
            num_embeddings
        )

        super().__init__(num_embeddings_per_tp_rank, embedding_dim, std=std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_world_size > 1:
            # Build the mask.
            input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
            # Mask the input.
            masked_input = input - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output_parallel = super().forward(masked_input)

        if self.tp_world_size > 1:
            # Mask the output embedding.
            output_parallel[input_mask, :] = 0
            output_parallel = reduce_from_tensor_parallel_region(output_parallel)

        return output_parallel

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")[
            self.vocab_start_index : self.vocab_end_index, :
        ]
        if self.num_embeddings > weight.shape[0]:
            weight = torch.cat(
                [
                    weight,
                    torch.zeros((self.num_embeddings - weight.shape[0], weight.shape[1])),
                ]
            )

        self.load_state_dict({"weight": weight})


def get_tensor_parallel_vocab_info(vocab_size: int, make_vocab_size_divisible_by: int = 64) -> tuple[int, int, int]:
    tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    divide_if_divisible(make_vocab_size_divisible_by, tp_world_size, "")

    vocab_size_per_tensor_parallel_rank = (
        make_vocab_size_divisible_by * math.ceil(vocab_size / make_vocab_size_divisible_by)
    ) // tp_world_size

    vocab_start_index = tp_rank * vocab_size_per_tensor_parallel_rank
    vocab_end_index = min((tp_rank + 1) * vocab_size_per_tensor_parallel_rank, vocab_size)

    return vocab_start_index, vocab_end_index, vocab_size_per_tensor_parallel_rank

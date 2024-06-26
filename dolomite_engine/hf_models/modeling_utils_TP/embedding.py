import contextlib
import math

import torch

from ...utils import ProcessGroupManager, SafeTensorsWeightsManager, get_cuda_rng_tracker
from ..modeling_utils import ParameterizedEmbedding
from ..utils import divide_if_divisible
from .TP import reduce_from_tensor_parallel_region


class Embedding_TP(ParameterizedEmbedding):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, std: float = None, tensor_parallel_embeddings: bool = False
    ) -> None:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.tensor_parallel_embeddings = tensor_parallel_embeddings and self.tp_world_size > 1

        if self.tensor_parallel_embeddings:
            self.vocab_start_index, self.vocab_end_index, num_embeddings_per_tp_rank = get_tensor_parallel_vocab_info(
                num_embeddings
            )

            super().__init__(num_embeddings_per_tp_rank, embedding_dim, std=std)
        else:
            with get_cuda_rng_tracker().fork():
                super().__init__(num_embeddings, embedding_dim, std=std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tensor_parallel_embeddings:
            # Build the mask.
            input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
            # Mask the input.
            masked_input = input - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input

        output_parallel = super().forward(masked_input)

        if self.tensor_parallel_embeddings:
            # Mask the output embedding.
            output_parallel[input_mask, :] = 0
            output_parallel = reduce_from_tensor_parallel_region(output_parallel)

        return output_parallel

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        if self.tensor_parallel_embeddings:
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

        else:
            weight = safetensors_weight_manager.get_tensor(prefix + "weight")

        self.load_state_dict({"weight": weight})

    def reset_parameters(self) -> None:
        context = contextlib.nullcontext if self.tensor_parallel_embeddings else get_cuda_rng_tracker().fork

        with context():
            return super().reset_parameters()


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

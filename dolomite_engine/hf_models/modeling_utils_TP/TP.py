from typing import Tuple

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard

from ...utils import ProcessGroupManager
from ..utils import divide_if_divisible


def copy_to_tensor_parallel_region(input: torch.Tensor) -> DTensor:
    assert isinstance(input.placements[0], Replicate)
    return input


def reduce_from_tensor_parallel_region(input: DTensor) -> DTensor:
    assert isinstance(input.placements[0], Shard)
    input = input.redistribute(device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()])
    return input


def gather_from_tensor_parallel_region(input: DTensor) -> DTensor:
    assert isinstance(input.placements[0], Shard)
    input = input.full_tensor()
    return input


# class _CopyToTensorParallelRegion(torch.autograd.Function):
#     """Pass the input to the model parallel region."""

#     @staticmethod
#     def forward(ctx, input: torch.Tensor) -> torch.Tensor:
#         return input

#     @staticmethod
#     def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
#         return _tensor_parallel_all_reduce(grad_output)


# class _ReduceFromTensorParallelRegion(torch.autograd.Function):
#     """All-reduce the input from the model parallel region."""

#     @staticmethod
#     def forward(ctx, input: torch.Tensor) -> torch.Tensor:
#         return _tensor_parallel_all_reduce(input)

#     @staticmethod
#     def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
#         return grad_output


# class _GatherFromTensorParallelRegion(torch.autograd.Function):
#     """Gather the input from model parallel region and concatinate."""

#     @staticmethod
#     def forward(ctx, input: torch.Tensor) -> torch.Tensor:
#         return _gather_along_last_dim(input)

#     @staticmethod
#     def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
#         return _split_along_last_dim(grad_output)


# def _tensor_parallel_all_reduce(input: torch.Tensor) -> torch.Tensor:
#     if ProcessGroupManager.get_tensor_parallel_world_size() == 1:
#         return input

#     torch.distributed.all_reduce(input, group=ProcessGroupManager.get_tensor_parallel_group())

#     return input


# def _gather_along_last_dim(input: torch.Tensor) -> torch.Tensor:
#     world_size = ProcessGroupManager.get_tensor_parallel_world_size()

#     if world_size == 1:
#         return input

#     dim_size = list(input.size())
#     dim_size[0] = dim_size[0] * world_size

#     output = torch.empty(dim_size, dtype=input.dtype, device=torch.cuda.current_device())
#     torch.distributed.all_gather_into_tensor(output, input, group=ProcessGroupManager.get_tensor_parallel_group())

#     output = output.chunk(world_size)
#     output = torch.cat(output, dim=-1)

#     return output


# def _split_along_last_dim(input: torch.Tensor) -> torch.Tensor:
#     world_size = ProcessGroupManager.get_tensor_parallel_world_size()

#     if world_size == 1:
#         return input

#     input_list = input.chunk(world_size, dim=-1)
#     output = input_list[ProcessGroupManager.get_tensor_parallel_rank()]

#     return output


def tensor_parallel_split_safetensor_slice(slice, dim: int, start_end: Tuple[int, int] = None) -> torch.Tensor:
    shape = slice.get_shape()
    dimensionality = len(shape)
    assert dimensionality in [1, 2], f"tensor should be either 1 or 2 dimensional but {dimensionality} was found"

    tp_rank = ProcessGroupManager.get_tensor_parallel_rank()
    tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

    if start_end is None:
        stride = divide_if_divisible(
            shape[dim],
            tp_world_size,
            f"split dimension ({dim}) is not divisible by tensor parallel world size ({tp_world_size})",
        )
        start_index = tp_rank * stride
        end_index = (tp_rank + 1) * stride
    else:
        start_index = start_end[0]
        end_index = start_end[1]

    if dimensionality == 1:
        # bias tensor
        assert dim == 0, f"dim has to 0 for a bias tensor but dim ({dim}) was passed"
        return slice[start_index:end_index]
    elif dimensionality == 2:
        assert 0 <= dim <= 1, f"dim has to 0 or 1 for a weight tensor but dim ({dim}) was passed"
        # weight tensor
        if dim == 0:
            return slice[start_index:end_index, :]
        else:
            return slice[:, start_index:end_index]
    else:
        raise RuntimeError("this code should not be reachable")


def prepare_tensor_parallel_dtensor_input(
    module: nn.Module, inputs: tuple[torch.Tensor], placement: Placement
) -> DTensor:
    assert len(inputs) == 1
    input = inputs[0]

    input = DTensor.from_local(
        input, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[placement]
    )

    return (input,)


def prepare_tensor_parallel_tensor_output(
    module: nn.Module, outputs: list[DTensor], expected_placement: Placement
) -> torch.Tensor:
    assert len(outputs) == 1
    output = outputs[0]

    if isinstance(expected_placement, Replicate):
        assert output.placements[0].is_replicate()
    elif isinstance(expected_placement, Shard):
        assert output.placements[0].is_shard(expected_placement.dim)

    output = output.to_local()
    return (output,)

from typing import Tuple

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard
from torch.profiler import record_function

from ...utils import ProcessGroupManager
from ..utils import divide_if_divisible


def copy_to_tensor_parallel_region(input: torch.Tensor) -> torch.Tensor:
    with record_function("TP::copy_to_tensor_parallel_region"):
        return _CopyToTensorParallelRegion.apply(input)


def reduce_from_tensor_parallel_region(input: torch.Tensor) -> torch.Tensor:
    with record_function("TP::reduce_from_tensor_parallel_region"):
        return _ReduceFromTensorParallelRegion.apply(input)


def gather_from_tensor_parallel_region(input: torch.Tensor) -> torch.Tensor:
    with record_function("TP::gather_from_tensor_parallel_region"):
        return _GatherFromTensorParallelRegion.apply(input)


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return input


def reduce_from_tensor_parallel_region(input: DTensor) -> DTensor:
    with torch.profiler.record_function("TP::reduce_from_tensor_parallel_region"):
        assert isinstance(input.placements[0], Shard)
        input = input.redistribute(
            device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
        )
        return input


def gather_from_tensor_parallel_region(input: torch.Tensor) -> DTensor:
    tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

    with torch.profiler.record_function("TP::gather_from_tensor_parallel_region"):
        input = DTensor.from_local(input, device_mesh=tp_mesh, run_check=False, placements=[Shard(-1)])
        input = input.full_tensor()
        return input


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
    module: nn.Module, inputs: tuple[torch.Tensor], placement: Placement, desired_placement: Placement = None
) -> DTensor:
    assert len(inputs) == 1
    input = inputs[0]

    tp_mesh = ProcessGroupManager.get_tensor_parallel_mesh()

    with torch.profiler.record_function("TP::prepare_tensor_parallel_dtensor_input"):
        input = DTensor.from_local(input, device_mesh=tp_mesh, run_check=False, placements=[placement])

        if desired_placement is not None:
            input = input.redistribute(device_mesh=tp_mesh, placements=[desired_placement])

    return (input,)


def prepare_tensor_parallel_tensor_output(
    module: nn.Module,
    inputs: list[DTensor],
    output: DTensor,
    assert_placement: Placement = None,
    desired_placement: Placement = None,
) -> torch.Tensor:
    if assert_placement is not None:
        if isinstance(assert_placement, Replicate):
            assert output.placements[0].is_replicate()
        elif isinstance(assert_placement, Shard):
            dim = assert_placement.dim
            if dim == -1:
                dim = output.dim() - 1
            assert output.placements[0].is_shard(dim)

    with torch.profiler.record_function("TP::prepare_tensor_parallel_tensor_output"):
        if desired_placement is not None:
            output = output.redistribute(
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[desired_placement]
            )

        output = output.to_local()

    return output


def modify_state_dict_to_dtensor_dict(module: nn.Module, state_dict: dict) -> dict:
    result = {}
    for key, tensor in state_dict.items():
        device_mesh = getattr(module, key).device_mesh
        placements = getattr(module, key).placements
        result[key] = DTensor.from_local(tensor, device_mesh=device_mesh, placements=placements, run_check=False)
    return result

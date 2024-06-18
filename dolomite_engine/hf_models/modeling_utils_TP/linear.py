from functools import partial

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard

from ...utils import ProcessGroupManager, SafeTensorsWeightsManager, get_cuda_rng_tracker
from ..modeling_utils import ParameterizedLinear
from ..utils import divide_if_divisible
from .TP import (
    prepare_tensor_parallel_dtensor_input,
    prepare_tensor_parallel_tensor_output,
    tensor_parallel_split_safetensor_slice,
)


class ColumnParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.out_features_per_device = divide_if_divisible(
            out_features,
            tp_world_size,
            f"`out_features` ({out_features}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )

        super().__init__(
            in_features=in_features,
            out_features=self.out_features_per_device,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                run_check=False,
                placements=[Shard(0)],
            )
        )
        if bias:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    self.bias,
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    run_check=False,
                    placements=[Shard(0)],
                )
            )

        self.register_forward_pre_hook(partial(prepare_tensor_parallel_dtensor_input, placement=Replicate()))
        self.register_forward_hook(partial(prepare_tensor_parallel_tensor_output, assert_placement=Shard(-1)))

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=0)
        state_dict = {"weight": weight}

        if self.bias is not None:
            bias = safetensors_weight_manager.get_slice(prefix + "bias")
            bias = tensor_parallel_split_safetensor_slice(bias, dim=0)
            state_dict["bias"] = bias

        self.load_state_dict(state_dict)

    def extra_repr(self) -> str:
        return "in_features={}, out_features_per_device={}, bias={}".format(
            self.in_features, self.out_features_per_device, self.bias is not None
        )


class RowParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
    ) -> None:
        self.tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()

        self.in_features_per_device = divide_if_divisible(
            in_features,
            self.tp_world_size,
            f"`in_features` ({in_features}) must be divisible by `tensor_parallel_world_size` ({self.tp_world_size})",
        )

        super().__init__(
            in_features=self.in_features_per_device,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                run_check=False,
                placements=[Shard(1)],
            )
        )
        if bias:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    self.bias,
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    run_check=False,
                    placements=[Replicate()],
                )
            )

        self.register_forward_pre_hook(partial(prepare_tensor_parallel_dtensor_input, placement=Shard(-1)))
        self.register_forward_hook(partial(prepare_tensor_parallel_tensor_output, desired_placement=Replicate()))

    def load_from_safetensors_weights_manager(
        self, safetensors_weight_manager: SafeTensorsWeightsManager, prefix: str = ""
    ) -> None:
        weight = safetensors_weight_manager.get_slice(prefix + "weight")
        weight = tensor_parallel_split_safetensor_slice(weight, dim=1)
        state_dict = {"weight": weight}

        if self.bias is not None:
            state_dict["bias"] = safetensors_weight_manager.get_tensor(prefix + "bias")

        self.load_state_dict(state_dict)

    def extra_repr(self) -> str:
        return "in_features_per_device={}, out_features={}, bias={}".format(
            self.in_features_per_device, self.out_features, self.bias is not None
        )


class TensorParallelSharedLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        std: float = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype, std)

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                run_check=False,
                placements=[Replicate()],
            )
        )
        if bias:
            self.bias = nn.Parameter(
                DTensor.from_local(
                    self.bias,
                    device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                    run_check=False,
                    placements=[Replicate()],
                )
            )

        self.register_forward_pre_hook(partial(prepare_tensor_parallel_dtensor_input, placement=Replicate()))
        self.register_forward_hook(partial(prepare_tensor_parallel_tensor_output, assert_placement=Replicate()))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        with get_cuda_rng_tracker().fork():
            return super().reset_parameters()

import math
from typing import Any, Mapping

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed._tensor.placement_types import _Partial as Partial

from .....utils import ProcessGroupManager, SafeTensorsWeightsManager, is_scattermoe_available
from ....enums import InitMethod
from ....modeling_utils import ParameterizedLinear, get_activation_function, is_glu
from ....modeling_utils_TP.TP import (
    dtensor_to_tensor,
    modify_state_dict_to_dtensor_dict,
    tensor_parallel_split_safetensor_slice,
    tensor_to_dtensor,
)
from ....utils import divide_if_divisible
from ..config import MoEDolomiteConfig
from ..moe.base import SparseMoE
from ..moe.scatter import ScatterMoE, _ParameterizedScatteredExperts


if is_scattermoe_available():
    import scattermoe
    from scattermoe.parallel_experts import parallel_linear as scattered_experts


class ReplicatedParallelLinear(ParameterizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
        use_padding_free_transformer: bool = False,
        sequence_parallel: bool = False,
    ) -> None:

        super().__init__(
            in_features=in_features, out_features=out_features, device=device, dtype=dtype, std=std, bias=False
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight, device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(), placements=[Replicate()]
            )
        )
        self.input_placement = Replicate()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = tensor_to_dtensor(input, current_placement=self.input_placement)
        input = super().forward(input)
        input = dtensor_to_tensor(input, desired_placement=Replicate())
        return input

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

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class ScatterMoETP(SparseMoE):
    def __init__(
        self, config: MoEDolomiteConfig, use_padding_free_transformer: bool, layer_idx: int | None = None
    ) -> None:
        nn.Module.__init__(self)

        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.normalize_expert_weights = config.normalize_expert_weights
        self.use_padding_free_transformer = use_padding_free_transformer
        self.layer_idx = layer_idx

        self.gate = ParameterizedLinear(self.hidden_size, self.num_experts, bias=False)

        intermediate_size = config.n_inner
        activation_function = config.activation_function
        residual_dropout = config.resid_pdrop

        assert not config.add_bias, "ScatterMoE does not support add_bias"

        initializer_range = config.initializer_range
        m_width = config.m_width
        n_layer = config.n_layer
        init_method = InitMethod(config.init_method)

        std = initializer_range
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_fc = _ParameterizedScatteredExperts(
            self.num_experts,
            self.hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        std = initializer_range / math.sqrt(2 * n_layer)
        if init_method == InitMethod.mup:
            std /= math.sqrt(m_width)
        self.c_proj = _ParameterizedScatteredExperts(self.num_experts, intermediate_size, self.hidden_size, std=std)

        self.dropout = nn.Identity() if residual_dropout == 0 else nn.Dropout(residual_dropout)

    def _compute_expert_outputs(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = scattermoe.kernels.ops.flatten_and_sort(selected_experts)
            padded_block_idxs, expert_offsets = scattermoe.kernels.ops.padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )

        hidden_states = self.c_fc(
            hidden_states,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True,
        )
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(
            hidden_states,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_in=True,
            gates=routing_weights,
        )
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ColumnParallelScatteredExperts(_ParameterizedScatteredExperts):
    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.out_features_per_device = divide_if_divisible(
            output_size,
            tp_world_size,
            f"`output_size` ({output_size}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )
        super().__init__(
            num_experts=num_experts,
            output_size=self.out_features_per_device,
            input_size=input_size,
            std=std,
            device=device,
            dtype=dtype,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                placements=[Shard(1)],
                run_check=False,
            )
        )
        self.input_placement = Replicate()

    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        weight = dtensor_to_tensor(self.weight, desired_placement=Shard(1))
        results = scattered_experts(
            inputs,
            weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )
        return results

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class RowParallelScatteredExperts(_ParameterizedScatteredExperts):
    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        std: float | None = None,
    ) -> None:
        tp_world_size = ProcessGroupManager.get_tensor_parallel_world_size()
        self.in_features_per_device = divide_if_divisible(
            input_size,
            tp_world_size,
            f"`output_size` ({input_size}) must be divisible by `tensor_parallel_world_size` ({tp_world_size})",
        )
        super().__init__(
            num_experts=num_experts,
            output_size=output_size,
            input_size=self.in_features_per_device,
            device=device,
            dtype=dtype,
            std=std,
        )

        self.weight = nn.Parameter(
            DTensor.from_local(
                self.weight,
                device_mesh=ProcessGroupManager.get_tensor_parallel_mesh(),
                placements=[Shard(2)],
                run_check=False,
            )
        )
        self.input_placement = Shard(-1)

    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        print(inputs.size())
        weight = dtensor_to_tensor(self.weight, desired_placement=Shard(2))
        results = scattered_experts(
            inputs,
            weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )
        results = tensor_to_dtensor(results, current_placement=Partial())
        results = dtensor_to_tensor(results, desired_placement=Replicate())
        return results

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> None:
        state_dict = modify_state_dict_to_dtensor_dict(self, state_dict)
        return super().load_state_dict(state_dict, strict, assign)

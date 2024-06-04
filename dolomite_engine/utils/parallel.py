import os
from typing import Callable

import torch
from torch.distributed import barrier, get_process_group_ranks, get_rank, get_world_size
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


_DEVICE_MESH: DeviceMesh = None

_LOCAL_RANK: int = None
_GLOBAL_RANK: int = None
_WORLD_SIZE: int = None

_ZERO_HPZ_PARTITION_SIZE: int = None


class ProcessGroupManager:
    def __init__(
        self, tensor_parallel_size: int = None, data_parallel_size: int = None, zero_hpz_partition_size: int = None
    ) -> None:
        assert get_rank() == int(os.getenv("RANK", 0))

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        if tensor_parallel_size is None:
            tensor_parallel_size = 1

        if data_parallel_size is None:
            data_parallel_size = get_world_size() // tensor_parallel_size

        global _DEVICE_MESH, _ZERO_HPZ_PARTITION_SIZE

        _DEVICE_MESH = init_device_mesh(
            "cuda",
            (tensor_parallel_size, data_parallel_size),
            mesh_dim_names=("tp", "dp"),
        )

        _ZERO_HPZ_PARTITION_SIZE = zero_hpz_partition_size

    # global
    @staticmethod
    def barrier() -> None:
        barrier()

    @staticmethod
    def get_global_rank() -> int:
        global _GLOBAL_RANK

        if _GLOBAL_RANK is None:
            _GLOBAL_RANK = int(os.getenv("RANK", 0))
        return _GLOBAL_RANK

    @staticmethod
    def get_local_rank() -> int:
        global _LOCAL_RANK

        if _LOCAL_RANK is None:
            _LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
        return _LOCAL_RANK

    @staticmethod
    def get_world_size() -> int:
        global _WORLD_SIZE

        if _WORLD_SIZE is None:
            _WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
        return _WORLD_SIZE

    # tensor parallel
    @staticmethod
    def get_tensor_parallel_mesh() -> DeviceMesh:
        return _DEVICE_MESH["tp"]

    @staticmethod
    def get_tensor_parallel_group() -> DeviceMesh:
        return ProcessGroupManager.get_data_parallel_mesh().get_group()

    @staticmethod
    def get_tensor_parallel_rank() -> int:
        return ProcessGroupManager.get_tensor_parallel_mesh().get_rank()

    @staticmethod
    def get_tensor_parallel_world_size() -> int:
        return ProcessGroupManager.get_tensor_parallel_mesh().size()

    def get_process_group(self) -> torch.distributed.ProcessGroup:
        return self.process_group

    def get_ranks(self) -> List[int]:
        return self.ranks

    def get_world_size(self) -> int:
        return self.world_size

    @staticmethod
    def get_data_parallel_rank() -> int:
        return ProcessGroupManager.get_data_parallel_mesh().get_rank()

    def get_first_rank(self) -> int:
        return self.ranks[0]

    def get_rank(self) -> int:
        return self.rank

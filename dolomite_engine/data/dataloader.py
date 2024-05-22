from typing import Any, Callable, Iterable, List

import torch.distributed
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset, Sampler

from ..utils import get_global_rank


class DataLoader(_DataLoader):
    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict(), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict.get("dataset"))
        self.sampler.load_state_dict(state_dict.get("sampler"))


class DispatchingDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        sampler: Sampler | Iterable | None = None,
        collate_fn: Callable[[List], Any] | None = None,
        source_rank: int = None,
        broadcast_ranks: List[int] = None,
    ) -> None:
        self.source_rank = source_rank
        self.broadcast_process_group = torch.distributed.new_group(ranks=broadcast_ranks)
        self.broadcast_world_size = len(broadcast_ranks)
        self.is_source = get_global_rank() == self.source_rank
        self.local_rank_in_broadcast_group = broadcast_ranks.index(get_global_rank())
        self.local_batch_size = batch_size

        super().__init__(
            dataset,
            batch_size * self.broadcast_world_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )

        if self.is_source:
            self._length = torch.tensor(len(self), device=torch.cuda.current_device())
        else:
            self._length = torch.empty(1, device=torch.cuda.current_device())

        torch.distributed.broadcast(self._length, src=self.source_rank, group=self.broadcast_process_group)

    def __iter__(self):
        if self.is_source:
            iterator = super().__iter__()

            for batch in iterator:
                torch.distributed.broadcast_object_list(
                    [batch.shape], src=self.source_rank, group=self.broadcast_process_group
                )
                batch = batch.to(torch.cuda.current_device())

                torch.distributed.broadcast(batch, src=self.source_rank, group=self.broadcast_process_group)

                yield batch[: self.local_batch_size]
        else:
            for _ in range(self._length):
                batch_shape = [None]
                torch.distributed.broadcast_object_list(
                    batch_shape, src=self.source_rank, group=self.broadcast_process_group
                )

                batch = torch.empty_like(batch_shape[0], device=torch.cuda.current_device())
                torch.distributed.broadcast(batch, src=self.source_rank, group=self.broadcast_process_group)

                batch = batch[
                    self.local_rank_in_broadcast_group
                    * self.local_batch_size : (self.local_rank_in_broadcast_group + 1)
                    * self.local_batch_size
                ]
                yield batch

    def __len__(self) -> int:
        return self._length

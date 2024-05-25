from typing import Any, Callable, Iterable, List, Tuple

import torch
from torch.distributed import ProcessGroup, broadcast, broadcast_object_list
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset, Sampler

from ..utils import get_global_rank


class ResumableDataLoader(_DataLoader):
    def state_dict(self) -> dict:
        return {"dataset": self.dataset.state_dict(), "sampler": self.sampler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.dataset.load_state_dict(state_dict.get("dataset"))
        self.sampler.load_state_dict(state_dict.get("sampler"))


class DispatchingDataLoader(ResumableDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[List] | Iterable[List] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[List], Any] | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        source_rank: int = None,
        broadcast_ranks: List[int] = None,
        all_source_ranks_and_broadcast_groups: List[Tuple[int, ProcessGroup]] = None,
        keys: List[str] = ["input_ids", "attention_mask", "labels"],
    ) -> None:
        self.source_rank = source_rank
        self.broadcast_world_size = len(broadcast_ranks)
        self.is_source = get_global_rank() == self.source_rank
        self.local_rank_in_broadcast_group = broadcast_ranks.index(get_global_rank())
        self.local_batch_size = batch_size
        self.all_source_ranks_and_broadcast_groups = all_source_ranks_and_broadcast_groups

        super().__init__(
            dataset=dataset,
            batch_size=batch_size * self.broadcast_world_size if batch_sampler is None else 1,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        _length = torch.tensor(
            [super().__len__() if self.is_source else 0], dtype=torch.long, device=torch.cuda.current_device()
        )
        self._broadcast_all_groups(_length)
        self._length = _length.item()

        self.keys = keys

    def _broadcast_all_groups(self, item: torch.Tensor, is_tensor: bool = True) -> None:
        for src, grp in self.all_source_ranks_and_broadcast_groups:
            if is_tensor:
                broadcast(item, src=src, group=grp)
            else:
                broadcast_object_list(item, src=src, group=grp)

    def __iter__(self):
        if self.is_source:
            iterator = super().__iter__()
        else:
            iterator = range(self._length)

        for batch in iterator:
            # send/recv tensor shapes
            batch_shape = [batch[self.keys[0]].shape if self.is_source else None]
            self._broadcast_all_groups(batch_shape, is_tensor=False)

            # note batch is just a number on non source ranks for now, we need to fix it
            if not self.is_source:
                batch = {}

            for key in self.keys:
                # send/recv batch
                batch[key] = (
                    batch[key].to(torch.cuda.current_device())
                    if self.is_source
                    else torch.empty(batch_shape[0], dtype=torch.long, device=torch.cuda.current_device())
                )
                self._broadcast_all_groups(batch[key])

                # slice batch
                batch[key] = batch[key][
                    self.local_rank_in_broadcast_group
                    * self.local_batch_size : (self.local_rank_in_broadcast_group + 1)
                    * self.local_batch_size
                ]

            yield batch

    def __len__(self) -> int:
        return self._length

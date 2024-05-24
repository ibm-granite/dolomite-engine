from typing import Any, Callable, Iterable, List

import torch.distributed
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
        keys: List[str] = ["input_ids", "attention_mask", "labels"],
    ) -> None:
        self.source_rank = source_rank
        self.broadcast_process_group = torch.distributed.new_group(ranks=broadcast_ranks)
        self.broadcast_world_size = len(broadcast_ranks)
        self.is_source = get_global_rank() == self.source_rank
        self.local_rank_in_broadcast_group = broadcast_ranks.index(get_global_rank())
        self.local_batch_size = batch_size

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

        if self.is_source:
            self._length = torch.tensor([super().__len__()], dtype=torch.long, device=torch.cuda.current_device())
        else:
            self._length = torch.empty(1, dtype=torch.long, device=torch.cuda.current_device())

        torch.distributed.broadcast(self._length, src=self.source_rank, group=self.broadcast_process_group)
        self._length = self._length.item()

        self.keys = keys

    def __iter__(self):
        if self.is_source:
            iterator = super().__iter__()

            for batch in iterator:
                # send tensor shapes
                batch_shape = [batch[self.keys[0]].shape]
                torch.distributed.broadcast_object_list(
                    batch_shape, src=self.source_rank, group=self.broadcast_process_group
                )

                for i in self.keys:
                    # send batch
                    batch[i] = batch[i].to(torch.cuda.current_device())
                    torch.distributed.broadcast(batch[i], src=self.source_rank, group=self.broadcast_process_group)

                    # slice batch
                    batch[i] = batch[i][
                        self.local_rank_in_broadcast_group
                        * self.local_batch_size : (self.local_rank_in_broadcast_group + 1)
                        * self.local_batch_size
                    ]

                yield batch
        else:
            for _ in range(self._length):
                # receive tensor shapes
                batch_shape = [None]
                torch.distributed.broadcast_object_list(
                    batch_shape, src=self.source_rank, group=self.broadcast_process_group
                )

                batch = {}
                for i in self.keys:
                    # receive batch
                    batch[i] = torch.empty(batch_shape[0], dtype=torch.long, device=torch.cuda.current_device())
                    torch.distributed.broadcast(batch[i], src=self.source_rank, group=self.broadcast_process_group)

                    # slice batch
                    batch[i] = batch[i][
                        self.local_rank_in_broadcast_group
                        * self.local_batch_size : (self.local_rank_in_broadcast_group + 1)
                        * self.local_batch_size
                    ]

                yield batch

    def __len__(self) -> int:
        return self._length

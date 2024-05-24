from typing import Tuple

from ...utils import get_global_rank, get_world_size


class MegatronPretrainingSampler:
    def __init__(
        self, total_samples: int, consumed_samples: int, micro_batch_size: int, drop_last: bool = True
    ) -> None:
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * get_world_size()
        self.drop_last = drop_last
        self.data_parallel_rank = get_global_rank()

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.consumed_samples < self.total_samples, "no samples left to consume: {}, {}".format(
            self.consumed_samples, self.total_samples
        )
        assert self.micro_batch_size > 0

    def __len__(self) -> int:
        return self.total_samples

    def _get_start_end_idx(self) -> Tuple[int, int]:
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self._get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self._get_start_end_idx()
            yield batch[start_idx:end_idx]

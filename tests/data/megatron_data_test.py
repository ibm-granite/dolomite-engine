import os
import tempfile

import numpy as np
import torch

from dolomite_engine.data.megatron.indexed_dataset import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)

from .test_commons import TestCommons


class MegatronDatasetTest(TestCommons):
    def test_megatron_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "file")
            bin_path = get_bin_path(prefix)
            idx_path = get_idx_path(prefix)

            builder = MMapIndexedDatasetBuilder(bin_path)

            num_documents = 1000

            document = np.array([1, 2])

            for _ in range(num_documents):
                builder.add_item(torch.tensor(document))
                builder.end_document()

            builder.finalize(idx_path)

            assert os.path.exists(bin_path)
            assert os.path.exists(idx_path)

            dataset = MMapIndexedDataset(prefix)

            assert len(dataset) == num_documents
            for i in dataset:
                assert (i == document).all()

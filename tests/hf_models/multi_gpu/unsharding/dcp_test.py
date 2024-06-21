import subprocess
import tempfile

import torch
import torch.distributed
from parameterized import parameterized

from dolomite_engine.hf_models import AttentionHeadType

from ...test_common import TestCommons


class DCPTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            TestCommons.get_attention_head_types()[:1], ["gelu", "geglu"][:1], [False, True][:1]
        )
    )
    def test_dcp(
        self, attention_head_type: AttentionHeadType, activation_function: str, tensor_parallel_embeddings: bool
    ) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        gpus_per_node = torch.cuda.device_count()

        with tempfile.TemporaryDirectory() as tmp_path:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(gpus_per_node),
                "-m",
                "tests.hf_models.multi_gpu.unsharding.dcp",
                "--config",
                "tests/hf_models/multi_gpu/unsharding/dcp_config.yml",
                "--tmp-path",
                str(tmp_path),
            ]

            if tensor_parallel_embeddings:
                command.append("--tensor-parallel-embeddings")

            subprocess.run(command, check=True)

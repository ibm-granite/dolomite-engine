import os

import torch
from parameterized import parameterized
from transformers import set_seed

from dolomite_engine.hf_models.modeling_utils import flash_attention

from ...test_common import TestCommons


SEED = 1234


class FlashKernelTest(TestCommons):
    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], [torch.float16, torch.bfloat16]))
    def test_flash_kernel_no_mask(self, device: torch.device, torch_dtype: torch.dtype) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        query_pytorch = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        key_pytorch = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        value_pytorch = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)

        os.environ.update({"PYTORCH_NATIVE_FLASH_KERNEL": "1"})
        attention_pytorch = flash_attention(
            query_pytorch,
            key_pytorch,
            value_pytorch,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dropout_p=0,
            softmax_scale=None,
            causal=True,
        )

        query_tri_dao = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        query_tri_dao.data = query_pytorch.data
        key_tri_dao = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        key_tri_dao.data = key_pytorch.data
        value_tri_dao = torch.randn(4, 1024, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        value_tri_dao.data = value_pytorch.data

        os.environ.update({"PYTORCH_NATIVE_FLASH_KERNEL": "0"})
        attention_tri_dao = flash_attention(
            query_tri_dao,
            key_tri_dao,
            value_tri_dao,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dropout_p=0,
            softmax_scale=None,
            causal=True,
        )

        self.assert_equal_tensors(attention_pytorch, attention_tri_dao, exact_match=True)

        attention_pytorch.mean().backward()
        attention_tri_dao.mean().backward()

        tolerance = 6e-7

        self.assert_equal_tensors(
            query_pytorch.grad,
            query_tri_dao.grad,
            exact_match=False,
            rtol_float32=0,
            atol_float32=tolerance,
            rtol_float16=0,
            atol_float16=tolerance,
            rtol_bfloat16=0,
            atol_bfloat16=tolerance,
        )
        self.assert_equal_tensors(
            query_pytorch.grad,
            key_tri_dao.grad,
            exact_match=False,
            rtol_float32=0,
            atol_float32=tolerance,
            rtol_float16=0,
            atol_float16=tolerance,
            rtol_bfloat16=0,
            atol_bfloat16=tolerance,
        )
        self.assert_equal_tensors(
            query_pytorch.grad,
            value_tri_dao.grad,
            exact_match=False,
            rtol_float32=0,
            atol_float32=tolerance,
            rtol_float16=0,
            atol_float16=tolerance,
            rtol_bfloat16=0,
            atol_bfloat16=tolerance,
        )

    @parameterized.expand(TestCommons.make_args_matrix([torch.device("cuda")], [torch.float16, torch.bfloat16]))
    def test_flash_kernel(self, device: torch.device, torch_dtype: torch.dtype) -> None:
        self.skip_test_if_device_unavailable(device)

        set_seed(SEED)

        query_pytorch = torch.randn(4096, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        key_pytorch = torch.randn(4096, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        value_pytorch = torch.randn(4096, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)

        cu_seqlens = torch.tensor([1000, 999, 997, 996, 104], device=device)
        max_seqlen = torch.tensor(1000, device=device)

        os.environ.update({"PYTORCH_NATIVE_FLASH_KERNEL": "1"})
        attention_pytorch = flash_attention(
            query_pytorch,
            key_pytorch,
            value_pytorch,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0,
            softmax_scale=None,
            causal=True,
        )

        query_tri_dao = torch.randn(4096, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        query_tri_dao.data = query_pytorch.data
        key_tri_dao = torch.randn(4096, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        key_tri_dao.data = key_pytorch.data
        value_tri_dao = torch.randn(4096, 32, 128, device=device, requires_grad=True, dtype=torch_dtype)
        value_tri_dao.data = value_pytorch.data

        os.environ.update({"PYTORCH_NATIVE_FLASH_KERNEL": "0"})
        attention_tri_dao = flash_attention(
            query_tri_dao,
            key_tri_dao,
            value_tri_dao,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0,
            softmax_scale=None,
            causal=True,
        )

        self.assert_equal_tensors(attention_pytorch, attention_tri_dao, exact_match=True)

        attention_pytorch.mean().backward()
        attention_tri_dao.mean().backward()

        tolerance = 6e-7

        self.assert_equal_tensors(
            query_pytorch.grad,
            query_tri_dao.grad,
            exact_match=False,
            rtol_float32=0,
            atol_float32=tolerance,
            rtol_float16=0,
            atol_float16=tolerance,
            rtol_bfloat16=0,
            atol_bfloat16=tolerance,
        )
        self.assert_equal_tensors(
            query_pytorch.grad,
            key_tri_dao.grad,
            exact_match=False,
            rtol_float32=0,
            atol_float32=tolerance,
            rtol_float16=0,
            atol_float16=tolerance,
            rtol_bfloat16=0,
            atol_bfloat16=tolerance,
        )
        self.assert_equal_tensors(
            query_pytorch.grad,
            value_tri_dao.grad,
            exact_match=False,
            rtol_float32=0,
            atol_float32=tolerance,
            rtol_float16=0,
            atol_float16=tolerance,
            rtol_bfloat16=0,
            atol_bfloat16=tolerance,
        )

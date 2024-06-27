from typing import Tuple

import torch
import torch.nn.functional as F

from ....utils import is_flash_attention_available


if is_flash_attention_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func


class _FlashAttentionVarlenTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        attention_output, log_sum_exp, philox_seed, philox_offset, _ = torch.ops.aten._flash_attention_forward(
            query=query,
            key=key,
            value=value,
            cum_seq_q=cu_seqlens_q,
            cum_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_k,
            dropout_p=dropout_p,
            is_causal=causal,
            return_debug_mask=False,
            scale=softmax_scale,
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            attention_output,
            log_sum_exp,
            philox_seed,
            philox_offset,
        )

        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k

        return attention_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            attention_output,
            log_sum_exp,
            philox_seed,
            philox_offset,
        ) = ctx.saved_tensors

        dropout_p = ctx.dropout_p
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k

        query_grad, key_grad, value_grad = torch.ops.aten._flash_attention_backward(
            grad_out=grad_output,
            query=query,
            key=key,
            value=value,
            out=attention_output,
            logsumexp=log_sum_exp,
            cum_seq_q=cu_seqlens_q,
            cum_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_k,
            dropout_p=dropout_p,
            is_causal=causal,
            philox_seed=philox_seed,
            philox_offset=philox_offset,
            scale=softmax_scale,
        )

        return query_grad, key_grad, value_grad, None, None, None, None, None, None, None


def flash_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    use_pytorch_native_flash_attention: bool = False,
) -> torch.Tensor:
    if use_pytorch_native_flash_attention:
        attention_output = _FlashAttentionVarlenTorch.apply(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
        )
    else:
        attention_output = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )

    return attention_output


def get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

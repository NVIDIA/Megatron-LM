from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from megatron.core.chunked_pipeline_parallel_utils import ChunkedPipelineParallelParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType


def _slice(tensor: torch.Tensor, start: int, end: int, dim: int) -> torch.Tensor:
    """
    Returns a slice of the input tensor along the specified dimension.

    Args:
        tensor (torch.Tensor): Input tensor to be sliced.
        start (int): Start index of the slice (inclusive).
        end (int): End index of the slice (exclusive).
        dim (int): Dimension along which to perform the slice.

    Returns:
        torch.Tensor: A contiguous tensor slice of the input along the specified dimension.
    """
    slices = [slice(None)] * tensor.dim()
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]


class AttentionFuncionWithChunkedPipelineParallel(torch.autograd.Function):
    """TODO(yuzhongw): add docstring for this class"""

    """TODO(yuzhongw): refine the code"""

    @staticmethod
    def forward(
        ctx,
        module: Callable,
        chunked_pp_params: ChunkedPipelineParallelParams,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        assert chunked_pp_params is not None, (
            f"Chunked PP size = {self.config.chunked_pipeline_model_parallel_splits}, "
            "but chunked_pp_params is None."
        )
        assert (
            packed_seq_params is None
        ), "Packed sequence is not compatible with chunked PP for now."
        assert attention_mask is None, "Attention mask is not supported with chunked PP for now."

        # Switch to bottom-right mask for causal mask
        if attn_mask_type == AttnMaskType.causal:
            attn_mask_type = AttnMaskType.causal_bottom_right
        elif attn_mask_type == AttnMaskType.padding_causal:
            attn_mask_type = AttnMaskType.padding_causal_bottom_right

        ctx.module = module  # Save the self reference for backward
        ctx.chunked_pp_params = chunked_pp_params
        seq_dim = 0  # Currently only support SBHD layout
        span_idx = chunked_pp_params.span_idx_in_micro
        kv_cache_pool = chunked_pp_params.kv_cache
        if span_idx != 0:
            k_cache, v_cache = kv_cache_pool["k_cache"], kv_cache_pool["v_cache"]
            kv_cache_pool["tensor_ref"][span_idx - 1] = (k_cache, v_cache)
            offset = k_cache.shape[seq_dim]
            key = torch.cat([k_cache, key], dim=seq_dim).detach().requires_grad_()
            value = torch.cat([v_cache, value], dim=seq_dim).detach().requires_grad_()
            k_cache.data = torch.tensor([], device=key.device, dtype=key.dtype)
            v_cache.data = torch.tensor([], device=key.device, dtype=key.dtype)
        else:
            kv_cache_pool["tensor_ref"] = {}
            key = key.detach().requires_grad_()
            value = value.detach().requires_grad_()
            offset = 0
        seqlen_k = key.shape[seq_dim]
        seqlen_q = query.shape[seq_dim]
        ctx._seqlen_k = seqlen_k
        ctx._seqlen_q = seqlen_q
        ctx._offset = offset
        # Example for 4 spans
        # [0, 128, 256, 384, 512]
        # seqlen_k: [128, 256, 384, 512], which represent the actual key/value length in compute
        # seqlen_q: [128, 128, 128, 128], which represent the actual query/output/grad_output length in compute
        # offset: [0, 128, 256, 384], which represent the position of current span in sequence
        ctx.kv_cache_pool = kv_cache_pool

        with torch.enable_grad():
            core_attn_out = module(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        kv_cache_pool["k_cache"] = key
        kv_cache_pool["v_cache"] = value
        # if torch.distributed.get_rank() == 0:
        #     print(f"[rank=0] core_attn_out: {core_attn_out.abs().mean().item()}")
        ctx.save_for_backward(query, core_attn_out)
        return core_attn_out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        query, out = ctx.saved_tensors
        span_idx = ctx.chunked_pp_params.span_idx_in_micro
        span_num = len(ctx.chunked_pp_params.spans)
        seq_dim = 0
        last_idx = span_idx == span_num - 1

        pk = ctx.kv_cache_pool["k_cache"].contiguous()
        pv = ctx.kv_cache_pool["v_cache"].contiguous()
        if span_idx != 0:
            ctx.kv_cache_pool["k_cache"], ctx.kv_cache_pool["v_cache"] = (
                # pop up the KV cache for the current span_idx since it is no longer needed for precede span
                _slice(
                    pk, None, -ctx._seqlen_q, seq_dim
                ),  # pk[:, : -ctx._seqlen_q], when seq_dim == 1
                _slice(
                    pv, None, -ctx._seqlen_q, seq_dim
                ),  # pv[:, : -ctx._seqlen_q], when seq_dim == 1
            )
        else:
            del ctx.kv_cache_pool["k_cache"], ctx.kv_cache_pool["v_cache"]

        if not last_idx:
            key, value = ctx.kv_cache_pool["tensor_ref"][span_idx]
            key.data = pk
            value.data = pv
            del ctx.kv_cache_pool["tensor_ref"][span_idx]
        else:
            key = pk
            value = pv
        dq, dk, dv = torch.autograd.grad(
            outputs=out, inputs=(query, key, value), grad_outputs=grad_output
        )
        if not last_idx:
            k_grad_p, v_grad_p = (ctx.kv_cache_pool["k_grad"], ctx.kv_cache_pool["v_grad"])
            dk += k_grad_p
            dv += v_grad_p
        ctx.kv_cache_pool["k_grad"], ctx.kv_cache_pool["v_grad"] = (
            # rearrange the kv grad tensor
            _slice(
                dk, None, ctx._offset, seq_dim
            ).contiguous(),  # dk[:, : ctx._offset], index start to current span position
            _slice(
                dv, None, ctx._offset, seq_dim
            ).contiguous(),  # dv[:, : ctx._offset], index start to current span position
        )
        if span_idx == 0:
            del ctx.kv_cache_pool["k_grad"]
            del ctx.kv_cache_pool["v_grad"]
        dk = _slice(dk, ctx._offset, ctx._seqlen_k, seq_dim)
        dv = _slice(dv, ctx._offset, ctx._seqlen_k, seq_dim)
        return (
            None,  # module
            None,  # chunked_pp_params
            dq,  # query
            dk,  # key
            dv,  # value
            None,  # attention_mask
            None,  # attn_mask_type
            None,  # attention_bias
            None,  # packed_seq_params
        )

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from megatron.core.chunked_pipeline_parallel_utils import ChunkedPipelineParallelParams, KVCache
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class ChunkedKVCacheForAttention(KVCache):
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.key_cache = []
        self.value_cache = []
        self.grad_key_cache = []
        self.grad_value_cache = []

    def forward_with_kv_cache(
        self,
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
        assert len(self.key_cache) == chunked_pp_params.span_idx_in_micro
        assert len(self.value_cache) == chunked_pp_params.span_idx_in_micro
        assert len(self.grad_key_cache) == chunked_pp_params.span_idx_in_micro
        assert len(self.grad_value_cache) == chunked_pp_params.span_idx_in_micro
        return AttentionFuncionWithChunkedPipelineParallel.apply(
            module,
            query,
            key,
            value,
            self.key_cache,
            self.value_cache,
            self.grad_key_cache,
            self.grad_value_cache,
            attention_mask,
            attn_mask_type,
            attention_bias,
            packed_seq_params,
        )


class AttentionFuncionWithChunkedPipelineParallel(torch.autograd.Function):
    """Attention function with chunked pipeline parallel."""

    @staticmethod
    def forward(
        ctx,
        module: Callable,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        grad_key_cache: List[torch.Tensor],
        grad_value_cache: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        # Assertions
        assert (
            packed_seq_params is None
        ), "Packed sequence is not compatible with chunked PP for now."
        assert attention_mask is None, "Attention mask is not supported with chunked PP for now."

        # Switch to bottom-right mask for causal mask
        if attn_mask_type == AttnMaskType.causal:
            attn_mask_type = AttnMaskType.causal_bottom_right
        elif attn_mask_type == AttnMaskType.padding_causal:
            attn_mask_type = AttnMaskType.padding_causal_bottom_right

        # Manage KV cache.
        seq_dim = 0  # Currently only support SBHD layout
        key_cache.append(key)
        value_cache.append(value)
        grad_key_cache.append(None)
        grad_value_cache.append(None)
        key_joined = torch.cat(key_cache, dim=seq_dim).detach().requires_grad_()
        value_joined = torch.cat(value_cache, dim=seq_dim).detach().requires_grad_()

        # Forward pass.
        with torch.enable_grad():
            core_attn_out = module(
                query,
                key_joined,
                value_joined,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        # Release the data of joined key and value to save memory.
        key_joined.data = torch.tensor([], device=key_joined.device, dtype=key_joined.dtype)
        value_joined.data = torch.tensor([], device=value_joined.device, dtype=value_joined.dtype)

        # Save the context
        ctx.seq_dim = seq_dim
        ctx.key_cache = key_cache
        ctx.value_cache = value_cache
        ctx.grad_key_cache = grad_key_cache
        ctx.grad_value_cache = grad_value_cache
        ctx.save_for_backward(query, key_joined, value_joined, core_attn_out)

        core_attn_out = core_attn_out.clone()  # TODO: remove the clone
        return core_attn_out

    @staticmethod
    def backward(ctx, grad_output):
        # Recover the context
        query, key_joined, value_joined, out = ctx.saved_tensors
        seq_dim = ctx.seq_dim
        key_cache = ctx.key_cache
        value_cache = ctx.value_cache
        grad_key_cache = ctx.grad_key_cache
        grad_value_cache = ctx.grad_value_cache

        # Recover the key and value from the key and value cache.
        key_joined.data = torch.cat(key_cache, dim=seq_dim).data
        value_joined.data = torch.cat(value_cache, dim=seq_dim).data

        dq, dk_joined, dv_joined = torch.autograd.grad(
            outputs=out, inputs=(query, key_joined, value_joined), grad_outputs=grad_output
        )
        dk_slices = torch.split(dk_joined, [k.shape[seq_dim] for k in key_cache], dim=seq_dim)
        dv_slices = torch.split(dv_joined, [v.shape[seq_dim] for v in value_cache], dim=seq_dim)
        for i, dk in enumerate(dk_slices):
            if grad_key_cache[i] is None:
                grad_key_cache[i] = dk.contiguous().data
            else:
                grad_key_cache[i].add_(dk.contiguous().data)
        for i, dv in enumerate(dv_slices):
            if grad_value_cache[i] is None:
                grad_value_cache[i] = dv
            else:
                grad_value_cache[i].add_(dv)

        dk = grad_key_cache.pop(-1)
        dv = grad_value_cache.pop(-1)
        key_cache.pop(-1)
        value_cache.pop(-1)

        return (
            None,  # module
            dq,  # query
            dk,  # key
            dv,  # value
            None,  # key_cache
            None,  # value_cache
            None,  # grad_key_cache
            None,  # grad_value_cache
            None,  # attention_mask
            None,  # attn_mask_type
            None,  # attention_bias
            None,  # packed_seq_params
        )

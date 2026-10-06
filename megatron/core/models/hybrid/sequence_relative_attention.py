# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental dense CP adapter for the same unsplit FlashAttention arithmetic.

This defines a new numerical baseline. It intentionally does not reproduce the
old TE ring's length-dependent partial-output rounding.
"""

import itertools
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from flash_attn import flash_attn_varlen_func

from megatron.core.models.hybrid import shared_prefix_fused as attention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

if TYPE_CHECKING:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention


@lru_cache(maxsize=128)
def rank_major_mapping(
    boundaries: tuple[int, ...], cp_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map CP rank-major token ownership to canonical sequence order and back."""
    indices = []
    for rank in range(cp_size):
        for start, end in itertools.pairwise(boundaries):
            assert (end - start) % (2 * cp_size) == 0
            segment = (end - start) // (2 * cp_size)
            indices.extend(
                (
                    torch.arange(
                        start + rank * segment, start + (rank + 1) * segment, device=device
                    ),
                    torch.arange(end - (rank + 1) * segment, end - rank * segment, device=device),
                )
            )
    forward = torch.cat(indices)
    assert forward.numel() == boundaries[-1]
    return forward, forward.argsort()


def dense_attention_cp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    scale: float | None = None,
) -> torch.Tensor:
    """Exchange sequence shards for head shards, then attend each full sequence."""
    assert query.ndim == key.ndim == value.ndim == 3
    cp_size = cp_group.size()
    boundaries = tuple(cu_seqlens.tolist())
    assert boundaries[0] == 0 and query.shape[0] * cp_size == boundaries[-1]
    assert key.shape[0] == value.shape[0] == query.shape[0]
    assert key.shape[1] == value.shape[1]
    assert query.shape[2] == key.shape[2] == value.shape[2]
    to_rank_major, to_canonical = rank_major_mapping(boundaries, cp_size, query.device)
    slices = attention._cp_kv_head_slices_for_destinations(query.shape[1], key.shape[1], cp_size)

    def exchange(tensor, *, kv=False):
        if kv:
            tensor = torch.cat([tensor[:, part, :] for part in slices], dim=1)
        assert tensor.shape[1] % cp_size == 0
        heads = tensor.shape[1] // cp_size
        result = attention.all_to_all_sp2hp(tensor.reshape(tensor.shape[0], 1, -1), group=cp_group)
        return result.reshape(boundaries[-1], heads, tensor.shape[-1])[to_canonical]

    q, k, v = exchange(query), exchange(key, kv=True), exchange(value, kv=True)
    maximum = max(end - start for start, end in itertools.pairwise(boundaries))
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        maximum,
        maximum,
        causal=True,
        softmax_scale=scale,
        deterministic=True,
    )
    rank_major = output[to_rank_major].reshape(boundaries[-1], 1, -1)
    result = attention.all_to_all_hp2sp(rank_major, group=cp_group)
    return result.reshape(query.shape[0], query.shape[1] * value.shape[-1])


def sequence_relative_attention_forward(
    self: "TEDotProductAttention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    attn_mask_type: AttnMaskType,
    attention_bias: torch.Tensor | None = None,
    packed_seq_params: PackedSeqParams | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Use the common sequence-relative backend for supported causal THD attention."""
    if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
        raise ValueError("sequence-relative attention requires packed THD input")
    if attention_mask is not None or attention_bias is not None or num_splits is not None:
        raise ValueError(
            "sequence-relative attention does not support mask/bias/num_splits overrides"
        )
    assert attn_mask_type.name in ("causal", "padding_causal")
    assert self.config.attention_dropout == 0 and self.config.window_size is None
    assert not self.config.qk_clip and not self.config.log_max_attention_logit
    assert self.config.softmax_type == "vanilla"
    assert packed_seq_params.local_cp_size is None
    group = packed_seq_params.cp_group
    if group is None:
        group = self.cp_group
    assert group is not None
    # Shared-prefix MTP already aligns each dense branch to the CP/TP quantum
    # and supplies cu_seqlens without the optional *_padded fields.
    cu_q = packed_seq_params.cu_seqlens_q_padded
    cu_k = packed_seq_params.cu_seqlens_kv_padded
    if cu_q is None:
        cu_q = packed_seq_params.cu_seqlens_q
    if cu_k is None:
        cu_k = packed_seq_params.cu_seqlens_kv
    assert cu_q is not None and cu_k is not None and torch.equal(cu_q, cu_k)
    return dense_attention_cp(
        query, key, value, cu_seqlens=cu_q, cp_group=group, scale=self.config.softmax_scale
    )

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional MagiAttention primitive for load-balanced context parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn as nn


class _MagiAttentionAPI(NamedTuple):
    calc_attn: Any
    dispatch: Any
    undispatch: Any
    magi_attn_varlen_key: Any
    DispatchConfig: Any
    DistAttnConfig: Any
    OverlapConfig: Any
    AttnOverlapMode: Any


def _load_magi_attention_api() -> _MagiAttentionAPI:
    """Import MagiAttention lazily so it remains an optional dependency."""

    try:
        from magi_attention.api import calc_attn, dispatch, magi_attn_varlen_key, undispatch
        from magi_attention.common.enum import AttnOverlapMode
        from magi_attention.config import DispatchConfig, DistAttnConfig, OverlapConfig
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Megatron Lite's MagiAttention backend requires a working MagiAttention "
            "installation for the current CUDA architecture. Install "
            "SandAI-org/MagiAttention before selecting attention_backend_override='magi'."
        ) from exc

    return _MagiAttentionAPI(
        calc_attn=calc_attn,
        dispatch=dispatch,
        undispatch=undispatch,
        magi_attn_varlen_key=magi_attn_varlen_key,
        DispatchConfig=DispatchConfig,
        DistAttnConfig=DistAttnConfig,
        OverlapConfig=OverlapConfig,
        AttnOverlapMode=AttnOverlapMode,
    )


@dataclass(frozen=True, slots=True)
class MagiAttentionConfig:
    """Load-balancing and communication-overlap controls for MagiAttention.

    Both fields default to ``None`` = trust MagiAttention's own judgement:
    the chunk size is derived from the sequence length, and the overlap
    degree is solved per microbatch by the dynamic overlap solver. Explicit
    values are expert overrides for when profiling shows the automatic
    choice is wrong on a given workload or machine.
    """

    chunk_size: int | None = None
    overlap_degree: int | None = None

    def __post_init__(self) -> None:
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("MagiAttention chunk_size must be positive when provided.")
        if self.overlap_degree is not None and self.overlap_degree < 0:
            raise ValueError(
                "MagiAttention overlap_degree must be non-negative or None (dynamic)."
            )


def resolve_magi_attention_config(
    config: MagiAttentionConfig, *, total_tokens: int, cp_size: int
) -> MagiAttentionConfig:
    """Resolve the effective per-microbatch MagiAttention options.

    This is the single policy seam for overriding MagiAttention's own
    automatic judgement. The default implementation trusts upstream auto:
    ``None`` fields pass through, so magi derives the chunk size and the
    dynamic overlap solver picks the overlap degree. If profiling ever shows
    that judgement is wrong on a given machine (its cost model constants are
    calibrated upstream), encode the locally calibrated policy here — e.g. a
    lookup keyed on ``total_tokens // cp_size`` and the device capability —
    instead of scattering tuning logic across callers.
    """
    del total_tokens, cp_size
    return config


def build_magi_attention_runtime_key(
    cu_seqlens: torch.Tensor,
    *,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    config: MagiAttentionConfig | None = None,
) -> Any:
    """Build the per-microbatch MagiAttention runtime and dispatch plan.

    ``config`` defaults to fully automatic behaviour; explicit values exist
    for tests and for ``resolve_magi_attention_config`` calibration policies.
    """

    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(
            "MagiAttention cu_seqlens must be one-dimensional with at least two entries."
        )
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"MagiAttention cu_seqlens must be int32, got {cu_seqlens.dtype}.")
    if cp_group is None:
        raise ValueError("MagiAttention requires an explicit context-parallel process group.")
    if num_heads_q < 1 or num_heads_kv < 1 or num_heads_q % num_heads_kv != 0:
        raise ValueError(
            "MagiAttention requires positive query/KV head counts with query heads "
            "divisible by KV heads."
        )

    api = _load_magi_attention_api()
    config = resolve_magi_attention_config(
        config if config is not None else MagiAttentionConfig(),
        total_tokens=int(cu_seqlens[-1].item()),
        cp_size=int(cp_group.size()),
    )

    def build_key(chunk_size: int | None):
        if config.overlap_degree is None:
            # Dynamic mode: the overlap solver picks the degree per microbatch.
            overlap_config = api.OverlapConfig(mode=api.AttnOverlapMode.DYNAMIC, degree=None)
        else:
            overlap_config = api.OverlapConfig(degree=config.overlap_degree)
        dist_attn_config = api.DistAttnConfig(
            dispatch_config=api.DispatchConfig(chunk_size=chunk_size),
            overlap_config=overlap_config,
        )
        return api.magi_attn_varlen_key(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=0,
            cp_group_or_mesh=cp_group,
            causal=True,
            dist_attn_config=dist_attn_config,
        )

    runtime_key = build_key(config.chunk_size)
    if int(getattr(runtime_key, "pad_size", 0)) == 0:
        return runtime_key

    # Lite already pads each THD sequence to a CP-compatible length. If Magi's
    # automatically capped chunk size would add another tail pad, lower it to
    # the nearest divisor of tokens-per-rank. This keeps every CP shard equal
    # without introducing extra loss-bearing tokens.
    total_tokens = int(cu_seqlens[-1].item())
    cp_size = int(cp_group.size())
    if total_tokens % cp_size != 0:
        raise ValueError(
            f"MagiAttention requires total tokens ({total_tokens}) divisible by CP={cp_size}."
        )
    tokens_per_rank = total_tokens // cp_size
    adjusted_chunk_size = min(int(runtime_key.chunk_size), tokens_per_rank)
    while adjusted_chunk_size > 1 and tokens_per_rank % adjusted_chunk_size != 0:
        adjusted_chunk_size -= 1
    runtime_key = build_key(adjusted_chunk_size)
    if int(getattr(runtime_key, "pad_size", 0)) != 0:
        raise RuntimeError("Failed to derive a padding-free MagiAttention chunk size.")
    return runtime_key


def dispatch_magi_attention_tensor(
    tensor: torch.Tensor, runtime_key: Any, *, pad_value: float = 0.0
) -> torch.Tensor:
    """Apply a MagiAttention runtime's token permutation and CP dispatch."""

    return _load_magi_attention_api().dispatch(tensor, runtime_key, pad_value=pad_value)


def undispatch_magi_attention_tensor(tensor: torch.Tensor, runtime_key: Any) -> torch.Tensor:
    """Restore a CP-local MagiAttention tensor to global token order."""

    return _load_magi_attention_api().undispatch(tensor, runtime_key)


class MagiDotProductAttention(nn.Module):
    """Core attention adapter consuming a runtime key prepared before QKV projection."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        if head_dim < 1:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")
        self.softmax_scale = head_dim**-0.5

    def forward(self, query, key, value, *, packed_seq_params):
        runtime_key = getattr(packed_seq_params, "magi_runtime_key", None)
        if runtime_key is None:
            raise ValueError(
                "MagiAttention runtime metadata is missing. Prepare the microbatch with "
                "pack_magi_forward_kwargs before model.forward."
            )
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("MagiAttention expects Q/K/V in THD layout [tokens, heads, head_dim].")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"MagiAttention requires fp16 or bf16 Q/K/V tensors, got {query.dtype}."
            )
        if key.shape != value.shape:
            raise ValueError(
                f"MagiAttention key/value shapes must match, got {key.shape} and {value.shape}."
            )
        if hasattr(runtime_key, "num_heads_q") and runtime_key.num_heads_q != query.size(1):
            raise ValueError(
                f"MagiAttention runtime expects {runtime_key.num_heads_q} query heads, "
                f"got {query.size(1)}."
            )
        if hasattr(runtime_key, "num_heads_kv") and runtime_key.num_heads_kv != key.size(1):
            raise ValueError(
                f"MagiAttention runtime expects {runtime_key.num_heads_kv} KV heads, "
                f"got {key.size(1)}."
            )

        output, _meta = _load_magi_attention_api().calc_attn(
            query, key, value, runtime_key, softmax_scale=self.softmax_scale
        )
        return output


__all__ = [
    "MagiAttentionConfig",
    "MagiDotProductAttention",
    "build_magi_attention_runtime_key",
    "dispatch_magi_attention_tensor",
    "resolve_magi_attention_config",
    "undispatch_magi_attention_tensor",
]

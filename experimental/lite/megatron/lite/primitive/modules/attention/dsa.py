# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Dynamic Sparse Attention.

The module is model-agnostic: callers pass architecture dimensions directly and
keep model config classes out of the primitive layer.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from megatron.lite.primitive.kernels import dsa_kernels as _dsa_kernels
from megatron.lite.primitive.modules.attention.cp import iter_cp_sources
from megatron.lite.primitive.parallel.cp import (
    contiguous_position_ids_for_cp,
    contiguous_slice_for_cp,
)

if TYPE_CHECKING:
    from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention


# cuDNN FE 1.27's indexer_top_k decode-varlen kernel loads score columns in
# 512-thread tiles. CP aligns gathered projected KV once so the physical
# score width cannot end in a partial tile; the causal mask preserves the
# original logical sequence boundary.
_CUDNN_DSA_TOPK_COLUMN_ALIGNMENT = 512


def _fused_indexer_sparse_attn(*args, value_dim: int | None = None, **kwargs):
    try:
        return _dsa_kernels.fused_indexer_sparse_attn(
            *args, value_dim=value_dim, **kwargs
        )
    except TypeError as exc:
        if "value_dim" not in str(exc):
            raise
        return _dsa_kernels.fused_indexer_sparse_attn(*args, **kwargs)


def _fused_indexer_sparse_attn_with_topk(*args, value_dim: int | None = None, **kwargs):
    try:
        return _dsa_kernels.fused_indexer_sparse_attn_with_topk(
            *args, value_dim=value_dim, **kwargs
        )
    except TypeError as exc:
        if "value_dim" not in str(exc):
            raise
        return _dsa_kernels.fused_indexer_sparse_attn_with_topk(*args, **kwargs)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """Attach the DSA indexer loss to the output without changing forward values."""

    main_loss_backward_scale: torch.Tensor | None = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = (
            torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        )
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


RMSNorm = te.RMSNorm


def _hadamard_transform_torch(x: torch.Tensor, scale: float) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dim, got {n}")
    original_shape = x.shape
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (h * 2), h * 2)
        left = y[..., :h]
        right = y[..., h:]
        y = torch.cat([left + right, left - right], dim=-1)
        h *= 2
    return y.reshape(original_shape) * scale


try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard_transform
except Exception:  # pragma: no cover - optional CUDA extension
    _fast_hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
    scale = x.shape[-1] ** -0.5
    if _fast_hadamard_transform is not None and x.is_cuda:
        return _fast_hadamard_transform(x, scale=scale)
    return _hadamard_transform_torch(x, scale=scale)


def build_rope_cache(
    *,
    dim: int,
    max_position_embeddings: int,
    rope_theta: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    positions = torch.arange(
        max_position_embeddings, dtype=torch.float32, device=device
    )
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def build_rotary_embeddings(
    *, position_ids: torch.Tensor, dim: int, rope_theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    device = position_ids.device
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(torch.float32)
            / dim
        )
    )
    inv_freq_expanded = (
        inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = (
        device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, unsqueeze_dim: int
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    interleaved: bool = True,
) -> torch.Tensor:
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    input_dtype = x.dtype
    x = x.float()
    cos = cos.to(device=x.device)[position_ids].float().unsqueeze(2)
    sin = sin.to(device=x.device)[position_ids].float().unsqueeze(2)
    if interleaved:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out.to(input_dtype)

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(input_dtype)


def _rotary_embeddings_from_cache(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3 and sin.dim() == 3:
        return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)

    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    cos = cos.to(device=device)[position_ids].float()
    sin = sin.to(device=device)[position_ids].float()
    if cos.shape[-1] * 2 == dim:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _all_gather_cp(
    tensor: torch.Tensor, *, cp_size: int, cp_group
) -> list[torch.Tensor]:
    if cp_size <= 1:
        return [tensor]
    if cp_group is None:
        raise RuntimeError("CP>1 requires a context-parallel process group.")
    from torch.distributed.nn.functional import all_gather

    return list(all_gather(tensor.contiguous(), group=cp_group))


def _dense_cp_layout(
    *, local_seq: int, cp_size: int, cp_rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous local-query positions and global KV order."""
    full_seq = local_seq * cp_size
    query_positions = contiguous_position_ids_for_cp(
        full_seq, cp_rank, cp_size, device
    ).flatten()
    return query_positions, torch.arange(full_seq, device=device, dtype=torch.long)


def _packed_cp_layout(
    cu_seqlens: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous THD local-query positions and global KV order."""
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
    total = int(cu_seqlens[-1].item())
    if total % cp_size:
        raise ValueError(
            f"Packed sequential CP requires total tokens divisible by cp_size; "
            f"got total={total}, cp_size={cp_size}."
        )
    query_positions = contiguous_position_ids_for_cp(
        total, cp_rank, cp_size, device
    ).flatten()
    # Sequential shards are gathered in rank order, which is already global KV order.
    return query_positions, torch.arange(total, device=device, dtype=torch.long)


def _pad_cp_projected_kv(
    kv: torch.Tensor,
    index_k: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Physically align gathered CP KV for cuDNN top-k score loads."""
    logical_rows = kv.shape[0]
    aligned_rows = (
        (logical_rows + _CUDNN_DSA_TOPK_COLUMN_ALIGNMENT - 1)
        // _CUDNN_DSA_TOPK_COLUMN_ALIGNMENT
        * _CUDNN_DSA_TOPK_COLUMN_ALIGNMENT
    )
    pad_rows = aligned_rows - logical_rows
    if pad_rows == 0:
        return kv, index_k

    kv = torch.cat((kv, kv.new_zeros((pad_rows, *kv.shape[1:]))), dim=0)
    if index_k is not None:
        index_k = torch.cat(
            (index_k, index_k.new_zeros((pad_rows, *index_k.shape[1:]))), dim=0
        )
    return kv, index_k


def _build_cp_causal_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the explicit local-Q/global-K causal validity mask used by DSA CP."""
    query_positions = query_positions.to(dtype=torch.long)
    key_positions = key_positions.to(device=query_positions.device, dtype=torch.long)
    valid = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=query_positions.device, dtype=torch.long)
        query_segments = torch.bucketize(query_positions, cu_seqlens[1:], right=True)
        key_segments = torch.bucketize(key_positions, cu_seqlens[1:], right=True)
        valid = valid & (query_segments.unsqueeze(1) == key_segments.unsqueeze(0))
    zeros = torch.zeros((), device=query_positions.device, dtype=torch.float32)
    return torch.where(valid, zeros, torch.full_like(zeros, float("-inf")))


def _index_scores_and_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    mask: torch.Tensor,
    topk: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel-routed CP indexer over local Q and gathered projected K."""
    result = _dsa_kernels.indexer_topk_with_mask(
        q,
        k,
        weights,
        topk,
        mask,
        indexer_softmax_scale=scale,
    )
    if q.is_cuda:
        # cuDNN FE 1.27's decode-varlen wrapper owns an asynchronous scratch
        # buffer that is not returned to PyTorch.  Finish that launch before
        # compactify can reuse the allocation; CUDA_LAUNCH_BLOCKING previously
        # hid this lifetime bug during the focused THD run.
        torch.cuda.current_stream(q.device).synchronize()
    return result


def _cp_indexer_loss(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    *,
    mask: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    calculate_per_token_loss: bool,
) -> torch.Tensor:
    """Kernel-layer KL target for explicit local-Q/global-K CP masking."""
    return _dsa_kernels.cp_indexer_loss(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        query,
        kv,
        mask=mask,
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        sparse_loss=sparse_loss,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def is_dsa_skip_topk_layer(
    layer_number: int, skip_topk_offset: int, topk_freq: int
) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA indexer top-k."""
    if layer_number < 1:
        raise ValueError(f"layer_number must be >= 1, got {layer_number}")
    if topk_freq < 1:
        raise ValueError(f"topk_freq must be >= 1, got {topk_freq}")
    if skip_topk_offset < 0:
        raise ValueError(f"skip_topk_offset must be >= 0, got {skip_topk_offset}")
    if topk_freq == 1:
        return False
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_dsa_compute_layer(
    layer_number: int, skip_topk_offset: int, topk_freq: int
) -> int:
    """Return the 1-indexed full/indexer layer used by ``layer_number``."""
    if not is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq):
        return layer_number
    source_layer = layer_number - (max(layer_number - skip_topk_offset, 0) % topk_freq)
    if source_layer < 1:
        raise ValueError(
            "DSA IndexShare schedule makes layer "
            f"{layer_number} shared before any full source layer."
        )
    return source_layer


def dsa_indexer_type_for_layer(
    layer_number: int, skip_topk_offset: int, topk_freq: int
) -> str:
    return (
        "shared"
        if is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
        else "full"
    )


class DSAIndexShareState:
    """Per-forward top-k holder for DSA cross-layer IndexShare.

    Only one source layer is resident on GPU at a time.  When activation
    checkpointing needs old source groups during backward recomputation, they
    are retained on CPU and paged back one group at a time.  This keeps the GPU
    working set bounded instead of letting checkpoint closures retain every
    source layer's ``[B, S, topk]`` tensor.
    """

    def __init__(self, *, retain_for_recompute: bool = True):
        self.retain_for_recompute = retain_for_recompute
        self._resident_source_layer: int | None = None
        self._resident_topk_by_layer: dict[
            tuple[int, Hashable | None], torch.Tensor
        ] = {}
        self._cpu_topk_by_layer: dict[tuple[int, Hashable | None], torch.Tensor] = {}
        self._sealed = False

    @staticmethod
    def _key(
        layer_number: int, sequence_key: Hashable | None
    ) -> tuple[int, Hashable | None]:
        return layer_number, sequence_key

    def save_topk(
        self,
        layer_number: int,
        topk_indices: torch.Tensor,
        *,
        sequence_key: Hashable | None = None,
    ) -> None:
        if self._sealed:
            # A full source layer is being recomputed in backward.  Its shared
            # consumers have already run in reverse order, so no later layer
            # needs the newly produced top-k.
            if self._resident_source_layer == layer_number:
                self._resident_topk_by_layer.clear()
                self._resident_source_layer = None
            return

        if self._resident_source_layer not in (None, layer_number):
            self._evict_resident(retain=self.retain_for_recompute)
        self._resident_source_layer = layer_number
        self._resident_topk_by_layer[self._key(layer_number, sequence_key)] = (
            topk_indices.detach()
        )

    def finish_forward(self) -> None:
        """Evict the final GPU source group before returning model outputs."""
        if self._sealed:
            return
        self._evict_resident(retain=self.retain_for_recompute)
        self._sealed = True

    def _evict_resident(self, *, retain: bool) -> None:
        if retain:
            for key, topk in self._resident_topk_by_layer.items():
                self._cpu_topk_by_layer[key] = topk.detach().to(device="cpu")
        self._resident_topk_by_layer.clear()
        self._resident_source_layer = None

    @property
    def _topk_by_layer(self) -> dict[tuple[int, Hashable | None], torch.Tensor]:
        """Private compatibility view used by focused validation harnesses."""
        return {**self._cpu_topk_by_layer, **self._resident_topk_by_layer}

    def get_topk(
        self,
        layer_number: int,
        source_layer: int,
        *,
        sequence_key: Hashable | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        key = self._key(source_layer, sequence_key)
        if key in self._resident_topk_by_layer:
            return self._resident_topk_by_layer[key]
        if key not in self._cpu_topk_by_layer:
            available = sorted(self._topk_by_layer)
            raise AssertionError(
                "DSA IndexShare shared layer "
                f"{layer_number} needs top-k indices from source layer {source_layer}, "
                "but that source did not run earlier in this forward. "
                "Cross-PP top-k sharing is not supported. "
                f"Available cache keys: {available}."
            )

        if self._resident_source_layer != source_layer:
            # During backward, groups are consumed in reverse order.  The CPU
            # copy already remains authoritative, so dropping the previous GPU
            # group does not require another device-to-host transfer.
            self._resident_topk_by_layer.clear()
            self._resident_source_layer = source_layer
        topk = self._cpu_topk_by_layer[key]
        if device is not None and topk.device != device:
            topk = topk.to(device=device)
        self._resident_topk_by_layer[key] = topk
        return topk


def validate_dsa_index_share_pipeline_split(
    layer_indices: list[int],
    *,
    topk_freq: int,
    skip_topk_offset: int,
    indexer_types: list[str] | None = None,
) -> None:
    """Fail fast if a local PP stage starts on a shared DSA IndexShare layer."""
    if topk_freq <= 1 and not indexer_types:
        return

    positions = {layer_idx: pos for pos, layer_idx in enumerate(layer_indices)}
    for layer_idx in layer_indices:
        if indexer_types is not None and layer_idx < len(indexer_types):
            indexer_type = indexer_types[layer_idx]
        else:
            indexer_type = dsa_indexer_type_for_layer(
                layer_idx + 1, skip_topk_offset, topk_freq
            )
        if indexer_type != "shared":
            continue

        source_idx = (
            source_dsa_compute_layer(layer_idx + 1, skip_topk_offset, topk_freq) - 1
        )
        if source_idx not in positions:
            raise ValueError(
                "DSA IndexShare cannot cross pipeline stages: layer "
                f"{layer_idx} is shared from source layer {source_idx}, but this "
                f"stage only owns layers {layer_indices}."
            )
        if positions[source_idx] >= positions[layer_idx]:
            raise ValueError(
                "DSA IndexShare source layer must execute before its shared layer "
                f"within a pipeline stage, got source={source_idx}, layer={layer_idx}, "
                f"stage layers={layer_indices}."
            )


class DSAIndexer(nn.Module):
    """Compute per-token top-k key indices for Dynamic Sparse Attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rope_interleaved: bool = True,
        layer_norm_eps: float = 1e-5,
        rope_first: bool = False,
        use_hadamard: bool = True,
    ):
        super().__init__()
        if index_head_dim < qk_rope_head_dim:
            raise ValueError("index_head_dim must be >= qk_rope_head_dim")
        self.num_heads = index_n_heads
        self.head_dim = index_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = index_head_dim - qk_rope_head_dim
        self.index_topk = index_topk
        self.rope_interleaved = rope_interleaved
        self.rope_first = rope_first
        self.use_hadamard = use_hadamard

        self.wq_b = nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(index_head_dim, eps=layer_norm_eps)
        self.weights_proj = nn.Linear(hidden_size, index_n_heads, bias=False)
        self.softmax_scale = index_head_dim**-0.5

    def forward_before_topk(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project GLM5 indexer inputs for Megatron's fused DSA kernels."""
        if attention_mask is not None:
            raise NotImplementedError(
                "GLM5 fused DSA indexer only supports causal masking; custom "
                "attention_mask is not supported."
            )
        batch, seq_len, _ = x.shape
        cos, sin = _rotary_embeddings_from_cache(
            cos,
            sin,
            position_ids,
            device=x.device,
            dtype=x.dtype,
            dim=self.qk_rope_head_dim,
        )

        q = self.wq_b(q_resid).view(batch, seq_len, self.num_heads, self.head_dim)

        k = self.k_norm(self.wk(x))
        if self.rope_first:
            q_pe, q_nope = torch.split(
                q, [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1
            )
            k_pe, k_nope = torch.split(
                k, [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1
            )
        else:
            q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            k_nope, k_pe = torch.split(
                k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2)
        k_pe = k_pe.squeeze(2)

        if self.rope_first:
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)
        else:
            q = torch.cat([q_nope, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
        if self.use_hadamard:
            q = rotate_activation(q)
            k = rotate_activation(k)

        weights = self.weights_proj(x) * (self.num_heads**-0.5)
        return (
            q.transpose(0, 1).contiguous(),
            k.transpose(0, 1).contiguous(),
            weights.transpose(0, 1).contiguous(),
        )

    def forward(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, weights = self.forward_before_topk(
            x, q_resid, cos, sin, position_ids, attention_mask=attention_mask
        )
        topk_indices, _ = _dsa_kernels.indexer_topk(
            q,
            k,
            weights,
            min(self.index_topk, k.shape[0]),
            1,
            indexer_softmax_scale=self.softmax_scale,
        )
        return topk_indices


class DynamicSparseAttention(nn.Module):
    """Correctness-first DSA attention path."""

    @staticmethod
    def dense_attention_cls() -> type[MultiLatentAttention]:
        from megatron.lite.primitive.modules.attention.mla import MultiLatentAttention

        return MultiLatentAttention

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        rms_norm_eps: float,
        rope_interleaved: bool = True,
        latent_rms_norm_eps: float | None = None,
        indexer_layer_norm_eps: float = 1e-5,
        indexer_rope_interleaved: bool | None = None,
        indexer_rope_first: bool = False,
        indexer_use_hadamard: bool = True,
        layer_number: int | None = None,
        index_topk_freq: int = 1,
        index_skip_topk_offset: int = 0,
        indexer_type: str | None = None,
        indexer_loss_coeff: float = 0.0,
        indexer_use_sparse_loss: bool = False,
        calculate_per_token_loss: bool = False,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_group=None,
        cp_mode: str = "native",
    ):
        super().__init__()
        if cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {cp_size}")
        if not 0 <= cp_rank < cp_size:
            raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
        if cp_mode not in {"native", "legacy_gather_all"}:
            raise ValueError(
                "cp_mode must be 'native' or 'legacy_gather_all', " f"got {cp_mode!r}"
            )
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_interleaved = rope_interleaved
        self.softmax_scale = self.qk_head_dim**-0.5
        self.index_topk = index_topk
        self.indexer_softmax_scale = index_head_dim**-0.5
        self.indexer_loss_coeff = indexer_loss_coeff
        self.indexer_use_sparse_loss = indexer_use_sparse_loss
        self.calculate_per_token_loss = calculate_per_token_loss
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_group = cp_group
        self.cp_mode = cp_mode
        self.layer_number = 1 if layer_number is None else layer_number
        self.index_topk_freq = index_topk_freq
        self.index_skip_topk_offset = index_skip_topk_offset
        inferred_indexer_type = dsa_indexer_type_for_layer(
            self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
        )
        if indexer_type is None:
            indexer_type = inferred_indexer_type
        if indexer_type not in {"full", "shared"}:
            raise ValueError(
                f"indexer_type must be 'full' or 'shared', got {indexer_type!r}"
            )
        if indexer_type != inferred_indexer_type:
            raise ValueError(
                f"indexer_type={indexer_type!r} for layer {self.layer_number} does not match "
                f"IndexShare schedule {inferred_indexer_type!r}."
            )
        self.indexer_type = indexer_type
        self.index_share_enabled = self.index_topk_freq > 1
        self.skip_topk = indexer_type == "shared"
        self.index_share_source_layer = source_dsa_compute_layer(
            self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
        )
        latent_rms_norm_eps = (
            rms_norm_eps if latent_rms_norm_eps is None else latent_rms_norm_eps
        )
        indexer_rope_interleaved = (
            rope_interleaved
            if indexer_rope_interleaved is None
            else indexer_rope_interleaved
        )

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=latent_rms_norm_eps)
        self.q_b_proj = nn.Linear(
            q_lora_rank, num_attention_heads * self.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=latent_rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_attention_heads * v_head_dim, hidden_size, bias=False
        )
        self.indexer: DSAIndexer | None = None
        if not self.skip_topk:
            self.indexer = DSAIndexer(
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                rope_interleaved=indexer_rope_interleaved,
                layer_norm_eps=indexer_layer_norm_eps,
                rope_first=indexer_rope_first,
                use_hadamard=indexer_use_hadamard,
            )
        self.register_buffer(
            "attn_sink",
            torch.full((num_attention_heads,), -1.0e20, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params=None,
        index_share_state: DSAIndexShareState | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise NotImplementedError(
                "GLM5 fused DSA only supports causal masking; custom attention_mask "
                "is not supported."
            )
        if packed_seq_params is not None:
            if self.cp_size > 1 and self.cp_mode == "native":
                return self._forward_packed_cp_native(
                    x,
                    cos,
                    sin,
                    position_ids,
                    packed_seq_params,
                    index_share_state=index_share_state,
                )
            if self.cp_size > 1:
                x, position_ids = self._gather_packed_cp_inputs(
                    x, position_ids, packed_seq_params
                )
                cos, sin = self._gather_packed_cp_rotary(
                    cos, sin, packed_seq_params, x.device
                )
            out = self._forward_packed_full(
                x,
                cos,
                sin,
                position_ids,
                packed_seq_params,
                index_share_state=index_share_state,
            )
            if self.cp_size > 1:
                out = contiguous_slice_for_cp(
                    out, self.cp_rank, self.cp_size, seq_dim=1
                )
            return out

        if self.cp_size > 1 and self.cp_mode == "native":
            return self._forward_dense_cp_native(
                x,
                cos,
                sin,
                position_ids,
                index_share_state=index_share_state,
            )

        cp_restore = self.cp_size > 1
        if cp_restore:
            x, position_ids, attention_mask = self._gather_cp_inputs(
                x, position_ids, attention_mask
            )
            cos, sin = self._gather_dense_cp_rotary(cos, sin, full_seq=x.shape[1])

        out = self._forward_dense_full(
            x, cos, sin, position_ids, index_share_state=index_share_state
        )
        if cp_restore:
            out = contiguous_slice_for_cp(out, self.cp_rank, self.cp_size, seq_dim=1)
        return out

    def _project_cp_inputs(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Project rank-local hidden rows before any CP collective."""
        batch, local_seq, _ = x.shape
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).view(
            batch, local_seq, self.num_heads, self.qk_head_dim
        )
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        local_cos, local_sin = _rotary_embeddings_from_cache(
            cos,
            sin,
            position_ids,
            device=x.device,
            dtype=x.dtype,
            dim=self.qk_rope_head_dim,
        )
        q_pe = apply_rotary_pos_emb(q_pe, local_cos, local_sin, unsqueeze_dim=2)
        k_up_weight, v_up_weight = self._split_kv_b_weights()
        q_nope = torch.einsum("bshd,hdr->bshr", q_nope, k_up_weight)
        query = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1).contiguous()

        kv_latent, k_pe = torch.split(
            self.kv_a_proj_with_mqa(x),
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = apply_rotary_pos_emb(
            k_pe.unsqueeze(2), local_cos, local_sin, unsqueeze_dim=2
        ).squeeze(2)
        kv = torch.cat([kv_latent, k_pe], dim=-1).transpose(0, 1).contiguous()

        q_indexer = k_indexer = weights_indexer = None
        if not self.skip_topk:
            assert self.indexer is not None
            q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                x.detach(), q_resid.detach(), local_cos, local_sin, position_ids
            )
        return query, kv, v_up_weight, q_indexer, k_indexer, weights_indexer

    def _gather_projected_cp(
        self,
        tensor: torch.Tensor,
        kv_reorder: torch.Tensor,
        *,
        contiguous: bool = False,
    ) -> torch.Tensor:
        if not contiguous:
            parts = _all_gather_cp(
                tensor, cp_size=self.cp_size, cp_group=self.cp_group
            )
            rank_major = torch.cat(parts, dim=0)
            return rank_major.index_select(0, kv_reorder)
        local_positions = contiguous_position_ids_for_cp(
            tensor.shape[0] * self.cp_size,
            self.cp_rank,
            self.cp_size,
            tensor.device,
        )
        parts = [
            source
            for _rank, source, _positions in iter_cp_sources(
                tensor,
                local_positions,
                cp_rank=self.cp_rank,
                cp_size=self.cp_size,
                cp_group=self.cp_group,
            )
        ]
        rank_major = torch.cat(parts, dim=0)
        return rank_major.index_select(0, kv_reorder)

    def _run_cp_sparse_segment(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        q_indexer: torch.Tensor | None,
        k_indexer: torch.Tensor | None,
        weights_indexer: torch.Tensor | None,
        mask: torch.Tensor,
        *,
        index_share_state: DSAIndexShareState | None,
        index_share_cache_key: Hashable | None,
    ) -> torch.Tensor:
        batch = query.shape[1]
        topk_indices: torch.Tensor | None = None
        if self.skip_topk:
            if index_share_state is None:
                raise AssertionError(
                    "DSA IndexShare shared layers require a per-forward DSAIndexShareState."
                )
            topk_indices = index_share_state.get_topk(
                self.layer_number,
                self.index_share_source_layer,
                sequence_key=index_share_cache_key,
                device=query.device,
            )
        else:
            assert q_indexer is not None and k_indexer is not None
            assert weights_indexer is not None
            _scores, topk_indices = _index_scores_and_topk(
                q_indexer,
                k_indexer,
                weights_indexer,
                mask=mask,
                topk=self.index_topk,
                scale=self.indexer_softmax_scale,
            )
            if self.index_share_enabled and index_share_state is not None:
                index_share_state.save_topk(
                    self.layer_number,
                    topk_indices,
                    sequence_key=index_share_cache_key,
                )

        flat_idxs, flat_tlen = _dsa_kernels.build_flat_topk_idxs(
            topk_indices,
            batch_size=batch,
            seqlen_kv=kv.shape[0],
            compact=True,
        )
        out = _dsa_kernels.dsa_sparse_attn(
            query,
            kv,
            self.attn_sink.float(),
            flat_idxs,
            self.softmax_scale,
            topk_length=flat_tlen,
            value_dim=self.kv_lora_rank,
        )
        if self.training and torch.is_grad_enabled() and not self.skip_topk:
            indexer_loss = _cp_indexer_loss(
                q_indexer,
                k_indexer,
                weights_indexer,
                topk_indices,
                query.detach(),
                kv.detach(),
                mask=mask,
                softmax_scale=self.softmax_scale,
                loss_coeff=self.indexer_loss_coeff,
                sparse_loss=self.indexer_use_sparse_loss,
                calculate_per_token_loss=self.calculate_per_token_loss,
            )
            out = DSAIndexerLossAutoScaler.apply(out, indexer_loss)
        return out

    def _project_cp_output(
        self, out: torch.Tensor, v_up_weight: torch.Tensor
    ) -> torch.Tensor:
        local_seq, batch = out.shape[:2]
        out = out.view(local_seq, batch, self.num_heads, self.kv_lora_rank)
        out = out.permute(1, 0, 2, 3).contiguous()
        out = torch.einsum("bshr,hvr->bshv", out, v_up_weight)
        out = out.reshape(batch, local_seq, self.num_heads * self.v_head_dim)
        return self.o_proj(out)

    def _forward_dense_cp_native(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        index_share_state: DSAIndexShareState | None,
    ) -> torch.Tensor:
        query_pos, kv_reorder = _dense_cp_layout(
            local_seq=x.shape[1],
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
            device=x.device,
        )
        query, kv_local, v_up_weight, q_idx, k_idx_local, idx_weights = (
            self._project_cp_inputs(x, cos, sin, position_ids)
        )
        kv = self._gather_projected_cp(kv_local, kv_reorder)
        k_idx = (
            self._gather_projected_cp(k_idx_local, kv_reorder)
            if k_idx_local is not None
            else None
        )
        if kv.is_cuda and not self.skip_topk:
            kv, k_idx = _pad_cp_projected_kv(kv, k_idx)
        mask = _build_cp_causal_mask(
            query_pos,
            torch.arange(kv.shape[0], device=x.device),
        )
        out = self._run_cp_sparse_segment(
            query,
            kv,
            q_idx,
            k_idx,
            idx_weights,
            mask,
            index_share_state=index_share_state,
            index_share_cache_key=None,
        )
        return self._project_cp_output(out, v_up_weight)

    def _forward_packed_cp_native(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        packed_seq_params,
        *,
        index_share_state: DSAIndexShareState | None,
    ) -> torch.Tensor:
        if x.shape[0] != 1:
            raise RuntimeError(
                "GLM5 THD+CP DSA expects batch-collapsed input [1, total, hidden]."
            )
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params, x.device)
        cp_layout = getattr(packed_seq_params, "cp_layout", "contiguous")
        if cp_layout != "contiguous":
            raise ValueError(
                "GLM5 THD+CP DSA requires contiguous allgather-CP layout, "
                f"got {cp_layout!r}."
            )
        query_pos, kv_reorder = _packed_cp_layout(
            cu_seqlens,
            cp_size=self.cp_size,
            cp_rank=self.cp_rank,
            device=x.device,
        )
        if query_pos.numel() != x.shape[1]:
            raise RuntimeError(
                "GLM5 THD+CP local token count does not match packed CP layout: "
                f"x={x.shape[1]}, layout={cp_layout!r}, expected={query_pos.numel()}."
            )
        query, kv_local, v_up_weight, q_idx, k_idx_local, idx_weights = (
            self._project_cp_inputs(x, cos, sin, position_ids)
        )
        kv = self._gather_projected_cp(kv_local, kv_reorder, contiguous=True)
        k_idx = (
            self._gather_projected_cp(k_idx_local, kv_reorder, contiguous=True)
            if k_idx_local is not None
            else None
        )
        if kv.is_cuda and not self.skip_topk:
            kv, k_idx = _pad_cp_projected_kv(kv, k_idx)
        key_pos = torch.arange(kv.shape[0], device=x.device, dtype=torch.long)
        mask = _build_cp_causal_mask(
            query_pos,
            key_pos,
            cu_seqlens=cu_seqlens,
        )
        out = self._run_cp_sparse_segment(
            query,
            kv,
            q_idx,
            k_idx,
            idx_weights,
            mask,
            index_share_state=index_share_state,
            index_share_cache_key=None,
        )
        return self._project_cp_output(out, v_up_weight)

    def _forward_packed_full(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        packed_seq_params,
        *,
        index_share_state: DSAIndexShareState | None,
    ) -> torch.Tensor:
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params, x.device)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] != x.shape[1]:
            raise ValueError(
                "GLM5 packed DynamicSparseAttention position_ids must cover the reconstructed packed tokens, "
                f"got {tuple(position_ids.shape)} for packed length {x.shape[1]}."
            )
        pieces = []
        for idx in range(int(cu_seqlens.numel()) - 1):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            if end <= start:
                continue
            seg_cos, seg_sin = self._slice_rotary_cache(cos, sin, start, end)
            pieces.append(
                self._forward_dense_full(
                    x[:, start:end, :],
                    seg_cos,
                    seg_sin,
                    position_ids[:, start:end],
                    index_share_state=index_share_state,
                    index_share_cache_key=idx,
                )
            )
        if pieces:
            return torch.cat(pieces, dim=1)
        return x.new_empty(x.shape[0], 0, self.o_proj.out_features)

    def _forward_dense_full(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        index_share_state: DSAIndexShareState | None = None,
        index_share_cache_key: Hashable | None = None,
    ) -> torch.Tensor:

        batch, seq_len, _ = x.shape
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).view(
            batch, seq_len, self.num_heads, self.qk_head_dim
        )
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        cos, sin = _rotary_embeddings_from_cache(
            cos,
            sin,
            position_ids,
            device=x.device,
            dtype=x.dtype,
            dim=self.qk_rope_head_dim,
        )

        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        k_up_weight, v_up_weight = self._split_kv_b_weights()
        q_nope = torch.einsum("bshd,hdr->bshr", q_nope, k_up_weight)
        query_states = torch.cat([q_nope, q_pe], dim=-1).transpose(0, 1).contiguous()

        kv_latent, k_pe = torch.split(
            self.kv_a_proj_with_mqa(x),
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_pe = apply_rotary_pos_emb(
            k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2
        ).squeeze(2)
        kv_full = torch.cat([kv_latent, k_pe], dim=-1).transpose(0, 1).contiguous()

        topk_indices: torch.Tensor | None = None
        q_indexer = k_indexer = weights_indexer = None
        if self.skip_topk:
            if index_share_state is None:
                raise AssertionError(
                    "DSA IndexShare shared layers require a per-forward "
                    "DSAIndexShareState."
                )
            topk_indices = index_share_state.get_topk(
                self.layer_number,
                self.index_share_source_layer,
                sequence_key=index_share_cache_key,
                device=x.device,
            )
        else:
            assert self.indexer is not None
            q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                x.detach(), q_resid.detach(), cos, sin, position_ids
            )
        effective_indexer_topk = min(self.index_topk, seq_len)

        if self.training and torch.is_grad_enabled() and not self.skip_topk:
            window_idxs = torch.empty(
                batch, seq_len, 0, device=x.device, dtype=torch.int32
            )
            assert (
                q_indexer is not None
                and k_indexer is not None
                and weights_indexer is not None
            )
            if self.index_share_enabled:
                out, indexer_loss, topk_indices = _fused_indexer_sparse_attn_with_topk(
                    query_states,
                    kv_full,
                    self.attn_sink.float(),
                    window_idxs,
                    q_indexer,
                    k_indexer,
                    weights_indexer,
                    self.index_topk,
                    1,
                    self.softmax_scale,
                    self.indexer_softmax_scale,
                    self.indexer_loss_coeff,
                    sparse_loss=self.indexer_use_sparse_loss,
                    kv_offset=0,
                    calculate_per_token_loss=self.calculate_per_token_loss,
                    value_dim=self.kv_lora_rank,
                )
                if index_share_state is not None:
                    index_share_state.save_topk(
                        self.layer_number,
                        topk_indices,
                        sequence_key=index_share_cache_key,
                    )
            else:
                out, indexer_loss = _fused_indexer_sparse_attn(
                    query_states,
                    kv_full,
                    self.attn_sink.float(),
                    window_idxs,
                    q_indexer,
                    k_indexer,
                    weights_indexer,
                    self.index_topk,
                    1,
                    self.softmax_scale,
                    self.indexer_softmax_scale,
                    self.indexer_loss_coeff,
                    sparse_loss=self.indexer_use_sparse_loss,
                    kv_offset=0,
                    calculate_per_token_loss=self.calculate_per_token_loss,
                    value_dim=self.kv_lora_rank,
                )
            if self.indexer_loss_coeff > 0:
                out = DSAIndexerLossAutoScaler.apply(out, indexer_loss)
        else:
            if topk_indices is None:
                assert (
                    q_indexer is not None
                    and k_indexer is not None
                    and weights_indexer is not None
                )
                topk_indices, _ = _dsa_kernels.indexer_topk(
                    q_indexer,
                    k_indexer,
                    weights_indexer,
                    effective_indexer_topk,
                    1,
                    indexer_softmax_scale=self.indexer_softmax_scale,
                )
                if self.index_share_enabled and index_share_state is not None:
                    index_share_state.save_topk(
                        self.layer_number,
                        topk_indices,
                        sequence_key=index_share_cache_key,
                    )
            flat_idxs, flat_tlen = _dsa_kernels.build_flat_topk_idxs(
                topk_indices, batch_size=batch, seqlen_kv=seq_len, compact=True
            )
            out = _dsa_kernels.dsa_sparse_attn(
                query_states,
                kv_full,
                self.attn_sink.float(),
                flat_idxs,
                self.softmax_scale,
                topk_length=flat_tlen,
                value_dim=self.kv_lora_rank,
            )

        out = out.view(seq_len, batch, self.num_heads, self.kv_lora_rank)
        out = out.permute(1, 0, 2, 3).contiguous()
        out = torch.einsum("bshr,hvr->bshv", out, v_up_weight)
        out = out.reshape(batch, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(out)

    def _split_kv_b_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        kv_b = self.kv_b_proj.weight.view(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        return (
            kv_b[:, : self.qk_nope_head_dim, :],
            kv_b[:, self.qk_nope_head_dim :, :],
        )

    def _gather_cp_inputs(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        local_batch, local_seq = x.shape[:2]
        x_parts = _all_gather_cp(x, cp_size=self.cp_size, cp_group=self.cp_group)
        full_x = torch.cat(x_parts, dim=1)
        full_seq = full_x.shape[1]

        full_position_ids = self._full_cp_position_ids(
            position_ids,
            batch=local_batch,
            local_seq=local_seq,
            full_seq=full_seq,
            device=x.device,
        )
        if attention_mask is not None:
            expected = (full_seq, full_seq)
            if tuple(attention_mask.shape[-2:]) != expected:
                raise NotImplementedError(
                    "GLM5 DynamicSparseAttention CP attention_mask must already cover the reconstructed "
                    f"full sequence {expected}, got {tuple(attention_mask.shape)}."
                )
        return full_x, full_position_ids, attention_mask

    def _gather_dense_cp_rotary(
        self, cos: torch.Tensor, sin: torch.Tensor, *, full_seq: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cos.dim() != 3 or sin.dim() != 3:
            return cos, sin
        if cos.shape[1] == full_seq and sin.shape[1] == full_seq:
            return cos, sin
        expected_local_seq = full_seq // self.cp_size
        if cos.shape[1] != expected_local_seq or sin.shape[1] != expected_local_seq:
            raise ValueError(
                "GLM5 DynamicSparseAttention CP rotary caches must be either local or full sequence "
                f"length, got cos={tuple(cos.shape)} sin={tuple(sin.shape)} for "
                f"local_seq={expected_local_seq}, full_seq={full_seq}."
            )
        cos_parts = _all_gather_cp(cos, cp_size=self.cp_size, cp_group=self.cp_group)
        sin_parts = _all_gather_cp(sin, cp_size=self.cp_size, cp_group=self.cp_group)
        return (
            torch.cat(cos_parts, dim=1),
            torch.cat(sin_parts, dim=1),
        )

    def _full_cp_position_ids(
        self,
        position_ids: torch.Tensor,
        *,
        batch: int,
        local_seq: int,
        full_seq: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids.dim() == 3:
            position_ids = position_ids[0]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        if position_ids.shape[-1] == full_seq:
            return position_ids.to(device=device, dtype=torch.long)
        if position_ids.shape[-1] != local_seq:
            raise ValueError(
                "GLM5 DynamicSparseAttention CP position_ids must be either local or full sequence length, "
                f"got {tuple(position_ids.shape)} for local_seq={local_seq}, full_seq={full_seq}."
            )

        pos_parts = _all_gather_cp(
            position_ids.to(device=device, dtype=torch.long),
            cp_size=self.cp_size,
            cp_group=self.cp_group,
        )
        return torch.cat(pos_parts, dim=1)

    def _gather_packed_cp_inputs(
        self, x: torch.Tensor, position_ids: torch.Tensor, packed_seq_params
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_seq = x.shape[1]
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params, x.device)
        full_seq = int(cu_seqlens[-1].item())
        x_parts = _all_gather_cp(x, cp_size=self.cp_size, cp_group=self.cp_group)
        full_x = torch.cat(x_parts, dim=1)

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.to(device=x.device, dtype=torch.long)
        if position_ids.shape[-1] == full_seq:
            return full_x, position_ids
        if position_ids.shape[-1] != local_seq:
            raise ValueError(
                "GLM5 packed DynamicSparseAttention CP position_ids must be either local or full packed length, "
                f"got {tuple(position_ids.shape)} for local_seq={local_seq}, full_seq={full_seq}."
            )
        pos_parts = _all_gather_cp(
            position_ids, cp_size=self.cp_size, cp_group=self.cp_group
        )
        full_position_ids = torch.cat(pos_parts, dim=1)
        return full_x, full_position_ids

    def _gather_packed_cp_rotary(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        packed_seq_params,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cos.dim() != 3 or sin.dim() != 3:
            return cos, sin
        cu_seqlens = self._packed_cu_seqlens(packed_seq_params, device)
        full_seq = int(cu_seqlens[-1].item())
        if cos.shape[1] == full_seq and sin.shape[1] == full_seq:
            return cos, sin
        cos_parts = _all_gather_cp(cos, cp_size=self.cp_size, cp_group=self.cp_group)
        sin_parts = _all_gather_cp(sin, cp_size=self.cp_size, cp_group=self.cp_group)
        full_cos = torch.cat(cos_parts, dim=1)
        full_sin = torch.cat(sin_parts, dim=1)
        return full_cos, full_sin

    @staticmethod
    def _packed_cu_seqlens(packed_seq_params, device: torch.device) -> torch.Tensor:
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
        if cu_seqlens is None:
            cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q", None)
        if cu_seqlens is None:
            raise ValueError(
                "GLM5 packed DynamicSparseAttention requires packed cu_seqlens."
            )
        return cu_seqlens.to(device=device, dtype=torch.int32)

    @staticmethod
    def _slice_rotary_cache(
        cos: torch.Tensor, sin: torch.Tensor, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cos.dim() == 3 and sin.dim() == 3:
            return cos[:, start:end, :], sin[:, start:end, :]
        return cos, sin


__all__ = [
    "DSAIndexShareState",
    "DSAIndexer",
    "DSAIndexerLossAutoScaler",
    "DynamicSparseAttention",
    "RMSNorm",
    "apply_rotary_emb",
    "apply_rotary_pos_emb",
    "build_rope_cache",
    "build_rotary_embeddings",
    "rotate_activation",
    "rotate_half",
    "validate_dsa_index_share_pipeline_split",
]

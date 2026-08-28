# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn

from megatron.core.fp8_utils import get_fp8_disabled_context
from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    rotate_activation,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

#: Bit-exact determinism status for the eager CSA operations introduced here.
#: The operations use CUDA reductions and indexed accumulation, but bit-exact
#: repeatability has not been certified, so the conservative status is unknown.
CSA_OPERATION_DETERMINISM: dict[str, str] = {
    "unfused_sparse_attention": "unknown",
    "non_compressed_lse": "unknown",
    "compressor_pooling": "unknown",
}


class CompressorBuilder(Protocol):
    """Typed builder for a CSA compressor with its submodules already bound."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        name: str | None = None,
    ) -> "Compressor": ...


class CSAIndexerBuilder(Protocol):
    """Typed builder for a CSA indexer with its submodules already bound."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        compress_ratio: int,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        name: str | None = None,
    ) -> "CSAIndexer": ...


class CompressedSparseAttentionBuilder(Protocol):
    """Typed builder for CSA core attention with its submodules already bound."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection | None = None,
        rotary_pos_emb: nn.Module | None = None,
        compress_ratio: int = 0,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> "CompressedSparseAttention": ...


# ---------------------------------------------------------------------------
# Helper functions for index computation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(window_size: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Compute sliding-window indices for a single sequence (cached).

    Returns:
        indices: [seqlen, window_size] int tensor, -1 for invalid positions.
    """
    base = torch.arange(seqlen, device=device_str).unsqueeze(1)
    offsets = torch.arange(window_size, device=device_str)
    matrix = (base - window_size + 1).clamp(min=0) + offsets
    matrix = torch.where(matrix > base, -1, matrix)
    return matrix


def get_window_topk_idxs(
    window_size: int, batch_size: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Sliding-window indices [batch, seqlen, window_size]."""
    matrix = _get_window_topk_idxs_cached(window_size, seqlen, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


@lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> torch.Tensor:
    """Compute all-compressed-positions indices for a single sequence (cached).

    Returns:
        indices: [seqlen, seqlen // ratio] int tensor, -1 for future positions.
    """
    n_compressed = seqlen // ratio
    matrix = torch.arange(n_compressed, device=device_str).repeat(seqlen, 1)
    mask = matrix >= torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix


def get_compress_topk_idxs(
    ratio: int, batch_size: int, seqlen: int, offset: int, device: torch.device
) -> torch.Tensor:
    """All-compressed-position indices [batch, seqlen, seqlen // ratio]."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, str(device))
    return matrix.unsqueeze(0).expand(batch_size, -1, -1)


@lru_cache(maxsize=8)
def _get_compress_causal_mask_cached(
    ratio: int, seqlen: int, n_compressed: int, device_str: str
) -> torch.Tensor:
    """Return the additive causal mask for compressed positions (cached)."""
    compressed_positions = torch.arange(n_compressed, device=device_str).unsqueeze(0)
    valid_counts = torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio
    return torch.where(compressed_positions >= valid_counts, float("-inf"), 0.0)


@lru_cache(maxsize=8)
def _get_compress_valid_counts_cached(ratio: int, seqlen: int, device_str: str) -> torch.Tensor:
    """Return the number of causally valid compressed positions per query (cached)."""
    return torch.arange(1, seqlen + 1, device=device_str).unsqueeze(1) // ratio


# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------


def _apply_rope(
    x: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module: RotaryEmbedding,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    cp_group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Apply RoPE to the last ``qk_pos_emb_head_dim`` dims, leaving the rest unchanged.

    Accepts both 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
    inputs.  When the input is 3-D a temporary head dimension is inserted for
    ``apply_rotary_pos_emb`` and removed before returning.
    """
    if ratio == 1:
        total_seq_len = rotary_seq_len
    else:
        total_seq_len = rotary_seq_len * ratio
    # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
    # concentration factor (mscale) is NOT part of the DSv4 model contract --
    # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0
    # regardless of which rotary class is in use.
    mscale = 1.0
    rotary_pos_cos = None
    rotary_pos_sin = None
    if config.apply_rope_fusion:
        # ``mscale=1.0`` keeps the cached cos/sin free of yarn's
        # concentration factor so the fused kernel sees the same
        # rotation as the unfused split-rotate path (DSv4 "pure
        # rotation" contract).
        rotary_pos_cos, rotary_pos_sin = rotary_pos_emb_module.get_cached_cos_sin(
            total_seq_len, dtype=x.dtype, packed_seq=False, mscale=mscale
        )
        rotary_pos_emb = None
        assert (
            fused_mla_rope_inplace is not None
        ), "Fused MLA RoPE apply is not imported successfully"
    else:
        # Compressed-attention callers instantiate ``YarnRotaryEmbedding``
        # whenever ``compress_ratio > 1`` (regardless of ``config.rope_type``);
        # its ``forward`` returns ``(emb, mscale)``. Base ``RotaryEmbedding``
        # returns a single tensor. Unpack either form uniformly; the
        # caller-side ``mscale=1.0`` keeps the yarn concentration factor
        # out of the rotation.
        result = rotary_pos_emb_module(total_seq_len, packed_seq=False)
        if isinstance(result, tuple):
            rotary_pos_emb = result[0]
        else:
            rotary_pos_emb = result
    if rotary_pos_emb is not None and ratio > 1:
        rotary_pos_emb = rotary_pos_emb[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_cos is not None and ratio > 1:
        rotary_pos_cos = rotary_pos_cos[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_sin is not None and ratio > 1:
        rotary_pos_sin = rotary_pos_sin[:total_seq_len:ratio][:rotary_seq_len]

    squeeze_head = x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)
    if config.apply_rope_fusion:
        out = fused_mla_rope_inplace(
            x,
            rotary_pos_cos,
            rotary_pos_sin,
            nope_dim,
            pos_dim,
            None,
            cp_group.rank(),
            cp_group.size(),
            remove_interleaving=True,
        )
    else:
        x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=cp_group,
            mla_rotary_interleaved=True,
            mla_output_remove_interleaving=True,
        )
        out = torch.cat([x_nope, x_pe], dim=-1)
    if squeeze_head:
        out = out.squeeze(-2)
    return out


# ---------------------------------------------------------------------------
# Sparse attention kernel (unfused, differentiable)
# ---------------------------------------------------------------------------


def unfused_compressed_sparse_attn(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Differentiable sparse attention with MQA and attention sink.

    Determinism:
        Unknown. Bit-exact forward and backward repeatability has not been certified.

    Args:
        query:        [sq, b, np, hn]   multi-head query.
        kv_full:      [n_kv, b, hn]     single-head KV (original + compressed).
        attn_sink:    [np]              per-head learnable bias.
        topk_indices: [b, sq, topk]     indices into kv_full (int32, -1 = invalid).
        softmax_scale: float

    Returns:
        output:       [sq, b, np * hn]
    """
    sq, b, np_, hn = query.size()
    if attn_sink.ndim != 1 or attn_sink.numel() != np_:
        raise ValueError(
            f"attn_sink must contain one value per query head ({np_}), "
            f"got shape {tuple(attn_sink.shape)}."
        )

    # --- Gather KV at topk positions ---
    # Flatten batch and KV position before gathering. Gathering from a logical
    # [b, sq, n_kv, hn] expanded view makes gather backward allocate that entire
    # dense shape before reducing the stride-0 query dimension.
    n_kv = kv_full.size(0)
    topk = topk_indices.size(-1)
    kv_flat = kv_full.permute(1, 0, 2).reshape(b * n_kv, hn)
    batch_offsets = (torch.arange(b, device=kv_full.device, dtype=torch.int64) * n_kv).view(b, 1, 1)
    safe_indices = topk_indices.clamp(min=0).to(dtype=torch.int64) + batch_offsets
    kv_gathered = kv_flat.index_select(0, safe_indices.reshape(-1)).view(b, sq, topk, hn)

    # --- Attention scores ---
    # query: [sq, b, np, hn] -> [b, np, sq, hn]
    q = query.permute(1, 2, 0, 3).float()
    kv_g = kv_gathered.float()  # [b, sq, topk, hn]

    # [b, np, sq, topk]
    scores = torch.einsum("bnsh,bskh->bnsk", q, kv_g) * softmax_scale

    # Mask invalid
    invalid_mask = (topk_indices < 0).unsqueeze(1)  # [b, 1, sq, topk]
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    # --- Softmax with attention sink ---
    sink = attn_sink.view(1, np_, 1, 1).float()
    scores_max = scores.max(dim=-1, keepdim=True).values  # [b, np, sq, 1]
    scores_max = torch.max(scores_max, sink)

    exp_scores = torch.exp(scores - scores_max)  # [b, np, sq, topk]
    exp_sink = torch.exp(sink - scores_max)  # [1, np, 1, 1]

    sum_exp = exp_scores.sum(dim=-1, keepdim=True) + exp_sink
    attn_weights = exp_scores / sum_exp  # [b, np, sq, topk]

    # --- Weighted sum ---
    output = torch.einsum("bnsk,bskh->bnsh", attn_weights, kv_g)
    output = output.to(query.dtype)

    # [b, np, sq, hn] -> [sq, b, np, hn] -> [sq, b, np * hn]
    output = output.permute(2, 0, 1, 3).contiguous()
    output = output.reshape(sq, b, np_ * hn)
    return output


@torch.no_grad()
def _compute_unfused_csa_non_compressed_lse(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    window_indices: torch.Tensor,
    softmax_scale: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Return the detached sliding-window-plus-sink log mass for the CSA teacher.

    Determinism:
        Unknown. Bit-exact CUDA reduction behavior has not been certified.

    Args:
        query: Query tensor in ``[sq, batch, heads, head_dim]`` layout.
        kv_full: Original (non-compressed) KV in ``[sk, batch, head_dim]`` layout.
        attn_sink: Per-head sink logits in ``[heads]`` layout.
        window_indices: Local per-batch window indices in ``[batch, sq, window]`` layout.
        softmax_scale: Scale applied to query-key logits.
        chunk_size: Maximum number of flattened query rows processed at once.

    Returns:
        Detached FP32 log-sum-exp values in ``[batch, heads, sq]`` layout.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if query.ndim != 4:
        raise ValueError(f"query must have shape [sq, batch, heads, dim], got {query.shape}")
    if attn_sink.ndim != 1:
        raise ValueError(f"attn_sink must be 1D, got shape {tuple(attn_sink.shape)}")

    seqlen_q, batch_size, num_heads, head_dim = query.shape
    if kv_full.ndim != 3 or kv_full.shape[1:] != (batch_size, head_dim):
        raise ValueError(
            "non-compressed KV must have shape "
            f"[sk, {batch_size}, {head_dim}], got {tuple(kv_full.shape)}"
        )
    if window_indices.ndim != 3 or window_indices.shape[:2] != (batch_size, seqlen_q):
        raise ValueError(
            "window_indices must have shape "
            f"[{batch_size}, {seqlen_q}, window], got {tuple(window_indices.shape)}"
        )
    if attn_sink.numel() != num_heads:
        raise ValueError(f"attn_sink must contain {num_heads} values, got {attn_sink.numel()}")
    if not (query.device == kv_full.device == attn_sink.device == window_indices.device):
        raise ValueError("query, kv_full, attn_sink, and window_indices must share a device")

    n_kv = kv_full.shape[0]
    q_flat = query.detach().permute(1, 0, 2, 3).reshape(-1, num_heads, head_dim)
    kv_flat = kv_full.detach().permute(1, 0, 2).reshape(-1, head_dim)
    batch_offsets = (
        torch.arange(batch_size, device=window_indices.device, dtype=torch.int64) * n_kv
    ).view(batch_size, 1, 1)
    window_indices_i64 = window_indices.to(dtype=torch.int64)
    global_indices = torch.where(
        window_indices_i64 >= 0, window_indices_i64 + batch_offsets, window_indices_i64
    ).reshape(batch_size * seqlen_q, -1)

    sink = attn_sink.detach().to(dtype=torch.float32).view(1, num_heads)
    lse_chunks = []
    for start in range(0, q_flat.shape[0], chunk_size):
        end = min(start + chunk_size, q_flat.shape[0])
        indices = global_indices[start:end]
        gathered_kv = kv_flat.index_select(0, indices.clamp(min=0).reshape(-1)).reshape(
            end - start, indices.shape[-1], head_dim
        )
        window_logits = torch.einsum("rhd,rkd->rhk", q_flat[start:end].float(), gathered_kv.float())
        window_logits = (window_logits * softmax_scale).masked_fill(
            (indices < 0).unsqueeze(1), float("-inf")
        )
        lse_chunks.append(torch.logaddexp(torch.logsumexp(window_logits, dim=-1), sink))

    if lse_chunks:
        lse_flat = torch.cat(lse_chunks, dim=0)
    else:
        lse_flat = torch.empty((0, num_heads), dtype=torch.float32, device=query.device)
    return lse_flat.reshape(batch_size, seqlen_q, num_heads).permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorSubmodules:
    """Submodule specs for CSA and HCA Compressor."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


def _pool_compressor_values(
    kv: torch.Tensor, score: torch.Tensor, output_dtype: torch.dtype
) -> torch.Tensor:
    """Pool compressor values with FP32 weights, products, and reduction."""
    weights = torch.softmax(score, dim=1, dtype=torch.float32)
    return (kv.float() * weights).sum(dim=1).to(output_dtype)


class Compressor(MegatronModule):
    """Gated pooling compressor for CSA and HCA sparse attention.

    Compresses a sequence of tokens into a shorter sequence by pooling groups of
    ``compress_ratio`` tokens using learned gated weights.

    For ``compress_ratio == 4``, overlapping compression is used (``coff = 2``).
    For ``compress_ratio == 128``, non-overlapping compression is used (``coff = 1``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its parent module
        """
        super().__init__(config=config)

        if pg_collection is None:
            raise ValueError("Compressor requires an explicit ProcessGroupCollection")
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        with get_fp8_disabled_context(config, is_init=True):
            self.linear_wkv = build_module(
                submodules.linear_wkv,
                config.hidden_size,
                proj_out_dim,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + ".linear_wkv") if name is not None else None,
            )

            self.linear_wgate = build_module(
                submodules.linear_wgate,
                config.hidden_size,
                proj_out_dim,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + ".linear_wgate") if name is not None else None,
            )

        # keep to high precision (FP32 in the reference DeepSeek V4 checkpoint)
        _ape = torch.empty(
            compress_ratio, proj_out_dim, device=torch.cuda.current_device(), dtype=torch.float32
        )
        config.init_method(_ape)
        self.ape = mark_keep_in_fp32(nn.Parameter(_ape))

        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.norm = build_module(
            submodules.norm, config=norm_config, hidden_size=head_dim, eps=config.layernorm_epsilon
        )

    def backward_dw(self):
        """Compute deferred weight gradients for the compressor projections."""
        self.linear_wkv.backward_dw()
        self.linear_wgate.backward_dw()

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        """Apply overlapping window transform for 4x compression.

        Input shape:  [n_groups, ratio, b, coff * head_dim]
        Output shape: [n_groups, 2 * ratio, b, head_dim]
        """
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, :d]
        return new_tensor

    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project compressor values and gates outside any enclosing FP8 context."""
        with get_fp8_disabled_context(self.config):
            kv, _ = self.linear_wkv(x)
            score, _ = self.linear_wgate(x)
        return kv, score

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compress hidden states into shorter KV sequence.

        Determinism:
            Unknown. Bit-exact CUDA pooling and gradient reductions have not been certified.

        Args:
            x: [sq, b, hidden_size]

        Returns:
            compressed_kv [sq // ratio, b, head_dim] or None if too short.
        """
        nvtx_range_push("compressor")

        sq, b, _ = x.size()
        ratio = self.compress_ratio

        if sq < ratio:
            nvtx_range_pop("compressor")
            return None

        kv, score = self._project(x)  # [sq, b, coff * head_dim]

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]

        n_compressed = cutoff // ratio

        # Reshape: [n_compressed, ratio, b, coff * head_dim]
        kv = kv.view(n_compressed, ratio, b, -1)
        score = score.view(n_compressed, ratio, b, -1)

        # APE: [ratio, coff * head_dim] -> [1, ratio, 1, coff * head_dim]
        score = score + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        kv = _pool_compressor_values(kv, score, x.dtype)  # [n_compressed, b, head_dim]

        kv = self.norm(kv)

        kv = _apply_rope(
            kv,
            self.head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            n_compressed,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        if self.rotate:
            kv = rotate_activation(kv)

        nvtx_range_pop("compressor")
        return kv  # [n_compressed, b, head_dim]


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


@dataclass
class CSAIndexerSubmodules:
    """Submodule specs for CSAIndexer."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: CompressorBuilder | None = None


class CSAIndexer(MegatronModule):
    """Learned top-k retrieval over compressed positions for CSA sparse attention.

    Computes index scores to select the most relevant compressed KV positions for each
    query.  Reuses the scoring logic from ``DSAIndexer`` (einsum -> relu -> weight -> sum
    -> topk) and ``rotate_activation`` (Hadamard transform) from ``dsa.py``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSAIndexerSubmodules,
        compress_ratio: int,
        rotary_pos_emb: nn.Module = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its parent module
        """
        super().__init__(config=config)

        if pg_collection is None:
            raise ValueError("CSAIndexer requires an explicit ProcessGroupCollection")
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        )

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        # Q projection
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
            name=(name + ".linear_wq_b") if name is not None else None,
        )

        # The reference DeepSeek V4 checkpoint keeps this projection in BF16.
        with get_fp8_disabled_context(config, is_init=True):
            self.linear_weights_proj = build_module(
                submodules.linear_weights_proj,
                self.hidden_size,
                self.index_n_heads,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + ".linear_weights_proj") if name is not None else None,
            )

        # Own compressor (smaller head_dim, with Hadamard rotation)
        if submodules.compressor is None:
            raise ValueError("CSAIndexer requires a compressor builder")
        self.compressor = submodules.compressor(
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
            pg_collection=pg_collection,
            name=(name + ".compressor") if name is not None else None,
        )

    def backward_dw(self):
        """Compute deferred weight gradients for the indexer projections."""
        self.linear_wq_b.backward_dw()
        self.linear_weights_proj.backward_dw()
        self.compressor.backward_dw()

    def _project_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Project indexer weights outside any enclosing FP8 context."""
        with get_fp8_disabled_context(self.config):
            weights, _ = self.linear_weights_proj(x)
        return weights

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, compressed K, and weights before top-k selection."""
        nvtx_range_push("indexer_before_topk")

        sq, bsz, _ = x.size()

        # Q path
        q, _ = self.linear_wq_b(qr)  # [sq, b, n_heads * head_dim]
        q = q.reshape(sq, bsz, self.index_n_heads, self.index_head_dim)
        q = _apply_rope(
            q,
            self.index_head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            sq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        q = rotate_activation(q)

        # K path: own compressor
        k = self.compressor(x)  # [sq//ratio, b, index_head_dim]

        weights = self._project_weights(x)  # [sq, b, n_heads]
        weights = weights * (self.index_n_heads**-0.5)

        nvtx_range_pop("indexer_before_topk")
        return q, k, weights

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (index_scores, topk_indices)."""
        nvtx_range_push("indexer")
        q, k, weights = self.forward_before_topk(x, qr)
        nvtx_range_push("indexer_qk_topk")
        effective_topk = min(self.index_topk, k.size(0))
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, effective_topk, mask)
        nvtx_range_pop("indexer_qk_topk")
        nvtx_range_pop("indexer")
        return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for CompressedSparseAttention."""

    compressor: CompressorBuilder | None = None
    indexer: CSAIndexerBuilder | None = None


class CompressedSparseAttention(MegatronModule):
    """Sparse core attention for CompressedSparseAttention.

    Combines sliding-window attention with compressed KV attention. The spec always
    provides compressor and indexer submodule specs; which are built depends on the
    ``compress_ratio`` passed by the caller:

    * ``ratio <= 1``: window-only (neither compressor nor indexer is built).
    * ``ratio > 1``: window + compressed KV via ``Compressor``.
    * ``ratio == 4`` and not ``config.csa_dense_mode``: additionally builds
      ``CSAIndexer`` for learned top-k retrieval over compressed positions. Otherwise,
      all causally valid compressed positions are attended.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module = None,
        compress_ratio: int = 0,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its parent module
        """
        super().__init__(config=config)

        if pg_collection is None:
            raise ValueError(
                "CompressedSparseAttention requires an explicit ProcessGroupCollection"
            )
        self.pg_collection = pg_collection

        tp_size = self.pg_collection.tp.size()
        if tp_size != 1:
            raise ValueError(
                "CompressedSparseAttention supports only tensor-parallel size 1 in the "
                f"native SBHD slice, got tp_size={tp_size}."
            )

        self.layer_number = layer_number + self.config.num_layers if is_mtp_layer else layer_number
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim

        self.num_attention_heads = config.num_attention_heads

        if softmax_scale is None:
            softmax_scale = config.v_head_dim**-0.5
        self.softmax_scale = softmax_scale

        # Learnable attention sink per head, kept in reference-checkpoint FP32.
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.num_attention_heads, dtype=torch.float32))
        )

        # Conditionally build Compressor (ratio > 1)
        if self.compress_ratio > 1 and submodules.compressor is not None:
            self.compressor = submodules.compressor(
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
                name=(name + ".compressor") if name is not None else None,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer (ratio == 4)
        if (
            self.compress_ratio == 4
            and not config.csa_dense_mode
            and submodules.indexer is not None
        ):
            self.indexer = submodules.indexer(
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
                name=(name + ".indexer") if name is not None else None,
            )
        else:
            self.indexer = None

    def backward_dw(self):
        """Compute deferred gradients for the optional compressor and indexer projections."""
        if self.compressor is not None:
            self.compressor.backward_dw()
        if self.indexer is not None:
            self.indexer.backward_dw()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor = None,
        qr: torch.Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """Forward pass for CompressedSparseAttention.

        Args:
            query:  [sq, b, np, v_head_dim]
            key:    [sq, b, 1, v_head_dim]  (single-head MQA; head dim squeezed internally)
            value:  unused (key == value in MQA)
            attention_mask: Must be None; causal masking is applied internally.
            x:      [sq, b, hidden_size]  original hidden states.
            qr:     [sq, b, q_lora_rank]  compressed query representation.

        Returns:
            output: [sq, b, np * v_head_dim]
        """
        if attention_mask is not None:
            raise ValueError(
                "CompressedSparseAttention supports only an implicit causal mask in the "
                "native SBHD slice; padding and document-boundary masks are not supported."
            )
        if query.ndim != 4:
            raise ValueError(
                "CompressedSparseAttention query must have shape [seq, batch, heads, dim], "
                f"got {tuple(query.shape)}."
            )
        if query.size(2) != self.num_attention_heads:
            raise ValueError(
                "CompressedSparseAttention query head count must match the unsharded "
                f"attention sink ({self.num_attention_heads}), got {query.size(2)}."
            )
        nvtx_range_push("compressed_sparse_attn")
        assert (
            packed_seq_params is None
        ), "Packed sequence not supported for CompressedSparseAttention"

        sq, b, np, hn = query.size()

        # --- Step 1: Prepare single-head KV (squeeze singleton head dim) ---
        kv = key.squeeze(-2)  # [sq, b, 1, v_head_dim] -> [sq, b, v_head_dim]

        # --- Step 2: Compression ---
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = self.compressor(x)  # [n_compressed, b, v_head_dim]
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed indices start after original positions

        # --- Step 3: Window indices ---
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)

        # --- Step 4: Compressed indices ---
        indexer_loss = None

        if self.compress_ratio > 1 and n_compressed > 0:
            nvtx_range_push("compressed_indices")
            if self.indexer is not None:
                x_det = x.detach()
                qr_det = qr.detach()

                causal_mask = (
                    _get_compress_causal_mask_cached(
                        self.compress_ratio, sq, n_compressed, str(x.device)
                    )
                    .unsqueeze(0)
                    .expand(b, -1, -1)
                )

                if self.training and torch.is_grad_enabled():
                    q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                        x_det, qr_det
                    )
                    indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
                    key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, np, -1)
                    weights_for_unfused = weights_indexer.float() * self.indexer.softmax_scale
                    non_compressed_lse = _compute_unfused_csa_non_compressed_lse(
                        query, kv, self.attn_sink, window_idxs, self.softmax_scale
                    )
                    # The native reference intentionally recomputes sliding-window
                    # logits in the final attention call below. The teacher needs its
                    # detached denominator before indexer top-k is available; sharing
                    # it without materializing the full window gather requires a larger
                    # data-flow change. TODO(#6404): its fused training backend avoids
                    # this duplicate, but the native fallback should share the gathered
                    # window KV/logits instead of retaining this correctness-first helper.
                    topk_indices_compressed, indexer_loss = FusedDSAIndexerLoss.apply(
                        q_indexer,
                        weights_for_unfused,
                        k_indexer,
                        query.detach(),
                        key_for_loss.detach(),
                        self.softmax_scale,
                        min(self.indexer.index_topk, n_compressed),
                        indexer_loss_coeff,
                        causal_mask,
                        self.config.dsa_indexer_use_sparse_loss,
                        self.indexer.pg_collection,
                        None,
                        None,
                        None,
                        None,
                        self.config.calculate_per_token_loss,
                        True,
                        non_compressed_lse,
                    )
                    if indexer_loss_coeff > 0:
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                        )
                else:
                    _, topk_indices_compressed = self.indexer(x_det, qr_det, mask=causal_mask)

                n_valid_per_pos = _get_compress_valid_counts_cached(
                    self.compress_ratio, sq, str(x.device)
                )
                valid = (topk_indices_compressed >= 0) & (topk_indices_compressed < n_valid_per_pos)
                compress_topk_idxs = torch.where(valid, topk_indices_compressed + offset, -1)
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio, b, sq, offset, query.device
                )

            topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
            nvtx_range_pop("compressed_indices")
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.int()

        # --- Step 5: Sparse attention ---
        nvtx_range_push("sparse_attn_kernel")
        output = unfused_compressed_sparse_attn(
            query, kv_full, self.attn_sink.float(), topk_idxs, self.softmax_scale
        )
        nvtx_range_pop("sparse_attn_kernel")

        # --- Step 6: Attach indexer loss ---
        if indexer_loss is not None and self.training and torch.is_grad_enabled():
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        nvtx_range_pop("compressed_sparse_attn")
        return output

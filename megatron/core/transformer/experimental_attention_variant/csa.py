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
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import async_gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    rotate_activation,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from .csa_utils import cp_utils, packed_layout
from .csa_utils.fused_sparse_attention import (
    build_flat_topk_idxs,
    csa_sparse_attn,
    fused_csa_indexer_sparse_attn,
    indexer_topk,
)
from .csa_utils.packed_sparse_attention import FusedCSAIndexerSparseAttnFromTopkFunc, batch_of_row
from .csa_utils.packed_sparse_attention import csa_sparse_attn as packed_csa_sparse_attn
from .csa_utils.packed_sparse_attention import (
    defer_reduce_scatter_wait,
    get_flash_mla_topk_alignment,
)
from .dsa import _compute_indexer_teacher_probabilities, _normalize_indexer_teacher_target

#: Bit-exact determinism status for the eager CSA operations introduced here.
#: The operations use CUDA reductions and indexed accumulation, but bit-exact
#: repeatability has not been certified, so the conservative status is unknown.
CSA_OPERATION_DETERMINISM: dict[str, str] = {
    "unfused_sparse_attention": "unknown",
    "non_compressed_lse": "unknown",
    "compressor_pooling": "unknown",
}


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


def _unfused_compressed_sparse_attn_sbhd(
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
def _compute_unfused_csa_non_compressed_lse_sbhd(
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


@torch.no_grad()
def _compute_unfused_csa_non_compressed_lse(
    query, kv_full, attn_sink, window_indices, softmax_scale, chunk_size=512
):
    """Compute the same detached window/sink log mass in SBHD or packed layout."""
    if query.ndim == 3:
        query, kv_full, window_indices = (
            query.unsqueeze(1),
            kv_full.unsqueeze(1),
            window_indices.unsqueeze(0),
        )
    return _compute_unfused_csa_non_compressed_lse_sbhd(
        query, kv_full, attn_sink, window_indices, softmax_scale, chunk_size
    )


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class CompressorInterface(Protocol):
    """Runtime interface exposed by a CSA compressor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        """Compress an input sequence."""
        ...

    def backward_dw(self) -> None:
        """Compute deferred weight gradients."""
        ...

    def forward_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_seqlen_q: int,
        compressed_group_ids: torch.Tensor,
        compressed_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compress explicitly grouped packed tokens."""
        ...


class CompressorBuilder(Protocol):
    """Builder protocol for CSA compressors."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: ProcessGroupCollection,
        name: str | None = None,
    ) -> CompressorInterface:
        """Build a CSA compressor."""
        ...


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
        rotary_pos_emb: nn.Module | None = None,
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

    def _overlap_transform_thd(
        self, tensor: torch.Tensor, is_first_in_seg: torch.Tensor, fill_value: float = 0
    ) -> torch.Tensor:
        """Batched overlapping window transform for THD packed layout.

        Like :meth:`_overlap_transform` but operates on the flat
        ``(total_comp, ratio, b, coff * head_dim)`` tensor from all segments
        at once. ``is_first_in_seg`` is a ``(total_comp,)`` bool mask that
        is ``True`` for each compressed entry that starts a new segment
        (i.e. has no predecessor group to pull from).

        Input shape:  [total_comp, ratio, b, coff * head_dim]
        Output shape: [total_comp, 2 * ratio, b, head_dim]
        """
        n, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        # Previous group's first-half data — shift by 1 along dim-0.
        prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
        # Zero-fill (or fill_value-fill) segment boundaries.
        prev_data[is_first_in_seg] = fill_value
        new_tensor[:, :ratio] = prev_data
        return new_tensor

    def forward_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_seqlen_q: int,
        compressed_group_ids: torch.Tensor,
        compressed_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Pool fixed-capacity complete groups and their overlapping predecessors.

        Input layout comes from prepare_cp_compressor_input. Metadata describes
        physical local groups, while position IDs remain document relative.
        Canonical ownership is assigned separately by the global compressed-row map.
        """
        total_comp = compressed_group_ids.shape[0]
        ratio = self.compress_ratio
        kv, score = self._project(x)
        kv = kv.reshape(total_comp, ratio, 1, -1)
        score = score.reshape(total_comp, ratio, 1, -1) + self.ape.view(1, ratio, 1, -1)
        if self.overlap:
            is_first = compressed_group_ids == 0
            kv = self._overlap_transform_thd(kv, is_first, fill_value=0)
            score = self._overlap_transform_thd(score, is_first, fill_value=float("-inf"))
        compressed = _pool_compressor_values(kv, score, x.dtype)
        compressed = apply_module(self.norm)(compressed)
        if self.config.apply_rope_fusion:
            cos, sin = self.rotary_pos_emb.get_cached_cos_sin(
                max_seqlen_q, dtype=compressed.dtype, packed_seq=True, mscale=1.0
            )
            compressed = fused_mla_rope_inplace(
                compressed,
                cos,
                sin,
                self.head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                cu_seqlens_q=cu_seqlens,
                remove_interleaving=True,
                position_ids=compressed_position_ids,
            )
        else:
            rope = self.rotary_pos_emb(max_seqlen_q, packed_seq=True)
            freqs = rope[0] if isinstance(rope, tuple) else rope
            compressed = _apply_unfused_rope(
                compressed,
                freqs.index_select(0, compressed_position_ids.long()),
                self.head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                self.config,
                None,
                self.pg_collection.cp,
            )
        return rotate_activation(compressed) if self.rotate else compressed


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


class CSAIndexerInterface(Protocol):
    """Runtime interface exposed by a CSA indexer."""

    index_topk: int
    index_n_heads: int
    index_head_dim: int
    qk_pos_emb_head_dim: int
    rotary_pos_emb: nn.Module
    compressor: CompressorInterface
    softmax_scale: float
    pg_collection: ProcessGroupCollection

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute indexer projections before top-k selection."""
        ...

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute index scores and selected indices."""
        ...

    def backward_dw(self) -> None:
        """Compute deferred weight gradients."""
        ...

    def project_packed_queries(self, x: torch.Tensor, qr: torch.Tensor):
        """Project local packed indexer queries and weights."""
        ...


class CSAIndexerBuilder(Protocol):
    """Builder protocol for CSA indexers."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        compress_ratio: int,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: ProcessGroupCollection,
        name: str | None = None,
    ) -> CSAIndexerInterface:
        """Build a CSA indexer."""
        ...


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
        rotary_pos_emb: nn.Module | None = None,
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
        self.compressor = not_none(submodules.compressor)(
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
        k = not_none(apply_module(self.compressor)(x))  # [sq//ratio, b, index_head_dim]

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

    def project_packed_queries(self, x: torch.Tensor, qr: torch.Tensor):
        """Project local packed Q and head weights, retaining FP8/BF16 ownership."""
        q, _ = self.linear_wq_b(qr)
        q = q.reshape(x.shape[0], self.index_n_heads, self.index_head_dim)
        weights = self._project_weights(x).squeeze(1) * (self.index_n_heads**-0.5)
        return q, weights


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


class CompressedSparseAttentionInterface(Protocol):
    """Runtime interface exposed by compressed sparse attention."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        x: torch.Tensor | None = None,
        qr: torch.Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        boundary_hidden: torch.Tensor | None = None,
        boundary_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply compressed sparse attention."""
        ...

    def backward_dw(self) -> None:
        """Compute deferred weight gradients."""
        ...


class CompressedSparseAttentionBuilder(Protocol):
    """Builder protocol for compressed sparse attention."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        softmax_scale: float | None,
        k_channels: int | None,
        v_channels: int | None,
        cp_comm_type: str | None,
        pg_collection: ProcessGroupCollection,
        rotary_pos_emb: nn.Module | None,
        compress_ratio: int,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> CompressedSparseAttentionInterface:
        """Build compressed sparse attention."""
        ...


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
        cp_comm_type: str | None = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module | None = None,
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
        self.use_fused_kernels = use_fused_dsa_kernels(config)

        # Learnable attention sink per head, kept in reference-checkpoint FP32.
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.num_attention_heads, dtype=torch.float32))
        )

        # Conditionally build Compressor (ratio > 1)
        self.compressor: CompressorInterface | None
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
        self.indexer: CSAIndexerInterface | None
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

    def _forward_fused_sbhd(
        self, query: torch.Tensor, key: torch.Tensor, x: torch.Tensor, qr: torch.Tensor
    ) -> torch.Tensor:
        """Run the cuDNN/FlashMLA CSA path for an SBHD input."""
        sq, batch, _num_heads, _head_dim = query.size()
        kv = key.squeeze(-2)

        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = self.compressor(x)
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                num_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                num_compressed = 0
        else:
            kv_full = kv
            num_compressed = 0

        compressed_offset = sq
        window_indices = get_window_topk_idxs(self.window_size, batch, sq, query.device)
        has_indexer_compressed = (
            self.compress_ratio > 1 and num_compressed > 0 and self.indexer is not None
        )

        indexer_loss = None
        if has_indexer_compressed and self.training and torch.is_grad_enabled():
            nvtx_range_push("compressed_indices")
            q_indexer, k_indexer, indexer_weights = self.indexer.forward_before_topk(
                x.detach(), qr.detach()
            )
            nvtx_range_pop("compressed_indices")

            indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
            nvtx_range_push("sparse_attn_kernel")
            output, indexer_loss = fused_csa_indexer_sparse_attn(
                query,
                kv_full,
                self.attn_sink.float(),
                window_indices,
                q_indexer,
                k_indexer,
                indexer_weights,
                self.indexer.index_topk,
                self.compress_ratio,
                self.softmax_scale,
                self.indexer.softmax_scale,
                indexer_loss_coeff,
                sparse_loss=self.config.dsa_indexer_use_sparse_loss,
                kv_offset=compressed_offset,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
            )
            nvtx_range_pop("sparse_attn_kernel")

            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                )
        elif has_indexer_compressed:
            nvtx_range_push("compressed_indices")
            q_indexer, k_indexer, indexer_weights = self.indexer.forward_before_topk(
                x.detach(), qr.detach()
            )
            compressed_indices, _ = indexer_topk(
                q_indexer,
                k_indexer,
                indexer_weights,
                self.indexer.index_topk,
                self.compress_ratio,
                indexer_softmax_scale=self.indexer.softmax_scale,
            )
            compressed_indices = torch.where(
                compressed_indices >= 0, compressed_indices + compressed_offset, -1
            )
            flat_indices, flat_topk_length = build_flat_topk_idxs(
                window_indices, compressed_indices, batch_size=batch, compact=True
            )
            nvtx_range_pop("compressed_indices")

            nvtx_range_push("sparse_attn_kernel")
            output = csa_sparse_attn(
                query,
                kv_full,
                self.attn_sink.float(),
                flat_indices,
                self.softmax_scale,
                topk_length=flat_topk_length,
            )
            nvtx_range_pop("sparse_attn_kernel")
        else:
            nvtx_range_push("compressed_indices")
            if self.compress_ratio > 1 and num_compressed > 0:
                compressed_indices = get_compress_topk_idxs(
                    self.compress_ratio, batch, sq, compressed_offset, query.device
                )
                flat_indices, _ = build_flat_topk_idxs(
                    window_indices, compressed_indices, batch_size=batch
                )
            else:
                flat_indices, _ = build_flat_topk_idxs(window_indices, batch_size=batch)
            nvtx_range_pop("compressed_indices")

            nvtx_range_push("sparse_attn_kernel")
            output = csa_sparse_attn(
                query, kv_full, self.attn_sink.float(), flat_indices, self.softmax_scale
            )
            nvtx_range_pop("sparse_attn_kernel")

        if indexer_loss is not None:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        x: torch.Tensor | None = None,
        qr: torch.Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        boundary_hidden: torch.Tensor | None = None,
        boundary_kv: torch.Tensor | None = None,
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
        if packed_seq_params is not None:
            cp_utils.validate_packed_inputs(packed_seq_params, self.config, self.pg_collection.cp)
            if attention_mask is not None:
                raise ValueError(
                    "Packed CSA uses its document and causal metadata, not attention_mask."
                )
            if query.ndim != 3 or query.shape[1] != self.num_attention_heads:
                raise ValueError("Packed CSA query must be [tokens, attention_heads, head_dim].")
            if boundary_hidden is None:
                if self.pg_collection.cp.size() > 1:
                    raise ValueError("Packed CP requires the projected left boundary.")
                # Empty CP1 boundaries must not alias the full inputs: Dynamo's
                # dynamic-shape guards otherwise track an unused view base when
                # the compiled compactor switches between CP and CP1 layouts.
                boundary_hidden = x.new_empty((0, *x.shape[1:]))
                boundary_kv = key.new_empty((0, *key.shape[1:]))
            return self._forward_packed(
                query, key, x, qr, boundary_hidden, boundary_kv, packed_seq_params
            )
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

        if self.use_fused_kernels:
            output = self._forward_fused_sbhd(query, key, x, qr)
            nvtx_range_pop("compressed_sparse_attn")
            return output

        sq, b, np, hn = query.size()

        # --- Step 1: Prepare single-head KV (squeeze singleton head dim) ---
        kv = key.squeeze(-2)  # [sq, b, 1, v_head_dim] -> [sq, b, v_head_dim]

        # --- Step 2: Compression ---
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = apply_module(self.compressor)(
                not_none(x)
            )  # [n_compressed, b, v_head_dim]
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
                x_det = not_none(x).detach()
                qr_det = not_none(qr).detach()

                causal_mask = (
                    _get_compress_causal_mask_cached(
                        self.compress_ratio, sq, n_compressed, str(x_det.device)
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
                    _, topk_indices_compressed = apply_module(self.indexer)(
                        x_det, qr_det, mask=causal_mask
                    )

                n_valid_per_pos = _get_compress_valid_counts_cached(
                    self.compress_ratio, sq, str(x_det.device)
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

    def _forward_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        boundary_hidden: Optional[torch.Tensor],
        boundary_kv: Optional[torch.Tensor],
        packed_seq_params: PackedSeqParams,
    ) -> torch.Tensor:
        """THD-packed context-parallel branch.

        Build this rank's local KV context from boundary rows and fixed-capacity
        compressed KV, then run sparse attention with optional indexer loss.
        """
        # ---- Step 1: CP metadata and THD shape contract ----------------------
        cp_group = self.pg_collection.cp
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()

        l_local = query.shape[0]
        if l_local != key.shape[0]:
            raise RuntimeError("DSv4 THD CP path currently supports self-attention only.")
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        max_seqlen_q = int(packed_seq_params.max_seqlen_q)

        # ---- Step 2: local CP rows, local KV, and boundary tensors ------------
        global_start = cp_rank * l_local
        kv_local = key.squeeze(-2).squeeze(1)
        if boundary_hidden is None or boundary_kv is None:
            raise RuntimeError(
                "DSv4 THD CP path requires boundary_hidden and boundary_kv from "
                "the hidden-only boundary exchange and boundary KV projection path."
            )
        boundary_kv = boundary_kv.squeeze(-2).squeeze(1)
        d_window = boundary_hidden.shape[0]
        # Window-only defaults; compression fills the rank-major KV buffer.
        compressed_kv_rank_major = kv_local.new_empty((0, kv_local.shape[-1]))
        cu_seqlens_compressed = None

        # ``compressed_topk`` records which compressed blocks each query selects
        # within its sequence. ``seq_to_rank_row`` maps each compressed block to
        # the row where its K and KV are stored in the all-gathered buffer.
        compressed_topk = seq_to_rank_row = None
        ratio = self.compress_ratio
        indexer = self.indexer
        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
        training_with_grad = self.training and torch.is_grad_enabled()
        reconstruct_kv_for_backward = (
            torch.is_grad_enabled()
            and self.use_fused_kernels
            and self.config.recompute_granularity == "selective"
            and "mla_up_proj" in (self.config.recompute_modules or [])
        )
        sparse_indexer_loss = self.config.dsa_indexer_use_sparse_loss
        local_k_indexer_grad_edge = None
        indexer_k_rs_state = None
        local_compressed_kv_grad_edge = None
        compressed_kv_rs_state = None
        if self.compressor is not None and ratio > 1:
            # ---- Step 3: build fixed-capacity compressor input ----------------

            # ``hidden_compact`` packs the local and boundary tokens needed by the
            # Compressor. Tensor operations also provide RoPE positions, global
            # compressed prefixes and the sequence-major -> rank-major gather map.
            (
                hidden_compact,
                compressed_group_ids,
                compressed_position_ids,
                cu_seqlens_compressed,
                seq_to_rank_row,
            ) = cp_utils.prepare_cp_compressor_input(
                x, boundary_hidden, cu_seqlens, global_start, cp_size, ratio
            )

            if indexer is not None:
                # ---- Step 4: optional indexer compressed path -----------------
                indexer_x, indexer_qr = x.detach(), qr.detach()
                if indexer_x.shape[1] != 1:
                    raise RuntimeError(
                        f"DSv4 THD CP indexer expects bsz=1, got {indexer_x.shape[1]}."
                    )

                nvtx_range_push("dsv4_cp_indexer_k_compressor")
                indexer_compressed_local = indexer.compressor.forward_packed(
                    hidden_compact.detach(),
                    cu_seqlens,
                    max_seqlen_q=max_seqlen_q,
                    compressed_group_ids=compressed_group_ids,
                    compressed_position_ids=compressed_position_ids,
                )
                nvtx_range_pop("dsv4_cp_indexer_k_compressor")
                # Build this edge before the independent attention
                # compressor and indexer projection branches. Their newer
                # backward nodes can then run while Indexer-K RS completes.
                # The unfused and inference paths simply never consume it.
                local_k_indexer_grad_edge, indexer_k_rs_state = defer_reduce_scatter_wait(
                    indexer_compressed_local.squeeze(1),
                    "dsv4_cp_indexer_k_reduce_scatter_consumer_wait",
                )
                # Launch before the attention compressor and local indexer
                # projections so NCCL can overlap with both independent paths.
                nvtx_range_push("dsv4_cp_indexer_k_all_gather_launch")
                k_indexer_gather = async_gather_from_sequence_parallel_region(
                    indexer_compressed_local.squeeze(1), group=cp_group
                )
                nvtx_range_pop("dsv4_cp_indexer_k_all_gather_launch")

            # ---- Step 5: attention compressed KV path -------------------------
            nvtx_range_push("dsv4_cp_attention_kv_compressor")
            compressed_kv_local = self.compressor.forward_packed(
                hidden_compact,
                cu_seqlens,
                max_seqlen_q=max_seqlen_q,
                compressed_group_ids=compressed_group_ids,
                compressed_position_ids=compressed_position_ids,
            )
            nvtx_range_pop("dsv4_cp_attention_kv_compressor")
            if indexer is not None:
                # Create this edge before the independent indexer projection
                # nodes. Autograd then executes those newer backward nodes
                # before this edge waits for the compressed-KV RS.
                local_compressed_kv_grad_edge, compressed_kv_rs_state = defer_reduce_scatter_wait(
                    compressed_kv_local.squeeze(1),
                    "dsv4_cp_attention_kv_reduce_scatter_consumer_wait",
                )
                # Queue the attention-KV gather as soon as its local producer
                # completes. Indexer-K was enqueued first on every CP rank, so
                # collective ordering remains identical across the group while
                # both gathers can overlap independent local work.
                nvtx_range_push("dsv4_cp_attention_kv_all_gather_launch")
                compressed_kv_gather = async_gather_from_sequence_parallel_region(
                    compressed_kv_local.squeeze(1), group=cp_group
                )
                nvtx_range_pop("dsv4_cp_attention_kv_all_gather_launch")

                # Local indexer Q/W projections run here: after both RS edges
                # (their backward nodes must be newer than the edges) and while
                # the Indexer-K all-gather is still in flight.
                nvtx_range_push("dsv4_cp_indexer_q_weights")
                q_indexer_cp, weights_indexer_cp = indexer.project_packed_queries(
                    indexer_x, indexer_qr
                )
                if self.config.apply_rope_fusion:
                    rotary_pos_cos, rotary_pos_sin = indexer.rotary_pos_emb.get_cached_cos_sin(
                        max_seqlen_q, dtype=q_indexer_cp.dtype, packed_seq=True, mscale=1.0
                    )
                    q_indexer_cp = cp_utils.apply_thd_cp_local_rope_fused(
                        q_indexer_cp,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        indexer.index_head_dim - indexer.qk_pos_emb_head_dim,
                        indexer.qk_pos_emb_head_dim,
                        cu_seqlens,
                        global_start,
                    )
                else:
                    rope_result = indexer.rotary_pos_emb(max_seqlen_q, packed_seq=True)
                    rotary_pos_emb = (
                        rope_result[0] if isinstance(rope_result, tuple) else rope_result
                    )
                    q_indexer_cp = cp_utils.apply_thd_cp_local_rope_unfused(
                        q_indexer_cp,
                        rotary_pos_emb,
                        indexer.index_head_dim - indexer.qk_pos_emb_head_dim,
                        indexer.qk_pos_emb_head_dim,
                        cu_seqlens,
                        global_start,
                        self.config,
                    )
                q_indexer_cp = rotate_activation(q_indexer_cp)
                nvtx_range_pop("dsv4_cp_indexer_q_weights")

                nvtx_range_push("dsv4_cp_indexer_k_all_gather_wait")
                k_indexer_rank_major = k_indexer_gather.wait()
                nvtx_range_pop("dsv4_cp_indexer_k_all_gather_wait")

                # Indexer top-k consumes sequence-major K rows. Capacity-tail
                # map entries are unused by cu_seqlens_compressed.
                nvtx_range_push("dsv4_cp_indexer_topk")
                k_indexer_seq_major = torch.index_select(
                    k_indexer_rank_major, 0, seq_to_rank_row.clamp_min(0)
                )
                compressed_topk, indexer_layout = cp_utils.compute_cp_indexer_topk(
                    q_indexer_cp,
                    weights_indexer_cp,
                    k_indexer_seq_major,
                    cu_seqlens,
                    cu_seqlens_compressed,
                    global_start,
                    ratio,
                    indexer.index_topk,
                    indexer.softmax_scale,
                    max_seqlen_q=max_seqlen_q,
                    use_fused=self.use_fused_kernels,
                )
                nvtx_range_pop("dsv4_cp_indexer_topk")

                nvtx_range_push("dsv4_cp_attention_kv_all_gather_wait")
                compressed_kv_rank_major = compressed_kv_gather.wait()
                nvtx_range_pop("dsv4_cp_attention_kv_all_gather_wait")
            else:
                compressed_kv_rank_major = async_gather_from_sequence_parallel_region(
                    compressed_kv_local.squeeze(1), group=cp_group
                ).wait()

        use_indexer_loss = (
            training_with_grad and compressed_topk is not None and indexer_loss_coeff > 0
        )
        # ``use_indexer_loss`` implies the RS consumer edges above were built,
        # so the fused indexer-loss path always runs with backward overlap.
        overlap_cp_backward = use_indexer_loss and self.use_fused_kernels

        # ---- Step 6: concatenate the raw local KV sources -------------------
        # The fused training path owns both CP reduce-scatters. Detach the
        # gathered buffer here and return each local gradient through a
        # branch-local deferred consumer edge below.
        compressed_kv_for_attention = (
            compressed_kv_rank_major.detach() if overlap_cp_backward else compressed_kv_rank_major
        )
        kv_full_thd = torch.cat((boundary_kv, kv_local, compressed_kv_for_attention), dim=0)
        # ``kv_full_thd`` stays the autograd input so its cat edge owns dKV.
        # Saving the direct producers only changes which values survive until backward.
        kv_reconstruction_parts = (
            (boundary_kv, kv_local, compressed_kv_for_attention)
            if reconstruct_kv_for_backward
            else None
        )
        compressed_width = (
            compressed_topk.shape[-1]
            if compressed_topk is not None
            else (max_seqlen_q // ratio if ratio > 1 else 0)
        )
        cu_seqlens_q_unpadded = None
        if (
            packed_seq_params.cu_seqlens_q is not None
            and packed_seq_params.cu_seqlens_q_padded is not None
            and packed_seq_params.cu_seqlens_q.data_ptr()
            != packed_seq_params.cu_seqlens_q_padded.data_ptr()
        ):
            cu_seqlens_q_unpadded = packed_seq_params.cu_seqlens_q
        # Lower the logical ids into two physical spaces: indexer_topk_rank_major
        # addresses the rank-major compressed buffers without kv_full_thd's
        # compressed base, while topk_idxs addresses final rows in kv_full_thd.
        # The same row scan also emits the CUDA-graph padding mask when needed.
        build_indices = packed_layout.build_attention_indices
        topk_idxs, topk_length, indexer_topk_rank_major, q_padding_mask = build_indices(
            cu_seqlens,
            global_start,
            l_local,
            d_window,
            self.window_size,
            ratio,
            compressed_width,
            compressed_topk,
            cu_seqlens_compressed=cu_seqlens_compressed,
            seq_to_rank_row=seq_to_rank_row,
            for_indexer_loss=use_indexer_loss,
            compressed_rows=compressed_kv_rank_major.shape[0],
            cu_seqlens_unpadded=cu_seqlens_q_unpadded,
            output_alignment=(get_flash_mla_topk_alignment() if self.use_fused_kernels else 1),
        )
        if use_indexer_loss:
            # ---- Step 7a: indexer-loss path ----------------------------------
            k_indexer_for_loss = k_indexer_rank_major
            compressed_kv_for_loss = compressed_kv_rank_major
            if not sparse_indexer_loss:
                k_indexer_for_loss = k_indexer_seq_major
                compressed_kv_for_loss = torch.index_select(
                    compressed_kv_rank_major, 0, seq_to_rank_row.clamp_min(0)
                )
            if overlap_cp_backward:
                k_indexer_for_loss = k_indexer_for_loss.detach()
                compressed_kv_for_loss = compressed_kv_for_loss.detach()
            loss_divisor = 1 if self.config.calculate_per_token_loss else l_local * cp_size
            if not self.config.calculate_per_token_loss and cu_seqlens_q_unpadded is not None:
                # Match the reference DSA mean over real query rows. Keep the
                # scalar on device so padded THD remains CUDA-graph safe.
                loss_divisor = cu_seqlens_q_unpadded[-1].clamp_min(1)
            indexer_loss_args = (
                query,
                kv_full_thd,
                self.attn_sink.float(),
                topk_idxs,
                q_indexer_cp,
                k_indexer_for_loss,
                weights_indexer_cp,
                indexer_topk_rank_major,
                compressed_kv_for_loss,
                self.softmax_scale,
                indexer.softmax_scale,
                indexer_loss_coeff,
                loss_divisor,
                sparse_indexer_loss,
                ratio,
                max_seqlen_q,
                indexer_layout,
                q_padding_mask,
            )
            if overlap_cp_backward:
                output, indexer_loss = FusedCSAIndexerSparseAttnFromTopkFunc.apply(
                    *indexer_loss_args,
                    local_k_indexer_grad_edge,
                    local_compressed_kv_grad_edge,
                    cp_group,
                    d_window + l_local,
                    seq_to_rank_row if not sparse_indexer_loss else None,
                    indexer_k_rs_state,
                    compressed_kv_rs_state,
                    self.window_size,
                    kv_reconstruction_parts,
                )
            else:
                output, indexer_loss = _unfused_indexer_sparse_attn_from_topk(
                    *indexer_loss_args, tp_group=indexer.pg_collection.tp
                )
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                    reduce_group=cp_group,
                )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return output.unsqueeze(1)

        # ---- Step 7b: sparse attention path ----------------------------------
        if self.use_fused_kernels:
            output = packed_csa_sparse_attn(
                query,
                kv_full_thd,
                self.attn_sink.float(),
                topk_idxs,
                self.softmax_scale,
                topk_length=topk_length,
                is_thd=True,
                kv_reconstruction_parts=kv_reconstruction_parts,
                q_padding_mask=q_padding_mask,
            )
        else:
            output = unfused_compressed_sparse_attn(
                query, kv_full_thd, self.attn_sink.float(), topk_idxs, self.softmax_scale
            )
        if training_with_grad and indexer is not None:
            # Zero loss or a pack without compressed keys still needs explicit
            # indexer gradients, without saving full-size zero gradients in forward.
            zero_loss = (
                q_indexer_cp.sum() + k_indexer_rank_major.sum() + weights_indexer_cp.sum()
            ) * 0.0
            output = DSAIndexerLossAutoScaler.apply(output, zero_loss.float())
        return output.unsqueeze(1)


def _apply_unfused_rope(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    config: TransformerConfig,
    cu_seqlens: Optional[torch.Tensor],
    cp_group: torch.distributed.ProcessGroup,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Apply unfused RoPE (split, rotate, concat) with 3-D / 4-D handling.

    DSv4 forces ``mscale=1.0`` — the model relies on Q/KV RMS-norm +
    unit-magnitude rotation, not Yarn's concentration factor.
    """
    packed_seq = cu_seqlens is not None

    # Drop dummy ``b=1`` from packed 4-D ``(total, 1, h, d)`` callers.
    squeezed_b = packed_seq and x.dim() == 4 and x.size(1) == 1
    # Packed 3-D ``(total, 1, d)``: collapse batch and add a temporary head dim.
    squeezed_b_3d = packed_seq and x.dim() == 3 and x.size(1) == 1
    if squeezed_b:
        x = x.squeeze(1)
    elif squeezed_b_3d:
        x = x.squeeze(1).unsqueeze(-2)

    # Non-packed 3-D ``(b, s, d)``: add a temporary head dim.
    squeeze_head = not packed_seq and x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)

    x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
    x_pe = apply_rotary_pos_emb(
        x_pe,
        rotary_pos_emb,
        config=config,
        cu_seqlens=cu_seqlens,
        mscale=1.0,
        cp_group=cp_group,
        mla_rotary_interleaved=True,
        mla_output_remove_interleaving=True,
        max_seqlen=max_seqlen,
    )
    out = torch.cat([x_nope, x_pe], dim=-1)

    if squeezed_b:
        out = out.unsqueeze(1)
    elif squeezed_b_3d:
        out = out.squeeze(-2).unsqueeze(1)
    elif squeeze_head:
        out = out.squeeze(-2)
    return out


def _unfused_indexer_sparse_attn_from_topk(
    query: torch.Tensor,
    kv_full: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_indices: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    indexer_topk_indices: torch.Tensor,
    compressed_kv: torch.Tensor,
    softmax_scale: float,
    indexer_softmax_scale: float,
    loss_coeff: float,
    loss_divisor: Union[float, torch.Tensor],
    sparse_loss: bool,
    ratio: int,
    _max_seqlen_q: int,
    indexer_layout: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    q_padding_mask: Optional[torch.Tensor] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch sparse attention plus caller-supplied indexer loss for THD CP.

    This mirrors the fused CP top-k path at the tensor-contract level:
    ``topk_indices`` indexes ``kv_full`` for sparse attention, and
    ``indexer_topk_indices`` indexes compressed K for sparse indexer loss.
    ``_max_seqlen_q`` is unused here; it is retained to match the fused
    callable's signature, with the leading underscore marking it as unused.
    """
    output = unfused_compressed_sparse_attn(query, kv_full, attn_sink, topk_indices, softmax_scale)

    if loss_coeff <= 0:
        zero_loss = (q_indexer.sum() + k_indexer.sum() + weights.sum()) * 0.0
        return output, zero_loss.float()

    total_q, np_, hn = query.shape
    cu_seqlens_q, cu_seqlens_k, q_causal_offsets = indexer_layout
    indexer_topk = indexer_topk_indices.shape[-1]
    attention_indices = topk_indices
    if q_padding_mask is not None:
        attention_indices = attention_indices.masked_fill(q_padding_mask.unsqueeze(-1), -1)
    non_compressed_lse = _compute_unfused_csa_non_compressed_lse(
        query, kv_full, attn_sink, attention_indices[:, indexer_topk:], softmax_scale
    )

    def compressed_teacher_target(
        logits: torch.Tensor, valid_mask: torch.Tensor, row_slice: slice
    ) -> torch.Tensor:
        lse_slice = non_compressed_lse[:, :, row_slice]
        probabilities = _compute_indexer_teacher_probabilities(
            logits.permute(1, 0, 2).unsqueeze(0),
            valid_mask.unsqueeze(0),
            non_compressed_lse=lse_slice,
        )
        target = probabilities.sum(dim=1).squeeze(0)
        if tp_group is not None and tp_group.size() > 1:
            torch.distributed.all_reduce(target, group=tp_group)
        return _normalize_indexer_teacher_target(target, lse_slice)

    def scale_local_loss(raw_local_loss: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(loss_divisor):
            divisor = loss_divisor.to(device=raw_local_loss.device, dtype=torch.float32).clamp_min(
                1.0
            )
        else:
            divisor = max(float(loss_divisor), 1.0)
        return raw_local_loss * float(loss_coeff) / divisor

    if not sparse_loss:
        q_rows = torch.arange(total_q, dtype=cu_seqlens_q.dtype, device=query.device)
        q_sequence_ids = batch_of_row(cu_seqlens_q, total_q=total_q)
        q_positions = q_rows - cu_seqlens_q[q_sequence_ids] + q_causal_offsets[q_sequence_ids]
        visible_k = torch.minimum(
            torch.div(q_positions + 1, ratio, rounding_mode="floor"),
            (cu_seqlens_k[1:] - cu_seqlens_k[:-1])[q_sequence_ids],
        )
        q_valid = q_rows < cu_seqlens_q[-1]
        if q_padding_mask is not None:
            q_valid = q_valid & ~q_padding_mask

        k_rows = torch.arange(k_indexer.shape[0], dtype=cu_seqlens_k.dtype, device=k_indexer.device)
        k_sequence_ids = torch.bucketize(
            k_rows, cu_seqlens_k[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_k.shape[0] - 2)
        k_positions = k_rows - cu_seqlens_k[k_sequence_ids]

        k_indexer_float = k_indexer.float()
        compressed_kv_float = compressed_kv.detach().float()
        weights_scaled = weights.float() * float(indexer_softmax_scale)
        raw_local_loss = query.new_zeros((), dtype=torch.float32)
        for start in range(0, total_q, 512):
            end = min(start + 512, total_q)
            valid = (
                (q_sequence_ids[start:end].unsqueeze(1) == k_sequence_ids.unsqueeze(0))
                & (k_positions.unsqueeze(0) < visible_k[start:end].unsqueeze(1))
                & q_valid[start:end].unsqueeze(1)
            )
            row_valid = valid.any(dim=-1, keepdim=True)

            predict_logits = torch.einsum(
                "rhd,kd->rhk", q_indexer[start:end].float(), k_indexer_float
            )
            predict_logits = torch.relu(predict_logits) * weights_scaled[start:end].unsqueeze(-1)
            predict_logits = predict_logits.sum(dim=1).masked_fill(~valid, float("-inf"))
            predict_logits = torch.where(row_valid, predict_logits, 0.0)
            predict = torch.softmax(predict_logits, dim=-1, dtype=torch.float32)
            predict = predict * row_valid.float()

            target_logits = torch.einsum(
                "rhd,kd->rhk", query[start:end].detach().float(), compressed_kv_float
            )
            target_logits = (target_logits * softmax_scale).masked_fill(
                ~valid.unsqueeze(1), float("-inf")
            )
            target = compressed_teacher_target(target_logits, valid, slice(start, end))
            eps = torch.finfo(torch.float32).tiny
            target = target * row_valid.float()
            target = target.clamp(min=eps)
            predict = predict.clamp(min=eps)

            kl_per_row = (target * (torch.log(target) - torch.log(predict))).sum(dim=-1)
            kl_per_row = torch.where(
                row_valid.squeeze(-1), kl_per_row, torch.zeros_like(kl_per_row)
            )
            raw_local_loss = raw_local_loss + kl_per_row.sum()

        return output, scale_local_loss(raw_local_loss)

    if q_padding_mask is not None:
        indexer_topk_indices = indexer_topk_indices.masked_fill(q_padding_mask.unsqueeze(-1), -1)

    valid = indexer_topk_indices >= 0
    row_valid = valid.any(dim=-1, keepdim=True)
    safe_indices = indexer_topk_indices.clamp(min=0).long()

    weights_scaled = weights.float() * float(indexer_softmax_scale)
    predict_chunks = []
    # Avoid materializing the full [local_q, index_heads, global_k] score tensor.
    for start in range(0, total_q, 512):
        end = start + 512
        chunk_indices = safe_indices[start:end]
        selected_k_indexer = k_indexer.index_select(0, chunk_indices.reshape(-1)).reshape(
            chunk_indices.shape[0], indexer_topk, -1
        )
        chunk_scores = torch.einsum(
            "rhd,rkd->rhk", q_indexer[start:end].float(), selected_k_indexer.float()
        )
        chunk_scores = torch.relu(chunk_scores) * weights_scaled[start:end].unsqueeze(-1)
        predict_chunks.append(chunk_scores.sum(dim=1))
    predict_logits = torch.cat(predict_chunks)
    predict_logits = predict_logits.masked_fill(~valid, float("-inf"))
    predict_logits = predict_logits.masked_fill(~row_valid, 0.0)
    predict = torch.softmax(predict_logits, dim=-1, dtype=torch.float32)
    predict = predict * row_valid.float()

    selected_kv = compressed_kv.detach().index_select(0, safe_indices.reshape(-1))
    selected_kv = selected_kv.reshape(total_q, indexer_topk, hn)

    # Normalize selected compressed logits with the same sliding-window and
    # sink mass used by the real CSA attention teacher.
    attn_scores = torch.einsum("rhd,rkd->rhk", query.detach().float(), selected_kv.float())
    attn_scores = attn_scores * softmax_scale
    attn_scores = attn_scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    target = compressed_teacher_target(attn_scores, valid, slice(None))
    target = target * row_valid.float()

    eps = torch.finfo(torch.float32).tiny
    target = target.clamp(min=eps)
    predict = predict.clamp(min=eps)
    kl_per_row = (target * (torch.log(target) - torch.log(predict))).sum(dim=-1)
    kl_per_row = torch.where(row_valid.squeeze(-1), kl_per_row, torch.zeros_like(kl_per_row))
    raw_local_loss = kl_per_row.sum()

    indexer_loss = scale_local_loss(raw_local_loss)
    return output, indexer_loss


def unfused_compressed_sparse_attn(query, kv_full, attn_sink, topk_indices, softmax_scale):
    """Native sparse attention for SBHD or flat packed physical indices."""
    if query.ndim == 3:
        return _unfused_compressed_sparse_attn_sbhd(
            query.unsqueeze(1),
            kv_full.unsqueeze(1),
            attn_sink,
            topk_indices.unsqueeze(0),
            softmax_scale,
        ).squeeze(1)
    return _unfused_compressed_sparse_attn_sbhd(
        query, kv_full, attn_sink, topk_indices, softmax_scale
    )

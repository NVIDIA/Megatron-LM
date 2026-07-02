# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN/FlashMLA wrappers for fused DeepSeek sparse attention."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import torch
from torch import Tensor

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_indexer_loss,
    dsa_layout,
    dsa_masking,
)
from megatron.core.utils import get_pg_size, round_up_to_nearest_multiple

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

_flash_mla_sparse_fwd = None


class CudnnDsaInterface(Protocol):
    """Subset of the cudnn-frontend ``DSA`` namespace these fused kernels call.

    Each wrapper returns a ``dict`` of named output tensors (e.g. ``"indices"``/``"values"``,
    ``"scores"``, ``"out"``/``"denom"``, ``"target"``, ``"dq"``/``"dkv"``,
    ``"d_index_q"``/``"d_weights"``/``"d_index_k"``).
    """

    def indexer_top_k_wrapper(
        self, scores: Tensor, seq_lens: Tensor, top_k: int, next_n: int, return_val: bool
    ) -> dict:
        """Select the top-``top_k`` key indices (and optional scores) per query row."""

    def indexer_forward_wrapper(
        self,
        index_q: Tensor,
        index_k: Tensor,
        weights: Tensor,
        ratio: int,
        sm_scale: float,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> dict:
        """Compute the head-summed indexer logits over the key axis (``"scores"``)."""

    def indexer_backward_wrapper(
        self,
        index_q: Tensor,
        weights: Tensor,
        index_k: Tensor,
        attn_score: Tensor,
        index_score: Tensor,
        topk_indices: Tensor,
        sm_scale: float,
        loss_coeff: float,
        grad_loss: Tensor,
        block_I: int,
    ) -> dict:
        """Backprop the sparse (top-k) indexer KL loss to the indexer q/k/weights."""

    def dense_attn_score_recompute_wrapper(
        self,
        query: Tensor,
        key: Tensor,
        lse: Tensor,
        softmax_scale: float,
        qhead_per_kv_head: int,
        ratio: int,
    ) -> dict:
        """Recompute the dense head-summed attention distribution and its L1 norm."""

    def dense_indexer_backward_wrapper(
        self,
        index_q: Tensor,
        weights: Tensor,
        index_k: Tensor,
        attn_score: Tensor,
        attn_l1norm: Tensor,
        index_score: Tensor,
        index_lse: Tensor,
        sm_scale: float,
        loss_coeff: float,
        grad_loss: Tensor,
        ratio: int,
        block_I: int,
    ) -> dict:
        """Backprop the dense (full-KV) indexer KL loss to the indexer q/k/weights."""

    def sparse_attention_backward_wrapper(
        self,
        query: Tensor,
        kv: Tensor,
        out: Tensor,
        d_out: Tensor,
        lse: Tensor,
        attn_sink: Tensor,
        indices: Tensor,
        softmax_scale: float,
        topk_length: Tensor,
    ) -> dict:
        """Backprop FlashMLA sparse attention to query and compressed KV (``"dq"``/``"dkv"``)."""

    def sparse_attn_score_recompute_wrapper(
        self,
        query: Tensor,
        key: Tensor,
        lse: Tensor,
        topk_indices: Tensor,
        softmax_scale: float,
        qhead_per_kv_head: int,
        topk_indices_global: bool,
        topk_length: Optional[Tensor] = None,
    ) -> dict:
        """Recompute the head-summed sparse attention target over the top-k keys."""


_cudnn_dsa: Optional[CudnnDsaInterface] = None
_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny
_KL_EPS = 1e-10
_TOPK_TIE_BREAK_EPS = 1.0e-12
_INDEXER_RATIO = 1
_INDEXER_SOFTMAX_SCALE = 1.0
_TOPK_WRAPPER_MAX_SCRATCH_BYTES = 2 * 1024 * 1024 * 1024
_TOPK_WRAPPER_SCRATCH_INT32_FACTOR = 2
_TOPK_WRAPPER_ROW_ALIGNMENT = 512
_INDEXER_SCORE_CHUNK_MAX_BYTES = 1024 * 1024 * 1024
_INDEXER_SCORE_CHUNK_ROW_ALIGNMENT = 512
_DENSE_ATTN_LSE_CHUNK_MAX_BYTES = 1024 * 1024 * 1024
_FLASH_MLA_REQUIRED_VALUE_DIM = 512


def _assert_supported_indexer_scoring(use_relu: bool) -> None:
    """Check that the cuDNN indexer scoring mode matches the fused kernel implementation."""
    if not use_relu:
        raise RuntimeError(
            "cuDNN fused DSA kernels currently require dsa_indexer_scoring_relu=True."
        )


def _supports_flash_mla_value_layout(kv_width: int, d_v: int) -> bool:
    """Return whether FlashMLA sparse prefill supports the requested value layout."""
    return d_v == _FLASH_MLA_REQUIRED_VALUE_DIM and kv_width >= _FLASH_MLA_REQUIRED_VALUE_DIM


def _get_multi_packed_cp_thd_metadata(
    use_local_indexer_varlen: bool,
    packed_seq_params: Optional["PackedSeqParams"],
    single_packed_thd_sequence: bool,
    cp_size: int,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[int], Optional[int]]:
    """Return packed metadata needed by the cuDNN multi-sequence THD indexer."""
    if (
        not use_local_indexer_varlen
        or packed_seq_params is None
        or single_packed_thd_sequence
        or cp_size <= 1
    ):
        return None, None, None, None
    cu_seqlens_q, cu_seqlens_k = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
    return (
        cu_seqlens_q,
        cu_seqlens_k,
        packed_seq_params.max_seqlen_q,
        packed_seq_params.max_seqlen_kv,
    )


def run_fused_dsa_attention(
    *,
    config: TransformerConfig,
    query: Tensor,
    key: Tensor,
    value: Optional[Tensor],
    up_v_weight: Optional[Tensor],
    q_indexer: Tensor,
    k_indexer: Tensor,
    indexer_weights: Tensor,
    indexer_topk: int,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    calculate_per_token_loss: bool,
    absorbed_mla: bool,
    cp_size: int,
    attn_mask_type: Optional[AttnMaskType],
    packed_seq_params: Optional[PackedSeqParams],
    varlen_starts: Optional[Tensor],
    varlen_ends: Optional[Tensor],
    key_positions: Optional[Tensor],
    query_valid_rows: Optional[Tensor],
    varlen_is_plain_causal: bool = False,
    use_relu: bool,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    pg_collection: Optional["ProcessGroupCollection"] = None,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Optional full fused DSA hook for backends that fuse indexer and attention together.

    The dispatcher only routes here for the cuDNN backend and ``dsa.py`` already gates on
    ``use_fused_dsa_kernels`` before calling, so this hook assumes it was selected (matching
    the TileLang backend) and only validates that the requested shapes/layout are supported.
    """
    _assert_supported_indexer_scoring(use_relu)
    if (
        not absorbed_mla
        or value is not None
        or attn_mask_type != AttnMaskType.causal
        or key.size(2) != 1
    ):
        return None
    if not torch.is_grad_enabled():
        loss_coeff = 0.0
    # build_dsattention_forward_mask emits explicit varlen bounds even for the plain (non-packed,
    # non-CP) causal case and flags them via ``varlen_is_plain_causal``. Those bounds are exactly
    # equivalent to the no-varlen causal path the dense fused indexer loss was written for, so
    # normalize them back to None: plain causal then takes the same fused path (relying on the
    # kernel's internal causal masking) it took before that mask change, instead of declining
    # dense loss and falling back to the slow reference-loss path. The flag carries the mask
    # builder's structural knowledge, so we avoid the per-forward host/device sync a ``torch.equal``
    # bounds comparison would force on this common path. Genuine packed/CP/custom-position varlen
    # is left untouched (``varlen_is_plain_causal`` is False there) and is gated below, where it
    # would otherwise trip the padded-row assert in FusedIndexerSparseAttnFunc.forward.
    if cp_size == 1 and packed_seq_params is None and varlen_is_plain_causal:
        varlen_starts = None
        varlen_ends = None
    has_varlen = varlen_starts is not None or varlen_ends is not None or key_positions is not None
    if has_varlen:
        if varlen_starts is None or varlen_ends is None:
            return None
        if loss_coeff > 0 and not sparse_loss:
            return None
        if not use_local_indexer_varlen and key_positions is not None:
            return None
    elif cp_size != 1 or packed_seq_params is not None or query.size(0) != key.size(0):
        return None

    latent_v_channels = int(getattr(config, "kv_lora_rank", 0) or 0)
    if latent_v_channels <= 0:
        return None
    if up_v_weight is None:
        return None
    if not _supports_flash_mla_value_layout(key.size(-1), latent_v_channels):
        return None
    if not _flash_mla_supports_head_count(query):
        return None
    sq, b, num_heads, _ = query.size()
    kv_full = key.squeeze(2).contiguous()
    packed_cu_seqlens_q, packed_cu_seqlens_k, packed_max_seqlen_q, packed_max_seqlen_k = (
        _get_multi_packed_cp_thd_metadata(
            use_local_indexer_varlen, packed_seq_params, single_packed_thd_sequence, cp_size
        )
    )
    packed_thd_kwargs = {}
    if packed_cu_seqlens_q is not None:
        packed_thd_kwargs = {
            "packed_cu_seqlens_q": packed_cu_seqlens_q,
            "packed_cu_seqlens_k": packed_cu_seqlens_k,
            "packed_max_seqlen_q": packed_max_seqlen_q,
            "packed_max_seqlen_k": packed_max_seqlen_k,
            "packed_cp_size": cp_size,
        }

    output, indexer_loss = fused_indexer_sparse_attn(
        query.contiguous(),
        kv_full,
        q_indexer.contiguous(),
        k_indexer.contiguous(),
        indexer_weights.contiguous(),
        min(indexer_topk, key.size(0)),
        softmax_scale,
        loss_coeff,
        sparse_loss=sparse_loss,
        calculate_per_token_loss=calculate_per_token_loss,
        d_v=latent_v_channels,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
        query_valid_rows=query_valid_rows,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        tp_group=getattr(pg_collection, "tp", None),
        **packed_thd_kwargs,
    )

    output = output.view(sq, b, num_heads, latent_v_channels)
    output = torch.einsum("sbhc,hdc->sbhd", output, up_v_weight).contiguous()
    return output.view(sq, b, -1), indexer_loss


def _ensure_flash_mla() -> None:
    """Lazily import the FlashMLA sparse-forward kernel.

    FlashMLA ships ``flash_mla_sparse_fwd`` with a multi-head-KV signature;
    :func:`_dsa_fwd_flash_mla` below adapts the DSA-shape inputs and pads
    ``topk`` to the alignment expected by the active FlashMLA sparse-prefill kernel.
    """
    global _flash_mla_sparse_fwd
    if _flash_mla_sparse_fwd is not None:
        return

    try:
        from flash_mla import flash_mla_sparse_fwd as _fwd
    except ImportError as e:
        raise ImportError(
            "FlashMLA is required for DSA sparse attention forward. "
            "Install from https://github.com/deepseek-ai/FlashMLA/tree/nv_dev "
            "so that `from flash_mla import flash_mla_sparse_fwd` succeeds."
        ) from e
    _flash_mla_sparse_fwd = _fwd


@functools.lru_cache(maxsize=None)
def _device_sm(device_index: int) -> Tuple[int, int]:
    """Memoized ``(major, minor)`` SM capability of a CUDA device (static for a device)."""
    return torch.cuda.get_device_capability(device_index)


def _current_sm() -> Tuple[int, int]:
    """Return the current CUDA device's SM capability, memoized per device index."""
    return _device_sm(torch.cuda.current_device())


def _get_topk_alignment() -> int:
    """Minimum top-K alignment required by the active FlashMLA sparse-prefill kernel."""
    sm = _current_sm()
    if sm[0] >= 9:
        return 512
    raise RuntimeError(f"cudnn fused DSA requires SM90+ (Hopper or later), got SM{sm[0]}{sm[1]}.")


def _flash_mla_head_padding(num_heads: int) -> Optional[int]:
    """Padded query-head count FlashMLA supports for ``num_heads``, or None if unsupported.

    Returns None when the device is pre-SM90 or the head count cannot be aligned, so callers
    can decline (return None) and fall back to the unfused path instead of raising.
    """
    sm = _current_sm()
    if sm[0] >= 10:
        for padded_heads in (64, 128):
            if num_heads == padded_heads:
                return num_heads
            if num_heads < padded_heads and padded_heads % num_heads == 0:
                return padded_heads
        head_align = 128
    elif sm[0] >= 9:
        head_align = 64
    else:
        return None

    if num_heads % head_align == 0:
        return num_heads
    if num_heads < head_align and head_align % num_heads == 0:
        return head_align
    return None


def _get_head_padding(num_heads: int) -> int:
    """Return the query-head count supported by the FlashMLA sparse prefill kernel."""
    sm = _current_sm()
    if sm[0] < 9:
        raise RuntimeError(
            f"cudnn fused DSA requires SM90+ (Hopper or later), got SM{sm[0]}{sm[1]}."
        )
    padded = _flash_mla_head_padding(num_heads)
    if padded is None:
        head_align = 128 if sm[0] >= 10 else 64
        raise RuntimeError(
            "FlashMLA sparse prefill requires the local query-head count to divide "
            f"{head_align}, got h_q={num_heads}."
        )
    return padded


def _flash_mla_supports_head_count(query: Tensor) -> bool:
    """Whether the attention query head count is FlashMLA-supported on the query's device.

    Non-CUDA tensors return True (the fused path is CUDA-only and declines elsewhere), so a
    caller can gate with ``if not _flash_mla_supports_head_count(query): return None``.
    """
    if not query.is_cuda:
        return True
    return _flash_mla_head_padding(query.size(2)) is not None


def _bytes_to_chunk_rows(n_rows: int, bytes_per_row: int, max_bytes: int, alignment: int) -> int:
    """Rows fitting in ``max_bytes``, rounded down to ``alignment`` when the budget permits.

    ``alignment == 1`` performs no rounding (the result is just the byte-budget cap).
    """
    chunk_rows = max(1, max_bytes // max(1, bytes_per_row))
    if chunk_rows >= alignment:
        chunk_rows = (chunk_rows // alignment) * alignment
    return min(n_rows, chunk_rows)


def _causal_seq_lens(q_positions: Tensor, ratio: int, sk: int) -> Tensor:
    """Per-query causal key length under the indexer ``ratio``, clamped to ``sk``."""
    return ((q_positions + 1) // ratio).clamp(max=sk)


def _topk_tie_break_bias(positions: Tensor, key_count: int, dtype: torch.dtype) -> Tensor:
    """Per-position tie-break bias; lower key indices get a larger bias.

    Shared by :func:`_add_indexer_topk_tie_break` and
    :func:`_remove_indexer_topk_tie_break` so the add and remove sides stay exact inverses.
    """
    step = _TOPK_TIE_BREAK_EPS / float(key_count - 1)
    return ((key_count - 1 - positions) * step).to(dtype=dtype)


def _add_indexer_topk_tie_break(scores: Tensor, *, inplace: bool = False) -> Tensor:
    """Prefer lower key indices for exact top-k score ties."""
    sk = scores.size(-1)
    if sk <= 1:
        return scores
    positions = torch.arange(sk, device=scores.device, dtype=torch.float32)
    bias = _topk_tie_break_bias(positions, sk, scores.dtype).view(*((1,) * (scores.ndim - 1)), sk)
    if inplace:
        return scores.add_(bias)
    return scores + bias


def _use_dense_indexer_topk_tie_break(scores: Tensor, topk_k: int) -> bool:
    """Return whether to bias the full score matrix before cuDNN top-k.

    The dense bias is useful for deterministic small-shape parity tests, but on
    long-context CUDA runs it is a full score-matrix elementwise pass before
    every top-k and dominates runtime. Real indexer scores are not expected to
    rely on exact finite ties for correctness, so keep the expensive full-matrix
    tie-break off the CUDA training path.
    """
    return topk_k < scores.size(-1) and not scores.is_cuda


def _remove_indexer_topk_tie_break(
    topk_scores: Tensor, topk_indices: Tensor, key_count: int
) -> Tensor:
    """Remove the deterministic top-k tie-break from selected score values."""
    if key_count <= 1:
        return topk_scores
    positions = topk_indices.clamp(min=0, max=key_count - 1).to(dtype=torch.float32)
    bias = _topk_tie_break_bias(positions, key_count, topk_scores.dtype)
    return torch.where(topk_indices >= 0, topk_scores - bias, topk_scores)


def _indexer_top_k_wrapper_chunked(
    scores_flat: Tensor, seq_lens: Tensor, topk_k: int, return_topk_scores: bool
) -> dict:
    """Run cuDNN top-k in row chunks to bound wrapper scratch allocation."""
    n_rows, sk = scores_flat.shape
    scratch_bytes_per_row = max(1, sk) * torch.iinfo(torch.int32).bits // 8
    scratch_bytes_per_row *= _TOPK_WRAPPER_SCRATCH_INT32_FACTOR
    chunk_rows = _bytes_to_chunk_rows(
        n_rows, scratch_bytes_per_row, _TOPK_WRAPPER_MAX_SCRATCH_BYTES, _TOPK_WRAPPER_ROW_ALIGNMENT
    )

    if chunk_rows >= n_rows:
        return _cudnn_dsa.indexer_top_k_wrapper(
            scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=return_topk_scores
        )

    indices_chunks = []
    values_chunks = [] if return_topk_scores else None
    for row_start in range(0, n_rows, chunk_rows):
        row_end = min(row_start + chunk_rows, n_rows)
        tk_result = _cudnn_dsa.indexer_top_k_wrapper(
            scores_flat[row_start:row_end].contiguous(),
            seq_lens[row_start:row_end].contiguous(),
            top_k=topk_k,
            next_n=1,
            return_val=return_topk_scores,
        )
        indices_chunks.append(tk_result["indices"])
        if return_topk_scores:
            values = tk_result["values"]
            if values is None:
                raise RuntimeError("cuDNN indexer_top_k_wrapper did not return values.")
            values_chunks.append(values)

    return {
        "indices": torch.cat(indices_chunks, dim=0),
        "values": torch.cat(values_chunks, dim=0) if return_topk_scores else None,
    }


def _indexer_score_chunk_rows(b: int, sq: int, sk: int) -> int:
    score_bytes_per_seq_row = max(1, b) * max(1, sk) * torch.finfo(torch.float32).bits // 8
    return _bytes_to_chunk_rows(
        sq,
        score_bytes_per_seq_row,
        _INDEXER_SCORE_CHUNK_MAX_BYTES,
        _INDEXER_SCORE_CHUNK_ROW_ALIGNMENT,
    )


def _compute_indexer_scores_chunk_with_global_rows(
    q_chunk_bshd: Tensor,
    k_bshd: Tensor,
    w_chunk_bsh: Tensor,
    *,
    row_start: int,
    indexer_ratio: int,
    sm_scale: float,
    seq_lens: Optional[Tensor] = None,
    k_bdk: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
) -> Tensor:
    """Compute indexer scores for a query chunk using global causal row positions.

    ``k_bdk`` (fp32 ``[b, idx_hd, sk]`` key) and ``key_positions`` (``arange(sk)``) are
    loop-invariant across query chunks; callers that chunk may precompute and pass them to
    avoid recomputing the key cast/transpose and arange on every chunk.
    """
    if indexer_ratio <= 0:
        raise RuntimeError(f"cuDNN DSA indexer ratio must be positive, got {indexer_ratio}.")
    if k_bshd.size(2) != 1:
        raise RuntimeError(f"cuDNN DSA indexer expects one key head, got {k_bshd.size(2)}.")

    b, sq, idx_nh, _idx_hd = q_chunk_bshd.shape
    sk = k_bshd.size(1)
    scores_chunk = torch.zeros((b, sq, sk), dtype=torch.float32, device=q_chunk_bshd.device)
    if k_bdk is None:
        k_bdk = k_bshd[:, :, 0, :].to(dtype=torch.float32).transpose(1, 2).contiguous()

    # Accumulate one head at a time to avoid materializing a [b, sq, idx_nh, sk] tensor.
    for head_idx in range(idx_nh):
        head_scores = torch.bmm(q_chunk_bshd[:, :, head_idx, :].to(dtype=torch.float32), k_bdk)
        head_scores.relu_()
        head_scores.mul_(w_chunk_bsh[:, :, head_idx].to(dtype=torch.float32).unsqueeze(-1))
        scores_chunk.add_(head_scores)

    if sm_scale != 1.0:
        scores_chunk.mul_(sm_scale)

    if seq_lens is None:
        query_positions = torch.arange(row_start, row_start + sq, device=q_chunk_bshd.device)
        seq_lens = _causal_seq_lens(query_positions, indexer_ratio, sk)
    else:
        if seq_lens.ndim != 1 or seq_lens.numel() != sq:
            raise RuntimeError(f"cuDNN DSA score chunk seq_lens must have shape ({sq},).")
        seq_lens = seq_lens.to(device=q_chunk_bshd.device, dtype=torch.int64).clamp(max=sk)
    if key_positions is None:
        key_positions = torch.arange(sk, device=q_chunk_bshd.device)
    scores_chunk.masked_fill_(
        key_positions.view(1, 1, sk) >= seq_lens.view(1, sq, 1), float("-inf")
    )
    return scores_chunk


def _indexer_topk_from_score_chunks(
    q_bshd: Tensor,
    k_bshd: Tensor,
    w_bsh: Tensor,
    seq_lens: Tensor,
    topk_k: int,
    return_topk_scores: bool,
    *,
    starts: Optional[Tensor] = None,
    ends: Optional[Tensor] = None,
    key_positions_i64: Optional[Tensor] = None,
    indexer_ratio: int = _INDEXER_RATIO,
    score_seq_lens: Optional[Tensor] = None,
    bottom_right_key_start: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bshd.size(1)
    if bottom_right_key_start is not None:
        if bottom_right_key_start < 0:
            raise RuntimeError("cuDNN DSA bottom-right key start must be non-negative.")
        if score_seq_lens is not None:
            raise RuntimeError(
                "cuDNN DSA score chunks cannot combine bottom-right key cropping "
                "with explicit score sequence lengths."
            )
    chunk_rows = _indexer_score_chunk_rows(b, sq, sk)
    seq_lens_b = seq_lens.view(b, sq)
    indices_chunks = []
    values_chunks = [] if return_topk_scores else None
    # Loop-invariant across chunks; computed lazily on first global-row chunk and reused.
    global_rows_k_bdk = None
    global_rows_key_positions = None

    for row_start in range(0, sq, chunk_rows):
        row_end = min(row_start + chunk_rows, sq)
        q_chunk = q_bshd[:, row_start:row_end].contiguous()
        w_chunk = w_bsh[:, row_start:row_end].contiguous()
        score_k_bshd = k_bshd
        score_sk = sk
        chunk_topk_k = topk_k
        if bottom_right_key_start is not None:
            score_sk = min(sk, bottom_right_key_start + row_end)
            score_k_bshd = k_bshd[:, :score_sk].contiguous()
            chunk_topk_k = min(topk_k, score_sk)
        score_chunk_seq_lens = (
            None if score_seq_lens is None else score_seq_lens[row_start:row_end].contiguous()
        )
        if bottom_right_key_start is not None:
            scores_chunk = _cudnn_dsa.indexer_forward_wrapper(
                q_chunk, score_k_bshd, w_chunk, ratio=indexer_ratio, sm_scale=_INDEXER_SOFTMAX_SCALE
            )["scores"]
        elif score_seq_lens is None and row_start == 0 and row_end == sq:
            scores_chunk = _cudnn_dsa.indexer_forward_wrapper(
                q_chunk, k_bshd, w_chunk, ratio=indexer_ratio, sm_scale=_INDEXER_SOFTMAX_SCALE
            )["scores"]
        else:
            if global_rows_k_bdk is None:
                global_rows_k_bdk = (
                    k_bshd[:, :, 0, :].to(dtype=torch.float32).transpose(1, 2).contiguous()
                )
                global_rows_key_positions = torch.arange(sk, device=q_bshd.device)
            scores_chunk = _compute_indexer_scores_chunk_with_global_rows(
                q_chunk,
                k_bshd,
                w_chunk,
                row_start=row_start,
                indexer_ratio=indexer_ratio,
                sm_scale=_INDEXER_SOFTMAX_SCALE,
                seq_lens=score_chunk_seq_lens,
                k_bdk=global_rows_k_bdk,
                key_positions=global_rows_key_positions,
            )
        if starts is not None:
            scores_chunk = dsa_masking.apply_starts_ends_mask_to_scores(
                scores_chunk, starts[row_start:row_end], ends[row_start:row_end], key_positions_i64
            )
        scores_flat = scores_chunk.reshape(b * (row_end - row_start), score_sk).contiguous()
        use_tie_break = _use_dense_indexer_topk_tie_break(scores_flat, chunk_topk_k)
        scores_for_topk = (
            _add_indexer_topk_tie_break(scores_flat, inplace=True) if use_tie_break else scores_flat
        )
        chunk_seq_lens = seq_lens_b[:, row_start:row_end].reshape(-1).contiguous()
        tk_result = _indexer_top_k_wrapper_chunked(
            scores_for_topk,
            chunk_seq_lens,
            topk_k=chunk_topk_k,
            return_topk_scores=return_topk_scores,
        )
        topk_indices = tk_result["indices"].view(b, row_end - row_start, chunk_topk_k)
        topk_values = None
        if return_topk_scores:
            topk_values = tk_result["values"]
            if topk_values is None:
                raise RuntimeError("cuDNN indexer_top_k_wrapper did not return values.")
            topk_values = topk_values.view(b, row_end - row_start, chunk_topk_k)
            if use_tie_break:
                topk_values = _remove_indexer_topk_tie_break(topk_values, topk_indices, score_sk)
        topk_indices, topk_values = _pad_topk_result(topk_indices, topk_values, topk_k)
        indices_chunks.append(topk_indices)
        if return_topk_scores:
            values_chunks.append(topk_values)
        del scores_for_topk, scores_flat, scores_chunk

    return (
        torch.cat(indices_chunks, dim=1),
        torch.cat(values_chunks, dim=1) if return_topk_scores else None,
    )


def _pad_topk_result(
    topk_indices: Tensor, topk_scores: Optional[Tensor], target_topk: int
) -> Tuple[Tensor, Optional[Tensor]]:
    """Pad segment-local top-k results to the caller-requested K."""
    pad = target_topk - topk_indices.size(-1)
    if pad <= 0:
        return topk_indices, topk_scores
    index_pad = torch.full(
        (*topk_indices.shape[:-1], pad), -1, dtype=topk_indices.dtype, device=topk_indices.device
    )
    topk_indices = torch.cat((topk_indices, index_pad), dim=-1)
    if topk_scores is not None:
        score_pad = torch.full(
            (*topk_scores.shape[:-1], pad),
            torch.finfo(torch.float32).min,
            dtype=topk_scores.dtype,
            device=topk_scores.device,
        )
        topk_scores = torch.cat((topk_scores, score_pad), dim=-1)
    return topk_indices, topk_scores


def _indexer_topk_multi_packed_cp_thd(
    q_bshd: Tensor,
    k_bshd: Tensor,
    w_bsh: Tensor,
    topk_k: int,
    return_topk_scores: bool,
    starts: Tensor,
    ends: Tensor,
    packed_cu_seqlens_q: Tensor,
    packed_cu_seqlens_k: Tensor,
    packed_max_seqlen_q: int,
    packed_max_seqlen_k: int,
    cp_size: int,
    cp_rank: int,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Run cuDNN's THD indexer on packed CP front/back query segments."""
    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bshd.size(1)
    if b != 1 or cp_size <= 1 or not 0 <= cp_rank < cp_size:
        raise RuntimeError("packed CP cuDNN THD indexer requires b=1 and a valid CP rank")
    if packed_cu_seqlens_q.numel() < 3:
        raise RuntimeError(
            "multi-sequence packed CP cuDNN THD indexer requires at least two sequences"
        )
    if packed_cu_seqlens_q.shape != packed_cu_seqlens_k.shape:
        raise RuntimeError("packed CP cuDNN THD query/key cu_seqlens shapes must match")
    if packed_max_seqlen_q <= 0 or packed_max_seqlen_k <= 0:
        raise RuntimeError("packed CP cuDNN THD indexer requires positive maximum sequence lengths")

    segment_divisor = 2 * cp_size
    if sk % segment_divisor != 0:
        raise RuntimeError(f"packed CP key length must be divisible by {segment_divisor}, got {sk}")

    device = q_bshd.device
    cu_q = packed_cu_seqlens_q.to(device=device, dtype=torch.int64).contiguous()
    cu_k = packed_cu_seqlens_k.to(device=device, dtype=torch.int64).contiguous()
    q_lengths = cu_q[1:] - cu_q[:-1]
    k_lengths = cu_k[1:] - cu_k[:-1]
    q_half = q_lengths // segment_divisor
    k_half = k_lengths // segment_divisor
    segment_q_lengths = torch.stack((q_half, q_half), dim=1).reshape(-1)
    segment_k_lengths = torch.stack(
        ((cp_rank + 1) * k_half, k_lengths - cp_rank * k_half), dim=1
    ).reshape(-1)

    zero_i32 = torch.zeros(1, dtype=torch.int32, device=device)
    segment_cu_q = torch.cat(
        (zero_i32, segment_q_lengths.cumsum(dim=0, dtype=torch.int32))
    ).contiguous()
    segment_cu_k = torch.cat(
        (zero_i32, segment_k_lengths.cumsum(dim=0, dtype=torch.int32))
    ).contiguous()

    segment_key_starts = cu_k[:-1].repeat_interleave(2)
    total_segment_k = sk + sk // segment_divisor
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_k_lengths.numel(), device=device),
        segment_k_lengths,
        output_size=total_segment_k,
    )
    segment_offsets = torch.arange(total_segment_k, device=device, dtype=torch.int64)
    segment_offsets -= torch.repeat_interleave(
        segment_cu_k[:-1].to(dtype=torch.int64), segment_k_lengths, output_size=total_segment_k
    )
    source_indices = segment_key_starts.index_select(0, segment_ids) + segment_offsets
    segmented_k = k_bshd[0].index_select(0, source_indices).contiguous()

    max_segment_q = packed_max_seqlen_q // segment_divisor
    max_k_half = packed_max_seqlen_k // segment_divisor
    max_segment_k = max((cp_rank + 1) * max_k_half, packed_max_seqlen_k - cp_rank * max_k_half)
    scores = _cudnn_dsa.indexer_forward_wrapper(
        q_bshd[0],
        segmented_k,
        w_bsh[0],
        ratio=_INDEXER_RATIO,
        sm_scale=_INDEXER_SOFTMAX_SCALE,
        cu_seqlens_q=segment_cu_q,
        cu_seqlens_k=segment_cu_k,
        max_seqlen_q=max_segment_q,
        max_seqlen_k=max_segment_k,
    )["scores"]

    segment_topk = min(topk_k, max_segment_k)
    local_seq_lens = (ends - starts).clamp(max=max_segment_k).to(torch.int32).contiguous()
    use_tie_break = _use_dense_indexer_topk_tie_break(scores, segment_topk)
    scores_for_topk = _add_indexer_topk_tie_break(scores, inplace=True) if use_tie_break else scores
    tk_result = _indexer_top_k_wrapper_chunked(
        scores_for_topk, local_seq_lens, topk_k=segment_topk, return_topk_scores=return_topk_scores
    )
    topk_indices = tk_result["indices"].view(1, sq, segment_topk)
    topk_scores = None
    if return_topk_scores:
        topk_scores = tk_result["values"]
        if topk_scores is None:
            raise RuntimeError("cuDNN indexer_top_k_wrapper did not return values.")
        topk_scores = topk_scores.view(1, sq, segment_topk)
        if use_tie_break:
            topk_scores = _remove_indexer_topk_tie_break(topk_scores, topk_indices, max_segment_k)

    valid = topk_indices >= 0
    topk_indices = torch.where(
        valid, topk_indices + starts.view(1, sq, 1).to(dtype=topk_indices.dtype), topk_indices
    )
    return _pad_topk_result(topk_indices, topk_scores, topk_k)


def _indexer_topk_single_packed_cp_segments(
    q_bshd: Tensor,
    k_bshd: Tensor,
    w_bsh: Tensor,
    topk_k: int,
    return_topk_scores: bool,
    local_packed_cp_rank: int,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Fast cuDNN top-k for one packed THD sequence split into CP front/back slices."""
    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bshd.size(1)
    local_query_len = local_packed_cp_query_len if local_packed_cp_query_len is not None else sq
    local_query_start = local_packed_cp_query_start
    local_query_end = local_query_start + sq
    if b != 1 or local_query_len % 2 != 0 or _INDEXER_RATIO != 1:
        raise RuntimeError(
            "single packed CP cuDNN fast path requires b=1, even local query length, ratio=1"
        )
    if local_query_start < 0 or local_query_end > local_query_len:
        raise RuntimeError(
            "single packed CP cuDNN fast path received an invalid query slice: "
            f"start={local_query_start}, sq={sq}, local_query_len={local_query_len}"
        )

    half = local_query_len // 2
    if sk == local_query_len:
        segment_specs = ((0, half, half), (half, local_query_len, local_query_len))
    else:
        segment_specs = (
            (0, half, (local_packed_cp_rank + 1) * half),
            (half, local_query_len, sk - local_packed_cp_rank * half),
        )
    indices_chunks = []
    values_chunks = [] if return_topk_scores else None
    for segment_start, segment_end, key_end in segment_specs:
        row_start_global = max(local_query_start, segment_start)
        row_end_global = min(local_query_end, segment_end)
        if row_start_global >= row_end_global:
            continue
        row_start = row_start_global - local_query_start
        row_end = row_end_global - local_query_start
        if key_end <= 0 or key_end > sk:
            raise RuntimeError(
                "single packed CP cuDNN fast path derived invalid key prefix: "
                f"key_end={key_end}, sk={sk}, cp_rank={local_packed_cp_rank}, sq={sq}"
            )
        segment_topk = min(topk_k, key_end)
        key_start = key_end - (segment_end - segment_start)
        rel_start = row_start_global - segment_start
        rel_end = row_end_global - segment_start
        seq_lens = torch.arange(
            key_start + rel_start + 1,
            key_start + rel_end + 1,
            dtype=torch.int32,
            device=q_bshd.device,
        )
        # Crop each score chunk's key prefix so cuDNN bottom-right masking matches
        # these absolute row lengths without materializing scores in PyTorch.
        topk_indices, topk_scores = _indexer_topk_from_score_chunks(
            q_bshd[:, row_start:row_end].contiguous(),
            k_bshd[:, :key_end].contiguous(),
            w_bsh[:, row_start:row_end].contiguous(),
            seq_lens,
            segment_topk,
            return_topk_scores,
            bottom_right_key_start=key_start + rel_start,
        )
        valid = (topk_indices >= 0) & (topk_indices < seq_lens.view(1, -1, 1))
        topk_indices = torch.where(valid, topk_indices, torch.full_like(topk_indices, -1))
        if topk_scores is not None:
            topk_scores = torch.where(
                valid, topk_scores, torch.full_like(topk_scores, torch.finfo(torch.float32).min)
            )
        topk_indices, topk_scores = _pad_topk_result(topk_indices, topk_scores, topk_k)
        indices_chunks.append(topk_indices)
        if return_topk_scores:
            values_chunks.append(topk_scores)

    if not indices_chunks:
        raise RuntimeError(
            "single packed CP cuDNN fast path query slice did not intersect any CP segment"
        )

    return (
        torch.cat(indices_chunks, dim=1),
        torch.cat(values_chunks, dim=1) if return_topk_scores else None,
    )


def _dsa_fwd_flash_mla(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int,
    attn_sink: Tensor,
    topk_length: Tensor,
) -> Tuple[Tensor, Tensor]:
    """DSA-shaped adapter around :func:`flash_mla.flash_mla_sparse_fwd`.

    Accepts flat tensors with global indices; pads ``topk`` to
    the GPU-specific alignment; returns ``(out, lse)``.
    """
    _ensure_flash_mla()

    topk_count = topk_idxs.shape[-1]
    topk_align = _get_topk_alignment()
    topk_padded = round_up_to_nearest_multiple(topk_count, topk_align)
    if topk_padded != topk_count:
        pad_width = topk_padded - topk_count
        topk_idxs = torch.nn.functional.pad(topk_idxs, (0, pad_width), value=-1)

    kv_3d = kv.unsqueeze(1)  # (total_S_kv, 1, D)  h_kv=1
    indices = topk_idxs.unsqueeze(1)  # (total_S_q, 1, topk_padded) h_kv=1

    actual_num_heads = q.size(1)
    padded_num_heads = _get_head_padding(actual_num_heads)
    if padded_num_heads != actual_num_heads:
        q_padded = q.new_zeros((q.size(0), padded_num_heads, q.size(2)))
        q_padded[:, :actual_num_heads, :] = q
        q = q_padded

        attn_sink_padded = attn_sink.new_full((padded_num_heads,), float("-inf"))
        attn_sink_padded[:actual_num_heads] = attn_sink
        attn_sink = attn_sink_padded

    with torch.cuda.nvtx.range("flash_mla_sparse_fwd"):
        out, _max_logits, lse = _flash_mla_sparse_fwd(
            q,
            kv_3d,
            indices,
            softmax_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=0,
        )

    if padded_num_heads != actual_num_heads:
        out = out[:, :actual_num_heads, :]
        lse = lse[:, :actual_num_heads]

    return out, lse


def _ensure_dsa_namespace() -> None:
    """Lazily import the cudnn-frontend DSA namespace."""
    global _cudnn_dsa
    if _cudnn_dsa is not None:
        return
    try:
        from cudnn import DSA as _ns
    except ImportError as e:
        raise ImportError(
            "cudnn-frontend DSA namespace not available. Install with "
            "`pip install nvidia-cudnn-frontend[cutedsl]`."
        ) from e
    _cudnn_dsa = _ns


def _local_to_global_flat(local_idxs: Tensor, batch_size: int) -> Tensor:
    """Convert local per-batch indices to global flat indices.

    Follows the convention used by FlashMLA / SparseAttentionBackward:
    flat row order is SBHD ``row[s * B + b]``; global index is
    ``local * B + b`` for valid entries and ``-1`` otherwise.

    Args:
        local_idxs: ``(b, sq, topk)`` int, values in the local KV range or -1.
        batch_size: ``B``.

    Returns:
        ``(sq*b, topk)`` int32.
    """
    b, sq, topk = local_idxs.shape
    assert b == batch_size

    idxs_sb = local_idxs.permute(1, 0, 2).reshape(sq * b, topk)
    valid = idxs_sb >= 0
    # int32 throughout: the global index must fit int32 for the kernel (skv*b < 2**31), so the
    # int32 product cannot overflow before the result would, and we avoid an int64 allocation.
    batch_ids = torch.arange(sq * b, dtype=torch.int32, device=local_idxs.device) % b
    batch_ids_exp = batch_ids.unsqueeze(1).expand_as(idxs_sb)
    idxs_sb = torch.where(valid, idxs_sb * b + batch_ids_exp, idxs_sb)
    return idxs_sb.int()


def _trailing_positions(t: Tensor) -> Tensor:
    """Return a ``[1, ..., 1, N]`` index vector along the last axis of ``t`` for broadcasting."""
    return torch.arange(t.size(-1), device=t.device).view(*((1,) * (t.ndim - 1)), -1)


def _stable_compaction_order(valid: Tensor) -> Tensor:
    """Return a gather index that moves valid entries left, preserving their order."""
    positions = _trailing_positions(valid)
    sentinel = torch.full_like(positions, valid.size(-1)).expand_as(valid)
    order_key = torch.where(valid, positions.expand_as(valid), sentinel)
    return order_key.argsort(dim=-1)


def _compact_valid_topk_indices(topk_indices: Tensor) -> Tuple[Tensor, Tensor]:
    """Move valid top-K entries to the left so ``topk_length`` is semantically correct."""
    valid = topk_indices >= 0
    topk_length = valid.sum(dim=-1).int()
    order = _stable_compaction_order(valid)
    compacted = torch.gather(topk_indices, dim=-1, index=order)
    compacted_valid = torch.gather(valid, dim=-1, index=order)
    compacted = compacted.masked_fill(~compacted_valid, -1)
    return compacted.contiguous(), topk_length.contiguous()


def _compact_valid_topk_indices_and_scores(
    topk_indices: Tensor, topk_scores: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Move valid top-K entries and their aligned score payload to the left."""
    if topk_indices.shape != topk_scores.shape:
        raise RuntimeError(
            "top-k indices and score payload must have the same shape, "
            f"got {topk_indices.shape} and {topk_scores.shape}."
        )
    valid = (topk_indices >= 0) & torch.isfinite(topk_scores)
    topk_length = valid.sum(dim=-1).int()
    order = _stable_compaction_order(valid)
    compacted_indices = torch.gather(topk_indices, dim=-1, index=order)
    compacted_scores = torch.gather(topk_scores, dim=-1, index=order)
    compacted_valid = torch.gather(valid, dim=-1, index=order)
    compacted_indices = compacted_indices.masked_fill(~compacted_valid, -1)
    compacted_scores = compacted_scores.masked_fill(
        ~compacted_valid, torch.finfo(torch.float32).min
    )
    return compacted_indices.contiguous(), topk_length.contiguous(), compacted_scores.contiguous()


def _sort_valid_topk_indices_by_index(topk_indices: Tensor, topk_length: Tensor, sk: int) -> Tensor:
    """Canonicalize consumed top-K indices while keeping ignored suffix slots invalid."""
    positions = _trailing_positions(topk_indices)
    valid = positions < topk_length.unsqueeze(-1)
    sort_key = torch.where(valid, topk_indices, torch.full_like(topk_indices, sk))
    order = sort_key.argsort(dim=-1)
    sorted_indices = torch.gather(topk_indices, dim=-1, index=order)
    sorted_valid = torch.gather(valid.expand_as(topk_indices), dim=-1, index=order)
    return sorted_indices.masked_fill(~sorted_valid, -1).contiguous()


def _sort_valid_topk_indices_and_scores_by_index(
    topk_indices: Tensor, topk_scores: Tensor, topk_length: Tensor, sk: int
) -> Tuple[Tensor, Tensor]:
    """Sort valid top-K indices and keep the selected score payload aligned."""
    positions = _trailing_positions(topk_indices)
    valid = positions < topk_length.unsqueeze(-1)
    sort_key = torch.where(valid, topk_indices, torch.full_like(topk_indices, sk))
    order = sort_key.argsort(dim=-1)
    sorted_indices = torch.gather(topk_indices, dim=-1, index=order)
    sorted_scores = torch.gather(topk_scores, dim=-1, index=order)
    sorted_valid = torch.gather(valid.expand_as(topk_indices), dim=-1, index=order)
    sorted_indices = sorted_indices.masked_fill(~sorted_valid, -1)
    sorted_scores = sorted_scores.masked_fill(~sorted_valid, torch.finfo(torch.float32).min)
    return sorted_indices.contiguous(), sorted_scores.contiguous()


def _prepare_attention_topk_indices(topk_indices: Tensor, sk: int) -> Tuple[Tensor, Tensor]:
    """Prepare top-K indices for FlashMLA without changing indexer-loss ordering."""
    compacted, topk_length = _compact_valid_topk_indices(topk_indices)
    sorted_indices = _sort_valid_topk_indices_by_index(compacted, topk_length, sk)
    return sorted_indices.int().contiguous(), topk_length.contiguous()


def _topk_in_bounds(
    topk_indices: Tensor,
    starts: Optional[Tensor],
    ends: Optional[Tensor],
    key_positions_i64: Optional[Tensor],
    seq_lens: Tensor,
    use_local_indexer_varlen: bool,
    sk: int,
    b: int,
    sq: int,
) -> Tensor:
    """Mask of top-K slots that are in ``[0, sk)`` and within causal/varlen key bounds."""
    in_range = (topk_indices >= 0) & (topk_indices < sk)
    if starts is not None and use_local_indexer_varlen:
        valid_bounds = (topk_indices >= starts.view(1, sq, 1)) & (
            topk_indices < ends.view(1, sq, 1)
        )
    elif starts is not None:
        safe_indices = topk_indices.clamp(min=0, max=sk - 1).long()
        selected_positions = key_positions_i64.index_select(0, safe_indices.reshape(-1))
        selected_positions = selected_positions.view_as(topk_indices)
        valid_bounds = (selected_positions >= starts.view(1, sq, 1)) & (
            selected_positions < ends.view(1, sq, 1)
        )
    else:
        valid_bounds = topk_indices < seq_lens.view(b, sq, 1)
    return in_range & valid_bounds


def _indexer_topk_bshd(
    q_bshd: Tensor,
    k_bsd: Tensor,
    w_bsh: Tensor,
    topk: int,
    varlen_starts: Optional[Tensor] = None,
    varlen_ends: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    return_scores: bool = True,
    return_topk_scores: bool = False,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_cu_seqlens_q: Optional[Tensor] = None,
    packed_cu_seqlens_k: Optional[Tensor] = None,
    packed_max_seqlen_q: Optional[int] = None,
    packed_max_seqlen_k: Optional[int] = None,
    packed_cp_size: int = 1,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """BSHD-layout indexer scoring and top-K selection.

    Args:
        q_bshd: ``(b, sq, idx_nh, idx_hd)`` bf16, C-contiguous.
        k_bsd:  ``(b, sk, idx_hd)`` bf16, C-contiguous.
        w_bsh:  ``(b, sq, idx_nh)`` bf16, C-contiguous raw weights.
        topk:   number of top-K indices to return per query.

    Returns:
        ``(topk_indices, topk_length, score_payload)`` where:

        * ``topk_indices``: ``(b, sq, topk)`` int32, invalid slots ``-1``.
        * ``topk_length``:  ``(b, sq)`` int32, per-row valid count.
        * ``score_payload``: full scores when ``return_scores=True``, top-k
          scores when ``return_topk_scores=True``, otherwise ``None``.

    ``use_local_indexer_varlen`` is an optimized packed-CP path for a single
    packed sequence. It scores the local query rows directly instead of
    scattering them to their global query positions first, then applies the
    explicit starts/ends mask before top-k selection.
    """
    _ensure_dsa_namespace()

    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bsd.shape[1]
    device = q_bshd.device
    topk_k = min(topk, sk)
    topk_indices = None
    topk_scores = None

    starts = ends = key_positions_i64 = None
    explicit_key_positions = key_positions is not None
    if varlen_starts is not None:
        starts, ends, key_positions_i64 = dsa_masking.normalize_varlen_bounds(
            mask=None,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            sk=sk,
            device=device,
        )
        if not use_local_indexer_varlen and explicit_key_positions:
            raise RuntimeError("cuDNN fused DSA varlen path expects identity key positions")

    k_bshd = k_bsd.unsqueeze(2)  # (b, sk, 1, idx_hd)

    if starts is None:
        q_idx = torch.arange(sq, device=device)
        seq_lens = _causal_seq_lens(q_idx, _INDEXER_RATIO, sk).to(torch.int32).repeat(b)
        if not return_scores:
            topk_indices, topk_scores = _indexer_topk_from_score_chunks(
                q_bshd, k_bshd, w_bsh, seq_lens, topk_k, return_topk_scores
            )
        else:
            scores = _cudnn_dsa.indexer_forward_wrapper(
                q_bshd, k_bshd, w_bsh, ratio=_INDEXER_RATIO, sm_scale=_INDEXER_SOFTMAX_SCALE
            )[
                "scores"
            ]  # (b, sq, sk) fp32, -inf on masked positions
    else:
        seq_lens = ends.clamp(max=sk).to(torch.int32).repeat(b)
        if not return_scores and use_local_indexer_varlen and single_packed_thd_sequence:
            topk_indices, topk_scores = _indexer_topk_single_packed_cp_segments(
                q_bshd,
                k_bshd,
                w_bsh,
                topk_k,
                return_topk_scores,
                local_packed_cp_rank,
                local_packed_cp_query_start,
                local_packed_cp_query_len,
            )
        elif (
            not return_scores
            and use_local_indexer_varlen
            and packed_cu_seqlens_q is not None
            and packed_cu_seqlens_k is not None
            and packed_max_seqlen_q is not None
            and packed_max_seqlen_k is not None
        ):
            topk_indices, topk_scores = _indexer_topk_multi_packed_cp_thd(
                q_bshd,
                k_bshd,
                w_bsh,
                topk_k,
                return_topk_scores,
                starts,
                ends,
                packed_cu_seqlens_q,
                packed_cu_seqlens_k,
                packed_max_seqlen_q,
                packed_max_seqlen_k,
                packed_cp_size,
                local_packed_cp_rank,
            )
        elif not return_scores:
            topk_indices, topk_scores = _indexer_topk_from_score_chunks(
                q_bshd,
                k_bshd,
                w_bsh,
                seq_lens,
                topk_k,
                return_topk_scores,
                starts=starts,
                ends=ends,
                key_positions_i64=key_positions_i64,
                indexer_ratio=_INDEXER_RATIO,
                score_seq_lens=ends,
            )
        else:
            # This branch runs only when ``starts`` is set; normalize_varlen_bounds then
            # guarantees ``ends`` is non-None, so the global-row score path always applies.
            scores = _compute_indexer_scores_chunk_with_global_rows(
                q_bshd,
                k_bshd,
                w_bsh,
                row_start=0,
                indexer_ratio=_INDEXER_RATIO,
                sm_scale=_INDEXER_SOFTMAX_SCALE,
                seq_lens=ends,
            )
            scores = dsa_masking.apply_starts_ends_mask_to_scores(
                scores, starts, ends, key_positions_i64
            )

    # Top-K selection via the TRT-LLM CuTe-DSL radix kernel.
    if topk_indices is None:
        n_rows = b * sq
        scores_flat = scores.reshape(n_rows, sk).contiguous()
        use_tie_break = _use_dense_indexer_topk_tie_break(scores_flat, topk_k)
        scores_for_topk = (
            _add_indexer_topk_tie_break(scores_flat, inplace=not return_scores)
            if use_tie_break
            else scores_flat
        )

        tk_result = _indexer_top_k_wrapper_chunked(
            scores_for_topk, seq_lens, topk_k=topk_k, return_topk_scores=return_topk_scores
        )
        del scores_for_topk, scores_flat
        if not return_scores:
            del scores
        topk_indices = tk_result["indices"].view(b, sq, topk_k)
        if return_topk_scores:
            topk_scores = tk_result["values"]
            if topk_scores is None:
                raise RuntimeError("cuDNN indexer_top_k_wrapper did not return values.")
            topk_scores = topk_scores.view(b, sq, topk_k)
            if use_tie_break:
                topk_scores = _remove_indexer_topk_tie_break(topk_scores, topk_indices, sk)

    if return_topk_scores:
        topk_scores = topk_scores.to(dtype=torch.float32)
        valid = _topk_in_bounds(
            topk_indices,
            starts,
            ends,
            key_positions_i64,
            seq_lens,
            use_local_indexer_varlen,
            sk,
            b,
            sq,
        ) & torch.isfinite(topk_scores)
        topk_indices = torch.where(valid, topk_indices, torch.full_like(topk_indices, -1))
        topk_scores = torch.where(
            valid, topk_scores, torch.full_like(topk_scores, torch.finfo(torch.float32).min)
        )
    elif return_scores:
        in_range = (topk_indices >= 0) & (topk_indices < sk)
        gather_indices = topk_indices.clamp(min=0, max=sk - 1).long()
        gathered_scores = torch.gather(scores, dim=2, index=gather_indices)
        topk_indices = torch.where(
            in_range & torch.isfinite(gathered_scores),
            topk_indices,
            torch.full_like(topk_indices, -1),
        )
    else:
        valid = _topk_in_bounds(
            topk_indices,
            starts,
            ends,
            key_positions_i64,
            seq_lens,
            use_local_indexer_varlen,
            sk,
            b,
            sq,
        )
        topk_indices = torch.where(valid, topk_indices, torch.full_like(topk_indices, -1))

    topk_indices, topk_scores = _pad_topk_result(topk_indices, topk_scores, topk)

    if return_topk_scores:
        topk_indices, topk_length, topk_scores = _compact_valid_topk_indices_and_scores(
            topk_indices, topk_scores
        )
        return topk_indices.int().contiguous(), topk_length, topk_scores

    if return_scores:
        topk_indices, topk_length = _compact_valid_topk_indices(topk_indices)
        return topk_indices.int(), topk_length, scores

    # FlashMLA consumes only the prefix described by topk_length. Keep that
    # prefix deterministic across the global-scatter and local packed-CP paths.
    topk_indices, topk_length = _compact_valid_topk_indices(topk_indices)
    topk_indices = _sort_valid_topk_indices_by_index(topk_indices, topk_length, sk)
    return topk_indices.int().contiguous(), topk_length, None


def _sbhd_to_bshd_indexer_inputs(
    q_indexer: Tensor, k_indexer: Tensor, weights: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Permute the indexer inputs SBHD→BSHD once."""
    q_bshd = q_indexer.permute(1, 0, 2, 3).contiguous()
    k_bsd = k_indexer.permute(1, 0, 2).contiguous()
    w_bsh = weights.permute(1, 0, 2).contiguous()
    return q_bshd, k_bsd, w_bsh


def run_fused_qk_topk(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    starts: Tensor,
    ends: Tensor,
    block_size: int,
    use_relu: bool = True,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional["PackedSeqParams"] = None,
    cp_size: int = 1,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run the cuDNN fused indexer and return top-k indices for split DSA."""
    _assert_supported_indexer_scoring(use_relu)
    del block_size
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        return None
    if q.size(0) != weights.size(0) or q.size(1) != k.size(1) or q.size(1) != weights.size(1):
        return None
    if q.size(2) != weights.size(2) or q.size(3) != k.size(2):
        return None
    if starts is None or ends is None:
        return None

    packed_cu_seqlens_q, packed_cu_seqlens_k, packed_max_seqlen_q, packed_max_seqlen_k = (
        _get_multi_packed_cp_thd_metadata(
            use_local_indexer_varlen, packed_seq_params, single_packed_thd_sequence, cp_size
        )
    )
    q_bshd, k_bsd, w_bsh = _sbhd_to_bshd_indexer_inputs(q, k, weights)
    topk_indices, topk_length, _score_payload = _indexer_topk_bshd(
        q_bshd,
        k_bsd,
        w_bsh,
        index_topk,
        varlen_starts=starts,
        varlen_ends=ends,
        key_positions=None,
        return_scores=False,
        return_topk_scores=False,
        use_local_indexer_varlen=use_local_indexer_varlen,
        single_packed_thd_sequence=single_packed_thd_sequence,
        local_packed_cp_rank=local_packed_cp_rank,
        local_packed_cp_query_start=local_packed_cp_query_start,
        local_packed_cp_query_len=local_packed_cp_query_len,
        packed_cu_seqlens_q=packed_cu_seqlens_q,
        packed_cu_seqlens_k=packed_cu_seqlens_k,
        packed_max_seqlen_q=packed_max_seqlen_q,
        packed_max_seqlen_k=packed_max_seqlen_k,
        packed_cp_size=cp_size,
    )
    return topk_indices, topk_length


def run_fused_qk_topk_with_loss(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    starts: Tensor,
    ends: Tensor,
    block_size: int,
    query: Tensor,
    key: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    pg_collection: "ProcessGroupCollection",
    query_valid_rows: Optional[Tensor] = None,
    calculate_per_token_loss: bool = False,
    use_relu: bool = True,
    config: Optional["TransformerConfig"] = None,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional["PackedSeqParams"] = None,
    cp_size: int = 1,
) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """Run cuDNN fused indexer and sparse indexer loss for split DSA."""
    del block_size
    tp_group = getattr(pg_collection, "tp", None)
    _assert_supported_indexer_scoring(use_relu)
    if loss_coeff <= 0:
        return None
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        return None
    if query.ndim != 4 or key.ndim != 4 or key.size(2) != 1:
        return None
    if q.size(0) != weights.size(0) or q.size(1) != k.size(1) or q.size(1) != weights.size(1):
        return None
    if q.size(2) != weights.size(2) or q.size(3) != k.size(2):
        return None
    if query.size(0) != q.size(0) or query.size(1) != q.size(1) or key.size(1) != q.size(1):
        return None
    if query.size(3) != key.size(3):
        return None
    if starts is None or ends is None:
        return None

    latent_v_channels = int(getattr(config, "kv_lora_rank", 0) or 0)
    if latent_v_channels <= 0:
        return None
    if not _supports_flash_mla_value_layout(key.size(-1), latent_v_channels):
        return None
    if not _flash_mla_supports_head_count(query):
        return None

    packed_cu_seqlens_q, packed_cu_seqlens_k, packed_max_seqlen_q, packed_max_seqlen_k = (
        _get_multi_packed_cp_thd_metadata(
            use_local_indexer_varlen, packed_seq_params, single_packed_thd_sequence, cp_size
        )
    )
    return FusedQKTopKWithSparseLossFunc.apply(
        q.contiguous(),
        k.contiguous(),
        weights.contiguous(),
        query.contiguous(),
        key.squeeze(2).contiguous(),
        index_topk,
        softmax_scale,
        loss_coeff,
        calculate_per_token_loss,
        latent_v_channels,
        starts.contiguous(),
        ends.contiguous(),
        query_valid_rows,
        use_local_indexer_varlen,
        single_packed_thd_sequence,
        local_packed_cp_rank,
        local_packed_cp_query_start,
        local_packed_cp_query_len,
        packed_cu_seqlens_q,
        packed_cu_seqlens_k,
        packed_max_seqlen_q,
        packed_max_seqlen_k,
        cp_size,
        tp_group,
    )


def _all_reduce_tp_target(target: Tensor, tp_group) -> Tensor:
    """Sum a local attention target across TP ranks when TP is enabled."""
    if get_pg_size(tp_group) > 1:
        torch.distributed.all_reduce(target, group=tp_group)
    return target


def _pad_indexer_heads_for_backward(
    q_bshd: Tensor, w_bsh: Tensor, min_heads: int = 64
) -> Tuple[Tensor, Tensor, int]:
    """Pad indexer heads for cuDNN backward kernels that require at least 64 heads."""
    actual_heads = q_bshd.size(2)
    if actual_heads >= min_heads:
        return q_bshd, w_bsh, actual_heads

    q_padded = q_bshd.new_zeros((q_bshd.size(0), q_bshd.size(1), min_heads, q_bshd.size(3)))
    q_padded[:, :, :actual_heads, :] = q_bshd

    w_padded = w_bsh.new_zeros((w_bsh.size(0), w_bsh.size(1), min_heads))
    w_padded[:, :, :actual_heads] = w_bsh
    return q_padded.contiguous(), w_padded.contiguous(), actual_heads


def _slice_indexer_backward_head_grads(
    grad_q_bshd: Tensor, grad_w_bsh: Tensor, actual_heads: int
) -> Tuple[Tensor, Tensor]:
    """Slice padded indexer head gradients back to the real indexer head count."""
    if grad_q_bshd.size(2) != actual_heads:
        grad_q_bshd = grad_q_bshd[:, :, :actual_heads, :].contiguous()
    if grad_w_bsh.size(2) != actual_heads:
        grad_w_bsh = grad_w_bsh[:, :, :actual_heads].contiguous()
    return grad_q_bshd, grad_w_bsh


def _pad_sparse_backward_topk(
    attn_score: Tensor, index_score: Tensor, topk_indices: Tensor, block_size: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad sparse indexer-backward top-k tensors to cuDNN's block-size multiple."""
    # The backward wrapper has no separate top-k length; zero-score slots still need valid indices.
    topk_indices = topk_indices.clamp_min(0)
    topk = topk_indices.size(-1)
    padded_topk = round_up_to_nearest_multiple(topk, block_size)
    if padded_topk == topk:
        return attn_score, index_score, topk_indices

    pad_width = padded_topk - topk
    attn_score = torch.nn.functional.pad(attn_score, (0, pad_width), value=0.0)
    index_score = torch.nn.functional.pad(index_score, (0, pad_width), value=0.0)
    topk_indices = torch.nn.functional.pad(topk_indices, (0, pad_width), value=0)
    return attn_score.contiguous(), index_score.contiguous(), topk_indices.contiguous()


def _pad_attn_target_heads(
    q_attn_bshd: Tensor, lse: Tensor, *, min_heads: int = 8, head_multiple: int = 8
) -> Tuple[Tensor, Tensor, int]:
    """Pad local query heads to cuDNN sparse-score-recompute MMA constraints."""
    actual_heads = q_attn_bshd.size(2)
    padded_heads = max(min_heads, round_up_to_nearest_multiple(actual_heads, head_multiple))
    if padded_heads == actual_heads:
        return q_attn_bshd, lse, actual_heads

    q_padded = q_attn_bshd.new_zeros(
        (q_attn_bshd.size(0), q_attn_bshd.size(1), padded_heads, q_attn_bshd.size(3))
    )
    q_padded[:, :, :actual_heads, :] = q_attn_bshd

    lse_padded = lse.new_full((lse.size(0), lse.size(1), padded_heads), float("inf"))
    lse_padded[:, :, :actual_heads] = lse
    return q_padded.contiguous(), lse_padded.contiguous(), padded_heads


def _compute_attn_target(
    q_attn_bshd: Tensor,
    k_attn_bsd: Tensor,
    lse: Tensor,
    topk_indices: Tensor,
    topk_length: Optional[Tensor],
    softmax_scale: float,
    qhead_per_kv_head: int,
) -> Tensor:
    """Compute sparse indexer-loss target with cuDNN Frontend DSA.

    ``topk_indices`` carries invalid suffix entries as ``-1`` and
    ``topk_length`` marks the consumed compact prefix for each row.
    """
    _ensure_dsa_namespace()
    q_attn_bshd, lse, qhead_per_kv_head = _pad_attn_target_heads(q_attn_bshd, lse)
    kwargs = {"qhead_per_kv_head": qhead_per_kv_head, "topk_indices_global": False}
    if topk_length is not None:
        kwargs["topk_length"] = topk_length.to(dtype=torch.int32, device=topk_indices.device)
    result = _cudnn_dsa.sparse_attn_score_recompute_wrapper(
        q_attn_bshd,
        k_attn_bsd,
        lse.contiguous(),
        topk_indices.contiguous(),
        softmax_scale,
        **kwargs,
    )
    return result["target"].contiguous()


def _dense_attn_lse_chunk_rows(b: int, qhead_per_kv_head: int, sk: int, sq: int) -> int:
    """Return a query chunk size that bounds dense attention LSE score scratch."""
    score_bytes_per_row = (
        max(1, b) * max(1, qhead_per_kv_head) * max(1, sk) * torch.finfo(torch.float32).bits // 8
    )
    return _bytes_to_chunk_rows(sq, score_bytes_per_row, _DENSE_ATTN_LSE_CHUNK_MAX_BYTES, 1)


def _compute_dense_attn_lse(
    q_attn_bshd: Tensor, k_attn_bshd: Tensor, softmax_scale: float, qhead_per_kv_head: int
) -> Tensor:
    """Compute full-KV per-query-head attention LSE for dense indexer loss."""
    b, sq, num_heads, _d = q_attn_bshd.shape
    kb, sk, num_kv_heads, _kd = k_attn_bshd.shape
    if kb != b:
        raise RuntimeError(f"attention Q/K batch sizes must match, got {b} and {kb}.")
    if num_heads != num_kv_heads * qhead_per_kv_head:
        raise RuntimeError(
            "query-head count must equal key-value heads times qhead_per_kv_head, "
            f"got h_q={num_heads}, h_kv={num_kv_heads}, qhead_per_kv_head={qhead_per_kv_head}."
        )

    lse = torch.empty((b, sq, num_heads), device=q_attn_bshd.device, dtype=torch.float32)
    key_positions = torch.arange(sk, device=q_attn_bshd.device).view(1, sk)
    q_positions = torch.arange(sq, device=q_attn_bshd.device)
    seq_lens = _causal_seq_lens(q_positions, _INDEXER_RATIO, sk)
    chunk_rows = _dense_attn_lse_chunk_rows(b, qhead_per_kv_head, sk, sq)

    for kv_head in range(num_kv_heads):
        head_start = kv_head * qhead_per_kv_head
        head_end = head_start + qhead_per_kv_head
        k_group = k_attn_bshd[:, :, kv_head, :].float()
        for q_start in range(0, sq, chunk_rows):
            q_end = min(q_start + chunk_rows, sq)
            q_group = q_attn_bshd[:, q_start:q_end, head_start:head_end, :].float()
            scores = torch.einsum("bqhd,bkd->bqhk", q_group, k_group) * softmax_scale
            valid = key_positions < seq_lens[q_start:q_end].view(-1, 1)
            scores.masked_fill_(~valid.view(1, q_end - q_start, 1, sk), float("-inf"))
            lse[:, q_start:q_end, head_start:head_end] = torch.logsumexp(scores, dim=-1)

    return lse.contiguous()


def _kl_loss_from_target_predict(
    target: Tensor,
    predict_log_probs: Tensor,
    loss_coeff: float,
    query_valid_rows: Optional[Tensor] = None,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) over selected top-K entries.

    Invalid top-K slots must already be zeroed in ``target`` and ``predict_log_probs``.
    """
    return dsa_indexer_loss.indexer_loss_from_target(
        target,
        predict_log_probs,
        loss_coeff,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
    )


def _compute_dense_attn_score(
    q_attn_bshd: Tensor,
    k_attn_bshd: Tensor,
    lse: Tensor,
    qhead_per_kv_head: int,
    softmax_scale: float,
) -> Tuple[Tensor, Tensor]:
    """Dense attention score forward over the full ``S_k`` axis.

    Wraps :attr:`cudnn.DSA.dense_attn_score_recompute_wrapper`. Returns
    ``(out, denom)`` where

    * ``out``   : ``(B, S_q, S_k)`` fp32, the head-summed unnormalized
      attention probability ``S[b,q,k] = sum_h exp(Q_h · K_k^T · scale - LSE[b,q,h])``
      with ``ratio`` causal mask applied.
    * ``denom`` : ``(B, S_q)`` fp32, the L1-norm denom ``sum_k S[b,q,:]``.
      ``target = out / denom[..., None]`` is the L1-normalized
      head-summed attention distribution.
    """
    _ensure_dsa_namespace()
    result = _cudnn_dsa.dense_attn_score_recompute_wrapper(
        q_attn_bshd,
        k_attn_bshd,
        lse,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=_INDEXER_RATIO,
    )
    return result["out"], result["denom"]


def _kl_loss_from_dense_scores(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) over the **full** KV axis, averaged over ``(B, S_q)``.

    Derives ``target = attn_score / attn_l1norm`` (L1-normalised, matches
    ``compute_dsa_indexer_loss``'s ``attention_scores / sum`` step) and
    ``log_predict = index_score - index_lse`` (LSE-normalised log-softmax),
    then computes ``KL = sum_k target * (log target - log predict)`` and
    scales by ``loss_coeff``.

    Rows where the kernel's ``ratio`` causal mask leaves no valid KV
    position have ``attn_l1norm <= 0`` (L1) or ``index_lse == -inf``
    (LSE); those rows contribute 0 to the loss — the same ``row_valid``
    semantics as the reference ``compute_dsa_indexer_loss``.
    """
    eps = _KL_EPS
    # row_valid: rows with at least one un-masked KV position.
    row_valid = (attn_l1norm > eps) & torch.isfinite(index_lse)

    # Safe denoms: replace invalid rows with a finite value so target /
    # log-predict don't produce NaN; the row mask zeroes their KL below.
    safe_l1 = attn_l1norm.clamp(min=eps)
    safe_lse = torch.where(row_valid, index_lse, torch.zeros_like(index_lse))

    target = attn_score / safe_l1.unsqueeze(-1)
    # Per-position validity: the indexer-score kernel emits -inf at
    # ratio-masked positions; those contribute 0 to KL by the
    # ``0 · log(0/p) = 0`` convention. Without this gate, the eps-clamp
    # on target makes the term ``eps · (log eps - (-inf)) = +inf``.
    position_valid = torch.isfinite(index_score)
    safe_index_score = torch.where(position_valid, index_score, torch.zeros_like(index_score))
    log_predict = safe_index_score - safe_lse.unsqueeze(-1)

    kl_terms = target * (torch.log(target.clamp_min(eps)) - log_predict)
    kl_terms = torch.where(position_valid, kl_terms, torch.zeros_like(kl_terms))
    kl_per_row = kl_terms.sum(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


def _compute_sparse_indexer_loss_and_grads(
    *,
    q_idx_bshd: Tensor,
    k_idx_bsd: Tensor,
    w_bsh: Tensor,
    topk_indices_cmp: Tensor,
    topk_length_cmp: Tensor,
    indexer_score_payload: Tensor,
    query: Tensor,
    kv_full: Tensor,
    lse: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    query_valid_rows: Optional[Tensor],
    calculate_per_token_loss: bool,
    tp_group,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute sparse indexer KL loss and precomputed indexer gradients."""
    sq, b, num_heads, _d = query.shape
    skv = kv_full.shape[0]

    q_attn_bshd = query.detach().permute(1, 0, 2, 3).contiguous()
    k_attn_compressed_bsd = kv_full.detach().permute(1, 0, 2).contiguous()
    lse_bsqh = lse.reshape(sq, b, num_heads).permute(1, 0, 2)
    indexer_loss_coeff = loss_coeff * (b * sq) if calculate_per_token_loss else loss_coeff
    unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)
    if query_valid_rows is not None and not calculate_per_token_loss:
        valid_row_count = query_valid_rows.sum().to(dtype=torch.float32).clamp_min(1.0)
        unit_grad_loss = unit_grad_loss * ((b * sq) / valid_row_count)

    q_idx_bshd_for_bwd, w_bsh_for_bwd, actual_indexer_heads = _pad_indexer_heads_for_backward(
        q_idx_bshd, w_bsh
    )

    if get_pg_size(tp_group) > 1:
        # TP ranks own the same query rows but different attention heads. Canonicalize
        # selected-key slots before the positional target all-reduce.
        topk_indices_cmp, indexer_score_payload = _sort_valid_topk_indices_and_scores_by_index(
            topk_indices_cmp, indexer_score_payload, topk_length_cmp, skv
        )
    positions = _trailing_positions(topk_indices_cmp)
    valid_positions = (positions < topk_length_cmp.unsqueeze(-1)) & (topk_indices_cmp >= 0)
    if query_valid_rows is not None:
        valid_positions = valid_positions & query_valid_rows.unsqueeze(-1)
    predict_log_probs = dsa_masking.masked_log_softmax(
        indexer_score_payload, valid_positions, dim=-1
    )
    predict = predict_log_probs.exp().masked_fill(~valid_positions, 0.0)

    target = _compute_attn_target(
        q_attn_bshd,
        k_attn_compressed_bsd,
        lse_bsqh,
        topk_indices_cmp,
        topk_length_cmp,
        softmax_scale,
        qhead_per_kv_head=num_heads,
    )
    target.masked_fill_(~valid_positions, 0.0)
    # Attention heads are sharded across TP ranks. Match native DSA by
    # combining each rank's attention target before final row normalization.
    _all_reduce_tp_target(target, tp_group)
    target = dsa_indexer_loss.normalize_indexer_target(target)
    target.masked_fill_(~valid_positions, 0.0)
    indexer_loss = _kl_loss_from_target_predict(
        target, predict_log_probs, loss_coeff, query_valid_rows, calculate_per_token_loss
    )

    topk_indices_for_bwd = topk_indices_cmp
    if b > 1:
        batch_offsets = (
            torch.arange(b, device=query.device, dtype=topk_indices_cmp.dtype)
            .view(b, 1, 1)
            .mul(skv)
        )
        topk_indices_for_bwd = topk_indices_for_bwd + batch_offsets
    topk_indices_for_bwd = topk_indices_for_bwd.masked_fill(~valid_positions, 0)
    attn_score_for_bwd = target
    # The cuDNN sparse indexer-backward score-grad kernel gates the KL gradient
    # on positive predicted probabilities. Preserve the mathematically correct
    # ``predict - target`` gradient for underflowed softmax entries by keeping
    # selected probabilities nonzero before handing them to cuDNN.
    index_score_for_bwd = predict.clamp_min(_CLIP_PROB_MIN)
    block_i = 128
    attn_score_for_bwd, index_score_for_bwd, topk_indices_for_bwd = _pad_sparse_backward_topk(
        attn_score_for_bwd, index_score_for_bwd, topk_indices_for_bwd, block_i
    )
    ig = _cudnn_dsa.indexer_backward_wrapper(
        q_idx_bshd_for_bwd,
        w_bsh_for_bwd,
        k_idx_bsd,
        attn_score_for_bwd,
        index_score_for_bwd,
        topk_indices_for_bwd,
        sm_scale=_INDEXER_SOFTMAX_SCALE,
        loss_coeff=indexer_loss_coeff,
        grad_loss=unit_grad_loss,
        block_I=block_i,
    )

    grad_q_bshd, grad_w_bsh = _slice_indexer_backward_head_grads(
        ig["d_index_q"], ig["d_weights"], actual_indexer_heads
    )
    precomputed_grad_q_indexer = grad_q_bshd.permute(1, 0, 2, 3).contiguous()
    precomputed_grad_k_indexer = ig["d_index_k"].permute(1, 0, 2).contiguous()
    precomputed_grad_weights = grad_w_bsh.permute(1, 0, 2).contiguous()
    return (
        indexer_loss,
        precomputed_grad_q_indexer,
        precomputed_grad_k_indexer,
        precomputed_grad_weights,
    )


def _compute_dense_indexer_loss_and_grads(
    *,
    query: Tensor,
    kv_full: Tensor,
    q_idx_bshd: Tensor,
    k_idx_bsd: Tensor,
    w_bsh: Tensor,
    indexer_score_payload: Optional[Tensor],
    softmax_scale: float,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    tp_group,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Dense (full-KV) indexer KL loss and precomputed indexer gradients.

    Sibling of :func:`_compute_sparse_indexer_loss_and_grads` for the non-sparse-loss path.
    """
    sq, b, num_heads, _d = query.shape
    # Attention-path tensors are detached; the KL loss trains only the indexer.
    q_attn_bshd = query.detach().permute(1, 0, 2, 3).contiguous()
    k_attn_compressed_bsd = kv_full.detach().permute(1, 0, 2).contiguous()
    k_attn_bshd = k_attn_compressed_bsd.unsqueeze(2)
    dense_lse_bsqh = _compute_dense_attn_lse(
        q_attn_bshd, k_attn_bshd, softmax_scale, qhead_per_kv_head=num_heads
    )
    indexer_loss_coeff = loss_coeff * (b * sq) if calculate_per_token_loss else loss_coeff
    unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)
    q_idx_bshd_for_bwd, w_bsh_for_bwd, actual_indexer_heads = _pad_indexer_heads_for_backward(
        q_idx_bshd, w_bsh
    )
    index_score = indexer_score_payload
    if index_score is None:
        raise RuntimeError("cuDNN dense indexer loss requires full indexer scores.")
    index_lse = torch.logsumexp(index_score, dim=-1)
    attn_score, attn_l1norm = _compute_dense_attn_score(
        q_attn_bshd,
        k_attn_bshd,
        dense_lse_bsqh,
        qhead_per_kv_head=num_heads,
        softmax_scale=softmax_scale,
    )
    _all_reduce_tp_target(attn_score, tp_group)
    _all_reduce_tp_target(attn_l1norm, tp_group)
    indexer_loss = _kl_loss_from_dense_scores(
        attn_score, attn_l1norm, index_score, index_lse, loss_coeff, calculate_per_token_loss
    )
    attn_score_for_bwd = attn_score.clone()
    index_score_for_bwd = index_score.clone()
    ig = _cudnn_dsa.dense_indexer_backward_wrapper(
        q_idx_bshd_for_bwd,
        w_bsh_for_bwd,
        k_idx_bsd,
        attn_score_for_bwd,
        attn_l1norm,
        index_score_for_bwd,
        index_lse,
        sm_scale=_INDEXER_SOFTMAX_SCALE,
        loss_coeff=indexer_loss_coeff,
        grad_loss=unit_grad_loss,
        ratio=_INDEXER_RATIO,
        block_I=128,
    )
    # BSHD -> SBHD (match input layout).
    grad_q_bshd, grad_w_bsh = _slice_indexer_backward_head_grads(
        ig["d_index_q"], ig["d_weights"], actual_indexer_heads
    )
    precomputed_grad_q_indexer = grad_q_bshd.permute(1, 0, 2, 3).contiguous()
    precomputed_grad_k_indexer = ig["d_index_k"].permute(1, 0, 2).contiguous()
    precomputed_grad_weights = grad_w_bsh.permute(1, 0, 2).contiguous()
    return (
        indexer_loss,
        precomputed_grad_q_indexer,
        precomputed_grad_k_indexer,
        precomputed_grad_weights,
    )


def _run_sparse_attention_forward(
    query: Tensor,
    kv_full: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    d_v: int,
    topk_length: Optional[Tensor] = None,
    sanitize_topk_for_backward_in_place: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run FlashMLA sparse attention from local DSA top-k indices."""
    sq, b, num_heads, d = query.shape
    skv = kv_full.shape[0]

    attn_sink = torch.full((num_heads,), float("-inf"), dtype=torch.float32, device=query.device)
    if topk_length is None:
        topk_indices_attn, topk_length_attn = _prepare_attention_topk_indices(topk_indices, skv)
    else:
        topk_indices_attn = topk_indices.int().contiguous()
        topk_length_attn = topk_length.to(dtype=torch.int32, device=query.device).contiguous()
    if b == 1:
        global_idxs = topk_indices_attn.reshape(sq * b, topk_indices_attn.size(-1))
    else:
        global_idxs = _local_to_global_flat(topk_indices_attn, b)
    topk_length_flat = topk_length_attn.permute(1, 0).reshape(-1)

    q_flat = query.reshape(sq * b, num_heads, d)
    kv_flat = kv_full.reshape(skv * b, kv_full.size(-1))
    out_flat, lse = _dsa_fwd_flash_mla(
        q_flat,
        kv_flat,
        global_idxs,
        softmax_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length_flat,
    )

    # ``topk_indices_attn`` is compacted by construction, so invalid suffix
    # entries are already ``-1``. Backward only needs non-negative placeholders
    # for ignored slots. The full-fused path owns its top-k tensor, so it can
    # sanitize in-place and avoid a large per-layer allocation.
    if sanitize_topk_for_backward_in_place:
        global_idxs.clamp_min_(0)
        global_idxs_for_bwd = global_idxs
    else:
        global_idxs_for_bwd = global_idxs.clamp_min(0).contiguous()
    return out_flat, lse, q_flat, kv_flat, attn_sink, global_idxs_for_bwd, topk_length_flat


def _run_sparse_attention_backward(
    *,
    q_flat: Tensor,
    kv_flat: Tensor,
    attn_sink: Tensor,
    global_idxs: Tensor,
    out_flat: Tensor,
    lse: Tensor,
    topk_length: Tensor,
    softmax_scale: float,
    sq: int,
    b: int,
    num_heads: int,
    d: int,
    skv: int,
    grad_output: Tensor,
    all_rows_nonempty: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Run sparse attention backward, skipping rows that have no valid top-k entries."""
    d_v = out_flat.shape[-1]
    dO_flat = grad_output.reshape(sq * b, num_heads, d_v)

    bwd_q_flat = q_flat
    bwd_out_flat = out_flat
    bwd_dO_flat = dO_flat
    bwd_lse = lse
    bwd_attn_sink = attn_sink

    _ensure_dsa_namespace()
    if all_rows_nonempty:
        attn_bwd = _cudnn_dsa.sparse_attention_backward_wrapper(
            bwd_q_flat,
            kv_flat,
            bwd_out_flat,
            bwd_dO_flat,
            bwd_lse,
            bwd_attn_sink,
            global_idxs,
            softmax_scale=softmax_scale,
            topk_length=topk_length,
        )
        grad_query_flat = attn_bwd["dq"]
        grad_kv_flat = attn_bwd["dkv"]
        grad_query = grad_query_flat.reshape(sq, b, num_heads, d)
        grad_kv_full = grad_kv_flat.reshape(skv, b, kv_flat.size(-1))
        return grad_query, grad_kv_full

    valid_row_indices = torch.nonzero(topk_length > 0, as_tuple=False).flatten()
    dummy_row_index = torch.full(
        (1,), bwd_q_flat.size(0), dtype=valid_row_indices.dtype, device=valid_row_indices.device
    )
    compact_row_indices = torch.cat((valid_row_indices, dummy_row_index), dim=0)

    # Add one zero-gradient row so the cuDNN wrapper never receives an empty query batch.
    bwd_q_flat = torch.cat((bwd_q_flat, torch.zeros_like(bwd_q_flat[:1])), dim=0)
    bwd_out_flat = torch.cat((bwd_out_flat, torch.zeros_like(bwd_out_flat[:1])), dim=0)
    bwd_dO_flat = torch.cat((bwd_dO_flat, torch.zeros_like(bwd_dO_flat[:1])), dim=0)
    bwd_lse = torch.cat((bwd_lse, torch.zeros_like(bwd_lse[:1])), dim=0)
    global_idxs = torch.cat((global_idxs, torch.zeros_like(global_idxs[:1])), dim=0)
    topk_length = torch.cat((topk_length, torch.ones_like(topk_length[:1])), dim=0)

    attn_bwd = _cudnn_dsa.sparse_attention_backward_wrapper(
        bwd_q_flat.index_select(0, compact_row_indices).contiguous(),
        kv_flat,
        bwd_out_flat.index_select(0, compact_row_indices).contiguous(),
        bwd_dO_flat.index_select(0, compact_row_indices).contiguous(),
        bwd_lse.index_select(0, compact_row_indices).contiguous(),
        bwd_attn_sink,
        global_idxs.index_select(0, compact_row_indices).contiguous(),
        softmax_scale=softmax_scale,
        topk_length=topk_length.index_select(0, compact_row_indices).contiguous(),
    )
    grad_query_valid = attn_bwd["dq"][:-1]
    grad_query_flat = torch.zeros_like(q_flat)
    grad_query_flat.index_copy_(0, valid_row_indices, grad_query_valid)
    grad_kv_flat = attn_bwd["dkv"]

    grad_query = grad_query_flat.reshape(sq, b, num_heads, d)
    grad_kv_full = grad_kv_flat.reshape(skv, b, kv_flat.size(-1))
    return grad_query, grad_kv_full


class FusedIndexerSparseAttnFunc(torch.autograd.Function):
    """Fused DSv3.2 indexer, sparse attention, and indexer-loss autograd.

    Differentiable w.r.t. ``query``, ``kv_full``, ``q_indexer``, ``k_indexer``, ``weights``.

    Two indexer-loss variants, selected by the ``sparse_loss`` argument
    (matches ``compute_dsa_indexer_loss`` in the reference ``dsa.py``):

    * **Sparse loss** (``sparse_loss=True``) — KL is computed only over
      the top-K KV positions the indexer has selected.
    * **Dense loss** (``sparse_loss=False``, the default) — KL is
      computed over *all* causally valid KV positions.

    Both variants share the FlashMLA sparse-attention forward + the
    cuDNN sparse-attn backward; only the indexer-loss path branches.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        kv_full: Tensor,
        q_indexer: Tensor,
        k_indexer: Tensor,
        weights: Tensor,
        indexer_topk: int,
        softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        calculate_per_token_loss: bool,
        d_v: int,
        varlen_starts: Optional[Tensor],
        varlen_ends: Optional[Tensor],
        key_positions: Optional[Tensor],
        query_valid_rows: Optional[Tensor],
        use_local_indexer_varlen: bool,
        single_packed_thd_sequence: bool,
        local_packed_cp_rank: int,
        local_packed_cp_query_start: int,
        local_packed_cp_query_len: Optional[int],
        packed_cu_seqlens_q: Optional[Tensor],
        packed_cu_seqlens_k: Optional[Tensor],
        packed_max_seqlen_q: Optional[int],
        packed_max_seqlen_k: Optional[int],
        packed_cp_size: int,
        tp_group,
    ) -> Tuple[Tensor, Tensor]:
        """Fused forward: indexer scoring, sparse attention, KL loss, and indexer backward."""
        _ensure_dsa_namespace()

        sq, b, num_heads, d = query.shape
        skv = kv_full.shape[0]
        n_comp = k_indexer.shape[0]
        query_valid_rows = dsa_masking.normalize_query_valid_rows(
            query_valid_rows, b=b, sq=sq, device=query.device
        )

        effective_topk = min(indexer_topk, n_comp)
        q_idx_bshd, k_idx_bsd, w_bsh = _sbhd_to_bshd_indexer_inputs(q_indexer, k_indexer, weights)

        need_indexer_loss = loss_coeff > 0
        need_dense_scores = need_indexer_loss and not sparse_loss
        need_sparse_scores = need_indexer_loss and sparse_loss
        topk_indices_cmp, topk_length_cmp, indexer_score_payload = _indexer_topk_bshd(
            q_idx_bshd,
            k_idx_bsd,
            w_bsh,
            effective_topk,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            return_scores=need_dense_scores,
            return_topk_scores=need_sparse_scores,
            use_local_indexer_varlen=use_local_indexer_varlen,
            single_packed_thd_sequence=single_packed_thd_sequence,
            local_packed_cp_rank=local_packed_cp_rank,
            local_packed_cp_query_start=local_packed_cp_query_start,
            local_packed_cp_query_len=local_packed_cp_query_len,
            packed_cu_seqlens_q=packed_cu_seqlens_q,
            packed_cu_seqlens_k=packed_cu_seqlens_k,
            packed_max_seqlen_q=packed_max_seqlen_q,
            packed_max_seqlen_k=packed_max_seqlen_k,
            packed_cp_size=packed_cp_size,
        )

        prepared_topk_length = (
            topk_length_cmp if not need_dense_scores and not need_sparse_scores else None
        )
        out_flat, lse, q_flat, kv_flat, attn_sink, global_idxs, topk_length_flat = (
            _run_sparse_attention_forward(
                query,
                kv_full,
                topk_indices_cmp,
                softmax_scale,
                d_v,
                topk_length=prepared_topk_length,
                sanitize_topk_for_backward_in_place=loss_coeff <= 0,
            )
        )

        if need_indexer_loss:
            if sparse_loss:
                if indexer_score_payload is None:
                    raise RuntimeError("cuDNN sparse indexer loss requires selected top-k scores.")
                (
                    indexer_loss,
                    precomputed_grad_q_indexer,
                    precomputed_grad_k_indexer,
                    precomputed_grad_weights,
                ) = _compute_sparse_indexer_loss_and_grads(
                    q_idx_bshd=q_idx_bshd,
                    k_idx_bsd=k_idx_bsd,
                    w_bsh=w_bsh,
                    topk_indices_cmp=topk_indices_cmp,
                    topk_length_cmp=topk_length_cmp,
                    indexer_score_payload=indexer_score_payload,
                    query=query,
                    kv_full=kv_full,
                    lse=lse,
                    softmax_scale=softmax_scale,
                    loss_coeff=loss_coeff,
                    query_valid_rows=query_valid_rows,
                    calculate_per_token_loss=calculate_per_token_loss,
                    tp_group=tp_group,
                )
            else:
                # The dense fused indexer loss has no query-row masking and is only
                # reached with query_valid_rows is None: run_fused_dsa_attention declines
                # dense loss for packed/varlen inputs, the only source of padded rows.
                # Guard so a future gating change fails loudly instead of silently
                # averaging padded rows into the loss/grads, unlike the reference
                # compute_dsa_indexer_loss which masks both loss modes. To support padded
                # rows here, thread query_valid_rows through
                # _compute_dense_indexer_loss_and_grads (mirroring the sparse branch).
                assert query_valid_rows is None, (
                    "dense fused indexer loss does not support padded query rows "
                    "(query_valid_rows must be None); thread query_valid_rows through "
                    "_compute_dense_indexer_loss_and_grads to add support"
                )
                (
                    indexer_loss,
                    precomputed_grad_q_indexer,
                    precomputed_grad_k_indexer,
                    precomputed_grad_weights,
                ) = _compute_dense_indexer_loss_and_grads(
                    query=query,
                    kv_full=kv_full,
                    q_idx_bshd=q_idx_bshd,
                    k_idx_bsd=k_idx_bsd,
                    w_bsh=w_bsh,
                    indexer_score_payload=indexer_score_payload,
                    softmax_scale=softmax_scale,
                    loss_coeff=loss_coeff,
                    calculate_per_token_loss=calculate_per_token_loss,
                    tp_group=tp_group,
                )
        else:
            indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)
            # No indexer loss this step: no precomputed grads to save. backward gates on
            # ctx.has_precomputed_indexer_grads (False here) and never reads these slots.
            precomputed_grad_q_indexer = None
            precomputed_grad_k_indexer = None
            precomputed_grad_weights = None

        ctx.save_for_backward(
            q_flat,
            kv_flat,
            attn_sink,
            global_idxs,
            out_flat,
            lse,
            precomputed_grad_q_indexer,
            precomputed_grad_k_indexer,
            precomputed_grad_weights,
        )
        ctx.has_precomputed_indexer_grads = need_indexer_loss
        ctx.softmax_scale = softmax_scale
        ctx.topk_length = topk_length_flat
        ctx.sq = sq
        ctx.b = b
        ctx.num_heads = num_heads
        ctx.d = d
        ctx.skv = skv
        ctx.all_sparse_bwd_rows_nonempty = (
            query_valid_rows is None
            and use_local_indexer_varlen
            and varlen_starts is not None
            and varlen_ends is not None
            and key_positions is None
        )

        output = out_flat.reshape(sq, b, num_heads, d_v).reshape(sq, b, num_heads * d_v)
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        """Backward: sparse attention bwd + scale pre-computed indexer grads."""
        (
            q_flat,
            kv_flat,
            attn_sink,
            global_idxs,
            out_flat,
            lse,
            precomputed_grad_q_indexer,
            precomputed_grad_k_indexer,
            precomputed_grad_weights,
        ) = ctx.saved_tensors

        sq, b, num_heads, d = ctx.sq, ctx.b, ctx.num_heads, ctx.d
        skv = ctx.skv

        grad_query, grad_kv_full = _run_sparse_attention_backward(
            q_flat=q_flat,
            kv_flat=kv_flat,
            attn_sink=attn_sink,
            global_idxs=global_idxs,
            out_flat=out_flat,
            lse=lse,
            topk_length=ctx.topk_length,
            softmax_scale=ctx.softmax_scale,
            sq=sq,
            b=b,
            num_heads=num_heads,
            d=d,
            skv=skv,
            grad_output=grad_output,
            all_rows_nonempty=ctx.all_sparse_bwd_rows_nonempty,
        )

        if ctx.has_precomputed_indexer_grads and grad_loss is not None:
            grad_q_indexer = precomputed_grad_q_indexer * grad_loss
            grad_k_indexer = precomputed_grad_k_indexer * grad_loss
            grad_weights = precomputed_grad_weights * grad_loss
        else:
            grad_q_indexer = None
            grad_k_indexer = None
            grad_weights = None

        # Grads: query, kv_full, q_indexer, k_indexer, weights, indexer_topk,
        #   softmax_scale, loss_coeff, sparse_loss, calculate_per_token_loss, d_v,
        #   varlen_starts, varlen_ends, key_positions, query_valid_rows,
        #   use_local_indexer_varlen, single_packed_thd_sequence,
        #   local_packed_cp_rank, local_packed_cp_query_start,
        #   local_packed_cp_query_len, packed_cu_seqlens_q, packed_cu_seqlens_k,
        #   packed_max_seqlen_q, packed_max_seqlen_k, packed_cp_size, tp_group.
        return (
            grad_query,
            grad_kv_full,
            grad_q_indexer,
            grad_k_indexer,
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedQKTopKWithSparseLossFunc(torch.autograd.Function):
    """Fused cuDNN indexer top-k plus sparse indexer-loss autograd."""

    @staticmethod
    def forward(
        ctx,
        q_indexer: Tensor,
        k_indexer: Tensor,
        weights: Tensor,
        query: Tensor,
        kv_full: Tensor,
        indexer_topk: int,
        softmax_scale: float,
        loss_coeff: float,
        calculate_per_token_loss: bool,
        d_v: int,
        varlen_starts: Tensor,
        varlen_ends: Tensor,
        query_valid_rows: Optional[Tensor],
        use_local_indexer_varlen: bool,
        single_packed_thd_sequence: bool,
        local_packed_cp_rank: int,
        local_packed_cp_query_start: int,
        local_packed_cp_query_len: Optional[int],
        packed_cu_seqlens_q: Optional[Tensor],
        packed_cu_seqlens_k: Optional[Tensor],
        packed_max_seqlen_q: Optional[int],
        packed_max_seqlen_k: Optional[int],
        packed_cp_size: int,
        tp_group,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute shared attention top-k metadata and sparse indexer loss."""
        _ensure_dsa_namespace()

        sq, b, _num_heads, _d = query.shape
        query_valid_rows = dsa_masking.normalize_query_valid_rows(
            query_valid_rows, b=b, sq=sq, device=query.device
        )

        q_idx_bshd, k_idx_bsd, w_bsh = _sbhd_to_bshd_indexer_inputs(q_indexer, k_indexer, weights)
        topk_indices_cmp, topk_length_cmp, indexer_score_payload = _indexer_topk_bshd(
            q_idx_bshd,
            k_idx_bsd,
            w_bsh,
            indexer_topk,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=None,
            return_scores=False,
            return_topk_scores=True,
            use_local_indexer_varlen=use_local_indexer_varlen,
            single_packed_thd_sequence=single_packed_thd_sequence,
            local_packed_cp_rank=local_packed_cp_rank,
            local_packed_cp_query_start=local_packed_cp_query_start,
            local_packed_cp_query_len=local_packed_cp_query_len,
            packed_cu_seqlens_q=packed_cu_seqlens_q,
            packed_cu_seqlens_k=packed_cu_seqlens_k,
            packed_max_seqlen_q=packed_max_seqlen_q,
            packed_max_seqlen_k=packed_max_seqlen_k,
            packed_cp_size=packed_cp_size,
        )

        if indexer_score_payload is None:
            raise RuntimeError("cuDNN sparse indexer loss requires selected top-k scores.")
        topk_indices_attn, topk_length_attn = _prepare_attention_topk_indices(
            topk_indices_cmp, kv_full.size(0)
        )
        _out_flat, lse, *_ = _run_sparse_attention_forward(
            query, kv_full, topk_indices_attn, softmax_scale, d_v, topk_length=topk_length_attn
        )
        (
            indexer_loss,
            precomputed_grad_q_indexer,
            precomputed_grad_k_indexer,
            precomputed_grad_weights,
        ) = _compute_sparse_indexer_loss_and_grads(
            q_idx_bshd=q_idx_bshd,
            k_idx_bsd=k_idx_bsd,
            w_bsh=w_bsh,
            topk_indices_cmp=topk_indices_cmp,
            topk_length_cmp=topk_length_cmp,
            indexer_score_payload=indexer_score_payload,
            query=query,
            kv_full=kv_full,
            lse=lse,
            softmax_scale=softmax_scale,
            loss_coeff=loss_coeff,
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=calculate_per_token_loss,
            tp_group=tp_group,
        )

        ctx.save_for_backward(
            precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights
        )
        ctx.mark_non_differentiable(topk_indices_attn, topk_length_attn)
        return topk_indices_attn, topk_length_attn, indexer_loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_topk_length, grad_loss):
        """Scale saved indexer-loss gradients for autograd."""
        del grad_topk_indices, grad_topk_length
        precomputed_grad_q_indexer, precomputed_grad_k_indexer, precomputed_grad_weights = (
            ctx.saved_tensors
        )
        if grad_loss is not None:
            grad_q_indexer = precomputed_grad_q_indexer * grad_loss
            grad_k_indexer = precomputed_grad_k_indexer * grad_loss
            grad_weights = precomputed_grad_weights * grad_loss
        else:
            grad_q_indexer = None
            grad_k_indexer = None
            grad_weights = None

        # Grads: q_indexer, k_indexer, weights, then None for query, kv_full,
        #   indexer_topk, softmax_scale, loss_coeff, calculate_per_token_loss, d_v,
        #   varlen_starts, varlen_ends, query_valid_rows, use_local_indexer_varlen,
        #   single_packed_thd_sequence, local_packed_cp_rank,
        #   local_packed_cp_query_start, local_packed_cp_query_len,
        #   packed_cu_seqlens_q, packed_cu_seqlens_k, packed_max_seqlen_q,
        #   packed_max_seqlen_k, packed_cp_size, tp_group.
        return (
            grad_q_indexer,
            grad_k_indexer,
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedSparseAttentionFunc(torch.autograd.Function):
    """Fused sparse attention with provided top-k indices for DSA index sharing."""

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        kv_full: Tensor,
        topk_indices: Tensor,
        softmax_scale: float,
        d_v: int,
        topk_length: Optional[Tensor],
    ) -> Tensor:
        """Run fused sparse attention for precomputed top-k metadata."""
        sq, b, num_heads, d = query.shape
        skv = kv_full.shape[0]
        out_flat, lse, q_flat, kv_flat, attn_sink, global_idxs, topk_length_flat = (
            _run_sparse_attention_forward(
                query, kv_full, topk_indices, softmax_scale, d_v, topk_length=topk_length
            )
        )

        ctx.save_for_backward(q_flat, kv_flat, attn_sink, global_idxs, out_flat, lse)
        ctx.softmax_scale = softmax_scale
        ctx.topk_length = topk_length_flat
        ctx.sq = sq
        ctx.b = b
        ctx.num_heads = num_heads
        ctx.d = d
        ctx.skv = skv

        return out_flat.reshape(sq, b, num_heads, d_v)

    @staticmethod
    def backward(ctx, grad_output):
        """Run sparse attention backward for saved cuDNN graph inputs."""
        q_flat, kv_flat, attn_sink, global_idxs, out_flat, lse = ctx.saved_tensors

        sq, b, num_heads, d = ctx.sq, ctx.b, ctx.num_heads, ctx.d
        skv = ctx.skv
        grad_query, grad_kv_full = _run_sparse_attention_backward(
            q_flat=q_flat,
            kv_flat=kv_flat,
            attn_sink=attn_sink,
            global_idxs=global_idxs,
            out_flat=out_flat,
            lse=lse,
            topk_length=ctx.topk_length,
            softmax_scale=ctx.softmax_scale,
            sq=sq,
            b=b,
            num_heads=num_heads,
            d=d,
            skv=skv,
            grad_output=grad_output,
        )

        return grad_query, grad_kv_full, None, None, None, None


def run_fused_absorbed_sparse_attention(
    query: Tensor,
    key: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    v_channels: int,
    topk_length: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """Run cuDNN/FlashMLA sparse attention using externally supplied top-k indices."""
    if query.ndim != 4 or key.ndim != 4 or topk_indices.ndim != 3:
        return None
    if key.size(2) != 1 or v_channels <= 0:
        return None
    if query.size(0) != topk_indices.size(1) or query.size(1) != topk_indices.size(0):
        return None
    if key.size(1) != query.size(1) or key.size(3) != query.size(3):
        return None
    if not _supports_flash_mla_value_layout(key.size(-1), v_channels):
        return None
    if not _flash_mla_supports_head_count(query):
        return None

    kv_full = key.squeeze(2).contiguous()
    return FusedSparseAttentionFunc.apply(
        query, kv_full, topk_indices, softmax_scale, v_channels, topk_length
    )


def fused_indexer_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk: int,
    softmax_scale: float,
    loss_coeff: float = 0.0,
    sparse_loss: bool = False,
    calculate_per_token_loss: bool = False,
    d_v: int = 0,
    varlen_starts: Optional[Tensor] = None,
    varlen_ends: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
    query_valid_rows: Optional[Tensor] = None,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_cu_seqlens_q: Optional[Tensor] = None,
    packed_cu_seqlens_k: Optional[Tensor] = None,
    packed_max_seqlen_q: Optional[int] = None,
    packed_max_seqlen_k: Optional[int] = None,
    packed_cp_size: int = 1,
    tp_group=None,
) -> Tuple[Tensor, Tensor]:
    """Fused DSv3.2 indexer, sparse attention, and optional indexer loss.

    Args:
        query: ``(sq, b, np, d)`` bf16 SBHD attention query.
        kv_full: ``(skv, b, d)`` bf16 SBD compressed KV.
        q_indexer: ``(sq, b, idx_nh, idx_hd)`` bf16 indexer query.
        k_indexer: ``(skv, b, idx_hd)`` bf16 indexer key.
        weights: ``(sq, b, idx_nh)`` bf16 indexer weights.
        indexer_topk: number of top-K compressed positions to select.
        softmax_scale: attention ``Q @ K^T`` scale.
        loss_coeff: coefficient scaling the KL divergence loss.
        sparse_loss: whether to compute KL only over selected top-K positions.
        calculate_per_token_loss: whether to report a raw local KL sum.
        d_v: number of value channels returned by FlashMLA.

    Returns:
        ``(output, indexer_loss)`` where ``output`` is ``(sq, b, np * d_v)``
        bf16 and ``indexer_loss`` is a scalar f32.
    """
    if not torch.is_grad_enabled():
        loss_coeff = 0.0
    return FusedIndexerSparseAttnFunc.apply(
        query,
        kv_full,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        calculate_per_token_loss,
        d_v,
        varlen_starts,
        varlen_ends,
        key_positions,
        query_valid_rows,
        use_local_indexer_varlen,
        single_packed_thd_sequence,
        local_packed_cp_rank,
        local_packed_cp_query_start,
        local_packed_cp_query_len,
        packed_cu_seqlens_q,
        packed_cu_seqlens_k,
        packed_max_seqlen_q,
        packed_max_seqlen_k,
        packed_cp_size,
        tp_group,
    )


__all__ = [
    "fused_indexer_sparse_attn",
    "run_fused_absorbed_sparse_attention",
    "run_fused_dsa_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_loss",
]

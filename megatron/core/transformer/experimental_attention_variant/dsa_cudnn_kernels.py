# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN/FlashMLA wrappers for fused DeepSeek sparse attention."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa_masking

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

_flash_mla_sparse_fwd = None
_DSA = None
_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny
_KL_EPS = 1e-10
_INDEXER_RATIO = 1
_INDEXER_SOFTMAX_SCALE = 1.0
_TOPK_WRAPPER_MAX_SCRATCH_BYTES = 2 * 1024 * 1024 * 1024
_TOPK_WRAPPER_SCRATCH_INT32_FACTOR = 2
_TOPK_WRAPPER_ROW_ALIGNMENT = 512
_INDEXER_SCORE_CHUNK_MAX_BYTES = 1024 * 1024 * 1024
_INDEXER_SCORE_CHUNK_ROW_ALIGNMENT = 512


def _use_fused_dsa_kernels(config: TransformerConfig) -> bool:
    """Return whether DSA should attempt optional fused kernels before falling back."""
    backend = config.attention_backend
    if backend == AttnBackend.unfused or backend == "unfused":
        return False
    return config.dsa_kernel_backend == "cudnn"


def _assert_supported_indexer_scoring(use_relu: bool) -> None:
    """Check that the cuDNN indexer scoring mode matches the fused kernel implementation."""
    if not use_relu:
        raise RuntimeError(
            "cuDNN fused DSA kernels currently require dsa_indexer_scoring_relu=True."
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
    use_relu: bool,
    use_local_indexer_varlen: bool = False,
    pg_collection: Optional["ProcessGroupCollection"] = None,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Optional full fused DSA hook for backends that fuse indexer and attention together."""
    if not _use_fused_dsa_kernels(config):
        return None
    _assert_supported_indexer_scoring(use_relu)
    if (
        not absorbed_mla
        or value is not None
        or attn_mask_type != AttnMaskType.causal
        or key.size(2) != 1
    ):
        return None
    has_varlen = varlen_starts is not None or varlen_ends is not None or key_positions is not None
    if has_varlen:
        if varlen_starts is None or varlen_ends is None:
            return None
        if loss_coeff > 0 and not sparse_loss:
            raise RuntimeError(
                "cuDNN fused DSA packed-varlen dense indexer loss is unsupported because "
                "dense_attn_score_recompute_wrapper has no query-position input."
            )
        if not use_local_indexer_varlen:
            key_positions_i64 = (
                torch.arange(key.size(0), dtype=torch.int64, device=key.device)
                if key_positions is None
                else key_positions.to(device=key.device, dtype=torch.int64)
            )
            expected_key_positions = torch.arange(key.size(0), dtype=torch.int64, device=key.device)
            if not torch.equal(key_positions_i64, expected_key_positions):
                return None
    elif cp_size != 1 or packed_seq_params is not None or query.size(0) != key.size(0):
        return None

    latent_v_channels = int(getattr(config, "kv_lora_rank", 0) or 0)
    if latent_v_channels <= 0:
        return None

    sq, b, num_heads, _ = query.size()
    kv_full = key.squeeze(2).contiguous()

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
        tp_group=getattr(pg_collection, "tp", None),
    )

    if up_v_weight is None:
        return output, indexer_loss
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


def _get_topk_alignment() -> int:
    """Minimum top-K alignment required by the active FlashMLA sparse-prefill kernel."""
    sm = torch.cuda.get_device_capability()
    if sm[0] >= 9:
        return 512
    raise RuntimeError(f"cudnn fused DSA requires SM90+ (Hopper or later), got SM{sm[0]}{sm[1]}.")


def _get_head_padding(num_heads: int) -> int:
    """Return the query-head count supported by the FlashMLA sparse prefill kernel."""
    sm = torch.cuda.get_device_capability()
    if sm[0] >= 10:
        if num_heads in (64, 128):
            return num_heads
        head_align = 128
    elif sm[0] >= 9:
        head_align = 64
    else:
        raise RuntimeError(
            f"cudnn fused DSA requires SM90+ (Hopper or later), got SM{sm[0]}{sm[1]}."
        )

    if num_heads % head_align == 0:
        return num_heads
    if num_heads < head_align and head_align % num_heads == 0:
        return head_align
    raise RuntimeError(
        "FlashMLA sparse prefill requires the local query-head count to divide "
        f"{head_align}, got h_q={num_heads}."
    )


def _indexer_top_k_wrapper_chunked(
    scores_flat: Tensor, seq_lens: Tensor, topk_k: int, return_topk_scores: bool
) -> dict:
    """Run cuDNN top-k in row chunks to bound wrapper scratch allocation."""
    n_rows, sk = scores_flat.shape
    scratch_bytes_per_row = max(1, sk) * torch.iinfo(torch.int32).bits // 8
    scratch_bytes_per_row *= _TOPK_WRAPPER_SCRATCH_INT32_FACTOR
    chunk_rows = max(1, _TOPK_WRAPPER_MAX_SCRATCH_BYTES // scratch_bytes_per_row)
    row_alignment = _TOPK_WRAPPER_ROW_ALIGNMENT
    if chunk_rows > row_alignment:
        chunk_rows = (chunk_rows // row_alignment) * row_alignment
    else:
        chunk_rows = row_alignment
    chunk_rows = min(n_rows, chunk_rows)

    if chunk_rows >= n_rows:
        return _DSA.indexer_top_k_wrapper(
            scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=return_topk_scores
        )

    indices_chunks = []
    values_chunks = [] if return_topk_scores else None
    for row_start in range(0, n_rows, chunk_rows):
        row_end = min(row_start + chunk_rows, n_rows)
        tk_result = _DSA.indexer_top_k_wrapper(
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
    chunk_rows = max(1, _INDEXER_SCORE_CHUNK_MAX_BYTES // score_bytes_per_seq_row)
    row_alignment = _INDEXER_SCORE_CHUNK_ROW_ALIGNMENT
    if chunk_rows > row_alignment:
        chunk_rows = (chunk_rows // row_alignment) * row_alignment
    else:
        chunk_rows = row_alignment
    return min(sq, max(1, chunk_rows))


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
) -> Tuple[Tensor, Optional[Tensor]]:
    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bshd.size(1)
    chunk_rows = _indexer_score_chunk_rows(b, sq, sk)
    seq_lens_b = seq_lens.view(b, sq)
    indices_chunks = []
    values_chunks = [] if return_topk_scores else None

    for row_start in range(0, sq, chunk_rows):
        row_end = min(row_start + chunk_rows, sq)
        scores_chunk = _DSA.indexer_forward_wrapper(
            q_bshd[:, row_start:row_end].contiguous(),
            k_bshd,
            w_bsh[:, row_start:row_end].contiguous(),
            ratio=indexer_ratio,
            sm_scale=_INDEXER_SOFTMAX_SCALE,
        )["scores"]
        if starts is not None:
            scores_chunk = dsa_masking.apply_starts_ends_mask_to_scores(
                scores_chunk, starts[row_start:row_end], ends[row_start:row_end], key_positions_i64
            )

        scores_flat = scores_chunk.reshape(b * (row_end - row_start), sk).contiguous()
        chunk_seq_lens = seq_lens_b[:, row_start:row_end].reshape(-1).contiguous()
        tk_result = _indexer_top_k_wrapper_chunked(
            scores_flat, chunk_seq_lens, topk_k=topk_k, return_topk_scores=return_topk_scores
        )
        del scores_flat, scores_chunk

        indices_chunks.append(tk_result["indices"].view(b, row_end - row_start, topk_k))
        if return_topk_scores:
            values = tk_result["values"]
            if values is None:
                raise RuntimeError("cuDNN indexer_top_k_wrapper did not return values.")
            values_chunks.append(values.view(b, row_end - row_start, topk_k))

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
    topk_padded = (topk_count + topk_align - 1) // topk_align * topk_align
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
    global _DSA
    if _DSA is not None:
        return
    try:
        from cudnn import DSA as _ns
    except ImportError as e:
        raise ImportError(
            "cudnn-frontend DSA namespace not available. Install with "
            "`pip install nvidia-cudnn-frontend[cutedsl]`."
        ) from e
    _DSA = _ns


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
    batch_ids = torch.arange(sq * b, device=local_idxs.device) % b
    batch_ids_exp = batch_ids.unsqueeze(1).expand_as(idxs_sb)
    idxs_sb = torch.where(valid, idxs_sb * b + batch_ids_exp, idxs_sb)
    return idxs_sb.int()


def _valid_topk_length(topk_indices: Tensor) -> Tensor:
    """Return the count of valid top-K entries in each row."""
    return (topk_indices >= 0).sum(dim=-1).int().contiguous()


def _compact_valid_topk_indices(topk_indices: Tensor) -> Tuple[Tensor, Tensor]:
    """Move valid top-K entries to the left so ``topk_length`` is semantically correct."""
    valid = topk_indices >= 0
    topk_length = valid.sum(dim=-1).int()
    positions = torch.arange(topk_indices.size(-1), device=topk_indices.device).view(
        *((1,) * (topk_indices.ndim - 1)), -1
    )
    sentinel = torch.full_like(positions, topk_indices.size(-1)).expand_as(topk_indices)
    order_key = torch.where(valid, positions.expand_as(topk_indices), sentinel)
    order = order_key.argsort(dim=-1)
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
    positions = torch.arange(topk_indices.size(-1), device=topk_indices.device).view(
        *((1,) * (topk_indices.ndim - 1)), -1
    )
    sentinel = torch.full_like(positions, topk_indices.size(-1)).expand_as(topk_indices)
    order_key = torch.where(valid, positions.expand_as(topk_indices), sentinel)
    order = order_key.argsort(dim=-1)
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
    positions = torch.arange(topk_indices.size(-1), device=topk_indices.device).view(
        *((1,) * (topk_indices.ndim - 1)), -1
    )
    valid = positions < topk_length.unsqueeze(-1)
    sort_key = torch.where(valid, topk_indices, torch.full_like(topk_indices, sk))
    order = sort_key.argsort(dim=-1)
    sorted_indices = torch.gather(topk_indices, dim=-1, index=order)
    sorted_valid = torch.gather(valid.expand_as(topk_indices), dim=-1, index=order)
    return sorted_indices.masked_fill(~sorted_valid, -1).contiguous()


def _prepare_attention_topk_indices(topk_indices: Tensor, sk: int) -> Tuple[Tensor, Tensor]:
    """Prepare top-K indices for FlashMLA without changing indexer-loss ordering."""
    compacted, topk_length = _compact_valid_topk_indices(topk_indices)
    sorted_indices = _sort_valid_topk_indices_by_index(compacted, topk_length, sk)
    return sorted_indices.int().contiguous(), topk_length.contiguous()


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
    if varlen_starts is not None:
        starts, ends, key_positions_i64 = dsa_masking.normalize_varlen_bounds(
            mask=None,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
            sk=sk,
            device=device,
        )
        if not use_local_indexer_varlen:
            expected_key_positions = torch.arange(sk, dtype=torch.int64, device=device)
            if not torch.equal(key_positions_i64, expected_key_positions):
                raise RuntimeError("cuDNN fused DSA varlen path expects identity key positions")

    k_bshd = k_bsd.unsqueeze(2)  # (b, sk, 1, idx_hd)

    if starts is None:
        q_idx = torch.arange(sq, device=device)
        seq_lens = ((q_idx + 1) // _INDEXER_RATIO).clamp(max=sk).to(torch.int32).repeat(b)
        if not return_scores:
            topk_indices, topk_scores = _indexer_topk_from_score_chunks(
                q_bshd, k_bshd, w_bsh, seq_lens, topk_k, return_topk_scores
            )
        else:
            scores = _DSA.indexer_forward_wrapper(
                q_bshd, k_bshd, w_bsh, ratio=_INDEXER_RATIO, sm_scale=_INDEXER_SOFTMAX_SCALE
            )[
                "scores"
            ]  # (b, sq, sk) fp32, -inf on masked positions
    else:
        query_positions = (ends - 1).to(dtype=torch.int64)
        local_query_positions = torch.arange(sq, dtype=torch.int64, device=device)
        if use_local_indexer_varlen:
            q_for_scores = q_bshd
            w_for_scores = w_bsh
            gather_positions = None
            indexer_ratio = _INDEXER_RATIO
        elif torch.equal(query_positions, local_query_positions):
            q_for_scores = q_bshd
            w_for_scores = w_bsh
            gather_positions = None
            indexer_ratio = _INDEXER_RATIO
        else:
            score_sq = int(query_positions.max().item()) + 1
            q_for_scores = torch.zeros(
                (b, score_sq, q_bshd.size(2), q_bshd.size(3)), dtype=q_bshd.dtype, device=device
            )
            w_for_scores = torch.zeros(
                (b, score_sq, w_bsh.size(2)), dtype=w_bsh.dtype, device=device
            )
            q_for_scores.index_copy_(1, query_positions, q_bshd)
            w_for_scores.index_copy_(1, query_positions, w_bsh)
            gather_positions = query_positions
            indexer_ratio = _INDEXER_RATIO

        seq_lens = ends.clamp(max=sk).to(torch.int32).repeat(b)
        if not return_scores and gather_positions is None:
            topk_indices, topk_scores = _indexer_topk_from_score_chunks(
                q_for_scores,
                k_bshd,
                w_for_scores,
                seq_lens,
                topk_k,
                return_topk_scores,
                starts=starts,
                ends=ends,
                key_positions_i64=key_positions_i64,
                indexer_ratio=indexer_ratio,
            )
        else:
            scores = _DSA.indexer_forward_wrapper(
                q_for_scores,
                k_bshd,
                w_for_scores,
                ratio=indexer_ratio,
                sm_scale=_INDEXER_SOFTMAX_SCALE,
            )["scores"]
            if gather_positions is not None:
                scores = scores.index_select(1, gather_positions)
            scores = dsa_masking.apply_starts_ends_mask_to_scores(
                scores, starts, ends, key_positions_i64
            )

    # Top-K selection via the TRT-LLM CuTe-DSL radix kernel.
    if topk_indices is None:
        n_rows = b * sq
        scores_flat = scores.reshape(n_rows, sk).contiguous()

        tk_result = _indexer_top_k_wrapper_chunked(
            scores_flat, seq_lens, topk_k=topk_k, return_topk_scores=return_topk_scores
        )
        del scores_flat
        if not return_scores:
            del scores
        topk_indices = tk_result["indices"].view(b, sq, topk_k)
        if return_topk_scores:
            topk_scores = tk_result["values"]
            if topk_scores is None:
                raise RuntimeError("cuDNN indexer_top_k_wrapper did not return values.")
            topk_scores = topk_scores.view(b, sq, topk_k)

    if return_topk_scores:
        topk_scores = topk_scores.to(dtype=torch.float32)

    if return_topk_scores:
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
        valid = in_range & valid_bounds & torch.isfinite(topk_scores)
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
        in_range = (topk_indices >= 0) & (topk_indices < sk)
        if starts is not None and use_local_indexer_varlen:
            valid = (
                in_range
                & (topk_indices >= starts.view(1, sq, 1))
                & (topk_indices < ends.view(1, sq, 1))
            )
        elif starts is not None:
            safe_indices = topk_indices.clamp(min=0, max=sk - 1).long()
            selected_positions = key_positions_i64.index_select(0, safe_indices.reshape(-1))
            selected_positions = selected_positions.view_as(topk_indices)
            valid = (
                in_range
                & (selected_positions >= starts.view(1, sq, 1))
                & (selected_positions < ends.view(1, sq, 1))
            )
        else:
            valid = in_range & (topk_indices < seq_lens.view(b, sq, 1))
        topk_indices = torch.where(valid, topk_indices, torch.full_like(topk_indices, -1))

    if topk_k < topk:
        pad = torch.full((b, sq, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)
        if topk_scores is not None:
            score_pad = torch.full(
                (b, sq, topk - topk_k),
                torch.finfo(torch.float32).min,
                dtype=torch.float32,
                device=device,
            )
            topk_scores = torch.cat([topk_scores, score_pad], dim=-1)

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
) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """Run cuDNN fused indexer and sparse indexer loss for split DSA."""
    del block_size
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
        getattr(pg_collection, "tp", None),
    )


def _tp_group_size(tp_group) -> int:
    """Return the TP size for optional target reduction."""
    return 1 if tp_group is None else tp_group.size()


def _all_reduce_tp_target(target: Tensor, tp_group) -> Tensor:
    """Sum a local attention target across TP ranks when TP is enabled."""
    if _tp_group_size(tp_group) > 1:
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
    padded_topk = (topk + block_size - 1) // block_size * block_size
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
    if q_attn_bshd.is_cuda:
        padded_heads = _get_head_padding(actual_heads)
    else:
        padded_heads = max(
            min_heads, ((actual_heads + head_multiple - 1) // head_multiple) * head_multiple
        )
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
    result = _DSA.sparse_attn_score_recompute_wrapper(
        q_attn_bshd,
        k_attn_bsd,
        lse.contiguous(),
        topk_indices.contiguous(),
        softmax_scale,
        **kwargs,
    )
    return result["target"].contiguous()


def _kl_loss_from_target_predict(
    target: Tensor,
    predict: Tensor,
    loss_coeff: float,
    query_valid_rows: Optional[Tensor] = None,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) over selected top-K entries.

    Invalid top-K slots must already be zeroed in ``target`` and ``predict``.
    """
    eps = _KL_EPS
    kl_terms = target * (torch.log(target + eps) - torch.log(predict + eps))
    kl_per_row = kl_terms.sum(dim=-1)
    if query_valid_rows is None:
        loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    else:
        row_mask = query_valid_rows.to(dtype=torch.float32, device=kl_per_row.device)
        if calculate_per_token_loss:
            loss = (kl_per_row * row_mask).sum()
        else:
            row_count = row_mask.sum().clamp_min(1.0)
            loss = (kl_per_row * row_mask).sum() / row_count
    return loss_coeff * loss


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
    result = _DSA.dense_attn_score_recompute_wrapper(
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
    target_clamped = target.clamp(min=eps)
    # Per-position validity: the indexer-score kernel emits -inf at
    # ratio-masked positions; those contribute 0 to KL by the
    # ``0 · log(0/p) = 0`` convention. Without this gate, the eps-clamp
    # on target makes the term ``eps · (log eps - (-inf)) = +inf``.
    position_valid = torch.isfinite(index_score)
    safe_index_score = torch.where(position_valid, index_score, torch.zeros_like(index_score))
    log_predict = safe_index_score - safe_lse.unsqueeze(-1)
    predict = torch.exp(log_predict)

    kl_terms = target_clamped * (torch.log(target_clamped) - torch.log(predict + eps))
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
    tp_group=None,
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

    positions = torch.arange(topk_indices_cmp.size(-1), device=topk_indices_cmp.device).view(
        *((1,) * (topk_indices_cmp.ndim - 1)), -1
    )
    valid_positions = (positions < topk_length_cmp.unsqueeze(-1)) & (topk_indices_cmp >= 0)
    if query_valid_rows is not None:
        valid_positions = valid_positions & query_valid_rows.unsqueeze(-1)
    predict = dsa_masking.masked_softmax(indexer_score_payload, valid_positions, dim=-1)

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
    _all_reduce_tp_target(target, tp_group)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(_KL_EPS)
    target.masked_fill_(~valid_positions, 0.0)
    indexer_loss = _kl_loss_from_target_predict(
        target, predict, loss_coeff, query_valid_rows, calculate_per_token_loss
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
    index_score_for_bwd = predict.clamp_min(torch.finfo(torch.float32).tiny)
    block_i = 128
    attn_score_for_bwd, index_score_for_bwd, topk_indices_for_bwd = _pad_sparse_backward_topk(
        attn_score_for_bwd, index_score_for_bwd, topk_indices_for_bwd, block_i
    )
    ig = _DSA.indexer_backward_wrapper(
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
) -> Tuple[Tensor, Tensor]:
    """Run sparse attention backward, skipping rows that have no valid top-k entries."""
    d_v = out_flat.shape[-1]
    dO_flat = grad_output.reshape(sq * b, num_heads, d_v)

    bwd_q_flat = q_flat
    bwd_out_flat = out_flat
    bwd_dO_flat = dO_flat
    bwd_lse = lse
    bwd_attn_sink = attn_sink
    padded_num_heads = num_heads
    if q_flat.is_cuda:
        padded_num_heads = _get_head_padding(num_heads)
    if padded_num_heads != num_heads:
        bwd_q_flat = q_flat.new_zeros((q_flat.size(0), padded_num_heads, q_flat.size(2)))
        bwd_q_flat[:, :num_heads, :] = q_flat

        bwd_out_flat = out_flat.new_zeros((out_flat.size(0), padded_num_heads, out_flat.size(2)))
        bwd_out_flat[:, :num_heads, :] = out_flat

        bwd_dO_flat = dO_flat.new_zeros((dO_flat.size(0), padded_num_heads, dO_flat.size(2)))
        bwd_dO_flat[:, :num_heads, :] = dO_flat

        bwd_lse = lse.new_zeros((lse.size(0), padded_num_heads))
        bwd_lse[:, :num_heads] = lse

        bwd_attn_sink = attn_sink.new_full((padded_num_heads,), float("-inf"))
        bwd_attn_sink[:num_heads] = attn_sink

    valid_rows = topk_length > 0
    if torch.all(valid_rows):
        _ensure_dsa_namespace()
        attn_bwd = _DSA.sparse_attention_backward_wrapper(
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
        if padded_num_heads != num_heads:
            grad_query_flat = grad_query_flat[:, :num_heads, :].contiguous()
        grad_kv_flat = attn_bwd["dkv"]
    elif torch.any(valid_rows):
        _ensure_dsa_namespace()
        valid_row_indices = torch.nonzero(valid_rows, as_tuple=False).flatten()
        attn_bwd = _DSA.sparse_attention_backward_wrapper(
            bwd_q_flat.index_select(0, valid_row_indices).contiguous(),
            kv_flat,
            bwd_out_flat.index_select(0, valid_row_indices).contiguous(),
            bwd_dO_flat.index_select(0, valid_row_indices).contiguous(),
            bwd_lse.index_select(0, valid_row_indices).contiguous(),
            bwd_attn_sink,
            global_idxs.index_select(0, valid_row_indices).contiguous(),
            softmax_scale=softmax_scale,
            topk_length=topk_length.index_select(0, valid_row_indices).contiguous(),
        )
        grad_query_valid = attn_bwd["dq"]
        if padded_num_heads != num_heads:
            grad_query_valid = grad_query_valid[:, :num_heads, :].contiguous()
        grad_query_flat = torch.zeros_like(q_flat)
        grad_query_flat.index_copy_(0, valid_row_indices, grad_query_valid)
        grad_kv_flat = attn_bwd["dkv"]
    else:
        grad_query_flat = torch.zeros_like(q_flat)
        grad_kv_flat = torch.zeros_like(kv_flat)

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

        need_dense_scores = loss_coeff > 0 and not sparse_loss
        need_sparse_scores = loss_coeff > 0 and sparse_loss
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

        if loss_coeff > 0:
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
                # Attention-path tensors are detached; the KL loss trains only the indexer.
                q_attn_bshd = query.detach().permute(1, 0, 2, 3).contiguous()
                k_attn_compressed_bsd = kv_full.detach().permute(1, 0, 2).contiguous()
                lse_bsqh = lse.reshape(sq, b, num_heads).permute(1, 0, 2)
                indexer_loss_coeff = (
                    loss_coeff * (b * sq) if calculate_per_token_loss else loss_coeff
                )
                unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)
                q_idx_bshd_for_bwd, w_bsh_for_bwd, actual_indexer_heads = (
                    _pad_indexer_heads_for_backward(q_idx_bshd, w_bsh)
                )
                index_score = indexer_score_payload
                if index_score is None:
                    raise RuntimeError("cuDNN dense indexer loss requires full indexer scores.")
                index_lse = torch.logsumexp(index_score, dim=-1)
                attn_score, attn_l1norm = _compute_dense_attn_score(
                    q_attn_bshd,
                    k_attn_compressed_bsd.unsqueeze(2),
                    lse_bsqh,
                    qhead_per_kv_head=num_heads,
                    softmax_scale=softmax_scale,
                )
                _all_reduce_tp_target(attn_score, tp_group)
                _all_reduce_tp_target(attn_l1norm, tp_group)
                indexer_loss = _kl_loss_from_dense_scores(
                    attn_score,
                    attn_l1norm,
                    index_score,
                    index_lse,
                    loss_coeff,
                    calculate_per_token_loss,
                )
                attn_score_for_bwd = attn_score.clone()
                index_score_for_bwd = index_score.clone()
                ig = _DSA.dense_indexer_backward_wrapper(
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
        else:
            indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)
            precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
            precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
            precomputed_grad_weights = torch.zeros_like(weights)

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
        ctx.softmax_scale = softmax_scale
        ctx.topk_length = topk_length_flat
        ctx.sq = sq
        ctx.b = b
        ctx.num_heads = num_heads
        ctx.d = d
        ctx.skv = skv

        d_v = out_flat.shape[-1]
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
        )

        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss

        # Grads: query, kv_full, q_indexer, k_indexer, weights, indexer_topk,
        #   softmax_scale, loss_coeff, sparse_loss, calculate_per_token_loss, d_v,
        #   varlen_starts, varlen_ends, key_positions, query_valid_rows,
        #   use_local_indexer_varlen, tp_group.
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
        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss

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
        tp_group,
    )


__all__ = [
    "fused_indexer_sparse_attn",
    "run_fused_absorbed_sparse_attention",
    "run_fused_dsa_attention",
    "run_fused_qk_topk",
    "run_fused_qk_topk_with_loss",
]

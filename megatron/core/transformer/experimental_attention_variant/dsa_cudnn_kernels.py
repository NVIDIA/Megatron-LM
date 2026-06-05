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
    from megatron.core.transformer.transformer_config import TransformerConfig

_flash_mla_sparse_fwd = None
_DSA = None
_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny
_INDEXER_RATIO = 1
_INDEXER_SOFTMAX_SCALE = 1.0


def _use_fused_dsa_kernels(config: TransformerConfig) -> bool:
    """Return whether DSA should attempt optional fused kernels before falling back."""
    backend = config.attention_backend
    if backend == AttnBackend.unfused or backend == "unfused":
        return False
    return config.dsa_kernel_backend == "cudnn"


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
    use_relu: bool,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Optional full fused DSA hook for backends that fuse indexer and attention together."""
    if not _use_fused_dsa_kernels(config):
        return None
    if (
        not absorbed_mla
        or value is not None
        or attn_mask_type != AttnMaskType.causal
        or key.size(2) != 1
        or not use_relu
    ):
        return None
    has_varlen = varlen_starts is not None or varlen_ends is not None or key_positions is not None
    if has_varlen:
        if varlen_starts is None or varlen_ends is None:
            return None
        if loss_coeff > 0 and not sparse_loss:
            return None
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
    ``topk`` to the alignment expected by
    FlashMLA's Blackwell kernel.
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
    """Minimum top-K alignment required by the Blackwell FlashMLA kernel."""
    sm = torch.cuda.get_device_capability()
    if sm[0] >= 10:
        return 64
    raise RuntimeError(
        f"cudnn fused DSA requires SM100+ (Blackwell or later), got SM{sm[0]}{sm[1]}."
    )


def _get_head_padding(num_heads: int) -> int:
    """Return the query-head count supported by the FlashMLA sparse prefill kernel."""
    sm = torch.cuda.get_device_capability()
    if sm[0] >= 10:
        head_align = 128
    elif sm[0] >= 9:
        head_align = 64
    else:
        raise RuntimeError(
            f"cudnn fused DSA requires SM100+ (Blackwell or later), got SM{sm[0]}{sm[1]}."
        )

    if num_heads % head_align == 0:
        return num_heads
    if num_heads < head_align and head_align % num_heads == 0:
        return head_align
    raise RuntimeError(
        "FlashMLA sparse prefill requires the local query-head count to divide "
        f"{head_align}, got h_q={num_heads}."
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


def _indexer_topk_bshd(
    q_bshd: Tensor,
    k_bsd: Tensor,
    w_bsh: Tensor,
    topk: int,
    varlen_starts: Optional[Tensor] = None,
    varlen_ends: Optional[Tensor] = None,
    key_positions: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """BSHD-layout indexer scoring and top-K selection.

    Args:
        q_bshd: ``(b, sq, idx_nh, idx_hd)`` bf16, C-contiguous.
        k_bsd:  ``(b, sk, idx_hd)`` bf16, C-contiguous.
        w_bsh:  ``(b, sq, idx_nh)`` bf16, C-contiguous raw weights.
        topk:   number of top-K indices to return per query.

    Returns:
        ``(topk_indices, topk_length, scores)`` where:

        * ``topk_indices``: ``(b, sq, topk)`` int32, invalid slots ``-1``.
        * ``topk_length``:  ``(b, sq)`` int32, per-row valid count.
        * ``scores``: ``(b, sq, sk)`` fp32, scaled scores from
          :attr:`cudnn.DSA.indexer_forward_wrapper` with ``-inf`` on
          causally-masked positions.
    """
    _ensure_dsa_namespace()

    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bsd.shape[1]
    device = q_bshd.device

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
        expected_key_positions = torch.arange(sk, dtype=torch.int64, device=device)
        if not torch.equal(key_positions_i64, expected_key_positions):
            raise RuntimeError("cuDNN fused DSA varlen path expects identity key positions")

    k_bshd = k_bsd.unsqueeze(2)  # (b, sk, 1, idx_hd)

    if starts is None:
        scores = _DSA.indexer_forward_wrapper(
            q_bshd, k_bshd, w_bsh, ratio=_INDEXER_RATIO, sm_scale=_INDEXER_SOFTMAX_SCALE
        )[
            "scores"
        ]  # (b, sq, sk) fp32, -inf on masked positions
        q_idx = torch.arange(sq, device=device)
        seq_lens = ((q_idx + 1) // _INDEXER_RATIO).clamp(max=sk).to(torch.int32).repeat(b)
    else:
        query_positions = (ends - 1).to(dtype=torch.int64)
        if torch.equal(query_positions, torch.arange(sq, dtype=torch.int64, device=device)):
            q_for_scores = q_bshd
            w_for_scores = w_bsh
            gather_positions = None
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

        scores = _DSA.indexer_forward_wrapper(
            q_for_scores,
            k_bshd,
            w_for_scores,
            ratio=_INDEXER_RATIO,
            sm_scale=_INDEXER_SOFTMAX_SCALE,
        )["scores"]
        if gather_positions is not None:
            scores = scores.index_select(1, gather_positions)
        scores = dsa_masking.apply_starts_ends_mask_to_scores(
            scores, starts, ends, key_positions_i64
        )
        seq_lens = ends.clamp(max=sk).to(torch.int32).repeat(b)

    # Top-K selection via the TRT-LLM CuTe-DSL radix kernel.
    n_rows = b * sq
    scores_flat = scores.reshape(n_rows, sk).contiguous()

    topk_k = min(topk, sk)
    tk_result = _DSA.indexer_top_k_wrapper(
        scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=False
    )
    topk_indices = tk_result["indices"].view(b, sq, topk_k)
    in_range = (topk_indices >= 0) & (topk_indices < sk)
    gather_indices = topk_indices.clamp(min=0, max=sk - 1).long()
    gathered_scores = torch.gather(scores, dim=2, index=gather_indices)
    topk_indices = torch.where(
        in_range & torch.isfinite(gathered_scores), topk_indices, torch.full_like(topk_indices, -1)
    )

    if topk_k < topk:
        pad = torch.full((b, sq, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    topk_indices, topk_length = _compact_valid_topk_indices(topk_indices)
    return topk_indices.int(), topk_length, scores


def _sbhd_to_bshd_indexer_inputs(
    q_indexer: Tensor, k_indexer: Tensor, weights: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Permute the indexer inputs SBHD→BSHD once."""
    q_bshd = q_indexer.permute(1, 0, 2, 3).contiguous()
    k_bsd = k_indexer.permute(1, 0, 2).contiguous()
    w_bsh = weights.permute(1, 0, 2).contiguous()
    return q_bshd, k_bsd, w_bsh


def _compute_attn_target(
    q_attn_bshd: Tensor,
    k_attn_bsd: Tensor,
    lse: Tensor,
    topk_indices: Tensor,
    topk_length: Tensor | None,
    softmax_scale: float,
    qhead_per_kv_head: int,
) -> Tensor:
    """Compute ``target`` distribution (L1-normalised head-sum softmax).

    Wraps :attr:`cudnn.DSA.sparse_attn_score_recompute_wrapper`.

    ``lse`` is ``(B, S_q, H_q)`` FP32 from the attention forward pass.
    """
    _ensure_dsa_namespace()
    result = _DSA.sparse_attn_score_recompute_wrapper(
        q_attn_bshd,
        k_attn_bsd,
        lse,
        topk_indices,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        topk_length=topk_length,
    )
    return result["target"]


def _kl_loss_from_target_predict(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    loss_coeff: float,
    calculate_per_token_loss: bool = False,
) -> Tensor:
    """KL(target || predict) reduced over ``(B, S_q)`` and scaled by loss_coeff.

    Rows with no valid top-K positions (early query rows with ratio causal
    masking) contribute 0 to the loss — the sparse score kernels produce
    garbage for those rows, mirroring ``compute_dsa_indexer_loss``'s
    ``row_valid`` handling. The default mean is taken over all ``(B, S_q)``
    positions. Per-token-loss mode returns a raw local sum so finalize can
    apply the global token divisor.
    """
    eps = _CLIP_PROB_MIN
    t = target.clamp(min=eps)
    p = predict.clamp(min=eps)
    kl_per_row = (t * (torch.log(t) - torch.log(p))).sum(dim=-1)  # (B, S_q)

    row_valid = (topk_indices >= 0).any(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
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
    eps = _CLIP_PROB_MIN
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

    kl_terms = target_clamped * (torch.log(target_clamped) - log_predict)
    kl_terms = torch.where(position_valid, kl_terms, torch.zeros_like(kl_terms))
    kl_per_row = kl_terms.sum(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    loss = kl_per_row.sum() if calculate_per_token_loss else kl_per_row.mean()
    return loss_coeff * loss


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
    ) -> Tuple[Tensor, Tensor]:
        """Fused forward: indexer scoring, sparse attention, KL loss, and indexer backward."""
        _ensure_dsa_namespace()

        sq, b, num_heads, d = query.shape
        skv = kv_full.shape[0]
        n_comp = k_indexer.shape[0]

        effective_topk = min(indexer_topk, n_comp)
        attn_sink = torch.full(
            (num_heads,), float("-inf"), dtype=torch.float32, device=query.device
        )

        q_idx_bshd, k_idx_bsd, w_bsh = _sbhd_to_bshd_indexer_inputs(q_indexer, k_indexer, weights)

        topk_indices_cmp, topk_length_cmp, indexer_scores = _indexer_topk_bshd(
            q_idx_bshd,
            k_idx_bsd,
            w_bsh,
            effective_topk,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        )

        if b == 1:
            global_idxs = topk_indices_cmp.reshape(sq * b, effective_topk)
        else:
            global_idxs = _local_to_global_flat(topk_indices_cmp, b)
        topk_length_flat = topk_length_cmp.permute(1, 0).reshape(-1)

        q_flat = query.reshape(sq * b, num_heads, d)
        kv_flat = kv_full.reshape(skv * b, d)
        out_flat, lse = _dsa_fwd_flash_mla(
            q_flat,
            kv_flat,
            global_idxs,
            softmax_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length_flat,
        )

        if loss_coeff > 0:
            # Attention-path tensors are detached; the KL loss trains only the indexer.
            q_attn_bshd = query.detach().permute(1, 0, 2, 3).contiguous()
            k_attn_compressed_bsd = kv_full.detach().permute(1, 0, 2).contiguous()
            lse_bsqh = lse.reshape(sq, b, num_heads).permute(1, 0, 2)
            indexer_loss_coeff = loss_coeff * (b * sq) if calculate_per_token_loss else loss_coeff
            unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)

            if sparse_loss:
                safe_indices = topk_indices_cmp.clamp(min=0).long()
                gathered_scores = torch.gather(indexer_scores, dim=2, index=safe_indices)
                gathered_scores = torch.where(
                    topk_indices_cmp >= 0, gathered_scores, torch.finfo(torch.float32).min
                )
                predict = torch.softmax(gathered_scores, dim=-1)

                target = _compute_attn_target(
                    q_attn_bshd,
                    k_attn_compressed_bsd,
                    lse_bsqh,
                    topk_indices_cmp,
                    topk_length_cmp,
                    softmax_scale,
                    qhead_per_kv_head=num_heads,
                )
                row_valid = (topk_indices_cmp >= 0).any(dim=-1, keepdim=True)
                target = torch.where(row_valid, target, torch.zeros_like(target))
                predict = torch.where(row_valid, predict, torch.zeros_like(predict))
                indexer_loss = _kl_loss_from_target_predict(
                    target, predict, topk_indices_cmp, loss_coeff, calculate_per_token_loss
                )
                attn_score_for_bwd = target.clone()
                index_score_for_bwd = predict.clone()
                ig = _DSA.indexer_backward_wrapper(
                    q_idx_bshd,
                    w_bsh,
                    k_idx_bsd,
                    attn_score_for_bwd,
                    index_score_for_bwd,
                    topk_indices_cmp,
                    sm_scale=_INDEXER_SOFTMAX_SCALE,
                    loss_coeff=indexer_loss_coeff,
                    grad_loss=unit_grad_loss,
                    block_I=128,
                )
            else:
                index_score = indexer_scores
                index_lse = torch.logsumexp(indexer_scores, dim=-1)
                attn_score, attn_l1norm = _compute_dense_attn_score(
                    q_attn_bshd,
                    k_attn_compressed_bsd.unsqueeze(2),
                    lse_bsqh,
                    qhead_per_kv_head=num_heads,
                    softmax_scale=softmax_scale,
                )
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
                    q_idx_bshd,
                    w_bsh,
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
            precomputed_grad_q_indexer = ig["d_index_q"].permute(1, 0, 2, 3).contiguous()
            precomputed_grad_k_indexer = ig["d_index_k"].permute(1, 0, 2).contiguous()
            precomputed_grad_weights = ig["d_weights"].permute(1, 0, 2).contiguous()
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

        d_v = out_flat.shape[-1]
        dO_flat = grad_output.reshape(sq * b, num_heads, d_v)
        valid_rows = ctx.topk_length > 0
        if not bool(valid_rows.all()):
            valid_rows_q = valid_rows.view(-1, 1, 1)
            valid_rows_lse = valid_rows.view(-1, 1)
            dO_flat = torch.where(valid_rows_q, dO_flat, torch.zeros_like(dO_flat))
            out_flat = torch.where(valid_rows_q, out_flat, torch.zeros_like(out_flat))
            lse = torch.where(valid_rows_lse, lse, torch.zeros_like(lse))

        attn_bwd = _DSA.sparse_attention_backward_wrapper(
            q_flat,
            kv_flat,
            out_flat,
            dO_flat,
            lse,
            attn_sink,
            global_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=ctx.topk_length,
        )
        grad_query = torch.nan_to_num(attn_bwd["dq"], nan=0.0, posinf=0.0, neginf=0.0).reshape(
            sq, b, num_heads, d
        )
        grad_kv_full = torch.nan_to_num(attn_bwd["dkv"], nan=0.0, posinf=0.0, neginf=0.0).reshape(
            skv, b, d
        )

        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss

        # Grads: query, kv_full, q_indexer, k_indexer, weights, indexer_topk,
        #   softmax_scale, loss_coeff, sparse_loss, calculate_per_token_loss, d_v,
        #   varlen_starts, varlen_ends, key_positions.
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
    )


__all__ = ["fused_indexer_sparse_attn", "run_fused_dsa_attention"]

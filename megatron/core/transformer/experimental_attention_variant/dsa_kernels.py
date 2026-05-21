# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
DSA kernel wrappers for Megatron's DSv4 sparse attention.

Mirrors the three integration paths of the old standalone ``dsa_kernels``
package, but built on top of

* :mod:`cudnn.deepseek_sparse_attention` (a.k.a. ``DSA``) — CuTe-DSL backward
  + indexer score kernels + TRT-LLM radix top-K, shipped as part of
  cuDNN Frontend.
* :mod:`flash_mla` — production sparse-attention forward kernel, expected to
  be available as a separate PyPI package.

Public API (same shape as the old ``dsa_kernels`` package):

* ``build_flat_topk_idxs`` / ``local_to_global_flat`` — index helpers.
* ``dsa_sparse_attn`` — Path A / Path C step 2, differentiable sparse attention.
* ``indexer_topk`` — Path C inference indexer scoring + top-K.
* ``fused_indexer_sparse_attn`` — Path B training, fused indexer loss +
  sparse attention with shared backward.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Lazy kernel imports
# ---------------------------------------------------------------------------


_flash_mla_sparse_fwd = None
_DSA = None


def _ensure_flash_mla():
    """Lazily import the FlashMLA sparse-forward kernel.

    FlashMLA ships ``flash_mla_sparse_fwd`` with a multi-head-KV signature;
    :func:`_dsa_fwd_flash_mla` below is a thin adapter that unbatches the
    DSA-shape inputs and pads ``TopK`` to the alignment expected by
    FlashMLA's SM90 / SM100 kernels.
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
    """Minimum ``TopK`` alignment required by the current GPU architecture.

    * SM90 : dual-warpgroup loop steps by 2 blocks → ``2 * B_TOPK = 128``
    * SM100: single-pipeline loop steps by 1 block → ``B_TOPK`` (64 for
      head64, 128 for head128). DSA uses ``D = 512`` which maps to the
      head64 kernel path → 64.
    """
    sm = torch.cuda.get_device_capability()
    if sm[0] >= 10:
        return 64
    return 128


def _dsa_fwd_flash_mla(
    q: Tensor,
    kv: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: Optional[Tensor] = None,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """DSA-shaped adapter around :func:`flash_mla.flash_mla_sparse_fwd`.

    Accepts flat (unbatched) tensors with global indices; pads ``TopK`` to
    the GPU-specific alignment; returns ``(out, lse, lse_indexer)``.
    """
    assert not (
        indexer_topk > 0 and topk_length is not None
    ), "indexer_topk > 0 requires non-compact mode (topk_length must be None)"
    _ensure_flash_mla()

    _total_S_q, _H, _D = q.shape
    TopK = topk_idxs.shape[-1]
    topk_align = _get_topk_alignment()
    TopK_padded = (TopK + topk_align - 1) // topk_align * topk_align
    if TopK_padded != TopK:
        pad_width = TopK_padded - TopK
        topk_idxs = torch.nn.functional.pad(topk_idxs, (0, pad_width), value=-1)

    kv_3d = kv.unsqueeze(1)  # (total_S_kv, 1, D)  h_kv=1
    indices = topk_idxs.unsqueeze(1)  # (total_S_q, 1, TopK_padded) h_kv=1

    with torch.cuda.nvtx.range("flash_mla_sparse_fwd"):
        res = _flash_mla_sparse_fwd(
            q,
            kv_3d,
            indices,
            softmax_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=indexer_topk,
        )
        if indexer_topk > 0:
            out, _max_logits, lse, lse_indexer = res
        else:
            out, _max_logits, lse = res
            lse_indexer = None

    if indexer_topk > 0:
        # When indexer_topk == total TopK, lse_indexer should equal lse but
        # the kernel may not snapshot correctly; fall back to lse.
        if indexer_topk >= TopK:
            return out, lse, lse.clone()
        return out, lse, lse_indexer
    return out, lse, None


def _ensure_dsa_namespace():
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


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def local_to_global_flat(local_idxs: Tensor, batch_size: int, seqlen_kv: int) -> Tensor:
    """Convert local per-batch indices to global flat indices.

    Follows the convention used by FlashMLA / SparseAttentionBackward:
    flat row order is SBHD ``row[s * B + b]``; global index is
    ``local * B + b`` for valid entries and ``-1`` otherwise.

    Args:
        local_idxs: ``(b, sq, topk)`` int, values in ``[0, seqlen_kv)`` or -1.
        batch_size: ``B``.
        seqlen_kv: KV sequence length per batch (used for shape assertions
            only; callers compute the values).

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


def build_flat_topk_idxs(
    *idx_groups: Tensor, batch_size: int, seqlen_kv: int, compact: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    """Combine local per-batch index groups and convert to flat global form.

    Each *idx_group* is ``(b, sq, topk_i)`` with local per-batch KV indices
    (already in ``kv_full`` index space, i.e. with any compressed-position
    offset applied). ``-1`` marks invalid positions.

    Args:
        *idx_groups: one or more ``(b, sq, topk_i)`` int tensors.
        batch_size: ``B``.
        seqlen_kv: total KV sequence length per batch.
        compact: if True, pack valid entries to the front of each row and
            additionally return ``topk_length``; if False, leave as-is and
            return ``None``.

    Returns:
        ``(topk_idxs, topk_length)`` where
        ``topk_idxs`` is ``(sq*b, total_topk)`` int32 (flat global) and
        ``topk_length`` is ``(sq*b,)`` int32 when ``compact``, else ``None``.
    """
    combined = torch.cat(idx_groups, dim=-1)  # (b, sq, total_topk)
    b, sq, total_topk = combined.shape

    # Globalize first, compact second. Both ops are element-wise + (-1)-preserving,
    # so swapping the order is a no-op for correctness; the win is that the
    # global indices come out already in (sq*b, total_topk) flat layout, which is
    # exactly the row order the cuDNN compactify kernel returns its per-row
    # ``length`` in — no extra permute on the length tensor.
    global_idxs = local_to_global_flat(combined, b, seqlen_kv)

    topk_length_flat = None
    if compact:
        if global_idxs.is_cuda:
            # Fast path: single warp-per-row CuTe DSL kernel from cuDNN's DSA
            # namespace. Replaces a stable argsort + gather + sum + permute
            # chain with one global-load + global-store per element.
            _ensure_dsa_namespace()
            res = _DSA.compactify_wrapper(global_idxs)
            global_idxs, topk_length_flat = res["indices"], res["topk_length"]
        else:
            # CPU fallback so the unit tests that exercise this helper without
            # CUDA still work. Production callers always go through the CUDA
            # path above.
            valid_mask = global_idxs >= 0
            sorted_indices = valid_mask.int().argsort(dim=-1, descending=True, stable=True)
            global_idxs = global_idxs.gather(-1, sorted_indices)
            topk_length_flat = valid_mask.sum(dim=-1).int()

    return global_idxs, topk_length_flat


# ---------------------------------------------------------------------------
# Path A + Path C step 2: differentiable sparse attention
# ---------------------------------------------------------------------------


class SparseAttnFunc(torch.autograd.Function):
    """SM100 sparse attention fwd + bwd on flat tensors.

    Forward uses :mod:`flash_mla`; backward uses cuDNN Frontend's
    :attr:`cudnn.DSA.sparse_attention_backward_wrapper`.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,  # (total_sq, H, D) bf16
        kv: Tensor,  # (total_skv, D) bf16
        attn_sink: Tensor,  # (H,) f32
        topk_idxs: Tensor,  # (total_sq, TopK) int32 global
        topk_length: Optional[Tensor],  # (total_sq,) int32 or None
        softmax_scale: float,
        indexer_topk: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        out, lse, lse_indexer = _dsa_fwd_flash_mla(
            q,
            kv,
            topk_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            indexer_topk=indexer_topk,
        )

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.topk_length = topk_length
        return out, lse, lse_indexer

    @staticmethod
    def backward(ctx, dO, d_lse, d_lse_indexer):
        _ensure_dsa_namespace()

        q, kv, attn_sink, topk_idxs, out, lse = ctx.saved_tensors

        result = _DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dO,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=ctx.topk_length,
        )
        dq, dkv, d_sink = result["dq"], result["dkv"], result["d_sink"]
        return dq, dkv, d_sink, None, None, None, None


def dsa_sparse_attn(
    query: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    softmax_scale: float,
    topk_length: Optional[Tensor] = None,
    indexer_topk: int = 0,
) -> Tensor:
    """Sparse attention (Path A / Path C step 2).

    Args:
        query: ``(sq, b, np, d)`` bf16 SBHD.
        kv:    ``(skv, b, d)`` bf16 SBD (K=V).
        attn_sink: ``(np,)`` f32.
        topk_idxs: ``(sq*b, topk)`` int32 — **flat global** indices produced
            by :func:`build_flat_topk_idxs`.
        softmax_scale: scalar float.
        topk_length: ``(sq*b,)`` int32 — optional compact fast-path. Must be
            ``None`` when ``indexer_topk > 0`` (FlashMLA constraint).
        indexer_topk: int; ``0`` for Paths A/C, positive for Path B to enable
            FlashMLA's ``lse_indexer`` output.

    Returns:
        ``(sq, b, np * d_v)`` bf16 output.
    """
    sq, b, np_, d = query.shape
    skv = kv.shape[0]

    q_flat = query.reshape(sq * b, np_, d)
    kv_flat = kv.reshape(skv * b, d)

    out_flat, _lse, _lse_indexer = SparseAttnFunc.apply(
        q_flat, kv_flat, attn_sink, topk_idxs, topk_length, softmax_scale, indexer_topk
    )

    d_v = out_flat.shape[-1]
    return out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)


# ---------------------------------------------------------------------------
# Path C inference: indexer scoring + top-K
# ---------------------------------------------------------------------------


def _indexer_topk_bshd(
    q_bshd: Tensor, k_bsd: Tensor, w_bsh: Tensor, topk: int, ratio: int = 4
) -> Tuple[Tensor, Tensor]:
    """BSHD-layout core for :func:`indexer_topk`.

    Internal entry point used by both the public SBHD wrapper and Path B's
    ``FusedIndexerSparseAttnFunc.forward`` so the SBHD→BSHD permute can be
    performed once at the call site and reused across both the indexer
    forward and the score-backward kernels (predict / target).

    Args:
        q_bshd: ``(b, sq, idx_nh, idx_hd)`` bf16, C-contiguous.
        k_bsd:  ``(b, sk, idx_hd)`` bf16, C-contiguous.
        w_bsh:  ``(b, sq, idx_nh)`` bf16, C-contiguous, **already
            ``indexer_softmax_scale``-scaled** by the caller.
        topk:   number of top-K indices to return per query.
        ratio:  compression ratio for the kernel's causal mask.

    Returns:
        ``(topk_indices (b, sq, topk) int32, topk_length (b, sq) int32)``.
        Invalid slots are ``-1``; ``topk_length`` is the per-row valid count.
    """
    _ensure_dsa_namespace()

    b, sq, _idx_nh, _idx_hd = q_bshd.shape
    sk = k_bsd.shape[1]
    device = q_bshd.device

    k_bshd = k_bsd.unsqueeze(2)  # (b, sk, 1, idx_hd)

    scores = _DSA.indexer_forward_wrapper(q_bshd, k_bshd, w_bsh, ratio=ratio)[
        "scores"
    ]  # (b, sq, sk) fp32, -inf on masked positions

    # Top-K selection via the TRT-LLM CuTe-DSL radix kernel.
    n_rows = b * sq
    scores_flat = scores.reshape(n_rows, sk).contiguous()
    q_idx = torch.arange(sq, device=device)
    valid_per_q = ((q_idx + 1) // ratio).clamp(max=sk).to(torch.int32)  # (sq,)
    seq_lens = valid_per_q.repeat(b)  # (b*sq,), row-major over (b, sq)

    topk_k = min(topk, sk)
    tk_result = _DSA.indexer_top_k_wrapper(
        scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=False
    )
    topk_indices = tk_result["indices"].view(b, sq, topk_k)

    if topk_k < topk:
        pad = torch.full((b, sq, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    topk_length = (topk_indices >= 0).sum(dim=-1).int()  # (b, sq)
    return topk_indices.int(), topk_length


def _sbhd_to_bshd_indexer_inputs(
    q_indexer: Tensor, k_indexer: Tensor, weights: Tensor, indexer_softmax_scale: float
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Permute the indexer inputs SBHD→BSHD once, returning both the raw
    BSHD weights and (when needed) a separate scaled copy.

    The ``relu(c·x) = c·relu(x)`` trick lets us push the indexer softmax
    scale onto ``W`` (``(B, S_q, H)``, small) instead of the score tensor
    (``(B, S_q, S_k)``, big). The raw ``w_bsh`` is preserved for the
    backward GEMM path, which takes ``sm_scale`` directly. When
    ``indexer_softmax_scale == 1.0`` the two views alias each other.

    Returns ``(q_bshd, k_bsd, w_bsh, w_bsh_scaled)``.
    """
    q_bshd = q_indexer.permute(1, 0, 2, 3).contiguous()
    k_bsd = k_indexer.permute(1, 0, 2).contiguous()
    w_bsh = weights.permute(1, 0, 2).contiguous()

    if indexer_softmax_scale != 1.0:
        w_bsh_scaled = (w_bsh.float() * indexer_softmax_scale).to(w_bsh.dtype)
    else:
        w_bsh_scaled = w_bsh

    return q_bshd, k_bsd, w_bsh, w_bsh_scaled


def indexer_topk(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk: int,
    ratio: int = 4,
    indexer_softmax_scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Score + top-K selection for inference (no KL loss, no backward).

    Built on cuDNN Frontend's CuTe-DSL indexer forward kernel followed by
    TRT-LLM's radix top-K kernel.

    Args:
        q_indexer: ``(sq, b, idx_nh, idx_hd)`` bf16 SBHD.
        k_indexer: ``(sk, b, idx_hd)`` bf16 SBD.
        weights:   ``(sq, b, idx_nh)`` bf16 SBH — raw (unscaled) weights.
        topk: number of top-K indices to select.
        ratio: compression ratio for the causal mask.
        indexer_softmax_scale: scale applied to the indexer ``Q @ K^T``
            scores (typically ``idx_hd ** -0.5``). Applied internally via
            the weights-scaling trick (``relu(c·x) = c·relu(x)`` for
            ``c > 0``) so the caller passes raw weights. Default ``1.0``
            means weights are treated as already-scaled.

    Returns:
        topk_indices: ``(b, sq, topk)`` int32 — local per-batch indices into
            ``k_indexer``; invalid positions are ``-1``.
        topk_length:  ``(b, sq)`` int32 — per-query valid count.
    """
    q_bshd, k_bsd, _w_bsh_raw, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
        q_indexer, k_indexer, weights, indexer_softmax_scale
    )
    return _indexer_topk_bshd(q_bshd, k_bsd, w_bsh_scaled, topk, ratio)


# ---------------------------------------------------------------------------
# Path B: fused indexer + sparse attention (training)
# ---------------------------------------------------------------------------


_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny  # kept compatible w/ cudnn kernel


def _compute_indexer_predict(
    q_indexer_bshd: Tensor,
    k_indexer_bsd: Tensor,
    weights_bsh: Tensor,
    topk_indices: Tensor,
    qhead_per_kv_head: int,
) -> Tensor:
    """Compute ``predict`` distribution (softmax over top-K of indexer scores).

    Wraps :attr:`cudnn.DSA.sparse_indexer_score_recompute_wrapper`.

    Args:
        q_indexer_bshd: ``(B, S_q, H_q, D)`` bf16.
        k_indexer_bsd:  ``(B, S_k, D)`` bf16.
        weights_bsh:    ``(B, S_q, H_q)`` bf16.
        topk_indices:   ``(B, S_q, topk)`` int32.
        qhead_per_kv_head: ``H_q`` (MQA).

    Returns:
        predict: ``(B, S_q, topk)`` fp32, softmax over the top-K axis.
    """
    _ensure_dsa_namespace()
    result = _DSA.sparse_indexer_score_recompute_wrapper(
        q_indexer_bshd,
        k_indexer_bsd,
        weights_bsh,
        topk_indices,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    return result["predict"]


def _compute_attn_target(
    q_attn_bshd: Tensor,
    k_attn_bsd: Tensor,
    lse: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    qhead_per_kv_head: int,
) -> Tensor:
    """Compute ``target`` distribution (L1-normalised head-sum softmax).

    Wraps :attr:`cudnn.DSA.sparse_attn_score_recompute_wrapper`.

    Shapes match :func:`_compute_indexer_predict`; ``lse`` is
    ``(B, S_q, H_q)`` FP32 (comes from the attention forward pass).
    """
    _ensure_dsa_namespace()
    result = _DSA.sparse_attn_score_recompute_wrapper(
        q_attn_bshd,
        k_attn_bsd,
        lse,
        topk_indices,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    return result["target"]


def _kl_loss_from_target_predict(
    target: Tensor, predict: Tensor, topk_indices: Tensor, loss_coeff: float
) -> Tensor:
    """KL(target || predict) averaged over ``(B, S_q)`` and scaled by loss_coeff.

    Rows with no valid top-K positions (early query rows with ratio causal
    masking) contribute 0 to the loss — the sparse score kernels produce
    garbage for those rows, mirroring ``compute_dsa_indexer_loss``'s
    ``row_valid`` handling. The mean is taken over all ``(B, S_q)``
    positions (matching the reference's ``kl_per_element.sum(-1).mean()``).
    """
    eps = _CLIP_PROB_MIN
    t = target.clamp(min=eps)
    p = predict.clamp(min=eps)
    kl_per_row = (t * (torch.log(t) - torch.log(p))).sum(dim=-1)  # (B, S_q)

    row_valid = (topk_indices >= 0).any(dim=-1)  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    return loss_coeff * kl_per_row.mean()


# ---------------------------------------------------------------------------
# Dense path (``sparse_loss=False``) — full-KV indexer loss
# ---------------------------------------------------------------------------


def _compute_dense_indexer_score(
    q_indexer_bshd: Tensor,
    k_indexer_bshd: Tensor,
    weights_bsh: Tensor,
    qhead_per_kv_head: int,
    indexer_softmax_scale: float,
    ratio: int,
) -> Tuple[Tensor, Tensor]:
    """Dense indexer score forward over the full ``S_k`` axis.

    Wraps :attr:`cudnn.DSA.dense_indexer_score_recompute_wrapper`. Returns
    ``(out, denom)`` where

    * ``out``    : ``(B, S_q, S_k)`` fp32, the raw head-reduced score
      ``S[b,q,k] = indexer_softmax_scale * sum_h ReLU(Q_h · K_k^T) · W_{b,q,h}``
      with the kernel's ``ratio``-causal mask applied to invalid columns.
    * ``denom``  : ``(B, S_q)`` fp32, the LSE denom of ``out`` along
      ``S_k`` — i.e. ``predict = exp(out - denom[..., None])`` is the
      indexer softmax distribution over the full KV.

    Both outputs are forwarded into :func:`_kl_loss_from_dense_scores`
    *and* saved for the dense-path backward, where the dense indexer-grad
    kernel consumes them directly.
    """
    _ensure_dsa_namespace()
    result = _DSA.dense_indexer_score_recompute_wrapper(
        q_indexer_bshd,
        k_indexer_bshd,
        weights_bsh,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=indexer_softmax_scale,
        ratio=ratio,
    )
    return result["out"], result["denom"]


def _compute_dense_attn_score(
    q_attn_bshd: Tensor,
    k_attn_bshd: Tensor,
    lse: Tensor,
    qhead_per_kv_head: int,
    softmax_scale: float,
    ratio: int,
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
        ratio=ratio,
    )
    return result["out"], result["denom"]


def _kl_loss_from_dense_scores(
    attn_score: Tensor,
    attn_l1norm: Tensor,
    index_score: Tensor,
    index_lse: Tensor,
    loss_coeff: float,
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
    log_predict = index_score - safe_lse.unsqueeze(-1)

    kl_per_row = (target_clamped * (torch.log(target_clamped) - log_predict)).sum(
        dim=-1
    )  # (B, S_q)
    kl_per_row = torch.where(row_valid, kl_per_row, torch.zeros_like(kl_per_row))
    return loss_coeff * kl_per_row.mean()


class FusedIndexerSparseAttnFunc(torch.autograd.Function):
    """Path B: fused indexer (+KL loss) + sparse attention in one autograd.

    Differentiable w.r.t. ``query``, ``kv_full``, ``attn_sink``,
    ``q_indexer``, ``k_indexer``, ``weights``.

    Two indexer-loss variants, selected by the ``sparse_loss`` argument
    (matches ``compute_dsa_indexer_loss`` in the reference ``dsa.py``):

    * **Sparse loss** (``sparse_loss=True``) — KL is computed only over
      the top-K KV positions the indexer has selected. Cheap; trains the
      indexer to match attention's distribution *among already-selected*
      positions. Uses
      :attr:`cudnn.DSA.sparse_indexer_score_recompute_wrapper` /
      :attr:`cudnn.DSA.sparse_attn_score_recompute_wrapper` for the
      forward target/predict, and
      :attr:`cudnn.DSA.indexer_backward_wrapper` for the backward.
    * **Dense loss** (``sparse_loss=False``, the default) — KL is
      computed over *all* causally valid KV positions. Pushes the
      indexer to learn the full attention distribution. Uses
      :attr:`cudnn.DSA.dense_indexer_score_recompute_wrapper` /
      :attr:`cudnn.DSA.dense_attn_score_recompute_wrapper` for the
      forward, and
      :attr:`cudnn.DSA.dense_indexer_backward_wrapper` for the
      backward. Memory cost is higher (intermediate ``(B, S_q, S_k)``
      fp32 tensors), but matches the DeepSeek-V3.2 paper formulation.

    Both variants share the FlashMLA sparse-attention forward + the
    cuDNN sparse-attn backward; only the indexer-loss path branches.
    """

    @staticmethod
    def forward(
        ctx,
        # Sparse attn inputs (differentiable)
        query: Tensor,  # (sq, b, np, d) bf16
        kv_full: Tensor,  # (skv, b, d) bf16
        attn_sink: Tensor,  # (np,) f32
        # Window indices (not differentiable)
        window_idxs: Tensor,  # (b, sq, win_topk) int32
        # Indexer inputs (differentiable)
        q_indexer: Tensor,  # (sq, b, idx_nh, idx_hd) bf16
        k_indexer: Tensor,  # (n_comp, b, idx_hd) bf16
        weights: Tensor,  # (sq, b, idx_nh) bf16 — raw (unscaled)
        # Scalars
        indexer_topk: int,
        ratio: int,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        kv_offset: int,
    ) -> Tuple[Tensor, Tensor]:
        _ensure_dsa_namespace()

        sq, b, np_, d = query.shape
        skv = kv_full.shape[0]
        n_comp = k_indexer.shape[0]
        idx_nh, idx_hd = q_indexer.shape[2], q_indexer.shape[3]

        effective_topk = min(indexer_topk, n_comp)

        # ---- 1. Permute indexer inputs SBHD->BSHD ONCE. -------------------
        q_idx_bshd, k_idx_bsd, w_bsh, w_bsh_scaled = _sbhd_to_bshd_indexer_inputs(
            q_indexer, k_indexer, weights, indexer_softmax_scale
        )

        # ---- 2. Indexer scoring + top-K (cudnn-frontend). -----------------
        topk_indices_cmp, _ = _indexer_topk_bshd(
            q_idx_bshd, k_idx_bsd, w_bsh_scaled, effective_topk, ratio
        )  # (b, sq, effective_topk) int32, -1 for invalid

        # ---- 3. Combine indices (indexer first, then window). --------------
        compress_topk_idxs = torch.where(topk_indices_cmp >= 0, topk_indices_cmp + kv_offset, -1)
        combined_local = torch.cat([compress_topk_idxs, window_idxs], dim=-1)
        global_idxs = local_to_global_flat(combined_local, b, skv)

        # ---- 4. FlashMLA forward (non-compact, indexer_topk > 0). ---------
        q_flat = query.reshape(sq * b, np_, d)
        kv_flat = kv_full.reshape(skv * b, d)
        out_flat, lse, lse_indexer = _dsa_fwd_flash_mla(
            q_flat,
            kv_flat,
            global_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=None,
            indexer_topk=effective_topk,
        )

        # Attention-path tensors (detached — loss is not differentiable through them).
        q_attn_bshd = query.detach().permute(1, 0, 2, 3).contiguous()
        k_attn_compressed_bsd = kv_full[kv_offset:].detach().permute(1, 0, 2).contiguous()

        lse_indexer_bsqh = lse_indexer.reshape(sq, b, np_).permute(1, 0, 2)

        # ---- 5. Compute predict + target + KL loss. -----------------------
        # Branch on ``sparse_loss``: sparse uses (predict, target) of shape
        # ``(B, S_q, topk)`` and trains only over selected positions; dense
        # uses raw scores + denoms of shape ``(B, S_q, S_k)`` and trains
        # over the full causal KV. See class docstring for details.
        if sparse_loss:
            predict = _compute_indexer_predict(
                q_idx_bshd, k_idx_bsd, w_bsh_scaled, topk_indices_cmp, qhead_per_kv_head=idx_nh
            )
            target = _compute_attn_target(
                q_attn_bshd,
                k_attn_compressed_bsd,
                lse_indexer_bsqh,
                topk_indices_cmp,
                softmax_scale,
                qhead_per_kv_head=np_,
            )

            if loss_coeff > 0:
                indexer_loss = _kl_loss_from_target_predict(
                    target, predict, topk_indices_cmp, loss_coeff
                )
            else:
                indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)

            ctx.save_for_backward(
                q_flat,
                kv_flat,
                attn_sink,
                global_idxs,
                out_flat,
                lse,
                q_idx_bshd,
                k_idx_bsd,
                w_bsh,
                topk_indices_cmp,
                target,
                predict,
            )
        else:
            # Dense kernels expect K as 4-D ``(B, S_k, H_kv, D)``; our K is
            # 3-D MQA. The unsqueeze is a free view (size-1 head dim).
            index_score, index_lse = _compute_dense_indexer_score(
                q_idx_bshd,
                k_idx_bsd.unsqueeze(2),
                w_bsh,
                qhead_per_kv_head=idx_nh,
                indexer_softmax_scale=indexer_softmax_scale,
                ratio=ratio,
            )
            attn_score, attn_l1norm = _compute_dense_attn_score(
                q_attn_bshd,
                k_attn_compressed_bsd.unsqueeze(2),
                lse_indexer_bsqh,
                qhead_per_kv_head=np_,
                softmax_scale=softmax_scale,
                ratio=ratio,
            )

            if loss_coeff > 0:
                indexer_loss = _kl_loss_from_dense_scores(
                    attn_score, attn_l1norm, index_score, index_lse, loss_coeff
                )
            else:
                indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)

            ctx.save_for_backward(
                q_flat,
                kv_flat,
                attn_sink,
                global_idxs,
                out_flat,
                lse,
                q_idx_bshd,
                k_idx_bsd,
                w_bsh,
                attn_score,
                attn_l1norm,
                index_score,
                index_lse,
            )

        # ---- 6. Save context. ---------------------------------------------
        ctx.sparse_loss = sparse_loss
        ctx.softmax_scale = softmax_scale
        ctx.indexer_softmax_scale = indexer_softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.ratio = ratio
        ctx.sq = sq
        ctx.b = b
        ctx.np_ = np_
        ctx.d = d
        ctx.skv = skv
        ctx.n_comp = n_comp
        ctx.idx_nh = idx_nh

        # ---- 7. Return. ---------------------------------------------------
        d_v = out_flat.shape[-1]
        output = out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)
        return output, indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        # Saved-tensor layout depends on the loss variant; unpack the common
        # prefix first, then the variant-specific tail.
        saved = ctx.saved_tensors
        (q_flat, kv_flat, attn_sink, global_idxs, out_flat, lse, q_idx_bshd, k_idx_bsd, w_bsh) = (
            saved[:9]
        )

        sq, b, np_, d = ctx.sq, ctx.b, ctx.np_, ctx.d
        skv = ctx.skv

        # ---- 1. Sparse attn backward. -------------------------------------
        d_v = out_flat.shape[-1]
        dO_flat = grad_output.reshape(sq * b, np_, d_v)

        attn_bwd = _DSA.sparse_attention_backward_wrapper(
            q_flat,
            kv_flat,
            out_flat,
            dO_flat,
            lse,
            attn_sink,
            global_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=None,
        )
        grad_query = attn_bwd["dq"].reshape(sq, b, np_, d)
        grad_kv_full = attn_bwd["dkv"].reshape(skv, b, d)
        d_sink = attn_bwd["d_sink"]

        # ---- 2. Indexer backward. -----------------------------------------
        # Both branches share: ``grad_scale = loss_coeff * grad_loss / (B * S_q)``
        # is folded inside the kernel, matching the ``.mean()`` reduction
        # applied to the KL tensor in the forward.
        if ctx.sparse_loss:
            # ``target`` and ``predict`` are mutated in-place by the kernel's
            # score-grad stage; clone to keep them intact in case anyone else
            # looks at them post-backward.
            topk_indices_cmp, target, predict = saved[9:]
            attn_score = target.clone()
            index_score = predict.clone()

            ig = _DSA.indexer_backward_wrapper(
                q_idx_bshd,
                w_bsh,
                k_idx_bsd,
                attn_score,
                index_score,
                topk_indices_cmp,
                sm_scale=ctx.indexer_softmax_scale,
                loss_coeff=ctx.loss_coeff,
                grad_loss=grad_loss,
                block_I=128,
            )
        else:
            # Dense path: kernel mutates ``attn_score`` / ``index_score`` in
            # place during its score-grad precompute stage, so we clone the
            # raw scores before passing. ``attn_l1norm`` / ``index_lse`` are
            # only read.
            attn_score, attn_l1norm, index_score, index_lse = saved[9:]
            attn_score_c = attn_score.clone()
            index_score_c = index_score.clone()

            ig = _DSA.dense_indexer_backward_wrapper(
                q_idx_bshd,
                w_bsh,
                k_idx_bsd,
                attn_score_c,
                attn_l1norm,
                index_score_c,
                index_lse,
                sm_scale=ctx.indexer_softmax_scale,
                loss_coeff=ctx.loss_coeff,
                grad_loss=grad_loss,
                ratio=ctx.ratio,
                block_I=128,
            )
        grad_q_indexer_bshd = ig["d_index_q"]
        grad_weights_bsh = ig["d_weights"]
        grad_k_indexer_bsd = ig["d_index_k"]

        # BSHD -> SBHD (match input layout).
        grad_q_indexer = grad_q_indexer_bshd.permute(1, 0, 2, 3).contiguous()
        grad_k_indexer = grad_k_indexer_bsd.permute(1, 0, 2).contiguous()
        grad_weights = grad_weights_bsh.permute(1, 0, 2).contiguous()

        # Grads: query, kv_full, attn_sink, window_idxs, q_indexer, k_indexer,
        #   weights, indexer_topk, ratio, softmax_scale, indexer_softmax_scale,
        #   loss_coeff, sparse_loss, kv_offset
        return (
            grad_query,
            grad_kv_full,
            d_sink,
            None,
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
        )


def fused_indexer_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    window_idxs: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk: int,
    ratio: int,
    softmax_scale: float,
    indexer_softmax_scale: float = 1.0,
    loss_coeff: float = 0.0,
    sparse_loss: bool = False,
    kv_offset: int = 0,
) -> Tuple[Tensor, Tensor]:
    """Path B (training): fused indexer (+KL loss) + sparse attention.

    See :class:`FusedIndexerSparseAttnFunc` for the detailed data flow.

    Args:
        query:        ``(sq, b, np, d)`` bf16 SBHD — attention query.
        kv_full:      ``(skv, b, d)`` bf16 SBD — original + compressed KV.
        attn_sink:    ``(np,)`` f32 — learnable sink per head.
        window_idxs:  ``(b, sq, win_topk)`` int32 — local window indices.
        q_indexer:    ``(sq, b, idx_nh, idx_hd)`` bf16 — indexer query.
        k_indexer:    ``(n_comp, b, idx_hd)`` bf16 — indexer key (compressed).
        weights:      ``(sq, b, idx_nh)`` bf16 — raw indexer weights.
        indexer_topk: number of top-K compressed positions to select.
        ratio:        compression ratio used for the causal mask.
        softmax_scale: attention ``Q @ K^T`` scale, typically
            ``1/sqrt(v_head_dim)``.
        indexer_softmax_scale: indexer ``Q @ K^T`` scale, typically
            ``1/sqrt(idx_hd)``. Applied internally — caller passes raw
            (unscaled) ``weights``.
        loss_coeff:   coefficient scaling the KL divergence loss.
        sparse_loss:  if ``True``, KL is computed only over the top-K
            positions (cheap, less informative); if ``False`` (the
            default, matches ``transformer_config.dsa_indexer_use_sparse_loss``),
            KL is computed over the full causally-valid KV (more
            informative, matches the DeepSeek-V3.2 paper, larger
            intermediate-tensor footprint). See
            :class:`FusedIndexerSparseAttnFunc` for the full data flow
            of each variant.
        kv_offset:    start of compressed region within ``kv_full``.

    Returns:
        ``(output, indexer_loss)`` where ``output`` is ``(sq, b, np * d_v)``
        bf16 and ``indexer_loss`` is a scalar f32.
    """
    return FusedIndexerSparseAttnFunc.apply(
        query,
        kv_full,
        attn_sink,
        window_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk,
        ratio,
        softmax_scale,
        indexer_softmax_scale,
        loss_coeff,
        sparse_loss,
        kv_offset,
    )


__all__ = [
    "build_flat_topk_idxs",
    "local_to_global_flat",
    "dsa_sparse_attn",
    "indexer_topk",
    "fused_indexer_sparse_attn",
]

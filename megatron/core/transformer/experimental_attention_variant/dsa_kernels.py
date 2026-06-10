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
* ``dsa_sparse_attn`` — differentiable sparse attention with precomputed indices.
* ``indexer_topk`` — inference indexer scoring + top-K.
* ``fused_indexer_sparse_attn`` — training fused indexer loss +
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


def batch_of_row(cu_seqlens_q: Tensor, total_q: Optional[int] = None) -> Tensor:
    """For a THD-packed query of length ``total_q``, return a ``(total_q,)``
    int64 tensor where entry ``i`` is the index of the segment that owns
    query row ``i`` (i.e. the unique ``b`` with
    ``cu_seqlens_q[b] <= i < cu_seqlens_q[b+1]``).

    Used by every helper that needs to translate between per-row indices
    and per-segment cumulative tensors.

    Args:
        cu_seqlens_q: ``(B+1,)`` int — cumulative Q lengths.
        total_q: optional row count override; defaults to
            ``int(cu_seqlens_q[-1].item())`` (forces a GPU→CPU sync).

    Returns:
        ``(total_q,)`` int64.
    """
    if total_q is None:
        total_q = int(cu_seqlens_q[-1].item())
    row_idx = torch.arange(total_q, device=cu_seqlens_q.device, dtype=torch.int64)
    return torch.bucketize(row_idx, cu_seqlens_q[1:], right=True)


def local_to_global_flat(
    local_idxs: Tensor,
    batch_size: int,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
) -> Tensor:
    """Convert local per-sequence indices to global flat indices.

    Follows the convention used by FlashMLA / SparseAttentionBackward:
    flat row order is SBHD ``row[s * B + b]``; global index is
    ``local * B + b`` for valid entries and ``-1`` otherwise.
    Two layouts are supported:

    * **SBHD-flat (default, ``cu_seqlens_*=None``)** — the convention used
      by FlashMLA / SparseAttentionBackward when packing a fixed-shape
      batch: flat row order is ``row[s * B + b]``; global index is
      ``local * B + b`` for valid entries and ``-1`` otherwise. Inputs
      are ``(b, sq, topk)``; outputs are ``(sq*b, topk)``.
    * **THD packed (``cu_seqlens_*`` supplied)** — for variable-length
      packed sequences. Both ``cu_seqlens_q`` and ``cu_seqlens_kv``
      must be supplied as 1-D int32 tensors of length ``B+1``. Flat row
      order is the natural ``(total_q,)`` order; global index is
      ``cu_seqlens_kv[batch_of_q] + local`` for valid entries and ``-1``
      otherwise. Inputs are ``(total_q, topk)``; outputs are
      ``(total_q, topk)``.

    Args:
        local_idxs: SBHD ``(b, sq, topk)`` or THD ``(total_q, topk)`` int.
        batch_size: ``B`` (only consulted in the SBHD branch).
        cu_seqlens_q: optional 1-D ``(B+1,)`` int32 — when present (with
            ``cu_seqlens_kv``), switches to the THD branch.
        cu_seqlens_kv: optional 1-D ``(B+1,)`` int32 — same.

    Returns:
        ``(sq*b, topk)`` int32 in SBHD mode; ``(total_q, topk)`` int32 in
        THD mode.
    """
    if (cu_seqlens_q is None) != (cu_seqlens_kv is None):
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_kv must both be provided for THD, or "
            "both None for SBHD."
        )

    if cu_seqlens_q is None:
        # ---- SBHD-flat path -------------------------------------------------
        b, sq, topk = local_idxs.shape
        assert b == batch_size

        idxs_sb = local_idxs.permute(1, 0, 2).reshape(sq * b, topk)
        valid = idxs_sb >= 0
        batch_ids = torch.arange(sq * b, device=local_idxs.device) % b
        batch_ids_exp = batch_ids.unsqueeze(1).expand_as(idxs_sb)
        idxs_sb = torch.where(valid, idxs_sb * b + batch_ids_exp, idxs_sb)
        return idxs_sb.int()

    # ---- THD packed path ----------------------------------------------------
    # Expect ``local_idxs`` to be (total_q, topk). For each row, look up its
    # batch index from ``cu_seqlens_q``, then add the corresponding KV offset
    # ``cu_seqlens_kv[batch]`` to every valid local index in the row.
    if local_idxs.ndim != 2:
        raise ValueError(f"THD local_idxs must be 2-D (total_q, topk), got {local_idxs.shape}")
    total_q, topk = local_idxs.shape
    if cu_seqlens_q.ndim != 1 or cu_seqlens_kv.ndim != 1:
        raise ValueError("cu_seqlens_q/kv must be 1-D")
    if cu_seqlens_q.shape != cu_seqlens_kv.shape:
        raise ValueError(
            f"cu_seqlens_q.shape={tuple(cu_seqlens_q.shape)} must equal "
            f"cu_seqlens_kv.shape={tuple(cu_seqlens_kv.shape)}"
        )

    row_batch_ids = batch_of_row(cu_seqlens_q, total_q=total_q)
    kv_offset = cu_seqlens_kv[row_batch_ids].unsqueeze(1)  # (total_q, 1)
    valid = local_idxs >= 0
    global_idxs = torch.where(valid, local_idxs + kv_offset, local_idxs)
    return global_idxs.int()


def build_flat_topk_idxs(
    *idx_groups: Tensor,
    batch_size: int,
    compact: bool = False,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Combine local per-sequence index groups and convert to flat global form.

    Each *idx_group* contains local per-sequence KV indices (already in
    ``kv_full`` index space, i.e. with any compressed-position offset
    applied). ``-1`` marks invalid positions. The shape of each group
    differs by layout:

    * **SBHD-flat** (``cu_seqlens_*=None``, default): each group is
      ``(b, sq, topk_i)``; outputs are ``(sq*b, total_topk)`` (flat
      SBHD with row order ``s*B + b``).
    * **THD packed** (``cu_seqlens_*`` supplied): each group is
      ``(total_q, topk_i)`` with ``total_q = cu_seqlens_q[-1]``;
      outputs are ``(total_q, total_topk)``.

    Args:
        *idx_groups: one or more index tensors, all of the same layout.
        batch_size: ``B`` (only consulted in SBHD).
        compact: if True, pack valid entries to the front of each row and
            additionally return ``topk_length``; if False, leave as-is and
            return ``None``.
        cu_seqlens_q: optional 1-D ``(B+1,)`` int32 — selects THD branch.
        cu_seqlens_kv: optional 1-D ``(B+1,)`` int32 — selects THD branch.

    Returns:
        ``(topk_idxs, topk_length)`` where the first axis of ``topk_idxs``
        is ``sq*b`` (SBHD) or ``total_q`` (THD), and ``topk_length`` is
        ``(rows,)`` int32 when ``compact``, else ``None``.
    """
    combined = torch.cat(idx_groups, dim=-1)

    # Globalize first, compact second. Both ops are element-wise +
    # ``-1``-preserving, so swapping the order is a no-op for correctness;
    # globalizing first puts the indices into the same flat row order the
    # cuDNN compactify kernel returns its per-row ``length`` in, so no
    # extra permute is needed afterward.
    global_idxs = local_to_global_flat(
        combined, batch_size, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv
    )

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
            # CPU reference branch for unit tests that exercise this helper
            # without CUDA. Production callers go through the CUDA path above.
            valid_mask = global_idxs >= 0
            sorted_indices = valid_mask.int().argsort(dim=-1, descending=True, stable=True)
            global_idxs = global_idxs.gather(-1, sorted_indices)
            topk_length_flat = valid_mask.sum(dim=-1).int()

    return global_idxs, topk_length_flat


# ---------------------------------------------------------------------------
# Differentiable sparse attention with precomputed indices
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
        """Run FlashMLA sparse-attention forward and save tensors for backward."""
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
        """Compute sparse-attention backward via cuDNN DSA wrapper."""
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
    is_thd: bool = False,
) -> Tensor:
    """Sparse attention with precomputed top-k indices.

    Two layouts:

    * **SBHD** (``is_thd=False``, default): ``query`` is ``(sq, b, np, d)``
      and ``kv`` is ``(skv, b, d)``; the wrapper reshapes them to
      ``(sq*b, np, d)`` / ``(skv*b, d)`` before passing to FlashMLA, and
      returns ``(sq, b, np * d_v)``.
    * **THD packed** (``is_thd=True``): ``query`` is already
      ``(total_sq, np, d)`` (3-D) and ``kv`` is ``(total_skv, d)`` (2-D).
      No reshape is needed; output is ``(total_sq, np * d_v)`` with a
      leading 2-D layout that the caller can fold into its own packed
      representation.

    Args:
        query: SBHD ``(sq, b, np, d)`` or THD ``(total_sq, np, d)`` bf16.
        kv:    SBD ``(skv, b, d)`` or THD ``(total_skv, d)`` bf16 (K=V).
        attn_sink: ``(np,)`` f32.
        topk_idxs: ``(rows, topk)`` int32 — **flat global** indices produced
            by :func:`build_flat_topk_idxs` in the matching layout.
        softmax_scale: scalar float.
        topk_length: ``(rows,)`` int32 — optional compact fast-path. Must be
            ``None`` when ``indexer_topk > 0`` (FlashMLA constraint).
        indexer_topk: int; ``0`` when no indexer columns are appended,
            positive when fused indexer-loss columns are appended.
        is_thd: when True, treat ``query`` and ``kv`` as already-packed
            THD tensors and skip the SBHD reshape steps.

    Returns:
        SBHD ``(sq, b, np * d_v)`` or THD ``(total_sq, np * d_v)`` bf16.
    """
    # Layout-specific input pre-reshape — the kernel always consumes a
    # flat ``(rows, np, d)`` query and ``(n_kv, d)`` KV; only the rows
    # axis interpretation differs (rows = ``total_sq`` for THD, rows =
    # ``sq * b`` for SBHD). ``topk_idxs`` is already a flat ``(rows, k)``
    # tensor in both layouts (built by :func:`build_flat_topk_idxs`).
    if is_thd:
        if query.ndim != 3:
            raise ValueError(
                f"THD dsa_sparse_attn expects query of shape "
                f"(total_sq, np, d), got {tuple(query.shape)}"
            )
        if kv.ndim != 2:
            raise ValueError(
                f"THD dsa_sparse_attn expects kv of shape (total_skv, d), " f"got {tuple(kv.shape)}"
            )
        q_flat, kv_flat = query, kv
    else:
        sq, b, np_, d = query.shape
        skv = kv.shape[0]
        q_flat = query.reshape(sq * b, np_, d)
        kv_flat = kv.reshape(skv * b, d)

    out_flat, _lse, _lse_indexer = SparseAttnFunc.apply(
        q_flat, kv_flat, attn_sink, topk_idxs, topk_length, softmax_scale, indexer_topk
    )  # (rows, np, d_v)

    # Layout-specific output reshape: collapse (np, d_v) → (np * d_v),
    # then THD stays flat (rows = total_sq); SBHD reflates the (sq, b) axes.
    np_, d_v = out_flat.shape[1], out_flat.shape[-1]
    if is_thd:
        return out_flat.reshape(-1, np_ * d_v)
    return out_flat.reshape(sq, b, np_ * d_v)


# ---------------------------------------------------------------------------
# Inference indexer scoring + top-K
# ---------------------------------------------------------------------------


def _indexer_topk_core(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    topk: int,
    ratio: int = 4,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    fixed_topk_width: Optional[int] = None,
    compute_topk_length: bool = True,
    precomputed_seq_lens: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Layout-agnostic core for :func:`indexer_topk`.

    Wraps cuDNN Frontend's CuTe-DSL indexer-forward kernel.
    The pipeline is indexer forward, per-row valid lengths, radix top-K, optional
    fixed-width output padding, and optionally ``topk_length``. Selected by
    ``cu_seqlens_q``.

    BSHD layout (``cu_seqlens_q is None``):
        q: ``(b, sq, idx_nh, idx_hd)`` bf16, C-contiguous.
        k: ``(b, sk, idx_hd)`` bf16, C-contiguous.
        w: ``(b, sq, idx_nh)`` bf16, C-contiguous, **already
           ``indexer_softmax_scale``-scaled by the caller**.
        Returns:
            ``(topk_indices (b, sq, topk) int32,
               topk_length  (b, sq)      int32)`` — invalid slots ``-1``.

    THD packed layout (``cu_seqlens_q is not None``):
        q: ``(total_q, idx_nh, idx_hd)`` bf16.
        k: ``(total_k, idx_hd)`` bf16.
        w: ``(total_q, idx_nh)`` bf16, already scaled.
        cu_seqlens_q/kv, max_seqlen_q/kv: standard packed args.
        Returns:
            ``(topk_indices (total_q, topk) int32,
               topk_length  (total_q,)     int32)`` — per-batch LOCAL ids
            in ``[0, seqlen_kv[batch])``; use :func:`local_to_global_flat`
            (with ``cu_seqlens_q/kv``) to promote to flat-global ids.

    Two internal entry points besides :func:`indexer_topk`:

    * ``FusedIndexerSparseAttnFunc.forward`` calls this directly
      so the SBHD→BSHD permute can be performed once and reused across
      the indexer forward and the score-recompute backward kernels.
    """
    _ensure_dsa_namespace()
    is_thd = cu_seqlens_q is not None
    device = q.device

    # ---------------- Layout-specific input prep ------------------------
    if is_thd:
        if q.ndim != 3:
            raise ValueError(f"THD q must be (total_q, idx_nh, idx_hd), got {q.shape}")
        if k.ndim != 2:
            raise ValueError(f"THD k must be (total_k, idx_hd), got {k.shape}")
        if w.ndim != 2:
            raise ValueError(f"THD w must be (total_q, idx_nh), got {w.shape}")

        # Kernel wants k as 3-D ``(total_k, h_kv, idx_hd)``.
        scores = _DSA.indexer_forward_wrapper(
            q,
            k.unsqueeze(1),
            w,
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_kv),
        )[
            "scores"
        ]  # (total_q, max_seqlen_kv) fp32, -inf on masked positions
        # Defensive contiguify (wrapper may return a stride-padded slice).
        scores_flat = scores.contiguous()
        sk = int(max_seqlen_kv)
        total_q = q.shape[0]

        if precomputed_seq_lens is not None:
            if precomputed_seq_lens.shape[0] != total_q:
                raise ValueError(
                    "precomputed_seq_lens must have one entry per THD query row: "
                    f"got={precomputed_seq_lens.shape[0]}, expected={total_q}."
                )
            if precomputed_seq_lens.dtype != torch.int32:
                raise ValueError(
                    f"precomputed_seq_lens must be int32, got {precomputed_seq_lens.dtype}."
                )
            if precomputed_seq_lens.device != device:
                raise ValueError(
                    "precomputed_seq_lens must be on the same device as q_indexer."
                )
            seq_lens = precomputed_seq_lens
        else:
            # The indexer kernel has already applied the causal mask using
            # cu_seqlens_q/cu_seqlens_k.  In CP, Q and K sequence lengths can
            # differ for a split sequence piece, which represents a trapezoid
            # causal mask.  Let top-k scan the whole visible K segment and filter
            # the kernel-masked ``-inf`` entries after selection.
            row_batch_ids = batch_of_row(cu_seqlens_q, total_q=total_q)
            seqlen_kv_per_row = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1])[row_batch_ids]
            seq_lens = seqlen_kv_per_row.to(torch.int32).contiguous()
    else:
        # Kernel wants k as 4-D ``(b, sk, h_kv, idx_hd)``.
        scores = _DSA.indexer_forward_wrapper(q, k.unsqueeze(2), w, ratio=ratio)[
            "scores"
        ]  # (b, sq, sk) fp32, -inf on masked positions
        b, sq = q.shape[:2]
        sk = k.shape[1]
        total_q = b * sq
        scores_flat = scores.reshape(total_q, sk).contiguous()

        # Per-row valid KV length: ((q_idx + 1) // ratio).clamp(max=sk),
        # tiled across the batch axis.
        q_idx = torch.arange(sq, device=device)
        valid_per_q = ((q_idx + 1) // ratio).clamp(max=sk).to(torch.int32)
        seq_lens = valid_per_q.repeat(b)  # (b*sq,), row-major over (b, sq)

    # ---------------- Shared: radix top-K + pad-to-topk -----------------
    topk_k = min(topk, sk)
    tk_result = _DSA.indexer_top_k_wrapper(
        scores_flat, seq_lens, top_k=topk_k, next_n=1, return_val=False
    )
    topk_indices = tk_result["indices"]  # (total_q, topk_k) int32

    output_topk = int(fixed_topk_width) if fixed_topk_width is not None else topk
    if output_topk < topk_k:
        raise ValueError(
            "fixed_topk_width must be greater than or equal to the computed top-k width: "
            f"fixed={output_topk}, computed={topk_k}."
        )

    if fixed_topk_width is None and topk_k < topk:
        pad = torch.full((total_q, topk - topk_k), -1, dtype=torch.int32, device=device)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)

    if is_thd:
        safe_topk = topk_indices.clamp(min=0, max=max(sk - 1, 0)).to(torch.long)
        selected_scores = torch.gather(scores_flat, dim=-1, index=safe_topk)
        selected_valid = (
            (topk_indices >= 0) & (topk_indices < sk) & torch.isfinite(selected_scores)
        )
        topk_indices = topk_indices.masked_fill(~selected_valid, -1)
        topk_length = (
            (topk_indices >= 0).sum(dim=-1).int()
            if compute_topk_length
            else torch.empty((0,), dtype=torch.int32, device=device)
        )
        if output_topk > topk_indices.shape[-1]:
            pad = torch.full(
                (total_q, output_topk - topk_indices.shape[-1]),
                -1,
                dtype=torch.int32,
                device=device,
            )
            topk_indices = torch.cat([topk_indices, pad], dim=-1)
    else:
        topk_length = (topk_indices >= 0).sum(dim=-1).int()  # (total_q,)

    # ---------------- Layout-specific output reshape --------------------
    if is_thd:
        return topk_indices.int(), topk_length, scores
    return (topk_indices.view(b, sq, topk).int(), topk_length.view(b, sq), scores)


def indexer_topk(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk: int,
    ratio: int = 4,
    indexer_softmax_scale: float = 1.0,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    fixed_topk_width: Optional[int] = None,
    compute_topk_length: bool = True,
    precomputed_seq_lens: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Score + top-K selection for inference (no KL loss, no backward).

    Built on cuDNN Frontend's CuTe-DSL indexer forward kernel followed by
    TRT-LLM's radix top-K kernel.

    Args:
        q_indexer: SBHD ``(sq, b, idx_nh, idx_hd)`` /
                   THD ``(total_q, idx_nh, idx_hd)`` bf16.
        k_indexer: SBHD ``(sk, b, idx_hd)`` / THD ``(total_k, idx_hd)`` bf16.
        weights:   SBHD ``(sq, b, idx_nh)`` / THD ``(total_q, idx_nh)``
            bf16 — raw (unscaled) weights.
        topk: number of top-K indices to select.
        ratio: compression ratio for the causal mask.
        indexer_softmax_scale: scale applied to the indexer ``Q @ K^T``
            scores (typically ``idx_hd ** -0.5``). Default ``1.0`` means
            weights are treated as already-scaled.
        cu_seqlens_q: THD only — ``(B+1,)`` int32 CUDA cumulative Q lens.
        cu_seqlens_kv: THD only — ``(B+1,)`` int32 CUDA cumulative KV lens.
        max_seqlen_q: THD only — per-batch max Q length.
        max_seqlen_kv: THD only — per-batch max KV length.
        fixed_topk_width: THD only — optional static output width. CP callers
            use this to fuse score filtering and fixed-width padding into one
            CuTeDSL kernel while keeping the radix top-K visible width smaller.
        compute_topk_length: THD only — set ``False`` when the caller only
            consumes the fixed-width top-k ids. In that case ``topk_length`` is
            returned as an empty int32 CUDA sentinel.
        precomputed_seq_lens: THD only — optional ``(total_q,)`` int32 CUDA
            tensor with the per-row visible K length for the radix top-K wrapper.
            CP callers pass this from the layout kernel to avoid rebuilding the
            same metadata with generic PyTorch indexing during CUDA graph replay.

    Returns:
        SBHD: ``(topk_indices (b, sq, topk),  topk_length (b, sq))`` int32
              — per-batch LOCAL ids into ``k_indexer`` (``-1`` invalid).
        THD:  ``(topk_indices (total_q, topk), topk_length (total_q,))``
              int32 — per-batch LOCAL ids in ``[0, seqlen_kv[batch])``.
    """
    is_thd = cu_seqlens_q is not None
    if is_thd and (cu_seqlens_kv is None or max_seqlen_q is None or max_seqlen_kv is None):
        raise ValueError(
            "indexer_topk THD mode requires cu_seqlens_q, cu_seqlens_kv, "
            "max_seqlen_q, and max_seqlen_kv to all be supplied."
        )
    if fixed_topk_width is not None and not is_thd:
        raise ValueError("fixed_topk_width is only supported in THD mode.")
    if not compute_topk_length and not is_thd:
        raise ValueError("compute_topk_length=False is only supported in THD mode.")
    if precomputed_seq_lens is not None and not is_thd:
        raise ValueError("precomputed_seq_lens is only supported in THD mode.")

    # ``indexer_softmax_scale`` is applied via the
    # ``relu(c·x) = c·relu(x)`` trick (the cudnn kernel does the relu),
    # so we push the scale onto the weights tensor (small) instead of the
    # score tensor (big). This is uniform across SBHD and THD; in SBHD
    # the subsequent permute carries the scaled values into BSHD order.
    if indexer_softmax_scale != 1.0:
        weights = (weights.float() * indexer_softmax_scale).to(weights.dtype)

    if is_thd:
        q, k, w = q_indexer, k_indexer, weights
    else:
        # SBHD → BSHD permute (one-shot copy each).
        q = q_indexer.permute(1, 0, 2, 3).contiguous()
        k = k_indexer.permute(1, 0, 2).contiguous()
        w = weights.permute(1, 0, 2).contiguous()

    topk_indices, topk_length, _ = _indexer_topk_core(
        q,
        k,
        w,
        topk=topk,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=int(max_seqlen_q) if max_seqlen_q is not None else None,
        max_seqlen_kv=int(max_seqlen_kv) if max_seqlen_kv is not None else None,
        fixed_topk_width=fixed_topk_width,
        compute_topk_length=compute_topk_length,
        precomputed_seq_lens=precomputed_seq_lens,
    )
    return topk_indices, topk_length


# ---------------------------------------------------------------------------
# Training fused indexer loss + sparse attention
# ---------------------------------------------------------------------------


_CLIP_PROB_MIN = torch.finfo(torch.float32).tiny  # kept compatible w/ cudnn kernel


def _compute_indexer_predict(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    topk_indices: Tensor,
    qhead_per_kv_head: int,
    *,
    topk_indices_global: bool = False,
) -> Tensor:
    """Compute ``predict`` distribution (softmax over top-K of indexer scores).

    Wraps `cudnn.DSA.sparse_indexer_score_recompute_wrapper`.
    This function is not used now, but it is kept for potential future use.

    Two layouts:

    * **BSHD** (default; 4-D q): ``q (B, S_q, H, D)``, ``k (B, S_k, D)``,
      ``w (B, S_q, H)``, ``topk (B, S_q, topk)``.
    * **THD packed** (3-D q): ``q (total_q, H, D)``, ``k (total_k, D)``,
      ``w (total_q, H)``, ``topk (total_q, topk)``. Internally
      fake-BSHD'd with ``B=1`` so the wrapper's 4-D-Q shape check
      passes; ``topk_indices_global=True`` is required (and enforced) so
      the kernel decodes the flat ids directly as positions into the
      ``(1*total_k, D)`` view.

    Output shape matches the layout: BSHD ``(B, S_q, topk)`` or
    THD ``(total_q, topk)``, fp32 softmax over the top-K axis.
    """
    _ensure_dsa_namespace()
    is_thd = q_indexer.ndim == 3
    if is_thd:
        if not topk_indices_global:
            raise ValueError(
                "THD ``_compute_indexer_predict`` requires "
                "``topk_indices_global=True`` so the kernel addresses K "
                "by flat ids over the packed ``(total_k, D)`` buffer."
            )
        total_q, h, d = q_indexer.shape
        total_k = k_indexer.shape[0]
        topk = topk_indices.shape[-1]
        q_bshd = q_indexer.view(1, total_q, h, d)
        k_bsd = k_indexer.view(1, total_k, d)
        w_bsh = weights.view(1, total_q, h)
        topk_bst = topk_indices.view(1, total_q, topk)
    else:
        q_bshd, k_bsd, w_bsh, topk_bst = q_indexer, k_indexer, weights, topk_indices

    result = _DSA.sparse_indexer_score_recompute_wrapper(
        q_bshd,
        k_bsd,
        w_bsh,
        topk_bst,
        qhead_per_kv_head=qhead_per_kv_head,
        topk_indices_global=topk_indices_global,
    )
    predict = result["predict"]
    if is_thd:
        predict = predict.view(total_q, topk)
    return predict


def _compute_attn_target(
    q_attn: Tensor,
    k_attn: Tensor,
    lse: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    qhead_per_kv_head: int,
    *,
    topk_indices_global: bool = False,
) -> Tensor:
    """Compute ``target`` distribution (L1-normalised head-sum softmax).

    Wraps :attr:`cudnn.DSA.sparse_attn_score_recompute_wrapper`. Same
    layout convention as :func:`_compute_indexer_predict`: 4-D q is
    BSHD; 3-D q is THD and gets fake-BSHD'd with ``B=1`` before the
    wrapper call (so the 4-D-Q shape check passes).
    """
    _ensure_dsa_namespace()
    is_thd = q_attn.ndim == 3
    if is_thd:
        if not topk_indices_global:
            raise ValueError(
                "THD ``_compute_attn_target`` requires "
                "``topk_indices_global=True`` so the kernel addresses K "
                "by flat ids over the packed ``(total_k, D)`` buffer."
            )
        total_q, h, d = q_attn.shape
        total_k = k_attn.shape[0]
        topk = topk_indices.shape[-1]
        q_bshd = q_attn.view(1, total_q, h, d)
        k_bsd = k_attn.view(1, total_k, d)
        lse_bsh = lse.view(1, total_q, h)
        topk_bst = topk_indices.view(1, total_q, topk)
    else:
        q_bshd, k_bsd, lse_bsh, topk_bst = q_attn, k_attn, lse, topk_indices

    result = _DSA.sparse_attn_score_recompute_wrapper(
        q_bshd,
        k_bsd,
        lse_bsh,
        topk_bst,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        topk_indices_global=topk_indices_global,
    )
    target = result["target"]
    if is_thd:
        target = target.view(total_q, topk)
    return target


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


# ---------------------------------------------------------------------------
# Dense path (``sparse_loss=False``) — full-KV indexer loss
# ---------------------------------------------------------------------------


def _compute_dense_indexer_score(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    qhead_per_kv_head: int,
    indexer_softmax_scale: float,
    ratio: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Dense indexer score forward over the full ``S_k`` axis (BSHD or THD).

    Wraps :attr:`cudnn.DSA.dense_indexer_score_recompute_wrapper`.
    This function is not used now, but it is kept for potential future use.
    Layout is selected by ``cu_seqlens_*`` kwargs:

    * **BSHD** (``cu_seqlens_*=None``): inputs are 4-D q ``(B, S_q, H, D)``,
      4-D k ``(B, S_k, H_kv, D)``, 3-D w ``(B, S_q, H)``. Outputs are
      ``out (B, S_q, S_k)`` + ``denom (B, S_q)``.
    * **THD** (``cu_seqlens_*`` supplied): inputs are 3-D q
      ``(total_q, H, D)``, 3-D k ``(total_k, H_kv, D)``, 2-D w
      ``(total_q, H)``. Outputs are ``out (total_q, max_seqlen_kv)`` +
      ``denom (total_q,)``.

    The kernel applies the bottom-right ratio causal mask
    ``col_limit = min(S_k, (q+1) // ratio)`` regardless of layout.
    """
    _ensure_dsa_namespace()
    result = _DSA.dense_indexer_score_recompute_wrapper(
        q_indexer,
        k_indexer,
        weights,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=indexer_softmax_scale,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
    )
    return result["out"], result["denom"]


def _compute_dense_attn_score(
    q_attn: Tensor,
    k_attn: Tensor,
    lse: Tensor,
    qhead_per_kv_head: int,
    softmax_scale: float,
    ratio: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Dense attention score forward over the full ``S_k`` axis (BSHD or THD).

    Wraps :attr:`cudnn.DSA.dense_attn_score_recompute_wrapper`. Same
    BSHD/THD layout convention as :func:`_compute_dense_indexer_score`.
    """
    _ensure_dsa_namespace()
    result = _DSA.dense_attn_score_recompute_wrapper(
        q_attn,
        k_attn,
        lse,
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_kv,
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
    """KL(target || predict) over the **full** KV axis, averaged over rows.

    Derives ``target = attn_score / attn_l1norm`` (L1-normalised, matches
    ``compute_dsa_indexer_loss``'s ``attention_scores / sum`` step) and
    ``log_predict = index_score - index_lse`` (LSE-normalised log-softmax),
    then computes ``KL = sum_k target * (log target - log predict)`` and
    scales by ``loss_coeff``.

    Layout-agnostic: works for both BSHD inputs (shapes
    ``attn_score (B, S_q, S_k)``, ``attn_l1norm (B, S_q)``, …) and THD
    inputs (shapes ``attn_score (total_q, max_seqlen_kv)``,
    ``attn_l1norm (total_q,)``, …). The final ``.mean()`` averages over
    all rows in either case.

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
    """Fused indexer KL loss + sparse attention in one autograd.

    Differentiable w.r.t. ``query``, ``kv_full``, ``attn_sink``,
    ``q_indexer``, ``k_indexer``, ``weights``.

    Layout is selected by the ``cu_seqlens_q`` kwarg passed to
    :func:`fused_indexer_sparse_attn`:

    * **SBHD** (``cu_seqlens_q is None``): inputs carry an explicit batch
      axis; the indexer pipeline runs in BSHD (after one SBHD→BSHD
      permute).
    * **THD packed** (``cu_seqlens_q`` supplied): inputs are flat
      packed-sequence tensors; the indexer pipeline runs directly on
      ``(total_q, …)`` / ``(total_kv, …)`` shapes with
      ``cu_seqlens_q/kv`` forwarded to every layout-aware kernel
      (``_indexer_topk_core`` THD branch, ``local_to_global_flat`` THD
      branch, ``_compute_dense_*_score`` THD branch,
      ``dense_indexer_backward_wrapper`` with ``cu_seqlens_q/k``).

    Two indexer-loss variants, selected by the ``sparse_loss`` argument
    (matches ``compute_dsa_indexer_loss`` in the reference ``dsa.py``):

    * **Sparse loss** (``sparse_loss=True``) — KL is computed only over
      the top-K KV positions the indexer has selected.
      **Supports both SBHD and THD.**
    * **Dense loss** (``sparse_loss=False``, the default) — KL is
      computed over *all* causally valid KV positions.
      **Supports both SBHD and THD.**

    The indexer backward is eagerly computed in the forward pass with
    ``grad_loss=1.0``; the actual backward simply scales the
    pre-computed gradients by ``grad_loss``.

    Both variants share the FlashMLA sparse-attention forward + the
    cuDNN sparse-attn backward (both of which are layout-agnostic — the
    flat shape they require is what the THD branch already passes
    directly, and what the SBHD branch reshapes into).
    """

    @staticmethod
    def forward(
        ctx,
        # Sparse attn inputs (differentiable)
        query: Tensor,  # SBHD (sq, b, np, d) / THD (total_q, np, d)
        kv_full: Tensor,  # SBHD (skv, b, d) / THD (total_kv_full, d)
        attn_sink: Tensor,  # (np,) f32
        # Window indices (not differentiable)
        window_idxs: Tensor,  # SBHD (b, sq, win_topk) / THD (total_q, win_topk)
        # Indexer inputs (differentiable)
        q_indexer: Tensor,  # SBHD (sq, b, idx_nh, idx_hd) / THD (total_q, idx_nh, idx_hd)
        k_indexer: Tensor,  # SBHD (n_comp, b, idx_hd) / THD (total_comp_idx, idx_hd)
        weights: Tensor,  # SBHD (sq, b, idx_nh) / THD (total_q, idx_nh) — raw (unscaled)
        # Scalars
        indexer_topk: int,
        ratio: int,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        kv_offset: int,  # SBHD only — start of compressed region in kv_full
        calculate_per_token_loss: bool,
        # THD packed-sequence args (all None for SBHD; all required for THD)
        cu_seqlens_q: Optional[Tensor],
        cu_seqlens_kv: Optional[Tensor],  # original (uncompressed) KV cu_seqlens
        cu_seqlens_kv_full: Optional[Tensor],  # original + compressed concat'd cu_seqlens
        cu_seqlens_compressed_idx: Optional[Tensor],  # indexer K cu_seqlens (== compressor's)
        max_seqlen_q: Optional[int],
        max_seqlen_compressed_idx: Optional[int],  # indexer K max
        compressed_kv: Optional[Tensor] = None,  # THD only — pre-packed compressed KV
    ) -> Tuple[Tensor, Tensor]:
        """Fused forward: indexer scoring, sparse attention, KL loss, and indexer backward."""
        _ensure_dsa_namespace()

        is_thd = cu_seqlens_q is not None

        # ---- Layout-specific input prep --------------------------------------
        # SBHD: permute SBHD→BSHD once and reuse the BSHD tensors for indexer
        # forward, dense score helpers, and the indexer backward.
        # THD: skip the permute; tensors are already flat.
        if is_thd:
            total_q = q_indexer.shape[0]
            idx_nh = q_indexer.shape[1]
            n_comp = k_indexer.shape[0]
            np_, d = query.shape[1], query.shape[2]

            q_indexer_flat = q_indexer
            k_indexer_flat = k_indexer
            w_indexer = weights
        else:
            sq, b, np_, d = query.shape
            skv = kv_full.shape[0]
            idx_nh = q_indexer.shape[2]
            n_comp = k_indexer.shape[0]

            q_indexer_flat = q_indexer.permute(1, 0, 2, 3).contiguous()
            k_indexer_flat = k_indexer.permute(1, 0, 2).contiguous()
            w_indexer = weights.permute(1, 0, 2).contiguous()

        # ``indexer_softmax_scale`` is applied via the
        # ``relu(c·x) = c·relu(x)`` trick (the cudnn kernel does the relu),
        # so we push the scale onto the weights tensor (small) instead of the
        # score tensor (big). This is uniform across SBHD and THD; in SBHD
        # the subsequent permute carries the scaled values into BSHD order.
        if indexer_softmax_scale != 1.0:
            w_indexer_scaled = (w_indexer.float() * indexer_softmax_scale).to(w_indexer.dtype)
        else:
            w_indexer_scaled = w_indexer

        # ---- 2. Indexer scoring + top-K (with scores retained). ---------------
        topk_indices_cmp, _, indexer_scores = _indexer_topk_core(
            q_indexer_flat,
            k_indexer_flat,
            w_indexer_scaled,
            indexer_topk,
            ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_compressed_idx,
            max_seqlen_q=int(max_seqlen_q) if max_seqlen_q is not None else None,
            max_seqlen_kv=(
                int(max_seqlen_compressed_idx) if max_seqlen_compressed_idx is not None else None
            ),
        )

        # ---- 3. Combine indices (indexer first, then window) + globalize. ----
        if is_thd:
            row_batch_ids = batch_of_row(cu_seqlens_q, total_q=total_q)
            offset_per_row = (
                (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1])[row_batch_ids].unsqueeze(1).to(torch.int32)
            )
            compress_topk_idxs = torch.where(
                topk_indices_cmp >= 0,
                topk_indices_cmp + offset_per_row,
                torch.full_like(topk_indices_cmp, -1),
            )
            combined_local = torch.cat(
                [compress_topk_idxs, window_idxs], dim=-1
            )  # (total_q, eff_topk + win_topk)
            global_idxs = local_to_global_flat(
                combined_local,
                batch_size=-1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_full,
            )
        else:
            compress_topk_idxs = torch.where(
                topk_indices_cmp >= 0, topk_indices_cmp + kv_offset, -1
            )
            combined_local = torch.cat([compress_topk_idxs, window_idxs], dim=-1)
            global_idxs = local_to_global_flat(combined_local, b)

        # ---- 4. FlashMLA forward (flat layout for both SBHD and THD). --------
        if is_thd:
            q_flat = query
            kv_flat = kv_full
        else:
            q_flat = query.reshape(sq * b, np_, d)
            kv_flat = kv_full.reshape(skv * b, d)
        out_flat, lse, lse_indexer = _dsa_fwd_flash_mla(
            q_flat,
            kv_flat,
            global_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=None,
            indexer_topk=indexer_topk,
        )

        # ---- 5. Derive predict from indexer_scores, compute target. ----------
        # Layout-specific attn tensors (detached — loss is not differentiable
        # through them).
        if is_thd:
            assert compressed_kv is not None, "compressed_kv is required for THD"
            q_attn_det = query.detach()
            k_attn_compressed_det = compressed_kv.detach()
            lse_indexer_det = lse_indexer.detach()
        else:
            q_attn_det = query.detach().permute(1, 0, 2, 3).contiguous()
            k_attn_compressed_det = kv_full[kv_offset:].detach().permute(1, 0, 2).contiguous()
            lse_indexer_det = lse_indexer.reshape(sq, b, np_).permute(1, 0, 2)

        if sparse_loss:
            # Derive predict: gather topk scores from indexer_scores → softmax.
            safe_indices = topk_indices_cmp.clamp(min=0).long()
            gathered_scores = torch.gather(indexer_scores, dim=-1, index=safe_indices)
            gathered_scores = torch.where(
                topk_indices_cmp >= 0, gathered_scores, torch.finfo(torch.float32).min
            )
            predict = torch.softmax(gathered_scores, dim=-1)

            # THD: _compute_attn_target's kernel addresses K by flat ids over
            # the packed (total_k, D) buffer, so promote per-segment-local
            # indices to flat-global against cu_seqlens_compressed_idx.
            if is_thd:
                topk_for_target = local_to_global_flat(
                    topk_indices_cmp,
                    batch_size=-1,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_compressed_idx,
                )
            else:
                topk_for_target = topk_indices_cmp

            target = _compute_attn_target(
                q_attn_det,
                k_attn_compressed_det,
                lse_indexer_det,
                topk_for_target,
                softmax_scale,
                qhead_per_kv_head=np_,
                topk_indices_global=is_thd,
            )

            if loss_coeff > 0:
                indexer_loss = _kl_loss_from_target_predict(
                    target, predict, topk_indices_cmp, loss_coeff, calculate_per_token_loss
                )
            else:
                indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)
        else:
            index_score = indexer_scores
            index_lse = torch.logsumexp(indexer_scores, dim=-1)

            k_unsqueeze_dim = 1 if is_thd else 2
            dense_attn_kwargs = {}
            if is_thd:
                dense_attn_kwargs = dict(
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_compressed_idx,
                    max_seqlen_q=int(max_seqlen_q),
                    max_seqlen_kv=int(max_seqlen_compressed_idx),
                )
            attn_score, attn_l1norm = _compute_dense_attn_score(
                q_attn_det,
                k_attn_compressed_det.unsqueeze(k_unsqueeze_dim),
                lse_indexer_det,
                qhead_per_kv_head=np_,
                softmax_scale=softmax_scale,
                ratio=ratio,
                **dense_attn_kwargs,
            )

            if loss_coeff > 0:
                indexer_loss = _kl_loss_from_dense_scores(
                    attn_score,
                    attn_l1norm,
                    index_score,
                    index_lse,
                    loss_coeff,
                    calculate_per_token_loss,
                )
            else:
                indexer_loss = torch.zeros((), device=query.device, dtype=torch.float32)

        # ---- 6. Eagerly compute indexer backward (grad_loss=1). ------------
        # The actual grad_loss scaling is deferred to backward (when
        # DSAIndexerLossAutoScaler provides the correct scale).
        indexer_loss_coeff = loss_coeff
        if calculate_per_token_loss:
            indexer_loss_coeff = loss_coeff * (total_q if is_thd else b * sq)

        unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)

        if loss_coeff > 0:
            if sparse_loss:
                attn_score_for_bwd = target.clone()
                index_score_for_bwd = predict.clone()
                if is_thd:
                    total_comp_idx = k_indexer_flat.shape[0]
                    topk = topk_indices_cmp.shape[-1]
                    idx_hd = q_indexer_flat.shape[-1]
                    topk_indices_cmp_global = local_to_global_flat(
                        topk_indices_cmp,
                        batch_size=-1,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_compressed_idx,
                    )
                    bwd_q = q_indexer_flat.view(1, total_q, idx_nh, idx_hd)
                    bwd_w = w_indexer.view(1, total_q, idx_nh)
                    bwd_k = k_indexer_flat.view(1, total_comp_idx, idx_hd)
                    bwd_attn = attn_score_for_bwd.view(1, total_q, topk)
                    bwd_idx = index_score_for_bwd.view(1, total_q, topk)
                    bwd_topk = topk_indices_cmp_global.view(1, total_q, topk)
                else:
                    bwd_q = q_indexer_flat
                    bwd_w = w_indexer
                    bwd_k = k_indexer_flat
                    bwd_attn = attn_score_for_bwd
                    bwd_idx = index_score_for_bwd
                    bwd_topk = topk_indices_cmp

                ig = _DSA.indexer_backward_wrapper(
                    bwd_q,
                    bwd_w,
                    bwd_k,
                    bwd_attn,
                    bwd_idx,
                    bwd_topk,
                    sm_scale=indexer_softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    grad_loss=unit_grad_loss,
                    block_I=128,
                )

                if is_thd:
                    precomputed_grad_q_indexer = ig["d_index_q"].view(total_q, idx_nh, idx_hd)
                    precomputed_grad_k_indexer = ig["d_index_k"].view(total_comp_idx, idx_hd)
                    precomputed_grad_weights = ig["d_weights"].view(total_q, idx_nh)
                else:
                    precomputed_grad_q_indexer = ig["d_index_q"].permute(1, 0, 2, 3).contiguous()
                    precomputed_grad_k_indexer = ig["d_index_k"].permute(1, 0, 2).contiguous()
                    precomputed_grad_weights = ig["d_weights"].permute(1, 0, 2).contiguous()
            else:
                attn_score_for_bwd = attn_score.clone()
                index_score_for_bwd = index_score.clone()
                dense_bwd_kwargs = {}
                if is_thd:
                    dense_bwd_kwargs = dict(
                        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_compressed_idx
                    )
                ig = _DSA.dense_indexer_backward_wrapper(
                    q_indexer_flat,
                    w_indexer,
                    k_indexer_flat,
                    attn_score_for_bwd,
                    attn_l1norm,
                    index_score_for_bwd,
                    index_lse,
                    sm_scale=indexer_softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    grad_loss=unit_grad_loss,
                    ratio=ratio,
                    block_I=128,
                    **dense_bwd_kwargs,
                )
                if is_thd:
                    precomputed_grad_q_indexer = ig["d_index_q"]
                    precomputed_grad_k_indexer = ig["d_index_k"]
                    precomputed_grad_weights = ig["d_weights"]
                else:
                    precomputed_grad_q_indexer = ig["d_index_q"].permute(1, 0, 2, 3).contiguous()
                    precomputed_grad_k_indexer = ig["d_index_k"].permute(1, 0, 2).contiguous()
                    precomputed_grad_weights = ig["d_weights"].permute(1, 0, 2).contiguous()
        else:
            precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
            precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
            precomputed_grad_weights = torch.zeros_like(weights)

        # ---- 7. Save context (only sparse-attn bwd tensors + indexer grads). -
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
        ctx.is_thd = is_thd
        ctx.np_ = np_
        ctx.d = d
        if is_thd:
            ctx.total_q = total_q
        else:
            ctx.sq = sq
            ctx.b = b
            ctx.skv = skv

        # ---- Output reshape: layout-specific. --------------------------------
        d_v = out_flat.shape[-1]
        if is_thd:
            output = out_flat.reshape(total_q, np_ * d_v)
        else:
            output = out_flat.reshape(sq, b, np_, d_v).reshape(sq, b, np_ * d_v)
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

        is_thd = ctx.is_thd
        np_, d = ctx.np_, ctx.d

        # ---- 1. Sparse attn backward (flat layout, layout-agnostic). --------
        d_v = out_flat.shape[-1]
        if is_thd:
            dO_flat = grad_output.reshape(ctx.total_q, np_, d_v)
        else:
            sq, b, skv = ctx.sq, ctx.b, ctx.skv
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
        if is_thd:
            grad_query = attn_bwd["dq"]
            grad_kv_full = attn_bwd["dkv"]
        else:
            grad_query = attn_bwd["dq"].reshape(sq, b, np_, d)
            grad_kv_full = attn_bwd["dkv"].reshape(skv, b, d)
        d_sink = attn_bwd["d_sink"]

        # ---- 2. Scale pre-computed indexer grads by grad_loss. ---------------
        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss

        # Grads: query, kv_full, attn_sink, window_idxs, q_indexer, k_indexer,
        #   weights, indexer_topk, ratio, softmax_scale, indexer_softmax_scale,
        #   loss_coeff, sparse_loss, kv_offset, calculate_per_token_loss,
        #   cu_seqlens_q, cu_seqlens_kv, cu_seqlens_kv_full,
        #   cu_seqlens_compressed_idx,
        #   max_seqlen_q, max_seqlen_compressed_idx,
        #   compressed_kv
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class PrecomputedIndexerSparseAttnFunc(torch.autograd.Function):
    """CP-aware fused sparse attention with precomputed indexer top-k.

    The caller owns CP-aware top-k selection. Sparse attention and
    indexer-loss backward still use FlashMLA / cuDNN DSA wrappers.
    """

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        kv_full: Tensor,
        attn_sink: Tensor,
        topk_idxs: Tensor,
        q_indexer: Tensor,
        k_indexer: Tensor,
        weights: Tensor,
        indexer_topk_idxs: Tensor,
        compressed_kv: Tensor,
        softmax_scale: float,
        indexer_softmax_scale: float,
        loss_coeff: float,
        calculate_per_token_loss: bool,
        global_query_rows: int,
    ) -> Tuple[Tensor, Tensor]:
        _ensure_dsa_namespace()

        total_q, np_, d = query.shape
        idx_nh, idx_hd = q_indexer.shape[1], q_indexer.shape[2]
        total_comp = k_indexer.shape[0]
        indexer_topk = indexer_topk_idxs.shape[-1]

        out_flat, lse, lse_indexer = _dsa_fwd_flash_mla(
            query,
            kv_full,
            topk_idxs,
            softmax_scale,
            attn_sink=attn_sink,
            topk_length=None,
            indexer_topk=indexer_topk,
        )
        if lse_indexer is None:
            raise RuntimeError("Precomputed indexer sparse attention requires lse_indexer.")

        if indexer_softmax_scale != 1.0:
            weights_scaled = (weights.float() * indexer_softmax_scale).to(weights.dtype)
        else:
            weights_scaled = weights
        predict = _compute_indexer_predict(
            q_indexer,
            k_indexer,
            weights_scaled,
            indexer_topk_idxs,
            qhead_per_kv_head=idx_nh,
            topk_indices_global=True,
        )
        target = _compute_attn_target(
            query.detach(),
            compressed_kv.detach(),
            lse_indexer.detach(),
            indexer_topk_idxs,
            softmax_scale,
            qhead_per_kv_head=np_,
            topk_indices_global=True,
        )

        raw_local_loss = _kl_loss_from_target_predict(
            target,
            predict,
            indexer_topk_idxs,
            loss_coeff,
            calculate_per_token_loss=True,
        )
        if calculate_per_token_loss:
            indexer_loss = raw_local_loss
            bwd_loss_coeff = loss_coeff * total_q
        else:
            if global_query_rows <= 0:
                raise RuntimeError(
                    f"global_query_rows must be positive, got {global_query_rows}."
                )
            indexer_loss = raw_local_loss / float(global_query_rows)
            bwd_loss_coeff = loss_coeff * float(total_q) / float(global_query_rows)

        unit_grad_loss = torch.ones((), device=query.device, dtype=torch.float32)
        if loss_coeff > 0:
            ig = _DSA.indexer_backward_wrapper(
                q_indexer.view(1, total_q, idx_nh, idx_hd),
                weights.view(1, total_q, idx_nh),
                k_indexer.view(1, total_comp, idx_hd),
                target.view(1, total_q, indexer_topk),
                predict.view(1, total_q, indexer_topk),
                indexer_topk_idxs.view(1, total_q, indexer_topk),
                sm_scale=indexer_softmax_scale,
                loss_coeff=bwd_loss_coeff,
                grad_loss=unit_grad_loss,
                block_I=128,
            )
            precomputed_grad_q_indexer = ig["d_index_q"].view(total_q, idx_nh, idx_hd)
            precomputed_grad_k_indexer = ig["d_index_k"].view(total_comp, idx_hd)
            precomputed_grad_weights = ig["d_weights"].view(total_q, idx_nh)
        else:
            precomputed_grad_q_indexer = torch.zeros_like(q_indexer)
            precomputed_grad_k_indexer = torch.zeros_like(k_indexer)
            precomputed_grad_weights = torch.zeros_like(weights)

        ctx.save_for_backward(
            query,
            kv_full,
            attn_sink,
            topk_idxs,
            out_flat,
            lse,
            precomputed_grad_q_indexer,
            precomputed_grad_k_indexer,
            precomputed_grad_weights,
        )
        ctx.softmax_scale = softmax_scale
        ctx.np_ = np_
        ctx.total_q = total_q

        return out_flat.reshape(total_q, np_ * out_flat.shape[-1]), indexer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss):
        _ensure_dsa_namespace()
        (
            query,
            kv_full,
            attn_sink,
            topk_idxs,
            out_flat,
            lse,
            precomputed_grad_q_indexer,
            precomputed_grad_k_indexer,
            precomputed_grad_weights,
        ) = ctx.saved_tensors

        d_v = out_flat.shape[-1]
        dO_flat = grad_output.reshape(ctx.total_q, ctx.np_, d_v)
        attn_bwd = _DSA.sparse_attention_backward_wrapper(
            query,
            kv_full,
            out_flat,
            dO_flat,
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=None,
        )
        grad_query = attn_bwd["dq"]
        grad_kv_full = attn_bwd["dkv"]
        d_sink = attn_bwd["d_sink"]

        grad_q_indexer = precomputed_grad_q_indexer * grad_loss
        grad_k_indexer = precomputed_grad_k_indexer * grad_loss
        grad_weights = precomputed_grad_weights * grad_loss

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


def fused_precomputed_indexer_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    indexer_topk_idxs: Tensor,
    compressed_kv: Tensor,
    softmax_scale: float,
    indexer_softmax_scale: float,
    loss_coeff: float,
    calculate_per_token_loss: bool,
    global_query_rows: int,
) -> Tuple[Tensor, Tensor]:
    return PrecomputedIndexerSparseAttnFunc.apply(
        query,
        kv_full,
        attn_sink,
        topk_idxs,
        q_indexer,
        k_indexer,
        weights,
        indexer_topk_idxs,
        compressed_kv,
        softmax_scale,
        indexer_softmax_scale,
        loss_coeff,
        calculate_per_token_loss,
        global_query_rows,
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
    calculate_per_token_loss: bool = False,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    cu_seqlens_kv_full: Optional[Tensor] = None,
    cu_seqlens_compressed_idx: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_compressed_idx: Optional[int] = None,
    compressed_kv: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Training fused indexer KL loss + sparse attention.

    Layout is selected by ``cu_seqlens_q``:

    * **SBHD** (``cu_seqlens_q is None``, default): inputs carry an
      explicit batch axis; the THD kwargs are ignored.
    * **THD packed** (``cu_seqlens_q`` supplied): all four
      ``cu_seqlens_*`` and four ``max_seqlen_*`` must be supplied (see
      below). Both ``sparse_loss=True`` and ``sparse_loss=False`` are
      supported — the sparse-loss path globalizes the per-segment-local
      topk indices via ``local_to_global_flat`` and the cuDNN
      sparse-indexer-backward kernel addresses K/dK by flat ids.

    See :class:`FusedIndexerSparseAttnFunc` for the detailed data flow.

    SBHD args:
        query:        ``(sq, b, np, d)`` bf16 SBHD — attention query.
        kv_full:      ``(skv, b, d)`` bf16 SBD — original + compressed KV.
        window_idxs:  ``(b, sq, win_topk)`` int32 — local window indices.
        q_indexer:    ``(sq, b, idx_nh, idx_hd)`` bf16 — indexer query.
        k_indexer:    ``(n_comp, b, idx_hd)`` bf16 — indexer key (compressed).
        weights:      ``(sq, b, idx_nh)`` bf16 — raw indexer weights.
        kv_offset:    start of compressed region within ``kv_full`` (== sq).

    SBHD return: ``output (sq, b, np * d_v)`` + scalar ``indexer_loss``.

    THD args (when ``cu_seqlens_q is not None``):
        query:        ``(total_q, np, d)`` bf16 — flat-packed Q.
        kv_full:      ``(total_kv_full, d)`` bf16 — per-segment-concat'd
                      ``[kv, compressed_kv]`` (built by
                      :func:`csa.cat_per_segment`).
        window_idxs:  ``(total_q, win_topk)`` int32 — local-per-segment
                      window indices.
        q_indexer:    ``(total_q, idx_nh, idx_hd)`` bf16.
        k_indexer:    ``(total_comp_idx, idx_hd)`` bf16 — compressed-only K
                      (== Compressor's output, packed flat).
        weights:      ``(total_q, idx_nh)`` bf16 — raw.
        kv_offset:    ignored.
        cu_seqlens_q:               ``(B+1,)`` int32 CUDA.
        cu_seqlens_kv:              ``(B+1,)`` int32 — original-KV cu_seqlens.
        cu_seqlens_kv_full:         ``(B+1,)`` int32 — built by
                                    :func:`csa.build_cu_seqlens_kv_full`.
        cu_seqlens_compressed_idx:  ``(B+1,)`` int32 — Compressor's
                                    second return value.
        max_seqlen_q / max_seqlen_compressed_idx:
                                    per-batch maxima for tile sizing.

    THD return: ``output (total_q, np * d_v)`` + scalar ``indexer_loss``.

    Common args:
        attn_sink:    ``(np,)`` f32 — learnable sink per head.
        indexer_topk: number of top-K compressed positions to select.
        ratio:        compression ratio used for the causal mask.
        softmax_scale: attention ``Q @ K^T`` scale, typically
            ``1/sqrt(v_head_dim)``.
        indexer_softmax_scale: indexer ``Q @ K^T`` scale, typically
            ``1/sqrt(idx_hd)``. Applied internally — caller passes raw
            (unscaled) ``weights``.
        loss_coeff:   coefficient scaling the KL divergence loss.
        sparse_loss:  if ``True``, KL is computed only over the top-K
            positions (cheap; **SBHD only**); if ``False`` (the default,
            matches ``transformer_config.dsa_indexer_use_sparse_loss``),
            KL is computed over the full causally-valid KV. See
            :class:`FusedIndexerSparseAttnFunc` for the full data flow.
        compressed_kv: THD only (required) — ``(total_compressed_kv, d)``
            bf16, the pre-packed compressed KV from the Compressor. Used
            by the loss path; THD ``kv_full`` is per-segment concatenated
            so it cannot be sliced uniformly the way SBHD ``kv_full`` is.
        calculate_per_token_loss: if True, report raw local KL sum and
            compensate the cuDNN backward wrappers' local averaging.
    """
    if cu_seqlens_q is not None:
        missing = [
            name
            for name, val in (
                ("cu_seqlens_kv", cu_seqlens_kv),
                ("cu_seqlens_kv_full", cu_seqlens_kv_full),
                ("cu_seqlens_compressed_idx", cu_seqlens_compressed_idx),
                ("max_seqlen_q", max_seqlen_q),
                ("max_seqlen_compressed_idx", max_seqlen_compressed_idx),
                ("compressed_kv", compressed_kv),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"fused_indexer_sparse_attn THD mode requires {missing} " "to all be supplied."
            )
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
        calculate_per_token_loss,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_kv_full,
        cu_seqlens_compressed_idx,
        max_seqlen_q,
        max_seqlen_compressed_idx,
        compressed_kv,
    )


__all__ = [
    "batch_of_row",
    "build_flat_topk_idxs",
    "local_to_global_flat",
    "dsa_sparse_attn",
    "indexer_topk",
    "fused_indexer_sparse_attn",
    "fused_precomputed_indexer_sparse_attn",
]

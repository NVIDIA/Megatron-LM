# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU stand-ins for the cuDNN frontend DSA indexer wrappers.

The fakes reproduce the masking and buffer contracts that MCore depends on rather than
the kernels' arithmetic order:

* ``BottomRightDefaultDSA`` models the default of cuDNN frontend <= 1.25: no
  ``q_causal_offsets`` means a bottom-right causal alignment, and the dense score buffer
  is prefilled with ``-inf``. Unlike a released frontend <= 1.25, its wrapper also accepts
  ``q_causal_offsets`` and honors explicit offsets.
* ``Fe131DSA`` mirrors cuDNN frontend 1.31.0: no ``q_causal_offsets`` means a zero
  (top-left) offset, the wrapper accepts ``precision='mxfp8'`` with packed E8M0 scales,
  and the buffer is prefilled with ``-inf``.
* ``Fe131DSA.sparse_indexer_score_recompute_wrapper`` is a torch reference of the FE
  sparse indexer score recompute: FP32 ``softmax`` over the selected keys of
  ``sum_h relu(q_h . k) * w_h`` from BF16 inputs, slots with a negative index masked (or
  past ``topk_length`` when given), top-k a multiple of 128; calls are recorded.
* ``Fe131KPoolLossDSA`` adds torch references of the two kernels of the cuDNN KPool indexer
  loss: ``sparse_attn_score_recompute_wrapper`` (head-summed attention mass per slot from the
  attention's LSE, L1-normalized; widths a multiple of 128 without ``topk_length``) and
  ``indexer_backward_wrapper`` (the sparse indexer KL backward, at least 64 heads, with the
  FE score-gradient stage); calls are recorded.
* ``Fe131CleanLogitsDSA`` adds the proposed ``clean_logits`` switch. With
  ``clean_logits=False`` the buffer starts as ``poison`` (NaN by default) and only the
  causal prefix ``[0, causal_len)`` of every covered row is written, as the FE 1.31
  kernel does.

``indexer_top_k_wrapper`` reads only ``[0, seq_len)`` of each row, matching the
out-of-bounds-lane guard that cuDNN frontend 1.29 added to the varlen radix top-k.
"""

from __future__ import annotations

from typing import Optional

import torch

# Bound before any test patches ``Tensor.tolist`` to detect host syncs in MCore code.
_TOLIST = torch.Tensor.tolist

BOTTOM_RIGHT = "bottom_right"
TOP_LEFT = "top_left"


def _dense_scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, sm_scale: float):
    """``sum_h relu(q_h . k) * w_h`` for ``q (..., sq, h, d)``, ``k (..., sk, d)``."""
    logits = torch.einsum("...qhd,...kd->...qhk", q.float(), k.float()).relu()
    return (logits * w.float().unsqueeze(-1)).sum(dim=-2) * sm_scale


def _forward(
    q,
    k,
    w,
    *,
    ratio,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_k,
    q_causal_offsets,
    default_offset,
    fill_value,
    calls,
    record,
):
    calls.append(record)
    if cu_seqlens_q is None:
        b, sq = q.shape[:2]
        sk = k.shape[1]
        if sq > sk * ratio:
            # The released BSHD kernels reject this layout (measured with frontend 1.31).
            raise ValueError(f"seqlen_q ({sq}) must be <= seqlen_k * ratio ({sk * ratio})")
        out = torch.full((b, sq, sk), fill_value, dtype=torch.float32)
        scores = _dense_scores(q, k[:, :, 0, :], w, sm_scale)
        rows = torch.arange(sq).view(1, sq, 1)
        if q_causal_offsets is not None:
            offsets = q_causal_offsets.to(torch.int64).view(b, 1, 1)
        elif default_offset == BOTTOM_RIGHT:
            offsets = torch.full((b, 1, 1), sk * ratio - sq, dtype=torch.int64)
        else:
            offsets = torch.zeros((b, 1, 1), dtype=torch.int64)
        limit = ((offsets + rows + 1) // ratio).clamp(min=0, max=sk)
        written = torch.arange(sk).view(1, 1, sk) < limit
        return torch.where(written, scores, out)

    total_q = q.shape[0]
    out = torch.full((total_q, int(max_seqlen_k)), fill_value, dtype=torch.float32)
    cu_q = _TOLIST(cu_seqlens_q)
    cu_k = _TOLIST(cu_seqlens_k)
    offsets = None if q_causal_offsets is None else _TOLIST(q_causal_offsets)
    for seg in range(len(cu_q) - 1):
        q_start, q_end = cu_q[seg], cu_q[seg + 1]
        k_start, k_end = cu_k[seg], cu_k[seg + 1]
        seg_q, seg_k = q_end - q_start, k_end - k_start
        if seg_q <= 0 or seg_k <= 0:
            continue
        if q_causal_offsets is not None:
            offset = offsets[seg]
        elif default_offset == BOTTOM_RIGHT:
            offset = seg_k * ratio - seg_q
        else:
            offset = 0
        scores = _dense_scores(q[q_start:q_end], k[k_start:k_end, 0, :], w[q_start:q_end], sm_scale)
        for i in range(seg_q):
            causal_len = max(0, min((offset + i + 1) // ratio, seg_k))
            out[q_start + i, :causal_len] = scores[i, :causal_len]
    return out


def trust_stand_in_top_k_for_any_k(monkeypatch):
    """Hand a test's own stand-in top-k every row, as if the real kernel were right for every K.

    Below ``_cudnn_topk_min_reliable_k`` the wrapper gives the kernel its short rows with
    length 0 and selects them in PyTorch. Tests of the call plumbing and of the post-processing
    of raw top-k output (lengths, chunks, ``return_val``, crafted raw ids, deterministic repair)
    that run small K on CPU stand-ins call this so the stand-in still sees and decides every
    row. The guard itself is covered by ``test_dsa_cudnn_topk_small_k.py`` and its GPU
    counterpart.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_topk_min_reliable_k", lambda dtype: 1)


def indexer_top_k_wrapper(input_values, seq_lens, top_k, next_n=1, return_val=True, **_unused):
    """Row-local top-k that never reads past each row's ``seq_len``.

    Below ``_cudnn_topk_min_reliable_k`` of the scores' dtype the real radix kernel returns
    duplicate and out-of-row ids (it does not raise) on rows longer than ``top_k`` and at most
    that many elements long; MCore must not hand it such a row, so this stand-in refuses them.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    reliable_k = dsa_cudnn_kernels._cudnn_topk_min_reliable_k(input_values.dtype)
    if top_k < reliable_k:
        lengths = seq_lens.to(torch.int64)
        exposed = (lengths > top_k) & (lengths <= reliable_k)
        if bool(exposed.any()):
            raise ValueError(
                f"top_k={top_k} on {input_values.dtype} scores reached the cuDNN radix top-k "
                f"with rows of {sorted(set(lengths[exposed].tolist()))} keys, which it gets wrong"
            )
    del next_n
    n_rows, n_cols = input_values.shape
    cols = torch.arange(n_cols).view(1, n_cols)
    lens = seq_lens.to(torch.int64).view(n_rows, 1)
    masked = torch.where(cols < lens, input_values, torch.full_like(input_values, float("-inf")))
    values, indices = masked.topk(top_k, dim=-1)
    slot = torch.arange(top_k).view(1, top_k)
    valid = slot < lens
    indices = torch.where(valid, indices, torch.full_like(indices, -1)).to(torch.int32)
    values = torch.where(valid, values, torch.full_like(values, float("-inf")))
    return {"indices": indices, "values": values if return_val else None}


class BottomRightDefaultDSA:
    """cuDNN frontend <= 1.25 default: bottom-right causal alignment, always prefilled.

    The wrapper also accepts ``q_causal_offsets``, which released frontends <= 1.25 lack.
    """

    calls: list = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def indexer_forward_wrapper(
        cls,
        q,
        k,
        w,
        ratio=4,
        qhead_per_kv_head=None,
        m_block_size=128,
        n_block_size=128,
        q_stage=2,
        kv_stage=4,
        sm_scale=1.0,
        stream=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
    ):
        scores = _forward(
            q,
            k,
            w,
            ratio=ratio,
            sm_scale=sm_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            default_offset=BOTTOM_RIGHT,
            fill_value=float("-inf"),
            calls=cls.calls,
            record={"q_causal_offsets": q_causal_offsets, "cu_seqlens_q": cu_seqlens_q},
        )
        return {"scores": scores}

    indexer_top_k_wrapper = staticmethod(indexer_top_k_wrapper)


def sparse_indexer_score_recompute(
    q_indexer,
    k_indexer,
    weights,
    topk_indices,
    *,
    qhead_per_kv_head=None,
    topk_length=None,
    topk_indices_global=False,
):
    """Torch reference of ``sparse_indexer_score_recompute_wrapper`` (``"predict"``)."""
    assert q_indexer.dtype == torch.bfloat16 and k_indexer.dtype == torch.bfloat16
    assert weights.dtype == torch.bfloat16 and topk_indices.dtype == torch.int32
    assert q_indexer.ndim == 4 and k_indexer.ndim == 3 and topk_indices.ndim == 3
    batch, rows, heads, _dim = q_indexer.shape
    keys = k_indexer.size(1)
    assert qhead_per_kv_head in (None, heads)
    assert topk_indices.size(-1) % 128 == 0, "topk must be a multiple of n_block_size (128)"
    indices = topk_indices.long()
    if topk_indices_global:
        indices = indices - torch.arange(batch).view(batch, 1, 1) * keys
    valid = indices >= 0
    if topk_length is not None:
        valid = torch.arange(indices.size(-1)).view(1, 1, -1) < topk_length.view(batch, rows, 1)
    safe = indices.clamp(min=0, max=keys - 1)
    selected_k = torch.stack([k_indexer[b].float()[safe[b]] for b in range(batch)])
    logits = torch.einsum("bqhd,bqkd->bqhk", q_indexer.float(), selected_k).relu()
    logits = (logits * weights.float().unsqueeze(-1)).sum(dim=2)
    masked = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    predict = torch.softmax(masked, dim=-1)
    return {"predict": torch.where(valid, predict, torch.zeros_like(predict))}


class Fe131DSA:
    """cuDNN frontend 1.31.0: top-left default offset, MXFP8 dense scorer, prefilled."""

    calls: list = []
    recompute_calls: list = []
    poison: float = float("nan")

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.recompute_calls = []

    @classmethod
    def sparse_indexer_score_recompute_wrapper(
        cls,
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        qhead_per_kv_head=None,
        topk_length=None,
        topk_indices_global=False,
        out=None,
        stream=None,
    ):
        cls.recompute_calls.append(
            dict(
                q=q_indexer,
                k=k_indexer,
                w=weights,
                topk_indices=topk_indices,
                qhead_per_kv_head=qhead_per_kv_head,
                topk_length=topk_length,
                topk_indices_global=topk_indices_global,
            )
        )
        return sparse_indexer_score_recompute(
            q_indexer,
            k_indexer,
            weights,
            topk_indices,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_length=topk_length,
            topk_indices_global=topk_indices_global,
        )

    @classmethod
    def _run(cls, q, k, w, *, fill_value, kwargs):
        precision = kwargs["precision"]
        if precision == "mxfp8":
            assert q.dtype == torch.float8_e4m3fn and k.dtype == torch.float8_e4m3fn
            assert w.dtype == torch.bfloat16
            assert kwargs["q_scale"] is not None and kwargs["k_scale"] is not None
            assert kwargs["q_scale"].dtype == torch.float8_e8m0fnu
            assert kwargs["k_scale"].dtype == torch.float8_e8m0fnu
            if kwargs["cu_seqlens_q"] is not None:
                assert kwargs["cu_seqlens_q_scale_padded"] is not None
                assert kwargs["cu_seqlens_k_scale_padded"] is not None
        else:
            assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
            assert kwargs["q_scale"] is None and kwargs["k_scale"] is None
        record = dict(kwargs)
        record.update(q=q, k=k, w=w)
        return _forward(
            q,
            k,
            w,
            ratio=kwargs["ratio"],
            sm_scale=kwargs["sm_scale"],
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            cu_seqlens_k=kwargs["cu_seqlens_k"],
            max_seqlen_k=kwargs["max_seqlen_k"],
            q_causal_offsets=kwargs["q_causal_offsets"],
            default_offset=TOP_LEFT,
            fill_value=fill_value,
            calls=cls.calls,
            record=record,
        )

    @classmethod
    def indexer_forward_wrapper(
        cls,
        q,
        k,
        w,
        ratio=4,
        qhead_per_kv_head=None,
        m_block_size=128,
        n_block_size=128,
        q_stage=2,
        kv_stage=4,
        sm_scale=1.0,
        out=None,
        stream=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
        precision="bf16",
        q_scale=None,
        k_scale=None,
        cu_seqlens_q_scale_padded=None,
        cu_seqlens_k_scale_padded=None,
        sf_vec_size=32,
        return_lse=False,
        lse_out=None,
    ):
        kwargs = dict(
            ratio=ratio,
            sm_scale=sm_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            sf_vec_size=sf_vec_size,
            clean_logits=True,
        )
        return {"scores": cls._run(q, k, w, fill_value=float("-inf"), kwargs=kwargs)}

    indexer_top_k_wrapper = staticmethod(indexer_top_k_wrapper)


_TINY = torch.finfo(torch.float32).tiny


def sparse_attn_score_recompute(
    q,
    k,
    lse,
    topk_indices,
    sm_scale,
    *,
    qhead_per_kv_head=None,
    topk_length=None,
    topk_indices_global=False,
):
    """Torch reference of ``sparse_attn_score_recompute_wrapper`` (``"target"``).

    ``S_j = sum_h exp(sm_scale * q_h . k_j - lse_h)`` over the slots with a non-negative id
    (and below ``topk_length`` when given), L1-normalized per row; other slots are zero. Like
    the FE 1.31 kernel at latent 512 and 64 heads: without ``topk_length`` the width must be a
    multiple of 128, and ``topk_length`` over a ``-1`` id is refused (the kernel would read
    the memory before ``k``).
    """
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert lse.dtype == torch.float32 and topk_indices.dtype == torch.int32
    assert q.ndim == 4 and k.ndim == 3 and topk_indices.ndim == 3
    batch, rows, heads, _dim = q.shape
    assert tuple(lse.shape) == (batch, rows, heads) and lse.is_contiguous()
    assert qhead_per_kv_head in (None, heads) and not topk_indices_global
    width = topk_indices.size(-1)
    indices = topk_indices.long()
    valid = indices >= 0
    if topk_length is None:
        assert width % 128 == 0, f"topk ({width}) must be a multiple of n_block_size (128)"
    else:
        inside = torch.arange(width).view(1, 1, width) < topk_length.view(batch, rows, 1)
        assert not bool((inside & ~valid).any()), "topk_length covers a -1 id"
        valid = valid & inside
    safe = indices.clamp(0, k.size(1) - 1)
    selected = torch.stack([k[b].float()[safe[b]] for b in range(batch)])
    logits = torch.einsum("bqhd,bqwd->bqhw", q.float(), selected) * sm_scale
    probs = torch.exp(logits - lse.unsqueeze(-1))
    probs = torch.where(valid.unsqueeze(2), probs, torch.zeros_like(probs))
    mass = probs.sum(dim=2)
    total = mass.sum(dim=-1, keepdim=True)
    return {"target": mass / torch.where(total > 0, total, torch.ones_like(total))}


def indexer_backward(
    q,
    w,
    k,
    attn_score,
    index_score,
    topk_indices,
    *,
    sm_scale,
    loss_coeff,
    grad_loss,
    block_I,
    topk_indices_global=False,
):
    """Torch reference of ``indexer_backward_wrapper`` (sparse indexer KL backward).

    The score is ``I = sum_h w_h * relu(sm_scale * q_h . k)``; the score gradient is the FE
    stage ``g = -max(T, tiny) * [p >= tiny] * gs``, ``dI = g - p * sum(g)`` with
    ``gs = grad_loss * loss_coeff / (B * S_q)``, and only slots with a valid id carry it. Like
    the kernel: at least 64 heads, a top-k that is a multiple of ``block_I``, BF16 outputs, and
    both score operands overwritten.
    """
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert q.size(2) >= 64, "assert heads >= 64"
    assert topk_indices.size(-1) % block_I == 0
    batch, rows, _heads, dim = q.shape
    keys = k.size(1)
    total = batch * keys
    ids = topk_indices.long()
    if not topk_indices_global:
        offsets = torch.arange(batch).view(batch, 1, 1) * keys
        ids = torch.where(ids >= 0, ids + offsets, ids)
    valid = (ids >= 0) & (ids < total)
    scale = grad_loss.float() * loss_coeff / (batch * rows)
    predict = index_score.float()
    gate = (predict >= _TINY).float()
    g = -attn_score.float().clamp_min(_TINY) * gate * scale
    dlogit = (g - predict * g.sum(dim=-1, keepdim=True)).masked_fill(~valid, 0.0)
    k_flat = k.reshape(total, dim).float()
    selected = k_flat[ids.clamp(0, total - 1)]
    queries = q.float()
    dots = torch.einsum("bqhd,bqkd->bqhk", queries, selected) * sm_scale
    slot_grad = dlogit.unsqueeze(2)
    d_weights = (dots.clamp_min(0) * slot_grad).sum(dim=-1)
    coeff = (dots > 0).float() * w.float().unsqueeze(-1) * slot_grad * sm_scale
    d_q = torch.einsum("bqhk,bqkd->bqhd", coeff, selected)
    d_selected = torch.einsum("bqhk,bqhd->bqkd", coeff, queries)
    d_k = torch.zeros(total, dim).index_add_(0, ids[valid], d_selected[valid])
    attn_score.fill_(float("nan"))
    index_score.fill_(float("nan"))
    return {
        "d_index_q": d_q.to(q.dtype),
        "d_weights": d_weights.to(w.dtype),
        "d_index_k": d_k.view(batch, keys, dim).to(k.dtype),
    }


class Fe131KPoolLossDSA(Fe131DSA):
    """FE 1.31 plus torch references of the two kernels of the cuDNN KPool indexer loss."""

    calls: list = []
    score_calls: list = []
    backward_calls: list = []

    @classmethod
    def reset(cls):
        super().reset()
        cls.score_calls = []
        cls.backward_calls = []

    @classmethod
    def sparse_attn_score_recompute_wrapper(
        cls,
        q,
        k,
        lse,
        topk_indices,
        sm_scale,
        qhead_per_kv_head=None,
        topk_indices_global=False,
        topk_length=None,
    ):
        cls.score_calls.append(
            dict(width=topk_indices.size(-1), topk_length=topk_length, heads=q.size(2))
        )
        return sparse_attn_score_recompute(
            q,
            k,
            lse,
            topk_indices,
            sm_scale,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_length=topk_length,
            topk_indices_global=topk_indices_global,
        )

    @classmethod
    def indexer_backward_wrapper(
        cls,
        q,
        w,
        k,
        attn_score,
        index_score,
        topk_indices,
        sm_scale=1.0,
        loss_coeff=1.0,
        grad_loss=None,
        block_I=128,
        topk_indices_global=False,
    ):
        cls.backward_calls.append(
            dict(sm_scale=sm_scale, heads=q.size(2), topk=topk_indices.size(-1))
        )
        return indexer_backward(
            q,
            w,
            k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            topk_indices_global=topk_indices_global,
        )


class Fe131CleanLogitsDSA(Fe131DSA):
    """FE 1.31 plus the proposed ``clean_logits`` switch on the dense indexer forward."""

    calls: list = []

    @classmethod
    def indexer_forward_wrapper(
        cls,
        q,
        k,
        w,
        ratio=4,
        qhead_per_kv_head=None,
        m_block_size=128,
        n_block_size=128,
        q_stage=2,
        kv_stage=4,
        sm_scale=1.0,
        out=None,
        stream=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        q_causal_offsets=None,
        precision="bf16",
        q_scale=None,
        k_scale=None,
        cu_seqlens_q_scale_padded=None,
        cu_seqlens_k_scale_padded=None,
        sf_vec_size=32,
        return_lse=False,
        lse_out=None,
        clean_logits: bool = True,
    ):
        kwargs = dict(
            ratio=ratio,
            sm_scale=sm_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            sf_vec_size=sf_vec_size,
            clean_logits=clean_logits,
        )
        fill_value = float("-inf") if clean_logits else cls.poison
        return {"scores": cls._run(q, k, w, fill_value=fill_value, kwargs=kwargs)}


def fake_quantize_indexer_mxfp8(
    x: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_scale_padded: Optional[torch.Tensor] = None,
    buffers=None,
    out_scale: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    calls: Optional[list] = None,
):
    """Stand-in for ``quantize_indexer_mxfp8``: E4M3 data, caller-owned scale storage."""
    del buffers
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert sf_vec_size == 32
    assert out_scale is not None, "the DSA scorer must pass static scale storage"
    if calls is not None:
        calls.append(
            dict(
                shape=tuple(x.shape),
                cu_seqlens=cu_seqlens,
                cu_seqlens_scale_padded=cu_seqlens_scale_padded,
                out_scale_shape=tuple(out_scale.shape),
            )
        )
    return x.to(torch.float8_e4m3fn), out_scale


def packed_zigzag_layout(seq_lens, cp_size, cp_rank, max_num_seqs=None, device="cpu"):
    """Local rows, key bounds and padded cu_seqlens of a zigzag-CP packed THD batch.

    Returns ``(starts, ends, cu_seqlens, max_seqlen)``: ``starts``/``ends`` are global
    packed key bounds of this CP rank's local query rows (front then back chunk of each
    sequence), and ``cu_seqlens`` is the global packed boundary table, padded to
    ``max_num_seqs + 1`` entries when given, as static CUDA-graph inputs are.
    """
    divisor = 2 * cp_size if cp_size > 1 else 1
    assert all(length % divisor == 0 for length in seq_lens)
    cu = [0]
    for length in seq_lens:
        cu.append(cu[-1] + length)
    if max_num_seqs is not None:
        cu += [cu[-1]] * (max_num_seqs + 1 - len(cu))
    starts, ends = [], []
    for seq, length in enumerate(seq_lens):
        if cp_size == 1:
            positions = list(range(length))
        else:
            half = length // divisor
            positions = list(range(cp_rank * half, (cp_rank + 1) * half)) + list(
                range((divisor - 1 - cp_rank) * half, (divisor - cp_rank) * half)
            )
        for position in positions:
            starts.append(cu[seq])
            ends.append(cu[seq] + position + 1)
    return (
        torch.tensor(starts, dtype=torch.int64, device=device),
        torch.tensor(ends, dtype=torch.int64, device=device),
        torch.tensor(cu, dtype=torch.int32, device=device),
        max(seq_lens),
    )

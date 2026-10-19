# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU stand-ins for the cuDNN frontend DSA indexer wrappers.

The fakes reproduce the masking and buffer contracts that MCore depends on rather than
the kernels' arithmetic order:

* ``LegacyOverlayDSA`` mirrors the cuDNN frontend 1.21.1 q-offset overlay: no
  ``q_causal_offsets`` means a bottom-right causal alignment, and the dense score buffer
  is prefilled with ``-inf``.
* ``Fe131DSA`` mirrors cuDNN frontend 1.31.0: no ``q_causal_offsets`` means a zero
  (top-left) offset, the wrapper accepts ``precision='mxfp8'`` with packed E8M0 scales,
  and the buffer is prefilled with ``-inf``.

``indexer_top_k_wrapper`` reads only ``[0, seq_len)`` of each row, matching the
out-of-bounds-lane guard that cuDNN frontend 1.29 added to the varlen radix top-k.
"""

from __future__ import annotations

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


def indexer_top_k_wrapper(input_values, seq_lens, top_k, next_n=1, return_val=True, **_unused):
    """Row-local top-k that never reads past each row's ``seq_len``."""
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


class LegacyOverlayDSA:
    """cuDNN frontend 1.21.1 q-offset overlay: bottom-right default, always prefilled."""

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


class Fe131DSA:
    """cuDNN frontend 1.31.0: top-left default offset, MXFP8 dense scorer, prefilled."""

    calls: list = []

    @classmethod
    def reset(cls):
        cls.calls = []

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

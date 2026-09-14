# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for CSA sparse-attention backward topk-width compile-key stabilization.

The cuDNN Frontend DSA sparse-attention backward (``flash_attn_bwd_sm90/sm100``)
includes ``max_topk = topk_idxs.shape[1]`` in its ``cute.compile`` cache key.
CSA topk widths are data-dependent (``window + ceil(max_seqlen/ratio)`` varies
per packed batch), so without stabilization every new width retriggers a
seconds-long, host-blocking JIT recompile inside backward.

``CSASparseAttnFunc`` therefore pads the topk indices saved for backward to the
next 64-aligned width with ``-1`` (the existing invalid-slot sentinel), which
collapses data-dependent widths onto stable 64-buckets without changing kernel
tile counts or numerics.

These tests cover the Megatron-side wiring only (the kernel itself is covered
in cudnn-frontend): the width/values the backward hands to the wrapper, for
both compact (``topk_length``) and non-compact modes. No CUDA or
cudnn-frontend is required — the FlashMLA forward and cuDNN backward wrappers
are stubbed.
"""

from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as fsa,
)


def _make_inputs(width: int, n: int = 64, h: int = 8, d: int = 64):
    """Build synthetic CSASparseAttnFunc inputs with ``width`` topk slots per row."""
    torch.manual_seed(width)
    q = torch.randn(n, h, d, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(n + 16, d, dtype=torch.bfloat16, requires_grad=True)
    sink = torch.full((h,), float("-inf"), dtype=torch.float32)
    topk_idxs = torch.randint(0, n + 16, (n, width), dtype=torch.int32)
    return q, kv, sink, topk_idxs


class _RecordingDSA:
    """Stub for the cudnn-frontend DSA namespace recording backward wrapper calls."""

    def __init__(self):
        self.calls = []

    def sparse_attention_backward_wrapper(
        self, q, kv, out, dout, lse, attn_sink, topk_idxs, softmax_scale, topk_length
    ):
        self.calls.append(topk_idxs.clone())
        return {
            "dq": torch.zeros_like(q),
            "dkv": torch.zeros_like(kv),
            "d_sink": torch.zeros_like(attn_sink),
        }


def _fake_fwd(q, kv, topk_idxs, softmax_scale, attn_sink=None, topk_length=None, indexer_topk=0):
    """Stub for ``_csa_fwd_flash_mla`` (FlashMLA is not exercised by these tests)."""
    out = torch.zeros_like(q)
    lse = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32)
    return out, lse, None


def _run_fwd_bwd(recorder, topk_length, width):
    q, kv, sink, topk_idxs = _make_inputs(width)
    with (
        patch.object(fsa, "_csa_fwd_flash_mla", _fake_fwd),
        patch.object(fsa, "_ensure_dsa_namespace", lambda: setattr(fsa, "_DSA", recorder)),
        patch.object(fsa, "_DSA", recorder),
    ):
        out, _, _ = fsa.CSASparseAttnFunc.apply(q, kv, sink, topk_idxs, topk_length, 1.0 / 8.0, 0)
        out.backward(torch.ones_like(out))
    return q, topk_idxs


@pytest.mark.parametrize("topk_length", [None, torch.full((64,), 100, dtype=torch.int32)])
def test_backward_topk_widths_collapse_to_64_buckets(topk_length):
    """Widths 130 and 189 both reach the backward wrapper as 192 (stable compile key)."""
    recorder = _RecordingDSA()
    for width in (130, 189, 130):
        _run_fwd_bwd(recorder, topk_length, width)
    assert len(recorder.calls) == 3
    seen_widths = [t.shape[-1] for t in recorder.calls]
    assert seen_widths == [
        192,
        192,
        192,
    ], f"backward topk widths must be 64-aligned stable buckets, got {seen_widths}"


def test_backward_topk_aligned_width_unchanged():
    """Already-64-aligned widths are passed through untouched (no copy, same values)."""
    recorder = _RecordingDSA()
    _, topk_idxs = _run_fwd_bwd(recorder, None, 256)
    (seen,) = recorder.calls
    assert seen.shape[-1] == 256
    assert torch.equal(seen, topk_idxs)


@pytest.mark.parametrize("topk_length", [None, torch.full((64,), 100, dtype=torch.int32)])
def test_backward_topk_padding_values(topk_length):
    """Original entries are preserved; padded suffix is the -1 invalid sentinel."""
    recorder = _RecordingDSA()
    _, topk_idxs = _run_fwd_bwd(recorder, topk_length, 130)
    (seen,) = recorder.calls
    assert seen.shape[-1] == 192
    assert torch.equal(seen[:, :130], topk_idxs)
    assert (seen[:, 130:] == -1).all()

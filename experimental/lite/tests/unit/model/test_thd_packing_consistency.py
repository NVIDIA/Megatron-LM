# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Reproduce the DS4 THD packing inconsistency crash (seq_len 206 > packed tensor 128).

``_nested_from_packed_tensor(tensor, seq_lens)`` narrows a 1-D packed tensor one
sequence at a time. When a declared ``seq_len`` exceeds the remaining length of
the packed tensor (i.e. ``seq_lens`` is inconsistent with ``input_ids.numel()``),
``tensor.narrow(0, offset, length)`` raises ``RuntimeError`` "...exceeds dimension
size..." -- exactly the 206 > 128 crash seen at the actor-update step.

These tests only reproduce the inconsistency (proving it must crash); they do not
exercise the fix (which restores full-length loss_mask upstream in the engine).
"""

from __future__ import annotations

import pytest
import torch

from megatron.lite.model.deepseek_v4.lite.protocol import _nested_from_packed_tensor


def test_reproduce_seqlen_exceeds_packed_tensor_206_over_128():
    """Packed tensor holds 128 tokens but seq_lens declares 206 -> narrow overruns."""
    packed = torch.arange(128, dtype=torch.long)           # 128 tokens
    seq_lens = torch.tensor([206], dtype=torch.long)        # declares 206 (> 128)
    with pytest.raises(RuntimeError) as ei:
        _nested_from_packed_tensor(packed, seq_lens)
    msg = str(ei.value).lower()
    assert "exceeds" in msg or "size" in msg or "range" in msg, str(ei.value)


def test_reproduce_multi_seq_second_overruns():
    """Multiple sequences: the second one pushes the cumulative offset past the end."""
    packed = torch.arange(128, dtype=torch.long)
    seq_lens = torch.tensor([100, 128], dtype=torch.long)   # 100+128=228 > 128
    with pytest.raises(RuntimeError):
        _nested_from_packed_tensor(packed, seq_lens)


def test_sum_mismatch_underrun_raises_valueerror():
    """seq_lens sum < numel (each fits, but the buffer is not filled) -> ValueError.

    This distinguishes the two inconsistency kinds: overrun -> RuntimeError (narrow),
    underrun -> ValueError (the offset != numel sum check).
    """
    packed = torch.arange(128, dtype=torch.long)
    seq_lens = torch.tensor([100], dtype=torch.long)        # 100 < 128
    with pytest.raises(ValueError, match="sizes sum to 100"):
        _nested_from_packed_tensor(packed, seq_lens)


def test_consistent_packing_succeeds():
    """Consistent case (sum(seq_lens) == numel) returns a nested tensor.

    Control case: proves the crash comes from the inconsistency, not the function.
    """
    packed = torch.arange(128, dtype=torch.long)
    seq_lens = torch.tensor([50, 78], dtype=torch.long)     # 50+78=128
    out = _nested_from_packed_tensor(packed, seq_lens)
    assert out is not None
    assert out.size(0) == 2                                  # 2 segments

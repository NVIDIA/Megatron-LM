# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for LogitsSaverHooks' cu_seqlens capture/persist plumbing (CP+packing fix).

These construct a bare LogitsSaverHooks via __new__ (bypassing __init__, which
needs an initialized parallel_state) and drive its cu_seqlens-related methods
directly at cp_size=1 -- the actual per-document CP zigzag math is covered
separately in test_logits_utils_packing.py against slice_tensor_for_cp_rank/
reassemble_cp_sequence, which _gather_full_cp_microbatch simply delegates to.
"""

import io
from collections import OrderedDict

import pytest
import torch

from megatron.training.distillation import logits_saver
from megatron.training.distillation.logits_saver import LogitsSaverHooks

zstandard = pytest.importorskip("zstandard")


def _bare_saver(cp_size=1, tp_rank=0, tp_size=1, dp_rank=0, dp_size=1):
    saver = LogitsSaverHooks.__new__(LogitsSaverHooks)
    saver.tp_rank = tp_rank
    saver.tp_size = tp_size
    saver.cp_rank = 0
    saver.cp_size = cp_size
    saver.dp_rank = dp_rank
    saver.dp_size = dp_size
    saver._save_dtype = torch.float16
    saver.k = 4
    saver.p = None
    saver.min_k = 1
    saver._accumulated_results = []
    saver._accumulated_cu_seqlens = []
    saver._pending_cu_seqlens = None
    saver._pending_writes = OrderedDict()
    saver._topp_kept_counts = []
    return saver


def _patch_buffer_deps(monkeypatch, *, consumed_samples=0, num_microbatches=1, mbs=1):
    monkeypatch.setattr(logits_saver, "get_consumed_train_samples", lambda: consumed_samples)
    monkeypatch.setattr(logits_saver, "get_num_microbatches", lambda: num_microbatches)

    class _Args:
        micro_batch_size = mbs

    monkeypatch.setattr(logits_saver, "get_args", lambda: _Args())


def _decode_buffered_payload(saver):
    assert len(saver._pending_writes) == 1
    (data,) = saver._pending_writes.values()
    return torch.load(io.BytesIO(data), weights_only=True)


# ---------------------------------------------------------------------------
# set_current_cu_seqlens / _forward_hook accumulator bookkeeping
# ---------------------------------------------------------------------------


def test_set_current_cu_seqlens_stores_pending():
    saver = _bare_saver()
    cu_seqlens = torch.tensor([0, 8, 24])
    saver.set_current_cu_seqlens(cu_seqlens)
    assert saver._pending_cu_seqlens is cu_seqlens


def test_set_current_cu_seqlens_none_for_unpacked_runs():
    saver = _bare_saver()
    saver.set_current_cu_seqlens(torch.tensor([0, 24]))
    saver.set_current_cu_seqlens(None)
    assert saver._pending_cu_seqlens is None


# ---------------------------------------------------------------------------
# _gather_full_cp_microbatch: cp_size == 1 passthrough (no collective needed)
# ---------------------------------------------------------------------------


def test_gather_full_cp_microbatch_cp_size_one_passes_through_ignoring_cu_seqlens():
    saver = _bare_saver(cp_size=1)
    values = torch.randn(4, 1, 2)
    indices = torch.randint(0, 4, (4, 1, 2))
    cu_seqlens = torch.tensor([0, 4])

    out_values, out_indices = saver._gather_full_cp_microbatch(values, indices, cu_seqlens)
    assert out_values is values
    assert out_indices is indices


# ---------------------------------------------------------------------------
# _save_accumulated_log_probs -> _buffer_iteration: cu_seqlens_padded persisted
# ---------------------------------------------------------------------------


def test_buffered_payload_includes_cu_seqlens_padded(monkeypatch):
    saver = _bare_saver(cp_size=1)
    _patch_buffer_deps(monkeypatch, num_microbatches=1)

    values = torch.randn(4, 1, 2)
    indices = torch.randint(0, 4, (4, 1, 2))
    cu_seqlens = torch.tensor([[0, 4]])
    saver._accumulated_results = [(values, indices)]
    saver._accumulated_cu_seqlens = [cu_seqlens]

    saver._save_accumulated_log_probs()

    payload = _decode_buffered_payload(saver)
    assert torch.equal(payload["cu_seqlens_padded"], cu_seqlens)


def test_buffered_payload_cu_seqlens_padded_none_for_unpacked_run(monkeypatch):
    saver = _bare_saver(cp_size=1)
    _patch_buffer_deps(monkeypatch, num_microbatches=1)

    values = torch.randn(4, 1, 2)
    indices = torch.randint(0, 4, (4, 1, 2))
    saver._accumulated_results = [(values, indices)]
    saver._accumulated_cu_seqlens = [None]

    saver._save_accumulated_log_probs()

    payload = _decode_buffered_payload(saver)
    assert payload["cu_seqlens_padded"] is None


def test_buffered_payload_pads_and_stacks_across_microbatches(monkeypatch):
    saver = _bare_saver(cp_size=1)
    _patch_buffer_deps(monkeypatch, num_microbatches=2)

    values = torch.randn(4, 1, 2)
    indices = torch.randint(0, 4, (4, 1, 2))
    # Two microbatches with different document counts -> different row widths.
    cu_a = torch.tensor([[0, 8, 24]])
    cu_b = torch.tensor([[0, 24]])
    saver._accumulated_results = [(values, indices), (values, indices)]
    saver._accumulated_cu_seqlens = [cu_a, cu_b]

    saver._save_accumulated_log_probs()

    payload = _decode_buffered_payload(saver)
    assert payload["cu_seqlens_padded"].tolist() == [[0, 8, 24], [0, 24, 24]]


def test_buffered_payload_rejects_mixed_none_and_present_cu_seqlens(monkeypatch):
    saver = _bare_saver(cp_size=1)
    _patch_buffer_deps(monkeypatch, num_microbatches=2)

    values = torch.randn(4, 1, 2)
    indices = torch.randint(0, 4, (4, 1, 2))
    saver._accumulated_results = [(values, indices), (values, indices)]
    saver._accumulated_cu_seqlens = [torch.tensor([[0, 4]]), None]

    with pytest.raises(RuntimeError, match="[Ss]ome but not all"):
        saver._save_accumulated_log_probs()

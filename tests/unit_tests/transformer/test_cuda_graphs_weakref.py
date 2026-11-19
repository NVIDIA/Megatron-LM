# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for `make_weakref` in megatron.core.transformer.cuda_graphs.

These tests are hermetic: they run on CPU and do not require TransformerEngine
to be installed. TE's `make_weak_ref` is substituted with a stub that either
succeeds or raises the documented RuntimeError for dtypes without a mapped
representation, which is the only TE behavior `make_weakref` depends on.
"""

import pytest
import torch

import megatron.core.transformer.cuda_graphs as cuda_graphs_module


def _mempool_tensor() -> torch.Tensor:
    """A tensor that passes the graph-mempool gate of `make_weakref`."""
    ten = torch.zeros(4, 4)
    # `make_weakref` only attempts the weak-ref conversion for tensors marked as
    # coming from the CUDA graph global mempool; setting the attribute is enough
    # to exercise that path here.
    ten.is_from_global_mempool = True
    return ten


@pytest.mark.parametrize("rank", [0, 1])
def test_make_weakref_fallback_keeps_strong_ref_when_te_weak_ref_fails(monkeypatch, rank):
    """The RuntimeError fallback must return the original tensor.

    Regression test: before the fix, the fallback path raised `NameError`
    (rank 0: the warning f-string referenced an undefined `arg` variable) or
    `UnboundLocalError` (rank != 0: `wr` was never assigned before `return wr`)
    instead of keeping a strong reference to the tensor.
    """
    monkeypatch.setattr(cuda_graphs_module, "HAVE_TE_GRAPHS", True)

    def failing_make_weak_ref(ten):
        # Mimics transformer_engine.pytorch.utils.make_weak_ref raising for
        # dtypes that have no mapped representation (e.g. torch.float64).
        raise RuntimeError("simd type not mapped for dtype")

    monkeypatch.setattr(cuda_graphs_module, "make_weak_ref", failing_make_weak_ref, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)

    ten = _mempool_tensor()
    # Repeated invocation mimics repeated graph (re)capture hitting the fallback.
    for _ in range(2):
        result = cuda_graphs_module.make_weakref(ten)
        assert result is ten
        # The fallback must not have re-bound `.data` to something else.
        assert result.dtype == ten.dtype
        assert result.data_ptr() == ten.data_ptr()


def test_make_weakref_inplace_rebinds_data_on_success(monkeypatch):
    """Primary (success) path: with inplace=True the original tensor is returned
    and its data pointer is preserved."""
    monkeypatch.setattr(cuda_graphs_module, "HAVE_TE_GRAPHS", True)

    def fake_make_weak_ref(ten):
        # Stand-in for TE's storage-free alias of the same memory.
        return ten.detach()

    monkeypatch.setattr(cuda_graphs_module, "make_weak_ref", fake_make_weak_ref, raising=False)

    ten = _mempool_tensor()
    ptr_before = ten.data_ptr()
    result = cuda_graphs_module.make_weakref(ten, inplace=True)
    assert result is ten
    assert result.data_ptr() == ptr_before


def test_make_weakref_without_inplace_returns_alias(monkeypatch):
    """Primary (success) path: with inplace=False the alias returned by
    TE's make_weak_ref is passed through unchanged."""
    monkeypatch.setattr(cuda_graphs_module, "HAVE_TE_GRAPHS", True)

    def fake_make_weak_ref(ten):
        return ten.detach()

    monkeypatch.setattr(cuda_graphs_module, "make_weak_ref", fake_make_weak_ref, raising=False)

    ten = _mempool_tensor()
    alias = cuda_graphs_module.make_weakref(ten, inplace=False)
    assert alias is not ten
    assert alias.data_ptr() == ten.data_ptr()


def test_make_weakref_passes_through_non_mempool_tensors(monkeypatch):
    """Tensors outside the graph mempool must keep strong refs and must not
    reach TE's make_weak_ref (use-after-free guard on replay)."""
    monkeypatch.setattr(cuda_graphs_module, "HAVE_TE_GRAPHS", True)

    def unexpected_make_weak_ref(ten):
        raise AssertionError("make_weak_ref must not be called for non-mempool tensors")

    monkeypatch.setattr(
        cuda_graphs_module, "make_weak_ref", unexpected_make_weak_ref, raising=False
    )

    plain = torch.zeros(2, 2)  # no `is_from_global_mempool` attribute
    assert cuda_graphs_module.make_weakref(plain) is plain

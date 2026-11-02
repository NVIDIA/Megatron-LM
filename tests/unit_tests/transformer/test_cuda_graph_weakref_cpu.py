# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression for the strong-reference fallback, without graph execution."""

import pytest
import torch

from megatron.core.transformer import cuda_graphs


def test_weakref_failure_retains_tensor_without_distributed_initialization(monkeypatch):
    tensor = torch.ones(4, dtype=torch.float64)
    tensor.is_from_global_mempool = True

    def fail_weakref(_tensor):
        raise RuntimeError("unsupported dtype")

    monkeypatch.setattr(cuda_graphs, "HAVE_TE_GRAPHS", True)
    monkeypatch.setattr(cuda_graphs, "make_weak_ref", fail_weakref, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: pytest.fail("no process group"))
    result = cuda_graphs.make_weakref(tensor)
    assert result is tensor
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, torch.ones(4, dtype=torch.float64))

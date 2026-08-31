# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression coverage for MLite VERL training-wide metric reductions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verl_mlite.engine.mlite_engine import MegatronLiteEngine

pytestmark = (pytest.mark.mlite, pytest.mark.optional)


def test_mtp_metric_averages_over_physical_pool_not_logical_singleton(monkeypatch):
    physical_pool = object()
    logical_singleton = object()
    engine = object.__new__(MegatronLiteEngine)
    engine.handle = SimpleNamespace(
        dp_group=logical_singleton,
        metric_group=physical_pool,
    )
    calls = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def average_across_pool(value, *, op, group):
        calls.append((op, group))
        assert group is physical_pool
        # Model a second pool rank with a different local MTP metric (3.0).
        value.copy_((value + 3.0) / 2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", average_across_pool)

    reduced = engine._reduce_mtp_metric(torch.tensor(1.0))

    assert torch.equal(reduced, torch.tensor(2.0))
    assert reduced.item() not in {1.0, 3.0}
    assert calls == [(torch.distributed.ReduceOp.AVG, physical_pool)]

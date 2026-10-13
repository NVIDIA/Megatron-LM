# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise Muon's actual GTP gather/split/restore contract without a GPU NS kernel."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import emerging_optimizers as eo_mod
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon


@pytest.mark.parametrize("mode", ["blockwise", "distributed", "duplicated", "auto"])
@pytest.mark.parametrize("rank", [0, 1])
def test_split_qkv_is_not_disabled_by_gtp_mode(mode, rank, monkeypatch):
    """Dev's explicit split-QKV request applies to complete projections in every mode."""
    group = SimpleNamespace(size=lambda: 2, rank=lambda: rank)
    monkeypatch.setattr(eo_mod, "get_pg_size", lambda pg: 1 if pg is None else pg.size())
    monkeypatch.setattr(eo_mod, "get_pg_rank", lambda pg: 0 if pg is None else pg.rank())
    full = torch.arange(24, dtype=torch.float32).view(8, 3)
    padded = torch.nn.functional.pad(full, (0, 0, 0, 2))
    grad = padded.chunk(2)[rank].clone()
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_gtp_weight_remat = True
    param.pad_length = 2
    param.partition_dim = 0
    param.qkv_split_shapes = [4, 2, 2]
    param.qkv_split_shapes_global = [4, 2, 2]
    param.qkv_split_groups_are_complete = True
    optimizer = TensorParallelMuon.__new__(TensorParallelMuon)
    optimizer.pg_collection = SimpleNamespace(tp=None, expt_tp=None, gtp_remat=group)
    optimizer.tp_mode = mode
    optimizer.split_qkv = True
    optimizer.split_qkv_per_head = False
    optimizer.is_qkv_fn = lambda _: True
    optimizer.qkv_split_shapes = None
    optimizer._warned_qkv_split_disabled = False
    optimizer._resolve_tp_mode = lambda *args: "duplicated"
    calls = []

    def orthogonalize_projection(value, tp_group, partition_dim, **kwargs):
        calls.append(tuple(value.shape))
        # A shape-sensitive stand-in isolates layout decisions from Newton-Schulz numerics.
        return value * value.shape[0]

    optimizer.scaled_orthogonalize_fn = orthogonalize_projection

    def all_gather(outputs, local, pg):
        assert pg is group
        torch.testing.assert_close(local, grad)
        for output, source in zip(outputs, padded.chunk(2)):
            output.copy_(source)

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    result = optimizer.orthogonalize(param, grad)
    expected = torch.cat([value * value.shape[0] for value in full.split([4, 2, 2])])
    expected = torch.nn.functional.pad(expected, (0, 0, 0, 2)).chunk(2)[rank]
    assert calls == [(4, 3), (2, 3), (2, 3)]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

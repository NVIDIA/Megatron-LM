# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import builtins
import random
from types import SimpleNamespace

import numpy as np
import torch


def test_generic_dcp_buffers_and_global_row_offsets(monkeypatch):
    from megatron.lite.primitive.ckpt import dcp

    model = torch.nn.Module()
    model.register_buffer("persistent", torch.tensor([-0.0, 1.0]))
    model.register_buffer("scratch", torch.ones(1), persistent=False)
    model.table = torch.nn.Module()
    model.table.lookup = SimpleNamespace(group=object(), boundaries=(0, 3, 5))
    model.table.weight = torch.nn.Parameter(torch.ones(2, 4))
    monkeypatch.setattr(dcp.dist, "get_world_size", lambda *args: 2)
    monkeypatch.setattr(dcp.dist, "get_rank", lambda *args: 1)
    monkeypatch.setattr(
        dcp.ShardedTensor, "_init_from_local_shards", lambda shards, *a, **k: shards
    )
    state, covered = dcp._local_state(model, "model")
    shard = state["model.table.weight"][0]
    assert (set(state), covered, shard.metadata.shard_offsets, shard.metadata.shard_sizes) == (
        {"model.persistent", "model.table.weight"},
        {id(model.table.weight)},
        [3, 0],
        [2, 16],
    )
    assert torch.equal(state["model.persistent"], model.persistent.view(torch.uint8))


def test_lazy_fused_import_preserves_rng(monkeypatch):
    from megatron.lite.primitive.ops import linear_cross_entropy as module

    original = builtins.__import__
    fused = object()

    def drawing(name, *args, **kwargs):
        if name == "verl.utils.kernel.linear_cross_entropy":
            random.random(), np.random.random(), torch.rand(1)
            return SimpleNamespace(linear_cross_entropy=fused)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", drawing)
    monkeypatch.setattr(module, "_VERL_LCE", None)
    before = random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state()
    assert module._verl_linear_cross_entropy() is fused
    assert (
        random.getstate() == before[0]
        and np.array_equal(np.random.get_state()[1], before[1])
        and torch.equal(torch.get_rng_state(), before[2])
    )

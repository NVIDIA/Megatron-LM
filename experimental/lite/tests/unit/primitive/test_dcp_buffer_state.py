# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Persistent buffers/row shards restore; plain-model serialization keys stay unchanged."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

WORLD = 2
ROWS = (0, 3, 5)  # rank 0 owns rows [0, 3), rank 1 owns rows [3, 5)


class _Table(nn.Module):

    def __init__(self, rank, boundaries=ROWS, rows=None):
        super().__init__()
        self.lookup = SimpleNamespace(group=dist.group.WORLD, boundaries=boundaries)
        rows = ROWS[rank + 1] - ROWS[rank] if rows is None else rows
        self.register_buffer("weight", torch.zeros(rows, 4, dtype=torch.float8_e4m3fn))
        self.register_buffer("scale", torch.zeros(rows, 1))
        self.master = nn.Parameter(torch.zeros(rows, 4))


class _Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scratch", torch.zeros(4), persistent=False)


class _Model(_Plain):
    def __init__(self, rank, **table):
        super().__init__()
        self.register_buffer("bias_state", torch.zeros(4))
        self.table = _Table(rank, **table)


def _fill(model, rank, value):
    with torch.no_grad():
        model.linear.weight.copy_(torch.arange(16.0).reshape(4, 4) / 7 + value)
        model.linear.bias.fill_(value)
        model.scratch.fill_(value)
        if hasattr(model, "table"):
            model.bias_state.copy_(torch.arange(4.0) + value)
            table = model.table
            rows = torch.arange(table.scale.shape[0], dtype=torch.float32) + ROWS[rank]
            table.scale.copy_(rows[:, None] + value)
            table.master.copy_(rows[:, None] * 10 + value)
            table.weight.copy_((rows[:, None] + value).expand(-1, 4).to(torch.float8_e4m3fn))


def _cpu_meshes(config):
    from torch.distributed.device_mesh import DeviceMesh

    ranks = torch.arange(dist.get_world_size()).reshape(1, -1, 1, 1)
    return DeviceMesh("cpu", ranks), DeviceMesh("cpu", ranks)


def _bytes_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _worker(rank, directory, kind):
    from megatron.lite.primitive.ckpt import dcp as checkpoint

    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/store",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=120),
    )
    try:
        checkpoint._build_meshes = _cpu_meshes  # CPU fixture, product still builds CUDA meshes.
        options = dict(
            config=SimpleNamespace(tp=1, ep=1, etp=1, cp=1, pp=1),
            ps=SimpleNamespace(pp_size=1, pp_rank=0),
        )
        path = str(root / "ckpt")
        bad = {
            "bad_rows": (dict(rows=4), "rows do not match row shard metadata"),
            "bad_cuts": (dict(boundaries=(1, 3, 5)), "Invalid checkpoint row shard boundaries"),
        }
        model = (
            _Plain() if kind in ("plain", "legacy") else _Model(rank, **bad.get(kind, ({}, ""))[0])
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        _fill(model, rank, 1.0)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        if kind in bad:
            with pytest.raises(ValueError, match=bad[kind][1]):
                checkpoint.save_training_checkpoint(model, optimizer, 1, path, **options)
            return
        checkpoint.save_training_checkpoint(model, optimizer, 1, path, **options)
        keys = set(
            checkpoint.dcp.FileSystemReader(str(root / "ckpt" / "step_1"))
            .read_metadata()
            .state_dict_metadata
        )
        if kind == "legacy":
            model = _Model(rank)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        _fill(model, rank, 7.0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        assert (
            checkpoint.load_training_checkpoint(
                model, optimizer, path, load_optimizer=kind != "legacy", **options
            )
            == 1
        )
        restored = model.state_dict()
        if kind == "plain":
            assert keys == {"model.linear.bias", "model.linear.weight", "step"}
            assert all(_bytes_equal(restored[k], saved[k]) for k in saved)
        elif kind == "full":
            assert {
                "model.bias_state",
                "model.table.weight",
                "model.table.scale",
                "model.table.master",
            } <= keys
            assert all(_bytes_equal(restored[k], saved[k]) for k in saved)
            assert float(model.scratch[0]) == 7.0
        else:
            assert all(
                _bytes_equal(restored[k], saved[k]) for k in ("linear.weight", "linear.bias")
            )
            assert all(_bytes_equal(restored[k], before[k]) for k in ("bias_state", "table.scale"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["plain", "full", "legacy", "bad_rows", "bad_cuts"])
def test_dcp_persistent_buffers_row_shards_and_legacy(tmp_path, kind):
    mp.start_processes(
        _worker, args=(str(tmp_path), kind), nprocs=WORLD, join=True, start_method="spawn"
    )

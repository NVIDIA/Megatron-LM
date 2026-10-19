# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DCP training checkpoints carry persistent buffers and keep row shards at their global rows.

A model without persistent buffers or row-sharded modules checkpoints exactly as
before: the same keys, parameters restored byte for byte.
"""

import json
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
    """A table whose rows are sharded over a group, as Engram's lookup is: frozen
    storage buffers plus a trainable master."""

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
    record = {}
    try:
        checkpoint._build_meshes = _cpu_meshes  # the DCP path builds CUDA meshes
        options = dict(
            config=SimpleNamespace(tp=1, ep=1, etp=1, cp=1, pp=1),
            ps=SimpleNamespace(pp_size=1, pp_rank=0),
        )
        path = str(root / "ckpt")
        if kind == "bad_rows":
            model = _Model(rank, rows=4)  # local rows disagree with the boundaries
        elif kind == "bad_cuts":
            model = _Model(rank, boundaries=(1, 3, 5))
        else:
            model = _Plain() if kind in ("plain", "legacy") else _Model(rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        _fill(model, rank, 1.0)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        try:
            checkpoint.save_training_checkpoint(model, optimizer, 1, path, **options)
        except ValueError as error:
            record["save_error"] = str(error)
            return
        reader = checkpoint.dcp.FileSystemReader(str(root / "ckpt" / "step_1"))
        record["keys"] = sorted(reader.read_metadata().state_dict_metadata)
        if kind == "legacy":
            # A checkpoint without buffer state, loaded into a model that has
            # persistent buffers and row shards.
            model = _Model(rank)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        _fill(model, rank, 7.0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        # The legacy model's optimizer has other parameters; only model state is under test.
        step = checkpoint.load_training_checkpoint(
            model, optimizer, path, load_optimizer=kind != "legacy", **options
        )
        restored = model.state_dict()
        record.update(
            step=int(step),
            restored={k: _bytes_equal(restored[k], saved[k]) for k in saved if k in restored},
            untouched={k: _bytes_equal(restored[k], before[k]) for k in restored},
            scratch=float(model.scratch[0]),
        )
    except Exception:
        import traceback

        record = dict(error=traceback.format_exc()[-4000:])
    finally:
        (root / f"rank-{rank}.json").write_text(json.dumps(record))
        dist.destroy_process_group()


def _records(tmp_path, kind):
    mp.start_processes(
        _worker, args=(str(tmp_path), kind), nprocs=WORLD, join=True, start_method="spawn"
    )
    records = [json.loads((tmp_path / f"rank-{r}.json").read_text()) for r in range(WORLD)]
    for record in records:
        assert "error" not in record, record["error"]
    return records


def test_dcp_checkpoint_of_a_plain_model_is_unchanged(tmp_path):
    for record in _records(tmp_path, "plain"):
        # Only the step and the parameters: no buffer or row-shard keys appear.
        assert record["keys"] == ["model.linear.bias", "model.linear.weight", "step"]
        assert record["step"] == 1
        assert record["restored"] == {"linear.weight": True, "linear.bias": True}


def test_dcp_checkpoint_restores_buffers_and_row_shards(tmp_path):
    for record in _records(tmp_path, "full"):
        assert record["step"] == 1
        expected = {"model.bias_state", "model.table.weight", "model.table.scale"}
        assert expected | {"model.table.master"} <= set(record["keys"])
        # Parameters, the persistent buffer and this rank's own table rows (buffers
        # and the row-sharded parameter) come back.
        assert all(record["restored"].values()), record["restored"]
        # Non-persistent buffers are not checkpoint state.
        assert record["scratch"] == 7.0


def test_dcp_checkpoint_without_buffer_state_still_loads(tmp_path):
    for record in _records(tmp_path, "legacy"):
        # Parameters come back; the buffers it never saved keep their values.
        assert record["restored"]["linear.weight"] and record["restored"]["linear.bias"]
        assert record["untouched"]["bias_state"] and record["untouched"]["table.scale"]


@pytest.mark.parametrize(
    "kind,message",
    [
        ("bad_rows", "rows do not match row shard metadata"),
        ("bad_cuts", "Invalid checkpoint row shard boundaries"),
    ],
)
def test_dcp_checkpoint_refuses_inconsistent_row_shards(tmp_path, kind, message):
    for record in _records(tmp_path, kind):
        assert message in record["save_error"], record

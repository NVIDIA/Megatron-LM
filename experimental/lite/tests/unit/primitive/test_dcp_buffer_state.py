# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DCP training checkpoints carry persistent buffers, row shards at their global rows."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

WORLD = 2
ROWS = (0, 3, 5)  # rank 0 owns rows [0, 3), rank 1 owns rows [3, 5)


class _Table(nn.Module):
    """A frozen table whose rows are sharded over a group, as Engram's lookup is."""

    def __init__(self, rank):
        super().__init__()
        self.lookup = SimpleNamespace(group=dist.group.WORLD, boundaries=ROWS)
        rows = ROWS[rank + 1] - ROWS[rank]
        self.register_buffer("weight", torch.zeros(rows, 4, dtype=torch.float8_e4m3fn))
        self.register_buffer("scale", torch.zeros(rows, 1))


class _Model(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("bias_state", torch.zeros(4))
        self.register_buffer("scratch", torch.zeros(4), persistent=False)
        self.table = _Table(rank)


def _fill(model, rank, value):
    with torch.no_grad():
        model.linear.weight.fill_(value)
        model.linear.bias.fill_(value)
        model.bias_state.copy_(torch.arange(4.0) + value)
        model.scratch.fill_(value)
        rows = torch.arange(model.table.scale.shape[0], dtype=torch.float32) + ROWS[rank]
        model.table.scale.copy_(rows[:, None] + value)
        model.table.weight.copy_((rows[:, None] + value).expand(-1, 4).to(torch.float8_e4m3fn))


def _cpu_meshes(config):
    from torch.distributed.device_mesh import DeviceMesh

    ranks = torch.arange(dist.get_world_size())
    return DeviceMesh("cpu", ranks.reshape(1, -1, 1, 1)), DeviceMesh(
        "cpu", ranks.reshape(1, -1, 1, 1)
    )


def _worker(rank, directory):
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
        config = SimpleNamespace(tp=1, ep=1, etp=1, cp=1, pp=1)
        ps = SimpleNamespace(pp_size=1, pp_rank=0)
        model = _Model(rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        _fill(model, rank, 1.0)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        checkpoint.save_training_checkpoint(
            model, optimizer, 1, str(root / "ckpt"), config=config, ps=ps
        )
        _fill(model, rank, 7.0)
        step = checkpoint.load_training_checkpoint(
            model, optimizer, str(root / "ckpt"), config=config, ps=ps
        )
        restored = model.state_dict()
        record = dict(
            step=int(step),
            equal={
                k: torch.equal(restored[k].view(torch.uint8), saved[k].view(torch.uint8))
                for k in saved
            },
            scratch=float(model.scratch[0]),
        )
    except Exception:
        import traceback

        record = dict(error=traceback.format_exc()[-4000:])
    finally:
        (root / f"rank-{rank}.json").write_text(json.dumps(record))
        dist.destroy_process_group()


def test_dcp_checkpoint_restores_persistent_and_row_sharded_buffers(tmp_path):
    mp.start_processes(
        _worker, args=(str(tmp_path),), nprocs=WORLD, join=True, start_method="spawn"
    )
    for rank in range(WORLD):
        record = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert "error" not in record, record["error"]
        assert record["step"] == 1
        # Parameters, the persistent buffer and this rank's table rows come back.
        assert all(record["equal"].values()), record["equal"]
        # Non-persistent buffers are not checkpoint state.
        assert record["scratch"] == 7.0

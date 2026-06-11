# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for `megatron_fsdp.dump_inputs.dump_optimizer_inputs`.

Run with torchrun:
    torchrun --nproc_per_node=4 -m pytest test_dump_inputs.py -v
    torchrun --nproc_per_node=2 -m pytest test_dump_inputs.py -v
"""

import json
import pathlib

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.dump_inputs import dump_optimizer_inputs


def _ensure_process_group(distributed_setup) -> None:
    """torchrun-launched, NCCL-backed PG if not already initialized."""
    if dist.is_initialized():
        return
    backend = "nccl" if distributed_setup.device.type == "cuda" else "gloo"
    dist.init_process_group(
        backend=backend, rank=distributed_setup.rank, world_size=distributed_setup.world_size
    )


def test_dump_optimizer_inputs_dtensor(distributed_setup, tmp_path: pathlib.Path) -> None:
    """Dump captures per-param mesh, placements, local/global shape, dtype, attrs."""
    _ensure_process_group(distributed_setup)
    mesh = init_device_mesh(
        distributed_setup.device.type, (distributed_setup.world_size,), mesh_dim_names=("dp",)
    )

    sharded_global = torch.randn(distributed_setup.world_size * 4, 8)
    sharded = distribute_tensor(sharded_global, mesh, [Shard(0)])
    replicated_global = torch.zeros(2, 3)
    replicated = distribute_tensor(replicated_global, mesh, [Replicate()])

    sharded.is_qkv = True  # exercise the optional attribute capture
    optimizer = torch.optim.SGD([sharded, replicated], lr=0.1, momentum=0.9)

    out_path = tmp_path / "dump.json"
    dump_optimizer_inputs(optimizer, out_path, extra_meta={"note": "hello"})

    written = out_path.with_suffix(f".rank{distributed_setup.rank}.json")
    assert written.exists(), f"per-rank dump file missing: {written}"
    spec = json.loads(written.read_text())

    assert spec["world_size"] == distributed_setup.world_size
    assert spec["rank"] == distributed_setup.rank
    assert spec["extra"] == {"note": "hello"}
    assert len(spec["params"]) == 2

    p_sharded, p_replicated = spec["params"]

    assert p_sharded["global_shape"] == [distributed_setup.world_size * 4, 8]
    assert p_sharded["local_shape"] == [4, 8]
    assert p_sharded["mesh_shape"] == [distributed_setup.world_size]
    assert p_sharded["mesh_dim_names"] == ["dp"]
    assert p_sharded["placements"] == ["Shard(dim=0)"]
    assert p_sharded["is_qkv"] is True
    assert p_sharded["dtype"].startswith("torch.float")
    # No `.step()` was called, so the optimizer state hasn't been populated yet.
    assert p_sharded["momentum_dtype"] is None
    assert p_sharded["momentum_local_shape"] is None

    assert p_replicated["global_shape"] == [2, 3]
    assert p_replicated["local_shape"] == [2, 3]
    assert p_replicated["placements"] == ["Replicate()"]
    assert p_replicated["is_qkv"] is False


def test_dump_optimizer_inputs_plain_tensor(distributed_setup, tmp_path: pathlib.Path) -> None:
    """Plain (non-DTensor) params: local_shape == global_shape, mesh fields are None."""
    _ensure_process_group(distributed_setup)

    param = torch.randn(4, 5, requires_grad=True, device=distributed_setup.device)
    optimizer = torch.optim.SGD([param], lr=0.01)

    out_path = tmp_path / f"plain_rank{distributed_setup.rank}.json"
    dump_optimizer_inputs(optimizer, out_path)

    written = out_path.parent / f"{out_path.name}.rank{distributed_setup.rank}.json"
    assert written.exists()
    spec = json.loads(written.read_text())
    p = spec["params"][0]
    assert p["global_shape"] == [4, 5]
    assert p["local_shape"] == [4, 5]
    assert p["mesh_shape"] is None
    assert p["mesh_dim_names"] is None
    assert p["placements"] is None


def test_dump_optimizer_inputs_rejects_multi_group(
    distributed_setup, tmp_path: pathlib.Path
) -> None:
    """Multi-param-group optimizers raise — the dump assumes a single group."""
    _ensure_process_group(distributed_setup)
    a = torch.randn(2, 2, requires_grad=True, device=distributed_setup.device)
    b = torch.randn(2, 2, requires_grad=True, device=distributed_setup.device)
    optimizer = torch.optim.SGD([{"params": [a]}, {"params": [b]}], lr=0.01)
    with pytest.raises(ValueError, match="exactly one parameter group"):
        dump_optimizer_inputs(optimizer, tmp_path / "rejects.json")

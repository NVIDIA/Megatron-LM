# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for `megatron.core.distributed.dump_parameters.dump_optimizer_parameters`.

Run with torchrun:
    torchrun --nproc_per_node=4 -m pytest test_dump_parameters.py -v
    torchrun --nproc_per_node=2 -m pytest test_dump_parameters.py -v
"""

import json
import pathlib

import torch
from torch.distributed._tensor import Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.dump_parameters import dump_optimizer_parameters


def _build_mesh(distributed_setup):
    """Build a 1-D DeviceMesh; this also lazy-inits the default process group."""
    return init_device_mesh(
        distributed_setup.device.type, (distributed_setup.world_size,), mesh_dim_names=("dp",)
    )


def test_dump_optimizer_parameters_dtensor(distributed_setup, tmp_path: pathlib.Path) -> None:
    """Dump captures per-param mesh, placements, local/global shape, dtype, attrs."""
    mesh = _build_mesh(distributed_setup)

    sharded_global = torch.randn(distributed_setup.world_size * 4, 8)
    sharded = distribute_tensor(sharded_global, mesh, [Shard(0)])
    replicated_global = torch.zeros(2, 3)
    replicated = distribute_tensor(replicated_global, mesh, [Replicate()])

    sharded.is_qkv = True  # exercise the optional attribute capture
    optimizer = torch.optim.SGD([sharded, replicated], lr=0.1, momentum=0.9)

    out_path = tmp_path / "dump.json"
    dump_optimizer_parameters(optimizer, out_path)

    written = out_path.with_suffix(f".rank{distributed_setup.rank}.json")
    assert written.exists(), f"per-rank dump file missing: {written}"
    spec = json.loads(written.read_text())

    assert spec["world_size"] == distributed_setup.world_size
    assert spec["rank"] == distributed_setup.rank
    assert len(spec["groups"]) == 1
    assert len(spec["groups"][0]["params"]) == 2

    p_sharded, p_replicated = spec["groups"][0]["params"]

    assert p_sharded["global_shape"] == [distributed_setup.world_size * 4, 8]
    assert p_sharded["local_shape"] == [4, 8]
    assert p_sharded["mesh_shape"] == [distributed_setup.world_size]
    assert p_sharded["mesh_dim_names"] == ["dp"]
    # Canonical mesh_ranks (== range(mesh.numel())) are omitted from the dump.
    # The full-world mesh used here always has canonical ranks.
    assert "mesh_ranks" not in p_sharded
    assert p_sharded["placements"] == ["Shard(dim=0)"]
    assert p_sharded["is_qkv"] is True
    assert p_sharded["dtype"].startswith("torch.float")

    assert p_replicated["global_shape"] == [2, 3]
    assert p_replicated["local_shape"] == [2, 3]
    assert p_replicated["placements"] == ["Replicate()"]
    assert p_replicated["is_qkv"] is False


def test_dump_optimizer_parameters_multi_group(distributed_setup, tmp_path: pathlib.Path) -> None:
    """Multi-param-group optimizers (e.g., Megatron's wd/no-wd split):
    structure is preserved in `groups`, no hyperparameters recorded."""
    mesh = _build_mesh(distributed_setup)
    a = distribute_tensor(torch.randn(distributed_setup.world_size * 2, 2), mesh, [Shard(0)])
    b = distribute_tensor(torch.randn(distributed_setup.world_size * 3, 3), mesh, [Shard(0)])

    optimizer = torch.optim.SGD(
        [{"params": [a], "weight_decay": 0.1}, {"params": [b], "weight_decay": 0.0}], lr=0.01
    )

    out_path = tmp_path / "multi.json"
    dump_optimizer_parameters(optimizer, out_path)

    written = out_path.with_suffix(f".rank{distributed_setup.rank}.json")
    spec = json.loads(written.read_text())
    assert len(spec["groups"]) == 2
    assert len(spec["groups"][0]["params"]) == 1
    assert len(spec["groups"][1]["params"]) == 1
    assert spec["groups"][0]["params"][0]["global_shape"] == [distributed_setup.world_size * 2, 2]
    assert spec["groups"][1]["params"][0]["global_shape"] == [distributed_setup.world_size * 3, 3]
    # Group hyperparameters (weight_decay, lr) are intentionally not recorded.
    assert "weight_decay" not in spec["groups"][0]
    assert "lr" not in spec["groups"][0]

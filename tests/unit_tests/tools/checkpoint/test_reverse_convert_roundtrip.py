# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Round-trip test for the fsdp_dtensor -> torch_dist reverse converter.

For each architecture archetype, build a synthetic native ``torch_dist`` checkpoint,
run it through both ``checkpoint_inspector.py`` CLIs

    torch_dist --(convert-torch-dist-to-fsdp-dtensor)--> fsdp_dtensor
               --(convert-fsdp-dtensor-to-torch-dist)--> torch_dist'

and assert ``torch_dist' == torch_dist`` tensor-for-tensor. This drives the whole
``reverse_convert_checkpoint`` pipeline (DCP load, key classification, SwiGLU merge,
expert re-stack, layer stacking, MTP rename, optimizer-key remap, DCP save) on CPU
in seconds. The end-to-end proof on real Megatron-FSDP checkpoints lives in the
opt-in suite under ``tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist``.

The converters run as subprocesses: each initializes its own single-rank gloo
group, which must not collide with the process group of the (torchrun-launched)
unit-test session. Only rank 0 runs the test; it has no collectives.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME

_INSPECTOR = (
    Path(__file__).resolve().parents[4] / "tools" / "checkpoint" / "checkpoint_inspector.py"
)
_DIST_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_cli(*cli_args):
    """Run a converter CLI as a fresh single-rank process (no inherited rendezvous)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _DIST_ENV_KEYS and not k.startswith("TORCHELASTIC")
    }
    env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()), RANK="0", WORLD_SIZE="1")
    proc = subprocess.run(
        [sys.executable, str(_INSPECTOR), *map(str, cli_args)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail = "\n".join(proc.stdout.splitlines()[-30:])
    assert proc.returncode == 0, f"{cli_args[0]} failed (rc={proc.returncode}):\n{tail}"


def _load_tensors(path):
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    out = {
        k: torch.empty(m.size, dtype=m.properties.dtype)
        for k, m in md.items()
        if isinstance(m, TensorStorageMetadata)
    }
    dcp.load(out, storage_reader=FileSystemReader(path), no_dist=True)
    return out


def _model_tensors(kind, num_layers=3, hidden=16, vocab=32, experts=4):
    """Model section of a native torch_dist checkpoint for one archetype."""
    g = torch.Generator().manual_seed(1234)

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    sd = {
        "embedding.word_embeddings.weight": rnd(vocab, hidden),
        "decoder.final_layernorm.weight": rnd(hidden),
        "output_layer.weight": rnd(vocab, hidden),
    }
    if kind in ("dense", "swiglu", "mtp"):
        # Homogeneous block -> mcore stores every layer param stacked on axis 0.
        fc1 = 2 * 4 * hidden if kind == "swiglu" else 4 * hidden
        sd.update(
            {
                "decoder.layers.self_attention.linear_qkv.weight": rnd(
                    num_layers, 3 * hidden, hidden
                ),
                "decoder.layers.self_attention.linear_proj.weight": rnd(num_layers, hidden, hidden),
                "decoder.layers.input_layernorm.weight": rnd(num_layers, hidden),
                "decoder.layers.pre_mlp_layernorm.weight": rnd(num_layers, hidden),
                "decoder.layers.mlp.linear_fc1.weight": rnd(num_layers, fc1, hidden),
                "decoder.layers.mlp.linear_fc2.weight": rnd(num_layers, hidden, 4 * hidden),
            }
        )
    elif kind == "moe":
        # Interleaved dense/MoE (layer 0 dense) -> non-homogeneous, stored per-layer;
        # the routed experts are stacked on axis 0.
        for i in range(num_layers):
            p = f"decoder.layers.{i}."
            sd[p + "self_attention.linear_qkv.weight"] = rnd(3 * hidden, hidden)
            sd[p + "input_layernorm.weight"] = rnd(hidden)
            if i == 0:
                sd[p + "mlp.linear_fc1.weight"] = rnd(4 * hidden, hidden)
                sd[p + "mlp.linear_fc2.weight"] = rnd(hidden, 4 * hidden)
                continue
            sd[p + "mlp.router.weight"] = rnd(experts, hidden)
            sd[p + "mlp.experts.experts.linear_fc1.weight"] = rnd(experts, 4 * hidden, hidden)
            sd[p + "mlp.experts.experts.linear_fc2.weight"] = rnd(experts, hidden, 4 * hidden)
    else:
        raise ValueError(kind)
    if kind == "mtp":
        # An MTP layer is its own block, always stored per-layer, and must not stop
        # the decoder block from being stacked.
        p = "mtp.layers.0.transformer_layer."
        sd[p + "self_attention.linear_qkv.weight"] = rnd(3 * hidden, hidden)
        sd[p + "mlp.linear_fc1.weight"] = rnd(4 * hidden, hidden)
    return sd


def _save_torch_dist(path, kind, with_optimizer):
    """Write a native torch_dist checkpoint: model + (optionally) fully-reshardable
    optimizer state (fp32 masters under ``optimizer.state.param`` plus Adam moments)."""
    model = _model_tensors(kind)
    full = dict(model)
    if with_optimizer:
        g = torch.Generator().manual_seed(4321)
        for k, v in model.items():
            full[f"optimizer.state.param.{k}"] = v.clone()
            full[f"optimizer.state.exp_avg.{k}"] = torch.randn(v.shape, generator=g)
            full[f"optimizer.state.exp_avg_sq.{k}"] = torch.rand(v.shape, generator=g)
    os.makedirs(path, exist_ok=True)
    dcp.save(full, storage_writer=FileSystemWriter(path), no_dist=True)
    # Written directly: ``save_common`` needs a process group and this test has none.
    torch.save(
        {
            "args": SimpleNamespace(num_layers=3, hidden_size=16),
            "checkpoint_version": 3.0,
            "iteration": 100,
            "optimizer": {
                "optimizer": {"param_groups": [{"lr": 1e-3, "params": list(range(len(model)))}]}
            },
        },
        path / COMMON_STATE_FNAME,
    )
    save_config(CheckpointingConfig(sharded_backend="torch_dist"), path)


# (archetype, forward-converter flags, with optimizer state). MoE optimizer state
# cannot go through the *forward* converter from a model-shaped synthetic source
# (its fc2 ETP transpose expects mcore's nd-reformulated layout), so the MoE case is
# weights-only; real MoE optimizer state is covered by the end-to-end suite.
_CASES = [
    pytest.param("dense", (), True, id="dense"),
    pytest.param("swiglu", ("--swiglu",), True, id="swiglu"),
    pytest.param("moe", (), False, id="moe"),
    pytest.param("mtp", ("--rename-mtp-keys",), True, id="mtp"),
]


@pytest.mark.parametrize("kind, forward_flags, with_optimizer", _CASES)
def test_forward_then_reverse_is_identity(tmp_path, kind, forward_flags, with_optimizer):
    if dist.is_initialized() and dist.get_rank() != 0:
        pytest.skip("single-process test; runs on rank 0 only")
    td, fsdp, td2 = tmp_path / "td", tmp_path / "fsdp", tmp_path / "td2"
    _save_torch_dist(td, kind, with_optimizer)

    _run_cli("convert-torch-dist-to-fsdp-dtensor", td, fsdp, *forward_flags)
    _run_cli("convert-fsdp-dtensor-to-torch-dist", fsdp, td2)

    before, after = _load_tensors(td), _load_tensors(td2)
    assert set(after) == set(before), (
        f"missing {sorted(set(before) - set(after))[:5]}, "
        f"extra {sorted(set(after) - set(before))[:5]}"
    )
    bad = [k for k in before if not torch.equal(before[k], after[k])]
    assert not bad, f"{len(bad)} tensors changed in the round trip: {bad[:8]}"

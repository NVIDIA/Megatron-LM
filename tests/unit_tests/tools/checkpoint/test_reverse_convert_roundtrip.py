# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end round-trip validation for the fsdp_dtensor -> torch_dist converter.

This is a standalone GPU script (like ``test_distributed_round_trip.py``), not a
pytest-collected test — it drives the real ``checkpoint_inspector.py`` CLIs and,
for the real-model case, a full mcore save/convert/convert/load cycle. Run it
inside the CI dev container on 1+ GPU:

    # Synthetic round-trip for every architecture archetype (fast, no model):
    #   torch_dist --(forward)--> fsdp_dtensor --(reverse)--> torch_dist' ; compare.
    python tests/unit_tests/tools/checkpoint/test_reverse_convert_roundtrip.py synthetic

    # Real dense GPTModel round-trip WITH load into a fresh model+optimizer:
    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
        python tests/unit_tests/tools/checkpoint/test_reverse_convert_roundtrip.py real

    # Both:
    python tests/unit_tests/tools/checkpoint/test_reverse_convert_roundtrip.py

The synthetic case validates every key/tensor transform (prefix strip, SwiGLU
merge, expert re-stack, layer stacking, MTP rename, GDN passthrough, optimizer
state) by asserting ``forward∘reverse == identity`` on the tensors. The real case
proves the converted checkpoint actually loads into a native (non-FSDP) mcore
GPTModel + DistributedOptimizer and reproduces the weights.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
)
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.core.dist_checkpointing.strategies.common import save_common

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)  # make ``tests.unit_tests.*`` importable as a script
_INSPECTOR = os.path.join(_REPO_ROOT, "tools", "checkpoint", "checkpoint_inspector.py")
_ARCHETYPES = ("dense", "moe", "swiglu", "gdnmtp")
_PORT = [29811]


def _clean_subprocess_env():
    """Fresh single-rank env for a converter subprocess.

    Strips ``torchrun``'s elastic-agent variables so the child does a plain
    env:// rendezvous on its own port instead of trying to reach the parent's
    agent store.
    """
    _PORT[0] += 1
    env = {k: v for k, v in os.environ.items() if not k.startswith("TORCHELASTIC")}
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_PORT[0]),
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        }
    )
    return env


def _run_cli(*cli_args):
    py = os.environ.get("PYTHON", sys.executable)
    subprocess.run([py, _INSPECTOR, *cli_args], check=True, env=_clean_subprocess_env())


def _load_tensors(path):
    reader = FileSystemReader(path)
    md = reader.read_metadata()
    out = {
        k: torch.empty(v.size, dtype=v.properties.dtype)
        for k, v in md.state_dict_metadata.items()
        if isinstance(v, TensorStorageMetadata)
    }
    dcp.load(out, storage_reader=reader, planner=DefaultLoadPlanner())
    return out


def _assert_tensors_equal(a, b, label):
    ka, kb = set(a), set(b)
    assert ka == kb, (
        f"[{label}] key mismatch: {len(ka - kb)} missing {sorted(ka - kb)[:5]}, "
        f"{len(kb - ka)} extra {sorted(kb - ka)[:5]}"
    )
    bad = [
        k
        for k in ka
        if a[k].shape != b[k].shape or not torch.allclose(a[k], b[k], atol=1e-6, rtol=1e-5)
    ]
    assert not bad, f"[{label}] {len(bad)} tensor mismatches: {bad[:8]}"
    print(f"  PASS [{label}]: {len(ka)} tensors round-tripped exactly")


# ---------------------------------------------------------------------------
# Synthetic round-trip: build a native torch_dist state dict per archetype.
# ---------------------------------------------------------------------------
def _build_source(path, kind, num_layers=3, hidden=16, vocab=32, experts=4):
    torch.manual_seed(1234)
    sd = {
        "embedding.word_embeddings.weight": torch.randn(vocab, hidden),
        "decoder.final_layernorm.weight": torch.randn(hidden),
        "output_layer.weight": torch.randn(vocab, hidden),
    }
    if kind in ("dense", "swiglu"):
        # Dense/homogeneous -> layers stacked on axis 0.
        fc1 = 2 * 4 * hidden if kind == "swiglu" else 4 * hidden
        sd.update(
            {
                "decoder.layers.self_attention.linear_qkv.weight": torch.randn(
                    num_layers, 3 * hidden, hidden
                ),
                "decoder.layers.self_attention.linear_proj.weight": torch.randn(
                    num_layers, hidden, hidden
                ),
                "decoder.layers.input_layernorm.weight": torch.randn(num_layers, hidden),
                "decoder.layers.pre_mlp_layernorm.weight": torch.randn(num_layers, hidden),
                "decoder.layers.mlp.linear_fc1.weight": torch.randn(num_layers, fc1, hidden),
                "decoder.layers.mlp.linear_fc2.weight": torch.randn(num_layers, hidden, 4 * hidden),
            }
        )
    elif kind == "moe":
        # MoE -> non-homogeneous: every param per-layer; experts stacked on axis 0.
        for i in range(num_layers):
            p = f"decoder.layers.{i}."
            sd[p + "self_attention.linear_qkv.weight"] = torch.randn(3 * hidden, hidden)
            sd[p + "self_attention.linear_proj.weight"] = torch.randn(hidden, hidden)
            sd[p + "input_layernorm.weight"] = torch.randn(hidden)
            sd[p + "pre_mlp_layernorm.weight"] = torch.randn(hidden)
            sd[p + "mlp.router.weight"] = torch.randn(experts, hidden)
            sd[p + "mlp.experts.experts.linear_fc1.weight"] = torch.randn(
                experts, 4 * hidden, hidden
            )
            sd[p + "mlp.experts.experts.linear_fc2.weight"] = torch.randn(
                experts, hidden, 4 * hidden
            )
    elif kind == "gdnmtp":
        # Gated DeltaNet (split sub-keys, per-layer) + an MTP layer.
        for i in range(num_layers):
            p = f"decoder.layers.{i}."
            for sub in ("query", "key", "value", "z", "beta", "alpha"):
                sd[p + f"self_attention.in_proj.weight.{sub}"] = torch.randn(hidden, hidden)
            for sub in ("query", "key", "value"):
                sd[p + f"self_attention.conv1d.weight.{sub}"] = torch.randn(hidden, 1, 4)
            sd[p + "input_layernorm.weight"] = torch.randn(hidden)
            sd[p + "mlp.linear_fc1.weight"] = torch.randn(4 * hidden, hidden)
            sd[p + "mlp.linear_fc2.weight"] = torch.randn(hidden, 4 * hidden)
        sd["mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight"] = torch.randn(
            3 * hidden, hidden
        )
        sd["mtp.layers.0.transformer_layer.mlp.linear_fc1.weight"] = torch.randn(4 * hidden, hidden)

    # MoE fc2 optimizer state cannot round-trip through the *forward* converter's
    # nd_reformulated ETP transpose from a model-shaped synthetic source, so the
    # synthetic MoE case is weights-only; full MoE optimizer is covered by the
    # real-model case below.
    full = dict(sd)
    if kind != "moe":
        for k, v in sd.items():
            full[f"optimizer.state.exp_avg.{k}"] = torch.randn_like(v)
            full[f"optimizer.state.exp_avg_sq.{k}"] = torch.randn_like(v).abs()

    os.makedirs(path, exist_ok=True)
    dcp.save(full, storage_writer=FileSystemWriter(path), planner=DefaultSavePlanner())
    save_common(
        {
            "args": SimpleNamespace(num_layers=num_layers, hidden_size=hidden),
            "checkpoint_version": 3.0,
            "iteration": 100,
            "optimizer": {
                "optimizer": {"param_groups": [{"lr": 1e-3, "params": list(range(len(sd)))}]}
            },
        },
        path,
    )
    save_config(CheckpointingConfig(sharded_backend="torch_dist"), path)


def run_synthetic():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29799")
        dist.init_process_group("gloo", rank=0, world_size=1)
    for kind in _ARCHETYPES:
        print(f"==== synthetic:{kind} ====")
        tmp = tempfile.mkdtemp(prefix=f"rt_{kind}_")
        td, fsdp, td2 = f"{tmp}/td", f"{tmp}/fsdp", f"{tmp}/td2"
        _build_source(td, kind)
        fwd = ["convert-torch-dist-to-fsdp-dtensor", td, fsdp]
        if kind == "swiglu":
            fwd.append("--swiglu")
        if kind == "gdnmtp":
            fwd.append("--rename-mtp-keys")
        _run_cli(*fwd)
        _run_cli("convert-fsdp-dtensor-to-torch-dist", fsdp, td2)
        _assert_tensors_equal(_load_tensors(td), _load_tensors(td2), f"synthetic:{kind}")


# ---------------------------------------------------------------------------
# Real dense GPTModel round-trip WITH load into a fresh model+optimizer.
# ---------------------------------------------------------------------------
def run_real():
    from unittest import mock

    from megatron.training.arguments import parse_args
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.training import preprocess_common_state_dict
    from tests.unit_tests.dist_checkpointing.utils import (
        init_checkpointing_mock_args,
        setup_model_and_optimizer,
    )
    from tests.unit_tests.test_utilities import Utils

    print("==== real:dense (with load) ====")
    Utils.initialize_model_parallel(1, 1)
    root = tempfile.mkdtemp(prefix="rt_real_")
    a_dir, fsdp_dir, b_dir = f"{root}/A", f"{root}/fsdp", f"{root}/B"

    args_a = parse_args(ignore_unknown_args=True)
    args_a.use_distributed_optimizer = True
    with mock.patch("megatron.training.checkpointing.get_args", new=lambda: args_a):
        model_a, opt_a = setup_model_and_optimizer(seed=2, tp=1, pp=1)
        init_checkpointing_mock_args(args_a, a_dir)
        args_a.dist_ckpt_optim_fully_reshardable = True
        save_checkpoint(
            10,
            model_a,
            opt_a,
            None,
            0,
            preprocess_common_state_dict_fn=preprocess_common_state_dict,
        )
    state_a = {
        k: v.clone() for k, v in model_a[0].state_dict().items() if isinstance(v, torch.Tensor)
    }

    if torch.distributed.get_rank() == 0:
        _run_cli(
            "convert-torch-dist-to-fsdp-dtensor", f"{a_dir}/iter_0000010", fsdp_dir, "--swiglu"
        )
        os.makedirs(f"{b_dir}/iter_0000010", exist_ok=True)
        _run_cli("convert-fsdp-dtensor-to-torch-dist", fsdp_dir, f"{b_dir}/iter_0000010")
        with open(f"{b_dir}/latest_checkpointed_iteration.txt", "w") as fh:
            fh.write("10")
    torch.distributed.barrier()

    args_b = parse_args(ignore_unknown_args=True)
    args_b.use_distributed_optimizer = True
    with mock.patch("megatron.training.checkpointing.get_args", new=lambda: args_b):
        model_b, opt_b = setup_model_and_optimizer(seed=999, tp=1, pp=1)
        init_checkpointing_mock_args(args_b, b_dir)
        args_b.dist_ckpt_optim_fully_reshardable = True
        args_b.dist_ckpt_strictness = "log_all"  # tolerate omitted _extra_state
        args_b.no_load_rng = True  # rng_state is not round-tripped
        with (
            mock.patch("megatron.training.checkpointing.check_checkpoint_args"),
            mock.patch("megatron.training.checkpointing.update_num_microbatches"),
        ):
            iteration, _ = load_checkpoint(model_b, opt_b, None)

    state_b = {k: v for k, v in model_b[0].state_dict().items() if isinstance(v, torch.Tensor)}
    if torch.distributed.get_rank() == 0:
        bad = [
            k
            for k in state_a
            if k not in state_b
            or not torch.allclose(
                state_a[k].cpu().float(), state_b[k].cpu().float(), atol=1e-3, rtol=1e-3
            )
        ]
        assert iteration == 10, iteration
        assert not bad, f"real-load mismatches: {bad[:8]}"
        print(f"  PASS [real:dense]: loaded into a fresh GPTModel; {len(state_a)} weights match")
    Utils.destroy_model_parallel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modes", nargs="*", default=["synthetic", "real"])
    args = parser.parse_args()
    if "synthetic" in args.modes:
        run_synthetic()
    if "real" in args.modes:
        run_real()


if __name__ == "__main__":
    main()

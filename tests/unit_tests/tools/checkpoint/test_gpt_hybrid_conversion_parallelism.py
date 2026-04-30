# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Parallelism-matrix integration tests for gpt_hybrid_conversion.py.

The converter operates on dist (``torch_dist`` / ``fsdp_dtensor``) checkpoints
only — DCP's metadata stores each tensor's ``global_shape``, so the on-disk
TP / PP / FSDP layout is abstracted away from the conversion logic. We
synthesize a DCP checkpoint via a single-rank ``dcp.save`` and round-trip
GPT -> Hybrid -> GPT through the conversion CLI, asserting attention and MLP
weights match exactly.

Each scenario is run as a distinct test to document the supported matrix and
catch regressions in dispatch logic. Designed to run on a single-GPU node via
SLURM (no torchrun needed).
"""

import argparse
import copy
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

# Make the conversion tool importable under both `python <file>` and `pytest`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))
sys.path.insert(0, _THIS_DIR)

from gpt_hybrid_conversion import main as conversion_main


# These scenarios are SYNTHETIC and single-rank by design: each one writes a
# tiny synthetic DCP checkpoint and round-trips it through the converter on
# rank 0. They share the default torch.distributed process group with whatever
# harness launched pytest. When that default PG is multi-rank (e.g. Megatron's
# CI/CD initialises NCCL with world_size>1 before pytest collection), the
# dcp.save/dcp.load collectives stall: each rank has its own
# tempfile.mkdtemp() path and its own torch.randn() tensors, so the metadata
# coordination across ranks never converges and the NCCL watchdog kills the
# job after 10 minutes (see ProcessGroupNCCL ALLGATHER timeout).
#
# Multi-rank coverage lives in test_distributed_round_trip.py, which uses a
# fresh single-rank gloo subgroup per scenario via SLURM/srun in
# run_slurm_ckpt_convert_tests.sh. Skip these synthetic tests whenever the
# default PG is already multi-rank.
@pytest.fixture(autouse=True)
def _skip_when_multi_rank_pg():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        pytest.skip(
            "Synthetic single-rank tests skipped under a multi-rank default "
            "process group; multi-rank coverage is in "
            "test_distributed_round_trip.py."
        )


# ---------------------------------------------------------------------------
# Synthetic-checkpoint helpers
# ---------------------------------------------------------------------------


def make_checkpoint_args(
    num_layers=4,
    hidden_size=128,
    num_attention_heads=4,
    seq_length=256,
    max_position_embeddings=256,
    iteration=100,
    num_moe_experts=None,
    moe_shared_expert_intermediate_size=None,
):
    """Build a minimal checkpoint 'args' namespace mirroring Megatron's.

    Set ``num_moe_experts`` to make the source/target a MoE GPT; the converter
    will then pass the MoE config through unchanged so the round-trip stays
    structurally consistent.
    """
    return SimpleNamespace(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=hidden_size * 4,
        seq_length=seq_length,
        max_position_embeddings=max_position_embeddings,
        iteration=iteration,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        train_iters=1000,
        train_samples=0,
        tokenizer_type='GPT2BPETokenizer',
        position_embedding_type='rope',
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
        num_moe_experts=num_moe_experts,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        moe_layer_freq=1,
    )


def make_gpt_state_dict(
    num_layers,
    hidden_size,
    vocab_size=1024,
    dtype=torch.float32,
    num_moe_experts=None,
    shared_expert_size=None,
):
    """Create a minimal GPT state dict with the standard Megatron keys.

    Dense MLP layout (default): ``mlp.linear_fc1`` / ``mlp.linear_fc2``.
    MoE layout (``num_moe_experts`` set): ``mlp.router`` plus N experts under
    ``mlp.experts.local_experts.<j>.linear_fc{1,2}``, optionally a shared
    expert under ``mlp.shared_experts.linear_fc{1,2}``. These are exactly the
    keys Megatron writes for non-grouped-GEMM MoE — they all live under
    ``decoder.layers.<i>.mlp.*`` so the converter ferries them through with no
    MoE-specific code.
    """
    sd = OrderedDict()
    sd['embedding.word_embeddings.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)

    for i in range(num_layers):
        p = f'decoder.layers.{i}.'
        sd[p + 'input_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
        sd[p + 'self_attention.linear_qkv.weight'] = torch.randn(
            3 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'self_attention.linear_proj.weight'] = torch.randn(
            hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'pre_mlp_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)

        if num_moe_experts is None:
            # Dense MLP
            sd[p + 'mlp.linear_fc1.weight'] = torch.randn(4 * hidden_size, hidden_size, dtype=dtype)
            sd[p + 'mlp.linear_fc2.weight'] = torch.randn(hidden_size, 4 * hidden_size, dtype=dtype)
        else:
            # MoE: router + N experts (+ optional shared expert)
            sd[p + 'mlp.router.weight'] = torch.randn(num_moe_experts, hidden_size, dtype=dtype)
            for j in range(num_moe_experts):
                ep = p + f'mlp.experts.local_experts.{j}.'
                sd[ep + 'linear_fc1.weight'] = torch.randn(
                    4 * hidden_size, hidden_size, dtype=dtype
                )
                sd[ep + 'linear_fc2.weight'] = torch.randn(
                    hidden_size, 4 * hidden_size, dtype=dtype
                )
            if shared_expert_size is not None:
                sp = p + 'mlp.shared_experts.'
                sd[sp + 'linear_fc1.weight'] = torch.randn(
                    shared_expert_size, hidden_size, dtype=dtype
                )
                sd[sp + 'linear_fc2.weight'] = torch.randn(
                    hidden_size, shared_expert_size, dtype=dtype
                )

    sd['decoder.final_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
    sd['output_layer.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)
    return sd


# ---------------------------------------------------------------------------
# Dist (torch_dist / fsdp_dtensor) fixture builders
# ---------------------------------------------------------------------------


def _save_dist_checkpoint(
    root_dir, full_sd, ckpt_args, iteration=100, prefix='model.', backend='torch_dist'
):
    """Write a full state dict as a single-rank DCP checkpoint.

    From the converter's POV, this is indistinguishable from a multi-rank
    TP+PP+FSDP save: DCP stores each tensor's global shape in its metadata
    and the read planner reassembles the full tensor regardless of how many
    processes wrote it.
    """
    from dist_checkpoint_io import (
        ensure_single_rank_process_group,
        save_dist_checkpoint_full,
        write_latest_iteration_marker,
    )

    ensure_single_rank_process_group()

    iter_dir = os.path.join(root_dir, f'iter_{iteration:07d}')
    common_state = {
        'args': copy.deepcopy(ckpt_args),
        'checkpoint_version': 3.0,
        'iteration': iteration,
    }
    save_dist_checkpoint_full(full_sd, common_state, iter_dir, model_prefix=prefix, backend=backend)
    write_latest_iteration_marker(iter_dir, iteration)


def _load_converted_dist(ckpt_dir):
    """Read a dist-format converted checkpoint back into a full state dict."""
    from dist_checkpoint_io import load_dist_checkpoint_full

    sd, common, prefix, backend, iteration = load_dist_checkpoint_full(ckpt_dir)
    return sd, common.get('args', None)


# ---------------------------------------------------------------------------
# Core scenario runner
# ---------------------------------------------------------------------------


def _run_scenario(
    label,
    source_format,
    target_format,
    num_layers=4,
    hidden_size=128,
    pattern="M*-M*-M*-M*-",
    source_prefix='model.',
    num_moe_experts=None,
    shared_expert_size=None,
):
    """Build a GPT source ckpt, convert GPT->Hybrid->GPT, verify round-trip."""
    print(f"\n=== {label} ===")
    print(f"  source={source_format} (prefix='{source_prefix}')")
    print(f"  target={target_format}")
    if num_moe_experts is not None:
        print(f"  MoE: num_experts={num_moe_experts} shared={shared_expert_size}")

    tmpdir = tempfile.mkdtemp(prefix=f'gpt_hybrid_{label.replace(" ", "_")}_')
    try:
        src_gpt_dir = os.path.join(tmpdir, 'gpt_src')
        hybrid_dir = os.path.join(tmpdir, 'hybrid_mid')
        dst_gpt_dir = os.path.join(tmpdir, 'gpt_dst')

        ckpt_args = make_checkpoint_args(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            moe_shared_expert_intermediate_size=shared_expert_size,
        )
        gpt_sd = make_gpt_state_dict(
            num_layers,
            hidden_size,
            num_moe_experts=num_moe_experts,
            shared_expert_size=shared_expert_size,
        )

        _save_dist_checkpoint(
            src_gpt_dir, gpt_sd, ckpt_args, prefix=source_prefix, backend=source_format
        )

        common_kwargs = dict(
            hybrid_layer_pattern=pattern,
            d_model=hidden_size,
            mamba_version=2,
            mamba_d_state=16,
            mamba2_n_groups=2,
            mamba2_head_dim=32,
            d_conv=4,
            init_method_std=0.02,
            reset_iterations=False,
            input_format='auto',
            output_format=target_format,
        )

        # --- GPT -> Hybrid ---
        conversion_main(
            argparse.Namespace(
                direction='gpt-to-hybrid', load_dir=src_gpt_dir, save_dir=hybrid_dir, **common_kwargs
            )
        )

        # --- Hybrid -> GPT ---
        conversion_main(
            argparse.Namespace(
                direction='hybrid-to-gpt', load_dir=hybrid_dir, save_dir=dst_gpt_dir, **common_kwargs
            )
        )

        # --- Verify ---
        recovered_sd, _ = _load_converted_dist(dst_gpt_dir)
        # The hybrid->gpt step renames decoder.final_norm -> decoder.final_layernorm,
        # mirroring the original GPT key. So recovered_sd should have the same
        # keys and tensor values as gpt_sd.

        mismatches = []
        for key, original in gpt_sd.items():
            if key not in recovered_sd:
                mismatches.append(f"MISSING: {key}")
                continue
            if not torch.equal(original, recovered_sd[key]):
                max_diff = (original - recovered_sd[key]).abs().max().item()
                mismatches.append(f"MISMATCH: {key} (max_diff={max_diff})")

        if mismatches:
            for m in mismatches[:10]:
                print(f"  FAIL: {m}")
            raise AssertionError(f"{label} failed with {len(mismatches)} weight mismatches")

        # SSM keys must be absent in the final GPT output.
        assert not any('mixer.' in k for k in recovered_sd), (
            f"SSM keys leaked into final GPT output: "
            f"{[k for k in recovered_sd if 'mixer.' in k][:5]}"
        )

        print(f"PASSED: {label}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test cases — one per (source backend, target backend, pattern) combo
# ---------------------------------------------------------------------------


def test_torch_dist_roundtrip():
    _run_scenario("torch_dist roundtrip", 'torch_dist', 'torch_dist')


def test_fsdp_dtensor_roundtrip():
    _run_scenario("fsdp_dtensor roundtrip", 'fsdp_dtensor', 'fsdp_dtensor')


def test_fsdp_dtensor_prefix():
    """fsdp_dtensor backend uses the 'model.module.' key prefix — verify we
    auto-detect and strip it correctly."""
    _run_scenario(
        "fsdp_dtensor prefix", 'fsdp_dtensor', 'fsdp_dtensor', source_prefix='model.module.'
    )


def test_torch_dist_alternating_pattern():
    """Pure transformer pattern (no SSM) round-trips."""
    _run_scenario("torch_dist alternating", 'torch_dist', 'torch_dist', pattern="*-*-*-*-")


def test_torch_dist_dense_ssm_pattern():
    """Dense SSM pattern still round-trips on the attn/MLP layers."""
    _run_scenario("torch_dist dense SSM", 'torch_dist', 'torch_dist', pattern="MM*-MM*-MM*-MM*-")


def test_torch_dist_moe_roundtrip():
    """MoE GPT (Mixtral-style) round-trips through an 'E'-bearing pattern.

    Source has num_moe_experts=4 and writes mlp.router / mlp.experts.* keys.
    The hybrid pattern 'M*EM*EM*E' has 3 'E' positions, one per source layer.
    The converter should ferry the router + every per-expert tensor through
    verbatim — no MoE-specific code path involved.
    """
    _run_scenario(
        "torch_dist MoE roundtrip",
        'torch_dist',
        'torch_dist',
        num_layers=3,
        pattern="M*EM*EM*E",
        num_moe_experts=4,
    )


def test_torch_dist_moe_with_shared_experts():
    """MoE + shared experts round-trip together (mlp.shared_experts.* keys)."""
    _run_scenario(
        "torch_dist MoE+shared",
        'torch_dist',
        'torch_dist',
        num_layers=3,
        hidden_size=64,
        pattern="*E*E*E",
        num_moe_experts=4,
        shared_expert_size=64 * 2,
    )


def test_fsdp_dtensor_moe_roundtrip():
    """MoE round-trips through fsdp_dtensor (covers the 'model.module.' prefix
    case combined with MoE keys)."""
    _run_scenario(
        "fsdp_dtensor MoE roundtrip",
        'fsdp_dtensor',
        'fsdp_dtensor',
        num_layers=3,
        pattern="M*EM*EM*E",
        num_moe_experts=4,
        source_prefix='model.module.',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("GPT <-> Hybrid Conversion Parallelism Matrix Tests")
    print("=" * 60)

    test_torch_dist_roundtrip()
    test_fsdp_dtensor_roundtrip()
    test_fsdp_dtensor_prefix()
    test_torch_dist_alternating_pattern()
    test_torch_dist_dense_ssm_pattern()
    test_torch_dist_moe_roundtrip()
    test_torch_dist_moe_with_shared_experts()
    test_fsdp_dtensor_moe_roundtrip()

    print("=" * 60)
    print("ALL PARALLELISM MATRIX TESTS PASSED")
    print("=" * 60)

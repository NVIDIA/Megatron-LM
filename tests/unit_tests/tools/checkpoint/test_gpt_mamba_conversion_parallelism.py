# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Parallelism-matrix integration tests for gpt_mamba_conversion.py.

The converter operates on dist (``torch_dist`` / ``fsdp_dtensor``) checkpoints
only — DCP's metadata stores each tensor's ``global_shape``, so the on-disk
TP / PP / FSDP layout is abstracted away from the conversion logic. We
synthesize a DCP checkpoint via a single-rank ``dcp.save`` and round-trip
GPT -> Mamba -> GPT through the conversion CLI, asserting attention and MLP
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

import torch

# Make the conversion tool importable under both `python <file>` and `pytest`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))
sys.path.insert(0, _THIS_DIR)

from gpt_mamba_conversion import main as conversion_main


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
):
    """Build a minimal checkpoint 'args' namespace mirroring Megatron's."""
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
    )


def make_gpt_state_dict(num_layers, hidden_size, vocab_size=1024, dtype=torch.float32):
    """Create a minimal GPT model state dict with the standard Megatron keys."""
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
        sd[p + 'mlp.linear_fc1.weight'] = torch.randn(
            4 * hidden_size, hidden_size, dtype=dtype
        )
        sd[p + 'mlp.linear_fc2.weight'] = torch.randn(
            hidden_size, 4 * hidden_size, dtype=dtype
        )

    sd['decoder.final_layernorm.weight'] = torch.randn(hidden_size, dtype=dtype)
    sd['output_layer.weight'] = torch.randn(vocab_size, hidden_size, dtype=dtype)
    return sd


# ---------------------------------------------------------------------------
# Dist (torch_dist / fsdp_dtensor) fixture builders
# ---------------------------------------------------------------------------

def _save_dist_checkpoint(root_dir, full_sd, ckpt_args, iteration=100,
                          prefix='model.', backend='torch_dist'):
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
    save_dist_checkpoint_full(
        full_sd, common_state, iter_dir,
        model_prefix=prefix, backend=backend,
    )
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
):
    """Build a GPT source ckpt, convert GPT->Mamba->GPT, verify round-trip."""
    print(f"\n=== {label} ===")
    print(f"  source={source_format} (prefix='{source_prefix}')")
    print(f"  target={target_format}")

    tmpdir = tempfile.mkdtemp(prefix=f'gpt_mamba_{label.replace(" ", "_")}_')
    try:
        src_gpt_dir = os.path.join(tmpdir, 'gpt_src')
        mamba_dir = os.path.join(tmpdir, 'mamba_mid')
        dst_gpt_dir = os.path.join(tmpdir, 'gpt_dst')

        ckpt_args = make_checkpoint_args(num_layers=num_layers, hidden_size=hidden_size)
        gpt_sd = make_gpt_state_dict(num_layers, hidden_size)

        _save_dist_checkpoint(
            src_gpt_dir, gpt_sd, ckpt_args,
            prefix=source_prefix, backend=source_format,
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

        # --- GPT -> Mamba ---
        conversion_main(argparse.Namespace(
            direction='gpt-to-mamba',
            load_dir=src_gpt_dir,
            save_dir=mamba_dir,
            **common_kwargs,
        ))

        # --- Mamba -> GPT ---
        conversion_main(argparse.Namespace(
            direction='mamba-to-gpt',
            load_dir=mamba_dir,
            save_dir=dst_gpt_dir,
            **common_kwargs,
        ))

        # --- Verify ---
        recovered_sd, _ = _load_converted_dist(dst_gpt_dir)
        # The mamba->gpt step renames decoder.final_norm -> decoder.final_layernorm,
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
            raise AssertionError(
                f"{label} failed with {len(mismatches)} weight mismatches"
            )

        # SSM keys must be absent in the final GPT output.
        assert not any('mixer.' in k for k in recovered_sd), \
            f"SSM keys leaked into final GPT output: " \
            f"{[k for k in recovered_sd if 'mixer.' in k][:5]}"

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
        "fsdp_dtensor prefix", 'fsdp_dtensor', 'fsdp_dtensor',
        source_prefix='model.module.',
    )


def test_torch_dist_alternating_pattern():
    """Pure transformer pattern (no SSM) round-trips."""
    _run_scenario(
        "torch_dist alternating", 'torch_dist', 'torch_dist',
        pattern="*-*-*-*-",
    )


def test_torch_dist_dense_ssm_pattern():
    """Dense SSM pattern still round-trips on the attn/MLP layers."""
    _run_scenario(
        "torch_dist dense SSM", 'torch_dist', 'torch_dist',
        pattern="MM*-MM*-MM*-MM*-",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("GPT <-> Mamba Conversion Parallelism Matrix Tests")
    print("=" * 60)

    test_torch_dist_roundtrip()
    test_fsdp_dtensor_roundtrip()
    test_fsdp_dtensor_prefix()
    test_torch_dist_alternating_pattern()
    test_torch_dist_dense_ssm_pattern()

    print("=" * 60)
    print("ALL PARALLELISM MATRIX TESTS PASSED")
    print("=" * 60)

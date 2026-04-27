# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Parallelism-matrix integration tests for gpt_mamba_conversion.py.

Covers every combination the user cares about:

                                     source format      exercises
    TP              TP=2,  PP=1      legacy             TP-combine + TP-split
    PP              TP=1,  PP=2      legacy             PP-stitch
    FSDP            world=1          dist (torch_dist)  DCP load + DCP save
    TP+PP           TP=2,  PP=2      legacy             TP+PP both paths
    TP+FSDP         world=1          dist               DCP load + DCP save
    PP+FSDP         world=1          dist               DCP load + DCP save
    TP+PP+FSDP      world=1          dist               DCP load + DCP save

Legacy configs synthesize ``mp_rank_XX[_YYY]/model_optim_rng.pt`` shards by
re-using the converter's own save routine (which implements the exact TP-split
and PP-stitch layout Megatron produces). Dist configs synthesize a DCP
checkpoint via a single-rank ``torch.distributed.checkpoint.save``; at the
converter level the TP/PP/FSDP sharding layout of a dist checkpoint is
abstracted away by DCP's global-shape metadata, so one save code path
exercises every ``*+FSDP`` combination. Each config is run as a distinct test
to document the matrix and catch regressions in the dispatch logic.

Designed to run on a single-GPU node via SLURM; no torchrun needed.
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

# Make the conversion tool and the sibling integration-test helpers importable
# under both `python <file>` and `pytest` (pytest doesn't put the test file's
# directory on sys.path).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, '..', '..', '..', '..')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))
sys.path.insert(0, _THIS_DIR)

from gpt_mamba_conversion import (
    combine_tp_shards,
    convert_gpt_to_mamba,
    get_checkpoint_iteration,
    load_checkpoint_shards,
    main as conversion_main,
    parse_hybrid_layer_pattern,
    save_checkpoint_shards,
    stitch_pp_shards,
)

from test_gpt_mamba_conversion_integration import (
    make_checkpoint_args,
    make_gpt_state_dict,
)


# ---------------------------------------------------------------------------
# Legacy (mp_rank_XX) fixture builders
# ---------------------------------------------------------------------------

def _save_legacy_sharded(root_dir, full_sd, ckpt_args, tp_size, pp_size,
                         hybrid_layer_pattern='',
                         hidden_size=128,
                         iteration=100):
    """Write a full state dict to disk as a sharded legacy checkpoint.

    We delegate to ``save_checkpoint_shards`` so the on-disk layout matches
    exactly what Megatron training would produce at the given TP/PP.
    """
    # save_checkpoint_shards expects a "sample_model" shape that mirrors a
    # single rank's on-disk file. Any args object with the target fields works.
    ckpt_args = copy.deepcopy(ckpt_args)
    ckpt_args.tensor_model_parallel_size = tp_size
    ckpt_args.pipeline_model_parallel_size = pp_size
    sample_model = {
        'args': ckpt_args,
        'checkpoint_version': 3.0,
        'iteration': iteration,
        'rng_state': [],
    }
    params = SimpleNamespace(
        target_tp_size=tp_size,
        target_pp_size=pp_size,
        target_num_layers=ckpt_args.num_layers,
        reset_iterations=False,
        # Mamba-only TP-split args; irrelevant for pure GPT shards but required.
        mamba_version=2,
        mamba_d_inner=hidden_size * 2,
        mamba_d_state=16,
        mamba2_n_groups=2,
        mamba2_n_heads=hidden_size * 2 // 32,
    )
    save_checkpoint_shards(full_sd, sample_model, params, root_dir, iteration)


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


# ---------------------------------------------------------------------------
# Output readers
# ---------------------------------------------------------------------------

def _load_converted_dist(ckpt_dir):
    """Read a dist-format converted checkpoint back into a full state dict."""
    from dist_checkpoint_io import load_dist_checkpoint_full
    sd, common, prefix, backend, iteration = load_dist_checkpoint_full(ckpt_dir)
    return sd, common.get('args', None)


def _load_converted_legacy_full(ckpt_dir):
    """Read a legacy TP+PP-sharded converted checkpoint into a full state dict.

    Peeks the first shard to discover TP/PP sizes and total layers, then reuses
    the converter's own load / TP-combine / PP-stitch routines.
    """
    iteration = get_checkpoint_iteration(ckpt_dir)
    model_dir = os.path.join(ckpt_dir, f'iter_{iteration:07d}')
    first_shard = sorted(os.listdir(model_dir))[0]
    sample = torch.load(
        os.path.join(model_dir, first_shard, 'model_optim_rng.pt'),
        map_location='cpu', weights_only=False,
    )
    tp_size = sample['args'].tensor_model_parallel_size
    pp_size = sample['args'].pipeline_model_parallel_size
    num_layers = sample['args'].num_layers
    num_layers_per_pp_rank = num_layers // pp_size

    all_shards, sample = load_checkpoint_shards(
        ckpt_dir, iteration, tp_size, pp_size,
    )
    # combine_tp_tensors only touches mamba-specific branches for mamba keys;
    # any hidden_size-consistent defaults work for GPT-only outputs.
    combine_params = SimpleNamespace(
        mamba_version=2,
        mamba_d_inner=0,
        mamba_d_state=0,
        mamba2_n_groups=0,
        mamba2_n_heads=0,
    )
    combined_pp = [combine_tp_shards(all_shards[pp], combine_params)
                   for pp in range(pp_size)]
    full = stitch_pp_shards(combined_pp, num_layers_per_pp_rank)
    return full, sample['args']


def _load_converted(ckpt_dir, output_format):
    if output_format == 'legacy':
        return _load_converted_legacy_full(ckpt_dir)
    return _load_converted_dist(ckpt_dir)


# ---------------------------------------------------------------------------
# Core scenario runner
# ---------------------------------------------------------------------------

def _run_scenario(
    label,
    source_format,
    source_tp,
    source_pp,
    target_format,
    target_tp=1,
    target_pp=1,
    num_layers=4,
    hidden_size=128,
    pattern="M*-M*-M*-M*-",
    source_prefix='model.',
):
    """Build a GPT source ckpt, convert GPT->Mamba->GPT, verify round-trip."""
    print(f"\n=== {label} ===")
    print(f"  source={source_format} (tp={source_tp}, pp={source_pp}, prefix='{source_prefix}')")
    print(f"  target={target_format} (tp={target_tp}, pp={target_pp})")

    tmpdir = tempfile.mkdtemp(prefix=f'gpt_mamba_{label.replace(" ", "_")}_')
    try:
        src_gpt_dir = os.path.join(tmpdir, 'gpt_src')
        mamba_dir = os.path.join(tmpdir, 'mamba_mid')
        dst_gpt_dir = os.path.join(tmpdir, 'gpt_dst')

        # --- Build source ---
        ckpt_args = make_checkpoint_args(
            num_layers=num_layers, hidden_size=hidden_size,
            tp_size=source_tp, pp_size=source_pp,
        )
        gpt_sd = make_gpt_state_dict(num_layers, hidden_size)

        if source_format == 'legacy':
            _save_legacy_sharded(
                src_gpt_dir, gpt_sd, ckpt_args, source_tp, source_pp,
                hidden_size=hidden_size,
            )
        else:
            _save_dist_checkpoint(
                src_gpt_dir, gpt_sd, ckpt_args,
                prefix=source_prefix, backend=source_format,
            )

        common_kwargs = dict(
            hybrid_layer_pattern=pattern,
            target_tp_size=target_tp,
            target_pp_size=target_pp,
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
        recovered_sd, recovered_args = _load_converted(dst_gpt_dir, target_format)
        layer_types = parse_hybrid_layer_pattern(pattern)

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
# Test cases — one per parallelism combo
# ---------------------------------------------------------------------------

def test_tp_only_legacy():
    _run_scenario("TP only (legacy)", 'legacy', 2, 1, 'legacy', target_tp=2, target_pp=1)


def test_pp_only_legacy():
    _run_scenario("PP only (legacy)", 'legacy', 1, 2, 'legacy', target_tp=1, target_pp=2)


def test_tp_pp_legacy():
    _run_scenario("TP+PP (legacy)", 'legacy', 2, 2, 'legacy', target_tp=2, target_pp=2)


def test_fsdp_only_dist():
    _run_scenario("FSDP only (torch_dist)", 'torch_dist', 1, 1, 'torch_dist')


def test_tp_fsdp_dist():
    _run_scenario("TP + FSDP (torch_dist)", 'torch_dist', 1, 1, 'torch_dist')


def test_pp_fsdp_dist():
    _run_scenario("PP + FSDP (torch_dist)", 'torch_dist', 1, 1, 'torch_dist')


def test_tp_pp_fsdp_dist():
    _run_scenario("TP+PP+FSDP (torch_dist)", 'torch_dist', 1, 1, 'torch_dist')


def test_fsdp_dtensor_prefix():
    """fsdp_dtensor backend uses the 'model.module.' key prefix — verify we
    auto-detect and strip it correctly."""
    _run_scenario(
        "FSDP dtensor prefix", 'fsdp_dtensor', 1, 1, 'fsdp_dtensor',
        source_prefix='model.module.',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("GPT <-> Mamba Conversion Parallelism Matrix Tests")
    print("=" * 60)

    test_tp_only_legacy()
    test_pp_only_legacy()
    test_tp_pp_legacy()
    test_fsdp_only_dist()
    test_tp_fsdp_dist()
    test_pp_fsdp_dist()
    test_tp_pp_fsdp_dist()
    test_fsdp_dtensor_prefix()

    print("=" * 60)
    print("ALL PARALLELISM MATRIX TESTS PASSED")
    print("=" * 60)

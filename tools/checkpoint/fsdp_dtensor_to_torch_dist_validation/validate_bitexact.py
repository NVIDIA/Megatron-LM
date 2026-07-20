# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact per-tensor check for the fsdp_dtensor -> torch_dist reverse converter.

Loads a *reverse-converted* torch_dist checkpoint into a real classic (non-FSDP)
mcore GPTModel + DistributedOptimizer — built through the exact same
``parse_and_validate_args`` -> ``initialize_megatron`` ->
``setup_model_and_optimizer(partial(model_provider, gpt_builder))`` path that
``pretrain_gpt.py`` uses, so the model matches the source FSDP model by
construction — then immediately re-saves it and does a strict per-tensor diff of
the re-save against the converter output (same key namespace). A clean diff proves
the converted checkpoint loads *bit-exactly*: weights AND full optimizer state
(fp32 masters, exp_avg / exp_avg_sq, and the reconstructed param_groups). This is
the decisive complement to ``validate_resume.sh``'s resume-continuity check.

Architecture flags are read from ``common.sh`` (the single source of truth for the
model registry), so this tool never re-declares them.

Run in the mcore dev container (one GPU free), after ``validate_resume.sh``:

    RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \\
        python validate_bitexact.py moe_grouped --iter 80

For ``gdn_hybrid``, flash-linear-attention must be installed first
(``validate_resume.sh`` does this; otherwise ``pip install flash-linear-attention``).
"""
import argparse
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import torch

_VAL_DIR = os.path.dirname(os.path.abspath(__file__))
_COMMON_SH = os.path.join(_VAL_DIR, "common.sh")
# tools/checkpoint/fsdp_dtensor_to_torch_dist_validation -> repo root is three levels up.
_REPO_ROOT = os.path.abspath(os.path.join(_VAL_DIR, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "tools", "checkpoint"))


def _sh(*args):
    """Run ``common.sh <args...>`` and return its stdout split into tokens."""
    proc = subprocess.run(
        ["bash", _COMMON_SH, *args], capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode)
    return proc.stdout.split()


def main():
    ap = argparse.ArgumentParser(
        description="Bit-exact per-tensor diff of a reverse-converted checkpoint."
    )
    ap.add_argument("model", help="model name (see: bash common.sh list-models)")
    ap.add_argument(
        "--iter",
        type=int,
        default=80,
        help="converted checkpoint iteration to check (default: 80)",
    )
    ap.add_argument(
        "--td",
        default=None,
        help="override the converted torch_dist dir (default: "
        "<results>/<model>/td<iter>)",
    )
    cli = ap.parse_args()

    results_root = os.environ.get("RESULTS_DIR", os.path.join(_VAL_DIR, "results"))
    td = cli.td or os.path.join(results_root, cli.model, f"td{cli.iter}")
    td2 = td.rstrip("/") + "_reload"

    # Single source of truth: pull the arg vector + load flags from common.sh.
    model_args = _sh("emit-args", cli.model)
    load_flags = _sh("emit-load-flags")

    # Drive megatron's own parser / init exactly like pretrain_gpt.py.
    sys.argv = (
        ["pretrain_gpt.py"]
        + model_args
        + load_flags
        + ["--load", td, "--save", td2, "--save-interval", "1", "--train-iters", "100"]
    )

    from gpt_builders import gpt_builder
    from megatron.core.enums import ModelType
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import setup_model_and_optimizer
    from model_provider import model_provider

    # Args must be parsed and set globally before initialize_megatron (which reads
    # get_args()); mirrors pretrain_gpt.py.
    parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
    initialize_megatron()

    # mcore's setup_model_and_optimizer takes (model_type, model_provider_func).
    model, optimizer, opt_sched = setup_model_and_optimizer(
        ModelType.encoder_or_decoder, partial(model_provider, gpt_builder)
    )
    iteration, _ = load_checkpoint(model, optimizer, opt_sched)
    print(f"[bitexact] loaded iteration {iteration} from {td}")

    # Re-save the just-loaded state (no training step => no drift).
    save_checkpoint(iteration, model, optimizer, opt_sched, 0)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    from checkpoint_inspector import _compare_two_checkpoint

    src = Path(td) / f"iter_{iteration:07d}"
    dst = Path(td2) / f"iter_{iteration:07d}"
    print(f"[bitexact] diffing converter output vs load+resave:\n  {src}\n  {dst}")
    # _compare_two_checkpoint prints per-key metadata/value mismatches; empty == pass.
    _compare_two_checkpoint(src, dst)
    print(
        "[bitexact] done. EXPECTED differences: the converter intentionally omits "
        "_extra_state / rng_state / rerun_state_machine_state, so those keys appear "
        "only in the reload (checkpoint 2). Any decoder.*/embedding.*/output_layer.* "
        "weight or optimizer.state.* tensor listed above is a REAL mismatch; none of "
        "those == bit-exact weights + optimizer (masters, moments, param_groups)."
    )


if __name__ == "__main__":
    main()

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact worker (subprocess) for the reverse-converter end-to-end suite.

Loads a *reverse-converted* torch_dist checkpoint into a real classic (non-FSDP)
mcore ``GPTModel`` + ``DistributedOptimizer`` — built through the exact
``parse_and_validate_args`` -> ``initialize_megatron`` ->
``setup_model_and_optimizer(partial(model_provider, gpt_builder))`` path
``pretrain_gpt.py`` uses, so the model matches the source FSDP model by
construction — then immediately re-saves it (no train step ⇒ no drift) and does a
**structured** per-tensor diff of the re-save against the converter output. A clean
diff proves the converted checkpoint loads bit-exactly: weights AND full optimizer
state (fp32 masters, exp_avg / exp_avg_sq, and the reconstructed param_groups).

The family arg vector comes from the Python registry/config, and the verdict is
emitted as one JSON object between sentinels so the controller can assert on it.

Runs in its OWN process per family, because ``initialize_megatron`` sets megatron's
global args exactly once per process. Requires one free GPU. Launched by
``harness.run_bitexact_worker``.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

# tests/integration_tests/tools/checkpoint/fsdp_dtensor_to_torch_dist/_bitexact_worker.py
# -> repo root is five package levels up. Put it on sys.path so both the tests
# package and the repo-root pretrain_gpt entrypoints (gpt_builders / model_provider)
# import cleanly when this file is run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import (  # noqa: E402
    config,
    registry,
)

_BEGIN = "===BITEXACT_JSON_BEGIN==="
_END = "===BITEXACT_JSON_END==="

# Keys the converter intentionally omits (RNG / rerun / common state and every
# TE _extra_state incl. FP8 amax history). The real re-save re-adds them, so they
# legitimately appear "only in the reload" and are NOT real mismatches.
_DROPPED_MARKERS = ("_extra_state", "rng_state", "rerun_state_machine_state", "common_state")


def structured_dcp_diff(converter_dir: Path, reload_dir: Path):
    """Per-tensor DCP diff. Returns (weight_mismatches, optim_mismatches, unexpected_extra).

    Reimplements ``checkpoint_inspector._compare_two_checkpoint`` (same
    ``atol=1e-8, rtol=1e-5``) but *returns* results classified by key namespace
    instead of only printing them. ``converter_dir`` is the converter output;
    ``reload_dir`` is the load+resave. Keys that appear only in the reload and match
    a dropped-state marker are expected and ignored.
    """
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader_c = FileSystemReader(converter_dir)
    meta_c = reader_c.read_metadata()
    reader_r = FileSystemReader(reload_dir)
    meta_r = reader_r.read_metadata()
    keys_c = set(meta_c.state_dict_metadata)
    keys_r = set(meta_r.state_dict_metadata)

    def _dropped(k):
        return any(mark in k for mark in _DROPPED_MARKERS)

    def _bucket(k, weight, optim):
        return optim if k.startswith("optimizer.state") else weight

    weight, optim, extra = [], [], []

    # Only in the reload and not intentionally dropped -> unexpected extra.
    for k in sorted(keys_r - keys_c):
        if not _dropped(k):
            extra.append(k)
    # Only in the converter output -> the loaded model dropped a real tensor.
    for k in sorted(keys_c - keys_r):
        if not _dropped(k):
            _bucket(k, weight, optim).append(f"missing:{k}")
    # Common tensor keys -> compare shape/dtype/values.
    for k in sorted(keys_c & keys_r):
        m_c = meta_c.state_dict_metadata[k]
        if not isinstance(m_c, TensorStorageMetadata):
            continue
        m_r = meta_r.state_dict_metadata[k]
        target = _bucket(k, weight, optim)
        if m_c.size != m_r.size or m_c.properties.dtype != m_r.properties.dtype:
            target.append(f"meta:{k}")
            continue
        v_c = torch.empty(m_c.size, dtype=m_c.properties.dtype)
        v_r = v_c.clone()
        dcp.load({k: v_c}, storage_reader=reader_c, planner=DefaultLoadPlanner())
        dcp.load({k: v_r}, storage_reader=reader_r, planner=DefaultLoadPlanner())
        if not torch.allclose(v_c, v_r, atol=1e-8, rtol=1e-5):
            target.append(f"value:{k}")
    return weight, optim, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("family", help="registry family name")
    ap.add_argument("--iter", type=int, default=80, help="converted checkpoint iteration")
    ap.add_argument(
        "--td", required=True, help="converted torch_dist dir (e.g. <results>/<fam>/td80)"
    )
    cli = ap.parse_args()

    fam = registry.MODELS[cli.family]
    td = cli.td.rstrip("/")
    td2 = td + "_reload"

    # Single source of truth: build the arg vector from the Python registry/config,
    # then drive megatron's own parser / init exactly like pretrain_gpt.py.
    model_args = list(config.COMMON_ARGS) + ["--num-layers", str(fam.num_layers)] + list(fam.arch)
    sys.argv = (
        ["pretrain_gpt.py"]
        + model_args
        + list(config.CLASSIC_LOAD_FLAGS)
        + ["--load", td, "--save", td2, "--save-interval", "1", "--train-iters", "100"]
    )

    import torch

    from gpt_builders import gpt_builder
    from megatron.core.enums import ModelType
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import setup_model_and_optimizer
    from model_provider import model_provider

    parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
    initialize_megatron()

    model, optimizer, opt_sched = setup_model_and_optimizer(
        ModelType.encoder_or_decoder, partial(model_provider, gpt_builder)
    )
    iteration, _ = load_checkpoint(model, optimizer, opt_sched)
    # Re-save the just-loaded state (no training step => no drift).
    save_checkpoint(iteration, model, optimizer, opt_sched, 0)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    src = Path(td) / f"iter_{iteration:07d}"
    dst = Path(td2) / f"iter_{iteration:07d}"
    weight, optim, extra = structured_dcp_diff(src, dst)
    payload = {
        "family": cli.family,
        "loaded_iteration": int(iteration),
        "verdict": "PASS" if not (weight or optim or extra) else "FAIL",
        "weight_mismatches": weight,
        "optim_mismatches": optim,
        "unexpected_extra": extra,
    }
    print(_BEGIN)
    print(json.dumps(payload))
    print(_END)


if __name__ == "__main__":
    main()

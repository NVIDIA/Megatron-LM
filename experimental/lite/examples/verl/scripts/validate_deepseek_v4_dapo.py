#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate the fixed DeepSeek-V4 DAPO launch contract."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

from packaging.version import Version


EXACT_DEPENDENCIES = {
    "vllm": "0.25.1",
    "flashinfer-python": "0.6.13",
    "nvidia-cutlass-dsl": "4.5.2",
    "tilelang": "0.1.9",
}
MINIMUM_DEPENDENCIES = {
    "transformer-engine": "2.15.0",
    "nvidia-cudnn-frontend": "1.27.0",
}
EXPECTED_VERL_COMMIT = "6a937b63"


def installed(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(f"DS4 dependency is missing: {name}") from exc


def validate_geometry(model_config: Path, rollout_tp: int) -> None:
    config = json.loads(model_config.read_text(encoding="utf-8"))
    o_groups = config.get("o_groups")
    if not isinstance(o_groups, int) or o_groups < 1 or o_groups % rollout_tp != 0:
        raise SystemExit(
            f"DS4 o_groups={o_groups} must be divisible by rollout_tp={rollout_tp}"
        )
    print(
        f"DS4_ROLLOUT_TP_PREFLIGHT_PASSED o_groups={o_groups} rollout_tp={rollout_tp}",
        flush=True,
    )


def verl_commit() -> str:
    declared = os.environ.get("VERL_COMMIT")
    root = os.environ.get("VERL_ROOT")
    if declared is None and root:
        try:
            declared = subprocess.check_output(
                ["git", "-C", root, "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if declared is None:
        raise SystemExit(
            f"VERL provenance is not verifiable; set VERL_COMMIT={EXPECTED_VERL_COMMIT}"
        )
    if not declared.startswith(EXPECTED_VERL_COMMIT):
        raise SystemExit(f"DS4 requires VERL {EXPECTED_VERL_COMMIT}, got {declared}")
    return declared


def validate_environment() -> None:
    actual = {
        name: installed(name) for name in EXACT_DEPENDENCIES | MINIMUM_DEPENDENCIES
    }
    bad_exact = {
        name: (actual[name], wanted)
        for name, wanted in EXACT_DEPENDENCIES.items()
        if Version(actual[name]) != Version(wanted)
    }
    bad_minimum = {
        name: (actual[name], wanted)
        for name, wanted in MINIMUM_DEPENDENCIES.items()
        if Version(actual[name]) < Version(wanted)
    }
    if bad_exact or bad_minimum:
        raise SystemExit(
            f"DS4 dependency mismatch: exact={bad_exact} minimum={bad_minimum}"
        )
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"DS4 requires Python 3.12, got {sys.version}")

    import cudnn
    import torch
    import transformer_engine.pytorch as te
    from cudnn import DSA

    if not torch.__version__.startswith("2.12.0a0") or torch.version.cuda != "13.2":
        raise SystemExit(
            "DS4 requires PyTorch 2.12 nv26.05 / CUDA 13.2, "
            f"got torch={torch.__version__} cuda={torch.version.cuda}"
        )
    if "q_causal_offsets" not in inspect.signature(
        DSA.indexer_forward_wrapper
    ).parameters:
        raise SystemExit(
            "nvidia-cudnn-frontend lacks q_causal_offsets required by fused DSA CP"
        )

    declared_verl = verl_commit()
    print(
        "DS4_DEPENDENCY_CONTRACT_PASSED "
        f"verl={declared_verl} python={sys.version.split()[0]} "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"vllm={actual['vllm']} te={actual['transformer-engine']} "
        f"cudnn_frontend={actual['nvidia-cudnn-frontend']} "
        f"flashinfer={actual['flashinfer-python']} "
        f"cutlass={actual['nvidia-cutlass-dsl']} tilelang={actual['tilelang']} "
        f"te_origin={te.__file__} cudnn_origin={cudnn.__file__}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    geometry = subparsers.add_parser("geometry")
    geometry.add_argument("--model-config", type=Path, required=True)
    geometry.add_argument("--rollout-tp", type=int, required=True)
    subparsers.add_parser("environment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "geometry":
        validate_geometry(args.model_config, args.rollout_tp)
    else:
        validate_environment()


if __name__ == "__main__":
    main()

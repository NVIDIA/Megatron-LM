# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in inventory of explicitly bound kernel entrypoints during a recipe run.

Run this module under torchrun, passing the original training script and args
after --. Wrapping records input metadata only; it neither compares tensors nor
adds CUDA synchronization. Captured calls are not a complete native-kernel trace.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import runpy
import subprocess
import sys
from pathlib import Path

from tools.determinism.recipe_coverage import signature_key

ENVIRONMENT_KEYS = (
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "MAMBA_DETERMINISTIC",
    "CAUSAL_CONV1D_DETERMINISTIC",
)


def runtime_signature(torch) -> dict:
    """Match the replay producer's schema-1 runtime settings."""
    return {
        "autocast": torch.is_autocast_enabled(),
        "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def input_signature(value, tensor_type) -> object:
    """Encode logical tensor metadata without reading device contents."""
    if isinstance(value, tensor_type):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        return {str(key): input_signature(item, tensor_type) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [input_signature(item, tensor_type) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"type": type(value).__qualname__}


def source_context(torch) -> dict:
    """Use the replay producer's versioned provenance fields."""
    versions: dict[str, str | None] = {}
    for name in (
        "torch",
        "triton",
        "transformer-engine",
        "causal-conv1d",
        "mamba-ssm",
        "flash-attn",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    driver: list[str] | None
    try:
        driver = sorted(
            set(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                )
                .strip()
                .splitlines()
            )
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = None
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"])
        ),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "python": platform.python_version(),
        "versions": versions,
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
    }


class Inventory:
    """Collect distinct signatures with bounded memory and visible truncation."""

    def __init__(self, torch, limit: int):
        self.torch = torch
        self.limit = limit
        self.operations: dict[str, dict] = {}
        self.truncated = False

    def wrap(self, function, binding):
        """Wrap a declared callable while preserving its result and exceptions."""

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            inputs = {"args": args, "kwargs": kwargs} if args and kwargs else kwargs or args
            signature = {
                "op_id": binding["op_id"],
                "implementation": binding["implementation"],
                "phase": "forward",
                "inputs": input_signature(inputs, self.torch.Tensor),
                "deterministic_algorithms": self.torch.are_deterministic_algorithms_enabled(),
                "runtime": runtime_signature(self.torch),
            }
            result = function(*args, **kwargs)
            self.record(signature, binding["target"])
            backward_seen = False

            def backward(gradient):
                nonlocal backward_seen
                if not backward_seen:
                    backward_seen = True
                    self.record({**signature, "phase": "forward_backward"}, binding["target"])
                return gradient

            def register(value):
                if isinstance(value, self.torch.Tensor) and value.requires_grad:
                    value.register_hook(backward)
                elif isinstance(value, dict):
                    for item in value.values():
                        register(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        register(item)

            register(result)
            return result

        return wrapped

    def record(self, signature, site):
        """Count a completed forward call or an observed backward traversal."""
        key = signature_key(signature)
        if key in self.operations:
            self.operations[key]["calls"] += 1
        elif len(self.operations) < self.limit:
            self.operations[key] = {"signature": signature, "calls": 1, "site": site}
        else:
            self.truncated = True


@contextlib.contextmanager
def install_bindings(inventory: Inventory, bindings: list[dict]):
    """Patch only explicitly selected module attributes and restore them on exit."""
    originals = []
    try:
        seen = set()
        for binding in bindings:
            if set(binding) != {"target", "op_id", "implementation"}:
                raise ValueError("Each binding requires target, op_id and implementation")
            target = binding["target"]
            if target in seen:
                raise ValueError(f"Duplicate binding: {target}")
            seen.add(target)
            module_name, attribute = target.split(":", 1)
            # Module-level functions only: binding a descriptor or a class
            # method changes call semantics and needs its own explicit adapter.
            if "." in attribute:
                raise ValueError("Bindings must select module-level functions")
            module = importlib.import_module(module_name)
            function = getattr(module, attribute)
            if not (inspect.isfunction(function) or inspect.isbuiltin(function)):
                raise ValueError(f"Binding must be a module-level function: {target}")
            originals.append((module, attribute, function))
            setattr(module, attribute, inventory.wrap(function, binding))
        yield
    finally:
        for module, attribute, function in reversed(originals):
            setattr(module, attribute, function)


def main(argv: list[str] | None = None) -> int:
    """Run a training entrypoint with explicit inventory bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--max-signatures", type=int, default=10000)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_signatures < 1:
        parser.error("A training script and positive signature limit are required")
    deterministic = "--deterministic-mode" in command
    if deterministic:
        # Seed defaults before importing Torch or Megatron: importing the
        # policy module can itself import libraries that cache these flags.
        for name, value in {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MAMBA_DETERMINISTIC": "1",
            "CAUSAL_CONV1D_DETERMINISTIC": "1",
        }.items():
            os.environ.setdefault(name, value)
    import torch

    if deterministic:
        torch.use_deterministic_algorithms(True)
        from megatron.training.determinism import apply_determinism_env

        apply_determinism_env(os.environ)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rank-{os.environ.get('RANK', '0')}.json"
    if path.exists():
        parser.error("Use a fresh output directory; existing rank evidence will not be overwritten")
    inventory = Inventory(torch, args.max_signatures)
    report = {
        "schema_version": 1,
        "kind": "determinism_inventory",
        "recipe_id": args.recipe_id,
        "rank": int(os.environ.get("RANK", "0")),
        "context": source_context(torch),
        "complete": False,
        "truncated": False,
        "operations": [],
    }

    def write():
        report.update(truncated=inventory.truncated, operations=list(inventory.operations.values()))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    write()
    previous_argv = sys.argv
    previous_path = sys.path[:]
    try:
        with install_bindings(inventory, json.loads(args.bindings.read_text())):
            sys.argv = command
            sys.path.insert(0, str(Path(command[0]).resolve().parent))
            try:
                runpy.run_path(command[0], run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
            report["complete"] = True
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
        write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in inventory of explicitly bound kernel entrypoints during a recipe run.

Run this module under torchrun, passing the original training script and args
after --. Default wrapping records metadata without tensor reads or CUDA
synchronization. Explicit collective capture also saves bounded tensor blobs
for later replay. Captured calls are not a complete native-kernel trace.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import inspect
import json
import math
import os
import runpy
import sys
from pathlib import Path

from tools.determinism import pytest_plugin
from tools.determinism.coverage import runtime_signature
from tools.determinism.recipe_coverage import signature_key

ROOT = Path(__file__).resolve().parents[2]


def source_context(torch, *, output_roots=()) -> dict:
    """Return the replay producer's provenance for this checkout.

    Inventories are joined to replay evidence only when both contexts are equal,
    so this delegates to the evidence plugin's definition. Untracked files under
    ``output_roots`` are outputs of the captured run, not source changes.
    """
    return pytest_plugin.source_context(ROOT, torch, output_roots=output_roots)


def _is_default(value, default) -> bool:
    if value is default:
        return True
    scalar = (type(None), bool, int, float, str)
    return type(value) is type(default) and isinstance(value, scalar) and value == default


def call_inputs(function, args, kwargs):
    """Return a call's arguments in the replay harness's input form.

    The harness records ``fn(*inputs)`` as the list of ``inputs``. Bind the call,
    omit arguments that equal their declared defaults (the harness passes only
    the arguments a case needs), and return the remaining positional arguments,
    plus keyword-only ones as ``{"args": ..., "kwargs": ...}``. Equivalent call
    styles therefore share one signature, and a production call that spells out
    default values matches a replay case that omits them.
    """
    signature = inspect.signature(function)
    bound = signature.bind(*args, **kwargs)
    for name, parameter in signature.parameters.items():
        if (
            name in bound.arguments
            and parameter.default is not inspect.Parameter.empty
            and _is_default(bound.arguments[name], parameter.default)
        ):
            del bound.arguments[name]
    return {"args": bound.args, "kwargs": bound.kwargs} if bound.kwargs else bound.args


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
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Signature mappings require string keys")
        return {str(key): input_signature(item, tensor_type) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [input_signature(item, tensor_type) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Opaque argument cannot establish a signature: {type(value).__qualname__}")


class Inventory:
    """Collect distinct signatures with bounded memory and visible truncation."""

    def __init__(self, torch, limit: int, *, collectives=None):
        self.torch = torch
        self.limit = limit
        self.operations: dict[str, dict] = {}
        self.truncated = False
        self.collectives = collectives
        self.issues: set[str] = set()

    def wrap(self, function, binding):
        """Wrap a declared callable while preserving its result and exceptions."""
        if binding.get("adapter") == "tensor_parallel_collective":
            from tools.determinism.collective_capture import wrap_collective

            return wrap_collective(self, function, binding)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            try:
                encoded = input_signature(call_inputs(function, args, kwargs), self.torch.Tensor)
            except (ValueError, TypeError) as error:
                self.issues.add(f"{binding['target']}: {error}")
                return function(*args, **kwargs)
            signature = {
                "op_id": binding["op_id"],
                "implementation": binding["implementation"],
                "phase": "forward",
                "inputs": encoded,
                "deterministic_algorithms": self.torch.are_deterministic_algorithms_enabled(),
                "runtime": runtime_signature(self.torch),
            }
            result = function(*args, **kwargs)
            self.record(signature, binding["target"])
            backward_seen: set[str] = set()

            def backward(gradient):
                backward_signature = {**signature, "phase": "forward_backward"}
                runtime = runtime_signature(self.torch)
                mode = self.torch.are_deterministic_algorithms_enabled()
                if runtime != signature["runtime"] or mode != signature["deterministic_algorithms"]:
                    backward_signature.update(
                        backward_runtime=runtime, backward_deterministic_algorithms=mode
                    )
                key = signature_key(backward_signature)
                if key not in backward_seen:
                    if len(backward_seen) >= self.limit:
                        self.truncated = True
                        return gradient
                    backward_seen.add(key)
                    self.record(backward_signature, binding["target"])
                return gradient

            def register(value):
                if isinstance(value, self.torch.Tensor) and value.requires_grad:
                    if value.is_leaf:
                        self.issues.add(
                            f"{binding['target']}: backward observation of a persistent leaf is unsupported"
                        )
                        return
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
            fields = {"target", "op_id", "implementation"}
            if set(binding) not in (fields, fields | {"adapter"}) or (
                "adapter" in binding and binding["adapter"] != "tensor_parallel_collective"
            ):
                raise ValueError(
                    "Each binding requires target, op_id and implementation, with an optional collective adapter"
                )
            target = binding["target"]
            if target in seen:
                raise ValueError(f"Duplicate binding: {target}")
            seen.add(target)
            module_name, attribute = target.split(":", 1)
            # Patch module attributes, never class descriptors. An exported
            # autograd apply alias keeps its original bound Function class.
            if "." in attribute:
                raise ValueError("Bindings must select module-level functions")
            module = importlib.import_module(module_name)
            function = getattr(module, attribute)
            owner = getattr(function, "__self__", None)
            autograd_base = getattr(getattr(inventory.torch, "autograd", None), "Function", None)
            autograd_alias = (
                (inspect.ismethod(function) or inspect.isbuiltin(function))
                and inspect.isclass(owner)
                and getattr(function, "__name__", None) == "apply"
                and autograd_base is not None
                and issubclass(owner, autograd_base)
            )
            native_function = inspect.isbuiltin(function) and (
                owner is None or inspect.ismodule(owner)
            )
            if not (inspect.isfunction(function) or native_function or autograd_alias):
                raise ValueError(
                    f"Binding must be a module-level function or autograd apply alias: {target}"
                )
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
    parser.add_argument(
        "--collective-capture",
        type=Path,
        help="Opt in to synchronized collective tensor snapshots in a fresh directory",
    )
    parser.add_argument("--max-collective-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_signatures < 1 or args.max_collective_bytes < 1:
        parser.error("A training script and positive signature limit are required")
    # Honor the effective CLI/YAML policy before bound modules can import Core,
    # initialize CUDA or cache backend settings. Training still validates all
    # model options through its normal parser.
    try:
        from megatron.determinism import bootstrap_training_determinism
    except ModuleNotFoundError as error:
        if error.name != "megatron.determinism":
            raise
        parser.exit(2, "Recipe capture requires the megatron.determinism startup API.\n")

    bootstrap_training_determinism(command[1:])
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rank-{os.environ.get('RANK', '0')}.json"
    if path.exists():
        parser.error("Use a fresh output directory; existing rank evidence will not be overwritten")
    collectives = None
    if args.collective_capture is not None:
        from tools.determinism.collective_capture import CollectiveCapture

        collectives = CollectiveCapture(
            args.collective_capture / f"rank-{os.environ.get('RANK', '0')}",
            max_bytes=args.max_collective_bytes,
            max_events=args.max_signatures,
        )
    inventory = Inventory(torch, args.max_signatures, collectives=collectives)
    output_roots = [args.output, args.collective_capture]
    # Megatron's checkpoint output is generated during the instrumented recipe.
    for index, argument in enumerate(command):
        if argument == "--save" and index + 1 < len(command):
            output_roots.append(Path(command[index + 1]))
        elif argument.startswith("--save="):
            output_roots.append(Path(argument.split("=", 1)[1]))
    report = {
        "schema_version": 1,
        "kind": "determinism_inventory",
        "recipe_id": args.recipe_id,
        "rank": int(os.environ.get("RANK", "0")),
        "context": source_context(torch, output_roots=output_roots),
        "complete": False,
        "truncated": False,
        "operations": [],
    }

    def write():
        report.update(truncated=inventory.truncated, operations=list(inventory.operations.values()))
        report["capture_issues"] = sorted(
            inventory.issues | (collectives.issues if collectives is not None else set())
        )
        if collectives is not None:
            collectives.write(report)
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
            report["context_after"] = source_context(torch, output_roots=output_roots)
            if report["context_after"] != report["context"]:
                raise RuntimeError("Source or environment context changed during recipe capture")
            report["complete"] = True
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
        write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

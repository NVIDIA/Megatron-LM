# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Every DSA ``autograd.Function`` must return one gradient per forward input.

Adding an argument to ``forward`` without extending the ``backward`` return tuple
raises only when a backward pass actually runs, so it survives any test that stops at
the forward. This check reads the source instead, and therefore needs no GPU.
"""

import ast
import inspect
from pathlib import Path

import pytest

MODULE_PATHS = [
    Path(inspect.getfile(__import__("megatron"))).parent
    / "core"
    / "transformer"
    / "experimental_attention_variant"
    / name
    for name in ("dsa_cudnn_kernels.py", "dsa_kernels.py", "dsa.py")
]


def _autograd_functions(path: Path):
    """Yield (class name, forward input count, backward gradient count) triples."""
    if not path.exists():
        return
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        forward = backward = None
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                forward = item
            if isinstance(item, ast.FunctionDef) and item.name == "backward":
                backward = item
        if forward is None or backward is None:
            continue
        # A tuple return is the only shape that pins the count; a bare ``return x``
        # single-gradient backward is checked by autograd itself.
        returns = [
            r
            for r in ast.walk(backward)
            if isinstance(r, ast.Return) and isinstance(r.value, ast.Tuple)
        ]
        if not returns:
            continue
        n_inputs = len(forward.args.args) - 1  # drop ctx
        for ret in returns:
            yield node.name, n_inputs, len(ret.value.elts), ret.lineno


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda p: p.name)
def test_backward_returns_one_gradient_per_forward_input(path):
    """A mismatch here is a runtime error that only appears during a backward pass."""
    mismatches = [
        f"{name}: forward takes {n_in} inputs but backward returns {n_out} "
        f"gradients (line {lineno})"
        for name, n_in, n_out, lineno in _autograd_functions(path)
        if n_in != n_out
    ]
    assert not mismatches, "\n".join(mismatches)

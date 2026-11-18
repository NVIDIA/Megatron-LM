# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay harness for operator-level bit-exact determinism tests.

A kernel is deterministic when re-running it on identical inputs yields byte-identical
outputs (and input gradients, for differentiable kernels). ``assert_replays_bit_exact``
does exactly that: it materialises one set of inputs, runs the kernel ``replays`` times on
fresh clones of them, and compares every output tensor and every input gradient against
the first replay byte for byte (``bytes_equal``: same shape, same dtype, identical bit
patterns -- signed zeros and NaN payloads must match too). Optional side-stream contention
perturbs kernel scheduling between replays so ordering-dependent reductions surface as
mismatches.

Sizing matters more than replay count: a reduction with two contending blocks can be
"deterministic" by accident. Pick shapes that put many blocks on the same output (see
``CONTENTION_TOKENS``) -- ``test_ssm_conv1d.py`` documents the measurement behind this.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from tests.unit_tests.determinism.utils import (
    RacingStreams,
    capture_rng_state,
    collect_grads,
    restore_rng_state,
    zero_grads,
)

# Shapes that make many CTAs contend on shared outputs. A token count of a few thousand
# with a hidden size of ~1-4k puts dozens of blocks on every reduction.
CONTENTION_TOKENS = 4096


def _clone_preserving_layout(t: torch.Tensor) -> torch.Tensor:
    """Copy ``t`` keeping its exact strides, including expanded (stride-0) views.

    ``Tensor.clone`` only preserves the strides of dense tensors; an ``expand``ed operand
    would come back as a dense copy and steer a kernel into a different specialisation
    than production uses (e.g. ``selective_state_update``'s ``TIE_HDIM`` path).
    """
    src = t.detach()
    if src.is_contiguous():
        return src.clone()
    extent = (
        src.storage_offset()
        + 1
        + sum((n - 1) * s for n, s in zip(src.shape, src.stride()) if n > 1)
    )
    flat = torch.empty(0, dtype=src.dtype, device=src.device).set_(src.untyped_storage())
    return flat[:extent].clone().as_strided(src.shape, src.stride(), src.storage_offset())


def clone_inputs(inputs: Any) -> Any:
    """Deep-copy tensors (preserving layout and ``requires_grad``); pass everything else through."""
    if isinstance(inputs, torch.Tensor):
        clone = _clone_preserving_layout(inputs)
        if inputs.requires_grad:
            clone.requires_grad_(True)
        return clone
    if isinstance(inputs, dict):
        return {k: clone_inputs(v) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        cloned = [clone_inputs(v) for v in inputs]
        return type(inputs)(cloned) if isinstance(inputs, tuple) else cloned
    return inputs


def flatten_tensors(obj: Any, prefix: str = "out") -> Dict[str, torch.Tensor]:
    """Flatten nested outputs into ``{name: tensor}``; non-tensors are ignored."""
    found: Dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        found[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.update(flatten_tensors(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.update(flatten_tensors(v, f"{prefix}[{i}]"))
    return found


def _leaf_inputs(inputs: Any, prefix: str = "in") -> Dict[str, torch.Tensor]:
    return {name: t for name, t in flatten_tensors(inputs, prefix).items() if t.requires_grad}


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """The logical contents of ``t`` as a flat ``uint8`` tensor (layout-independent)."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """True iff ``a`` and ``b`` have the same shape and dtype and identical bit patterns.

    Stricter than ``torch.equal``: ``+0.0`` and ``-0.0`` differ, and NaNs only match when
    their payloads match. Strides are not compared -- both operands are read in logical
    order -- so a kernel may return a differently laid out tensor with the same contents.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    return bool(torch.equal(_as_bytes(a), _as_bytes(b)))


def _describe_mismatch(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape or a.dtype != b.dtype:
        return (
            f"{name}: shape/dtype differ ({tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype})"
        )
    bits_differ = (_as_bytes(a) != _as_bytes(b)).view(a.numel(), a.element_size()).any(dim=1)
    count = int(bits_differ.sum().item())
    msg = f"{name}: {count}/{a.numel()} elements differ"
    if not a.is_complex():
        af, bf = a.detach().reshape(-1).float(), b.detach().reshape(-1).float()
        value_differs = (af != bf) & ~(af.isnan() & bf.isnan())
        diff = torch.where(value_differs, (af - bf).abs(), torch.zeros_like(af))
        max_diff = float(diff.max().item()) if count else 0.0
        msg += f" (max |diff| {max_diff:.3e}"
        bit_only = count - int(value_differs.sum().item())
        if bit_only:
            msg += f"; {bit_only} differ only in bit pattern, e.g. signed zero or NaN payload"
        msg += ")"
    return msg


def run_once(
    fn: Callable[..., Any],
    inputs: Any,
    grad_outputs: Optional[Dict[str, torch.Tensor]] = None,
    backward: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Run ``fn(**inputs)`` (or ``fn(*inputs)``) once and return outputs and input grads.

    ``grad_outputs`` maps flattened output names to the upstream gradient to use; outputs
    without an entry get ``ones_like``. Returns detached clones so later replays cannot
    alias them.
    """
    local = clone_inputs(inputs)
    result = fn(**local) if isinstance(local, dict) else fn(*local)
    outputs = {k: v for k, v in flatten_tensors(result).items()}
    grads: Dict[str, torch.Tensor] = {}
    if backward:
        leaves = _leaf_inputs(local)
        diff_outputs = [
            (name, t) for name, t in outputs.items() if t.requires_grad and t.is_floating_point()
        ]
        if leaves and diff_outputs:
            gos = [(grad_outputs or {}).get(name, torch.ones_like(t)) for name, t in diff_outputs]
            computed = torch.autograd.grad(
                [t for _, t in diff_outputs],
                list(leaves.values()),
                grad_outputs=gos,
                allow_unused=True,
            )
            for name, g in zip(leaves, computed):
                if g is not None:
                    grads[name] = g.detach().clone()
    torch.cuda.synchronize()
    return {k: v.detach().clone() for k, v in outputs.items()}, grads


def assert_replays_bit_exact(
    fn: Callable[..., Any],
    inputs: Any,
    *,
    replays: int = 3,
    backward: bool = True,
    grad_outputs: Optional[Dict[str, torch.Tensor]] = None,
    contention: bool = False,
    restore_rng: bool = False,
    what: str = "kernel",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Assert that ``replays`` runs of ``fn`` on identical inputs are byte-identical.

    Every output tensor and input gradient is compared with ``bytes_equal``.

    Args:
        fn: kernel entry point; called with ``**inputs`` if ``inputs`` is a dict, else
            ``*inputs``.
        inputs: materialised inputs. Tensors with ``requires_grad`` get their gradient
            compared too (when ``backward``).
        replays: number of runs; the first is the reference.
        backward: also run autograd and compare input gradients.
        grad_outputs: fixed upstream gradients keyed by flattened output name (``out``,
            ``out[0]``, ``out.name``). Defaults to ones, so the same value feeds every replay.
        contention: run replays 2.. under ``RacingStreams`` side-stream GEMM pressure.
        restore_rng: snapshot every RNG before the reference run and restore it before
            each replay -- for kernels that consume random numbers (dropout).
        what: label for error messages.

    Returns:
        The reference outputs and gradients (for follow-up assertions).
    """
    if replays < 2:
        raise ValueError("replays must be >= 2")
    rng = capture_rng_state() if restore_rng else None
    ref_out, ref_grad = run_once(fn, inputs, grad_outputs, backward)
    if not ref_out:
        raise AssertionError(f"{what} produced no tensor outputs; nothing to compare")
    for i in range(1, replays):
        if rng is not None:
            torch.cuda.synchronize()
            restore_rng_state(rng)
        if contention:
            with RacingStreams():
                out, grad = run_once(fn, inputs, grad_outputs, backward)
        else:
            out, grad = run_once(fn, inputs, grad_outputs, backward)
        _assert_replay_matches(i, ref_out, ref_grad, out, grad, what)
    return ref_out, ref_grad


def _assert_replay_matches(i, ref_out, ref_grad, out, grad, what) -> None:
    """Raise if replay ``i`` differs from the reference in any output or gradient tensor."""
    if out.keys() != ref_out.keys() or grad.keys() != ref_grad.keys():
        raise AssertionError(
            f"{what}: replay {i} returned different tensors "
            f"({sorted(out)}/{sorted(grad)} vs {sorted(ref_out)}/{sorted(ref_grad)})"
        )
    mismatches = [
        _describe_mismatch(name, ref_out[name], out[name])
        for name in ref_out
        if not bytes_equal(ref_out[name], out[name])
    ] + [
        _describe_mismatch(f"grad({name})", ref_grad[name], grad[name])
        for name in ref_grad
        if not bytes_equal(ref_grad[name], grad[name])
    ]
    if mismatches:
        raise AssertionError(
            f"{what} is not bit-exact across replays (replay {i} vs 1):\n  "
            + "\n  ".join(mismatches)
        )


def count_differing_replays(
    fn: Callable[..., Any], inputs: Any, *, replays: int = 8, backward: bool = True
) -> int:
    """Return how many of ``replays - 1`` re-runs differ from the first in any tensor.

    For *negative controls*: a test that documents a known non-deterministic default path
    asserts this is ``> 0`` so the paired deterministic assertion is known to be sensitive.
    Size such controls generously -- hardware can serialise small problems.
    """
    ref_out, ref_grad = run_once(fn, inputs, None, backward)
    differing = 0
    for _ in range(replays - 1):
        out, grad = run_once(fn, inputs, None, backward)
        same = all(bytes_equal(ref_out[k], out[k]) for k in ref_out) and all(
            bytes_equal(ref_grad[k], grad[k]) for k in ref_grad
        )
        differing += not same
    return differing


def seeded(seed: int = 1234) -> None:
    """Seed every RNG the kernels may consume so input construction is repeatable."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def deterministic_algorithms(enabled: bool):
    """Temporarily set ``torch.use_deterministic_algorithms`` (warn-only), then restore.

    Kernel wrappers select branches on ``torch.are_deterministic_algorithms_enabled()``;
    tests use this to exercise each branch explicitly instead of inheriting whatever an
    earlier test left behind.
    """
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def _zero_module_grads(module: torch.nn.Module) -> None:
    zero_grads(module)
    for p in module.parameters():
        main_grad = getattr(p, "main_grad", None)
        if main_grad is not None:
            main_grad.zero_()


def _module_fwd_bwd(module, inputs, grad_output, backward):
    local = clone_inputs(inputs)
    result = module(**local) if isinstance(local, dict) else module(*local)
    outputs = flatten_tensors(result)
    grads: Dict[str, torch.Tensor] = {}
    if backward:
        diff = [(n, t) for n, t in outputs.items() if t.requires_grad and t.is_floating_point()]
        if not diff:
            raise AssertionError("module produced no differentiable output")
        name, tensor = diff[0]
        g = grad_output if grad_output is not None else torch.ones_like(tensor)
        tensor.backward(g)
        grads = collect_grads([module])
        for n, t in _leaf_inputs(local).items():
            if t.grad is not None:
                grads[n] = t.grad.detach().clone()
    torch.cuda.synchronize()
    return {k: v.detach().clone() for k, v in outputs.items()}, grads


def assert_module_replays_bit_exact(
    module: torch.nn.Module,
    inputs: Any,
    *,
    replays: int = 3,
    backward: bool = True,
    grad_output: Optional[torch.Tensor] = None,
    contention: bool = False,
    restore_rng: bool = True,
    what: str = "module",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Module-level twin of ``assert_replays_bit_exact``.

    Runs ``module(**inputs)`` (or ``module(*inputs)``) ``replays`` times from identical
    parameters, RNG state and inputs; backward propagates ``grad_output`` (default: ones)
    through the first differentiable output. Compares every output tensor, every parameter
    gradient (``p.grad`` or ``p.main_grad``) and every input gradient bit-for-bit. Grads
    (including ``main_grad`` buffers) are zeroed between replays; parameters are not
    updated, so no optimizer state needs resetting.
    """
    if replays < 2:
        raise ValueError("replays must be >= 2")
    rng = capture_rng_state() if restore_rng else None
    _zero_module_grads(module)
    ref_out, ref_grad = _module_fwd_bwd(module, inputs, grad_output, backward)
    for i in range(1, replays):
        _zero_module_grads(module)
        if rng is not None:
            torch.cuda.synchronize()
            restore_rng_state(rng)
        if contention:
            with RacingStreams():
                out, grad = _module_fwd_bwd(module, inputs, grad_output, backward)
        else:
            out, grad = _module_fwd_bwd(module, inputs, grad_output, backward)
        _assert_replay_matches(i, ref_out, ref_grad, out, grad, what)
    return ref_out, ref_grad

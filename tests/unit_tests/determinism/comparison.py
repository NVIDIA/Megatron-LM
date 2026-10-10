# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Byte comparisons shared by kernel and model replay; also testable on CPU."""

import torch

from tools.determinism.coverage import ReplayMismatch, observe_replay, runtime_signature


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """Read logical contents in order, independent of the storage layout."""
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def bytes_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare shape, dtype and bits, including signed zeros and NaN payloads."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(_as_bytes(a), _as_bytes(b)))


def assert_bit_exact(out_a, grads_a, out_b, grads_b) -> None:
    """Compare two model outputs and parameter-gradient dictionaries.

    The runner supplies independent replay snapshots. This does not compare
    optimizer, RNG, or other training state. Explicit raises remain active under
    ``python -O``. Empty outputs or gradients cannot certify a backward replay.
    """
    signature = {
        "phase": "forward_backward",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "runtime": runtime_signature(torch),
    }
    protocol = {
        "replays": 2,
        "scope": "same_process_model_outputs_and_parameter_gradients",
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }
    with observe_replay(signature, protocol) as observation:
        if not out_a.numel() or not any(grad.numel() for grad in grads_a.values()):
            raise ValueError("Model replay requires nonempty output and parameter gradients")
        if not bytes_equal(out_a, out_b):
            raise ReplayMismatch("Output bytes differ between deterministic runs")
        if grads_a.keys() != grads_b.keys():
            raise ReplayMismatch("Grad keys differ between runs")
        for name, grad in grads_a.items():
            if not bytes_equal(grad, grads_b[name]):
                raise ReplayMismatch(f"Gradient bytes differ for {name}")
        observation.update(
            compared_outputs=1,
            compared_gradients=sum(grad.numel() > 0 for grad in grads_a.values()),
        )

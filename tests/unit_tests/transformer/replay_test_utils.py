# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CUDA memory comparisons for residual replay tests."""

import gc
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def _measure_cuda_peak(device: int) -> Iterator[dict[str, int]]:
    """Record the peak allocated bytes above the resting allocation for this block."""
    resting = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory = {"resting": resting}
    yield memory
    torch.cuda.synchronize(device)
    memory["peak"] = torch.cuda.max_memory_allocated(device) - resting


def _training_step(model: torch.nn.Module, hidden_states: torch.Tensor, device: int) -> int:
    """Run forward and backward, returning only the allocated bytes at the end of forward."""
    output = model(hidden_states=hidden_states, attention_mask=None)
    forward_allocated = torch.cuda.memory_allocated(device)
    output.square().mean().backward()
    assert hidden_states.grad is not None
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient for {name}"
    return forward_allocated


def _clear_gradients(model: torch.nn.Module, hidden_states: torch.Tensor, device: int) -> None:
    model.zero_grad(set_to_none=True)
    hidden_states.grad = None
    gc.collect()
    torch.cuda.synchronize(device)


def _measure_training_loop(
    model: torch.nn.Module, hidden_states: torch.Tensor, *, device: int
) -> dict[str, int]:
    # Warm up kernels and library workspaces before resetting the peak counter.
    for _ in range(2):
        _training_step(model, hidden_states, device)
        _clear_gradients(model, hidden_states, device)

    with _measure_cuda_peak(device) as memory:
        forward_allocated = _training_step(model, hidden_states, device)
        memory["end_forward"] = forward_allocated - memory["resting"]

    _clear_gradients(model, hidden_states, device)
    memory["after_cleanup"] = torch.cuda.memory_allocated(device)
    return memory


def measure_replay_peak_memory(build_model_and_input):
    """Measure eager and replay in both orders, with one fresh model/input per sample."""
    device = torch.cuda.current_device()
    measurements = []
    for recompute in (False, True, True, False):
        # Collect the previous sample before building the next model and input.
        gc.collect()
        torch.cuda.synchronize(device)
        memory = _measure_training_loop(*build_model_and_input(recompute), device=device)
        measurements.append({"recompute": recompute, **memory})
    return measurements


def assert_replay_peak_memory_saving(measurements, *, block_size, min_saving_bytes):
    """Compare the best eager peak with the worst replay peak above equal resting baselines."""
    diagnostics = f"block_size={block_size}, CUDA allocated bytes: {measurements}"
    print(diagnostics)
    for measurement in measurements:
        assert measurement["after_cleanup"] == measurement["resting"], diagnostics
    # Subtracting a larger replay baseline must not conceal retained activation storage.
    assert len({measurement["resting"] for measurement in measurements}) == 1, diagnostics

    eager_peaks = [m["peak"] for m in measurements if not m["recompute"]]
    replay_peaks = [m["peak"] for m in measurements if m["recompute"]]
    assert min(eager_peaks) - max(replay_peaks) >= min_saving_bytes, diagnostics

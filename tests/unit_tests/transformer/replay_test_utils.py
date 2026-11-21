# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CUDA memory comparisons for residual replay tests."""

import gc

import torch


def measure_replay_peak_memory(build_model_and_input):
    """Measure eager and replay in both orders, with one fresh model/input per sample."""
    device = torch.cuda.current_device()

    def measure(recompute):
        # Finish destruction of the previous model before constructing another one.
        gc.collect()
        torch.cuda.synchronize(device)
        model, hidden_states = build_model_and_input(recompute)

        def forward_backward():
            output = model(hidden_states=hidden_states, attention_mask=None)
            forward_allocated = torch.cuda.memory_allocated(device)
            output.square().mean().backward()
            assert hidden_states.grad is not None
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, f"Missing gradient for {name}"
            # Only return a number; no output, loss, or graph survives this call.
            return forward_allocated

        def clear_gradients():
            model.zero_grad(set_to_none=True)
            hidden_states.grad = None
            gc.collect()
            torch.cuda.synchronize(device)

        # Initialize kernels and library workspaces in both modes before resetting peaks.
        for _ in range(2):
            forward_backward()
            clear_gradients()

        resting = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        forward_allocated = forward_backward()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        clear_gradients()
        return {
            "recompute": recompute,
            "resting": resting,
            "end_forward": forward_allocated - resting,
            "peak": peak - resting,
            "after_cleanup": torch.cuda.memory_allocated(device),
        }

    # Run in both orders to expose lazy initialization or allocations retained between models.
    return [measure(recompute) for recompute in (False, True, True, False)]


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

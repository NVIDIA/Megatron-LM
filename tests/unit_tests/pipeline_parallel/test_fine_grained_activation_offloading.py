# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

import pytest
import torch

EPSILON = 0.1

# Skip all tests if CUDA is not available
cuda_available = torch.cuda.is_available()


def _reset_cuda_memory():
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()


class ToyModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 2048, num_layers: int = 4, dtype=torch.bfloat16):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                torch.nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device="cuda")
            )
        self.net = torch.nn.Sequential(*layers).to(device="cuda", dtype=dtype)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dtype = dtype

        # Prevent weights/bias from being considered activation tensors for offload;
        # ensure we only count activation tensors (inputs x) in memory accounting.
        for p in self.parameters():
            try:
                setattr(p, "offloading_activation", False)
            except Exception:
                pass

    def forward(self, x, use_offload: bool = False):
        from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

        if use_offload:
            # Initialize a new chunk (microbatch) and enable offload context.
            with off.get_fine_grained_offloading_context(True):
                off.fine_grained_offloading_init_chunk_handler(
                    vp_size=1, vp_stage=None, min_offloaded_tensor_size=1
                )
                for i, layer in enumerate(self.net):
                    # Group by module; with this linear-only model, each group corresponds to a layer.
                    off.fine_grained_offloading_set_last_layer(i == len(self.net) - 1)
                    x = off.fine_grained_offloading_group_start(x, name=f"layer_{i}")
                    x = layer(x)
                    # Commit the group; returns a tuple of tensors
                    (x,) = off.fine_grained_offloading_group_commit(
                        x, name=f"layer_{i}", forced_released_tensors=[]
                    )
                return x
        # Baseline path (no offload hooks)
        with (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.dtype in (torch.float16, torch.bfloat16)
            else torch.cuda.amp.autocast(enabled=False)
        ):
            for layer in self.net:
                x = layer(x)
            return x


@pytest.fixture(autouse=True)
def _monkeypatch_offload_deps(monkeypatch):
    # Avoid requiring torch.distributed initialization and NVML in tests
    import megatron.core.pipeline_parallel.fine_grained_activation_offload as off

    monkeypatch.setattr(off, "debug_rank", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(off, "set_ideal_affinity_for_current_gpu", lambda: None, raising=False)
    # Ensure a clean state each test
    off.fine_grained_offloading_reset()
    yield
    off.fine_grained_offloading_reset()


def test_fine_grained_activation_offload_memory_reduction():
    torch.manual_seed(1234)
    # Use a linear-only stack so theoretical saved memory equals sum of per-layer input x bytes.
    model = ToyModel(hidden_size=2048, num_layers=8, dtype=torch.bfloat16).eval()

    # Create input
    inp = torch.randn(
        (2048, model.hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    # Warmup to stabilize allocator behavior
    _reset_cuda_memory()
    out = model(inp, use_offload=False)
    (out.sum()).backward()
    torch.cuda.synchronize()
    _reset_cuda_memory()

    # Baseline memory measurement (no offload)
    _reset_cuda_memory()
    inp_baseline = inp.detach().clone().requires_grad_(True)
    baseline_mem_before = torch.cuda.memory_allocated() / (1024**2)
    out_base = model(inp_baseline, use_offload=False)
    baseline_mem_after = (torch.cuda.memory_allocated() - out_base.nbytes) / (1024**2)
    (out_base.sum()).backward()
    torch.cuda.synchronize()
    baseline_delta = baseline_mem_after - baseline_mem_before

    # Offload memory measurement
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    off.fine_grained_offloading_reset()
    _reset_cuda_memory()
    inp_off = inp.detach().clone().requires_grad_(True)
    offload_mem_before = torch.cuda.memory_allocated() / (1024**2)
    out_off = model(inp_off, use_offload=True)
    offload_mem_after = (torch.cuda.memory_allocated() - out_off.nbytes) / (1024**2)
    (out_off.sum()).backward()
    torch.cuda.synchronize()
    offload_delta = offload_mem_after - offload_mem_before

    # Offload should reduce peak cached memory usage after forward
    assert (
        offload_delta < baseline_delta
    ), f"offload did not reduce memory: off={offload_delta:.2f}MiB base={baseline_delta:.2f}MiB"

    # Theoretical savings: storing per-layer input x (same shape each layer).
    bytes_per_elem = inp.element_size()  # 2 for bfloat16
    input_bytes = inp.numel() * bytes_per_elem
    # -2 because the first and last activations are not offloaded
    expected_saved_mib = (model.num_layers - 2) * (input_bytes / (1024**2))

    # Actual savings ≈ baseline_delta - offload_delta (both exclude output tensor memory).
    actual_saved_mib = baseline_delta - offload_delta

    # Allow slack for allocator jitter and extra intermediates; magnitudes should match.
    rel_err = abs(actual_saved_mib - expected_saved_mib) / max(expected_saved_mib, 1e-6)
    assert (
        rel_err <= EPSILON
    ), f"saved mismatch: actual={actual_saved_mib:.2f}MiB expected~={expected_saved_mib:.2f}MiB (rel_err={rel_err:.2f})"


def test_fine_grained_activation_offload_output_and_grad_consistency():
    torch.manual_seed(2025)
    hidden = 1024
    layers = 3

    # Create identical models by resetting seed
    torch.manual_seed(2025)
    model_base = ToyModel(hidden_size=hidden, num_layers=layers, dtype=torch.bfloat16).train()
    torch.manual_seed(2025)
    model_off = ToyModel(hidden_size=hidden, num_layers=layers, dtype=torch.bfloat16).train()

    # Same input and target
    inp = torch.randn((32, hidden), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn_like(inp)

    # Baseline forward/backward
    out_base = model_base(inp, use_offload=False)
    loss_base = torch.nn.functional.mse_loss(out_base, target)
    loss_base.backward()
    grads_base = [
        p.grad.detach().clone() if p.grad is not None else None for p in model_base.parameters()
    ]

    # Offload forward/backward
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    off.fine_grained_offloading_reset()
    out_off = model_off(inp.detach().clone().requires_grad_(True), use_offload=True)
    loss_off = torch.nn.functional.mse_loss(out_off, target)
    loss_off.backward()
    grads_off = [
        p.grad.detach().clone() if p.grad is not None else None for p in model_off.parameters()
    ]

    # Compare outputs
    assert torch.allclose(out_off.float(), out_base.float(), rtol=1e-3, atol=1e-3)

    # Compare gradients parameter-wise
    for gb, go in zip(grads_base, grads_off):
        if gb is None and go is None:
            continue
        assert gb is not None and go is not None
        assert torch.allclose(go.float(), gb.float(), rtol=1e-3, atol=1e-3)

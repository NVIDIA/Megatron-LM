# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical and graph-safety tests for the fused MTP prefix objective."""

import pytest
import torch

from megatron.core.fusions.fused_mtp_prefix import (
    fused_mtp_prefix_unavailable_reason,
    mtp_e2e_prefix_objective,
)


def _native_prefix_objective(acceptances):
    prefix_losses = 1.0 - torch.cumprod(acceptances, dim=0)
    return prefix_losses.mean(dim=0), prefix_losses


def _run_objective_and_gradient(function, acceptance_data, grad_output):
    acceptances = acceptance_data.detach().clone().requires_grad_(True)
    output, prefix_losses = function(acceptances)
    (output * grad_output).sum().backward()
    return output.detach(), prefix_losses.detach(), acceptances.grad.detach()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_depths", [1, 2, 7])
def test_fused_mtp_prefix_matches_native_forward_and_backward(num_depths):
    """Dynamic draft depths match an independent cumprod oracle."""
    torch.manual_seed(51 + num_depths)
    acceptance_data = torch.rand(num_depths, 37, 2, device="cuda")
    grad_output = torch.randn(37, 2, device="cuda")

    assert fused_mtp_prefix_unavailable_reason(acceptance_data) is None
    actual, actual_prefix_losses, actual_grad = _run_objective_and_gradient(
        mtp_e2e_prefix_objective, acceptance_data, grad_output
    )
    reference, reference_prefix_losses, reference_grad = _run_objective_and_gradient(
        _native_prefix_objective, acceptance_data, grad_output
    )

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_prefix_losses, reference_prefix_losses, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("zero_depth", [0, 3, 6])
def test_fused_mtp_prefix_backward_is_zero_safe(zero_depth):
    """Exact zero acceptance never produces division artifacts or nonfinite gradients."""
    torch.manual_seed(71)
    acceptance_data = torch.rand(7, 19, device="cuda")
    acceptance_data[zero_depth] = 0
    grad_output = torch.randn(19, device="cuda")

    actual, actual_prefix_losses, actual_grad = _run_objective_and_gradient(
        mtp_e2e_prefix_objective, acceptance_data, grad_output
    )
    reference, reference_prefix_losses, reference_grad = _run_objective_and_gradient(
        _native_prefix_objective, acceptance_data, grad_output
    )

    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_prefix_losses, reference_prefix_losses, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_prefix_preserves_near_zero_loss():
    """The fused mean must not cancel a one-ULP rejection near convergence."""
    one = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    almost_one = torch.nextafter(one, torch.zeros_like(one))
    acceptance_data = torch.stack((one, almost_one)).reshape(2, 1)
    grad_output = torch.ones(1, device="cuda")

    actual, actual_prefix_losses, actual_grad = _run_objective_and_gradient(
        mtp_e2e_prefix_objective, acceptance_data, grad_output
    )
    reference, reference_prefix_losses, reference_grad = _run_objective_and_gradient(
        _native_prefix_objective, acceptance_data, grad_output
    )

    assert actual.item() > 0.0
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_prefix_losses, reference_prefix_losses, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)


def test_mtp_prefix_unsupported_input_uses_reference_fallback(monkeypatch):
    """CPU and noncontiguous inputs preserve the PyTorch reference path."""
    acceptance_data = torch.rand(2, 5, 3).transpose(1, 2)
    assert not acceptance_data.is_contiguous()
    assert fused_mtp_prefix_unavailable_reason(acceptance_data) is not None

    calls = 0
    original_cumprod = torch.cumprod

    def record_cumprod(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_cumprod(*args, **kwargs)

    monkeypatch.setattr(torch, "cumprod", record_cumprod)
    actual = mtp_e2e_prefix_objective(acceptance_data)
    reference = _native_prefix_objective(acceptance_data)
    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
    assert calls == 2


def test_mtp_prefix_empty_rows_use_reference_fallback():
    """An empty row dimension never dispatches a zero-program Triton grid."""
    acceptance_data = torch.empty(7, 0)
    assert fused_mtp_prefix_unavailable_reason(acceptance_data) is not None

    if torch.cuda.is_available():
        acceptance_data = acceptance_data.cuda()
        assert fused_mtp_prefix_unavailable_reason(acceptance_data) == (
            "acceptances must have at least one row"
        )
        actual, prefix_losses = mtp_e2e_prefix_objective(acceptance_data)
        assert actual.shape == (0,)
        assert prefix_losses.shape == (7, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mtp_prefix_bf16_acceptance_uses_reference_fallback():
    """BF16 model logits still reach this primitive as FP32 fused-TV outputs."""
    bf16_acceptances = torch.rand(2, 5, device="cuda", dtype=torch.bfloat16)
    assert fused_mtp_prefix_unavailable_reason(bf16_acceptances) == (
        "acceptance dtype torch.bfloat16 is not supported"
    )
    actual = mtp_e2e_prefix_objective(bf16_acceptances)
    reference = _native_prefix_objective(bf16_acceptances)
    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_prefix_is_deterministic_and_cuda_graph_safe():
    """Forward and analytical backward replay deterministically inside a CUDA Graph."""
    torch.manual_seed(83)
    acceptance_data = torch.rand(7, 67, device="cuda")
    grad_output = torch.randn(67, device="cuda")
    first = _run_objective_and_gradient(mtp_e2e_prefix_objective, acceptance_data, grad_output)
    second = _run_objective_and_gradient(mtp_e2e_prefix_objective, acceptance_data, grad_output)
    for first_tensor, second_tensor in zip(first, second):
        assert torch.equal(first_tensor, second_tensor)

    static_acceptances = acceptance_data.detach().clone().requires_grad_(True)
    static_acceptances.grad = torch.zeros_like(static_acceptances)
    graph = torch.cuda.CUDAGraph()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_acceptances.grad.zero_()
            warmup_output, _ = mtp_e2e_prefix_objective(static_acceptances)
            (warmup_output * grad_output).sum().backward()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    with torch.cuda.graph(graph):
        static_acceptances.grad.zero_()
        graph_output, graph_prefix_losses = mtp_e2e_prefix_objective(static_acceptances)
        (graph_output * grad_output).sum().backward()
    graph.replay()

    torch.testing.assert_close(graph_output, first[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(graph_prefix_losses, first[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(static_acceptances.grad, first[2], rtol=1e-5, atol=1e-6)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical and dispatch tests for the fused MTP TV primitive."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fusions import fused_mtp_tv as fused_mtp_tv_module
from megatron.core.fusions.fused_mtp_tv import (
    fused_mtp_tv_unavailable_reason,
    vocab_parallel_tv_distance,
)


def _native_tv_distance(draft_logits, target_logits):
    draft_prob = F.softmax(draft_logits.float(), dim=-1)
    target_prob = F.softmax(target_logits.detach().float(), dim=-1)
    return (1.0 - torch.minimum(draft_prob, target_prob).sum(dim=-1)).clamp(0.0, 1.0)


def _run_tv_and_gradient(function, draft_data, target_logits, grad_output):
    draft_logits = draft_data.detach().clone().requires_grad_(True)
    output = function(draft_logits, target_logits)
    (output * grad_output).sum().backward()
    return output.detach(), draft_logits.grad.detach()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab_size", [257, 154880])
def test_fused_mtp_tv_matches_native_forward_and_backward(dtype, vocab_size):
    """Cover an odd block tail and GLM-5.2's production vocabulary."""
    torch.manual_seed(1234)
    shape = (3, 2, vocab_size)
    draft_data = torch.randn(shape, device="cuda", dtype=dtype)
    target_logits = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad_output = torch.randn(shape[:-1], device="cuda", dtype=torch.float32)

    assert fused_mtp_tv_unavailable_reason(draft_data, target_logits) is None
    actual, actual_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )
    assert actual.dtype == torch.float32
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, target_logits, grad_output
    )

    rtol = 3e-3 if dtype == torch.bfloat16 else 1e-5
    atol = 3e-3 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=rtol, atol=atol)
    assert target_logits.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_tv_clamp_boundaries_and_zero_acceptance():
    """The fused path preserves both clamp endpoints and finite gradients."""
    vocab_size = 257
    identical = torch.randn(2, vocab_size, device="cuda", dtype=torch.float32)
    identical_output, identical_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        identical,
        identical,
        torch.ones(2, device="cuda"),
    )
    torch.testing.assert_close(
        identical_output, torch.zeros_like(identical_output), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(identical_grad, torch.zeros_like(identical_grad), atol=1e-6, rtol=0)

    draft = torch.full((1, vocab_size), -100.0, device="cuda")
    target = torch.full_like(draft, -100.0)
    draft[:, 0] = 100.0
    target[:, 1] = 100.0
    output, gradient = _run_tv_and_gradient(
        lambda draft_logits, target_logits: vocab_parallel_tv_distance(
            draft_logits, target_logits, logits_are_vocab_sharded=False
        ),
        draft,
        target,
        torch.ones(1, device="cuda"),
    )
    torch.testing.assert_close(output, torch.ones_like(output), atol=1e-6, rtol=0)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_noncontiguous_logits_use_reference_fallback(monkeypatch):
    """Unsupported strides retain the PyTorch reference behavior."""
    torch.manual_seed(17)
    draft_data = torch.randn(2, 257, 3, device="cuda").transpose(1, 2)
    target_logits = torch.randn(2, 257, 3, device="cuda").transpose(1, 2)
    grad_output = torch.randn(draft_data.shape[:-1], device="cuda")
    assert not draft_data.is_contiguous()
    assert fused_mtp_tv_unavailable_reason(draft_data, target_logits) == "logits are not contiguous"

    def fail_if_fused(*_args, **_kwargs):
        pytest.fail("The fused kernel must not receive unsupported noncontiguous logits")

    monkeypatch.setattr(fused_mtp_tv_module, "_fused_vocab_parallel_tv_distance", fail_if_fused)
    actual, actual_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, target_logits, grad_output
    )
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_noncontiguous_target_is_packed_before_fused_dispatch(monkeypatch):
    """A materialized-roll view is packed centrally when the draft supports Triton."""
    torch.manual_seed(19)
    draft_data = torch.randn(2, 3, 257, device="cuda")
    target_logits = torch.randn(2, 257, 3, device="cuda").transpose(1, 2)
    grad_output = torch.randn(2, 3, device="cuda")
    assert draft_data.is_contiguous()
    assert not target_logits.is_contiguous()
    assert fused_mtp_tv_unavailable_reason(draft_data, draft_data) is None

    original_fused = fused_mtp_tv_module._fused_vocab_parallel_tv_distance
    fused_calls = 0

    def record_fused(draft, target, *args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        assert target.is_contiguous()
        return original_fused(draft, target, *args, **kwargs)

    monkeypatch.setattr(fused_mtp_tv_module, "_fused_vocab_parallel_tv_distance", record_fused)
    actual, actual_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )
    reference, reference_grad = _run_tv_and_gradient(
        _native_tv_distance, draft_data, target_logits, grad_output
    )

    assert fused_calls == 1
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_tv_saves_only_compact_full_vocab_metadata():
    """Backward state must pack its only vocabulary-sized comparison mask."""
    torch.manual_seed(29)
    draft_logits = torch.randn(3, 2, 4097, device="cuda", requires_grad=True)
    target_logits = torch.randn_like(draft_logits)
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        output = vocab_parallel_tv_distance(
            draft_logits, target_logits, logits_are_vocab_sharded=False
        )
        output.sum().backward()

    full_vocab_tensors = [
        tensor for tensor in saved_tensors if tensor.numel() == draft_logits.numel()
    ]
    assert len(full_vocab_tensors) == 1
    assert full_vocab_tensors[0].data_ptr() == draft_logits.data_ptr()
    packed_masks = [tensor for tensor in saved_tensors if tensor.dtype == torch.uint8]
    assert [tuple(tensor.shape) for tensor in packed_masks] == [(6, (4097 + 7) // 8)]
    assert not any(
        tensor.dtype in (torch.bool, torch.uint8) and tensor.numel() == draft_logits.numel()
        for tensor in saved_tensors
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_mtp_tv_is_deterministic(dtype):
    """Repeated launches of the same implementation are bitwise stable."""
    torch.manual_seed(2026)
    draft_data = torch.randn(4, 1031, device="cuda", dtype=dtype)
    target_logits = torch.randn_like(draft_data)
    grad_output = torch.randn(4, device="cuda")

    def function(draft, target):
        return vocab_parallel_tv_distance(draft, target, logits_are_vocab_sharded=False)

    first_output, first_grad = _run_tv_and_gradient(
        function, draft_data, target_logits, grad_output
    )
    second_output, second_grad = _run_tv_and_gradient(
        function, draft_data, target_logits, grad_output
    )
    assert torch.equal(first_output, second_output)
    assert torch.equal(first_grad, second_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_mtp_tv_is_cuda_graph_safe():
    """The fused forward and analytical backward replay inside a CUDA Graph."""
    torch.manual_seed(2031)
    draft_data = torch.randn(4, 1031, device="cuda")
    target_logits = torch.randn_like(draft_data)
    grad_output = torch.randn(4, device="cuda")
    expected_output, expected_grad = _run_tv_and_gradient(
        lambda draft, target: vocab_parallel_tv_distance(
            draft, target, logits_are_vocab_sharded=False
        ),
        draft_data,
        target_logits,
        grad_output,
    )

    static_draft = draft_data.detach().clone().requires_grad_(True)
    static_draft.grad = torch.zeros_like(static_draft)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            static_draft.grad.zero_()
            warmup_output = vocab_parallel_tv_distance(
                static_draft, target_logits, logits_are_vocab_sharded=False
            )
            (warmup_output * grad_output).sum().backward()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_draft.grad.zero_()
        graph_output = vocab_parallel_tv_distance(
            static_draft, target_logits, logits_are_vocab_sharded=False
        )
        (graph_output * grad_output).sum().backward()
    graph.replay()

    torch.testing.assert_close(graph_output, expected_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(static_draft.grad, expected_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [(0, 257), (3, 0, 257)])
def test_empty_row_logits_use_reference_fallback(shape):
    """Empty leading dimensions never dispatch a zero-program Triton grid."""
    draft = torch.empty(shape)
    target = torch.empty_like(draft)
    assert fused_mtp_tv_unavailable_reason(draft, target) is not None

    if torch.cuda.is_available():
        draft = draft.cuda()
        target = target.cuda()
        assert fused_mtp_tv_unavailable_reason(draft, target) == "logits must have at least one row"
        actual = vocab_parallel_tv_distance(draft, target, logits_are_vocab_sharded=False)
        assert actual.shape == draft.shape[:-1]
        assert actual.numel() == 0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Check router-weighted activations and their mixed-precision memory bound."""

import os
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


@pytest.fixture
def implementation(monkeypatch):
    """Exercise chunk boundaries with small tensors without changing the public API."""
    from megatron.core.ops.moe import weighted_activation

    monkeypatch.setattr(weighted_activation, "_MAX_TEMP_ELEMENTS", 32)
    return vars(weighted_activation)


def _reference(activations: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return (activations * probs).to(activations.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("requires_grad", [(True, True), (True, False), (False, True)])
def test_outputs_and_gradients_match(
    implementation: dict[str, Any], dtype: torch.dtype, requires_grad: tuple[bool, bool]
) -> None:
    generator = torch.Generator().manual_seed(421)
    values = torch.randn(13, 7, generator=generator, dtype=dtype)
    weights = torch.rand(13, 1, generator=generator, dtype=torch.float32)
    upstream = torch.randn(values.shape, generator=generator, dtype=dtype)

    def evaluate(function: Callable[..., torch.Tensor]) -> tuple[torch.Tensor, ...]:
        activations = values.clone().requires_grad_(requires_grad[0])
        probs = weights.clone().requires_grad_(requires_grad[1])
        output = function(activations, probs)
        inputs = [tensor for tensor in (activations, probs) if tensor.requires_grad]
        return output, *torch.autograd.grad(output, inputs, upstream)

    actual = evaluate(implementation["apply_router_probs"])
    expected = evaluate(_reference)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


class _LargeFloatIntermediates(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if (
            isinstance(output, torch.Tensor)
            and output.dtype == torch.float32
            and output.ndim == 2
            and output.shape[-1] > 1
        ):
            self.largest = max(self.largest, output.numel())
        return output


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_promoted_intermediates_are_bounded(implementation: dict[str, Any], phase: str) -> None:
    activations = torch.randn(31, 7, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(31, 1, dtype=torch.float32, requires_grad=True)
    apply = implementation["apply_router_probs"]
    products = _LargeFloatIntermediates()
    if phase == "forward":
        with products:
            output = apply(activations, probs)
    else:
        output = apply(activations, probs)
        with products:
            torch.autograd.grad(output, (activations, probs), torch.ones_like(output))
    assert output.shape == activations.shape
    assert 0 < products.largest <= implementation["_MAX_TEMP_ELEMENTS"]


def test_existing_expression_exceeds_bound(implementation: dict[str, Any]) -> None:
    """The failed expression creates a full FP32 product before the BF16 cast."""
    activations = torch.randn(31, 7, dtype=torch.bfloat16)
    probs = torch.rand(31, 1, dtype=torch.float32)
    products = _LargeFloatIntermediates()
    with products:
        _reference(activations, probs)
    assert products.largest > implementation["_MAX_TEMP_ELEMENTS"]


def test_constant_router_weights_do_not_retain_activations(implementation: dict[str, Any]) -> None:
    activations = torch.randn(31, 7, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(31, 1, dtype=torch.float32)
    saved_bytes = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved_bytes
        saved_bytes += tensor.untyped_storage().nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = implementation["apply_router_probs"](activations, probs)
    assert output.requires_grad
    assert saved_bytes <= probs.untyped_storage().nbytes()
    expected = _reference(activations, probs)
    actual_grad = torch.autograd.grad(output, activations, torch.ones_like(output))[0]
    expected_grad = torch.autograd.grad(expected, activations, torch.ones_like(expected))[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 4, 5, 13])
def test_empty_and_chunk_boundary_sizes(implementation: dict[str, Any], rows: int) -> None:
    activations = torch.randn(rows, 8, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(rows, 1, dtype=torch.float32, requires_grad=True)
    output = implementation["apply_router_probs"](activations, probs)
    torch.testing.assert_close(output, _reference(activations, probs), rtol=0, atol=0)
    gradients = torch.autograd.grad(output, (activations, probs), torch.ones_like(output))
    assert gradients[0].shape == activations.shape
    assert gradients[1].shape == probs.shape


def test_recomputed_activation_matches(implementation: dict[str, Any]) -> None:
    from torch.utils.checkpoint import checkpoint

    values = torch.randn(13, 14, dtype=torch.bfloat16)
    weights = torch.rand(13, 1, dtype=torch.float32)

    def evaluate(function: Callable[..., torch.Tensor]) -> tuple[torch.Tensor, ...]:
        inputs = values.clone().requires_grad_()
        probs = weights.clone().requires_grad_()

        def layer(inputs, probs):
            gate, linear = inputs.chunk(2, dim=-1)
            activation = torch.nn.functional.silu(gate) * linear
            return function(activation, probs)

        output = checkpoint(layer, inputs, probs, use_reentrant=True)
        output.sum().backward()
        assert inputs.grad is not None
        assert probs.grad is not None
        return output, inputs.grad, probs.grad

    for actual, expected in zip(
        evaluate(implementation["apply_router_probs"]), evaluate(_reference), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_noncontiguous_activations_and_no_grad(implementation: dict[str, Any]) -> None:
    activations = torch.randn(7, 13, dtype=torch.bfloat16).t()
    probs = torch.rand(13, 1, dtype=torch.float32)
    with torch.no_grad():
        actual = implementation["apply_router_probs"](activations, probs)
        expected = _reference(activations, probs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_broadcasting_is_preserved(implementation: dict[str, Any]) -> None:
    activations = torch.randn(13, 7, dtype=torch.bfloat16)
    probs = torch.rand(7, dtype=torch.float32)
    actual = implementation["apply_router_probs"](activations, probs)
    torch.testing.assert_close(actual, _reference(activations, probs), rtol=0, atol=0)


def _glu(inputs: torch.Tensor, clamp: float | None = None) -> torch.Tensor:
    gate, linear = inputs.chunk(2, dim=-1)
    if clamp is not None:
        gate = gate.clamp(max=clamp)
        linear = linear.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * (linear + 0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_recomputed_activation_preserves_autocast(
    implementation: dict[str, Any], enabled: bool, dtype: torch.dtype
) -> None:
    """Backward must reuse forward precision after its autocast scope has ended."""
    from megatron.core.activations import squared_relu

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    generator = torch.Generator(device=device).manual_seed(421)
    values = torch.randn(13, 14, device=device, dtype=dtype, generator=generator)
    weights = torch.rand(13, 1, device=device, dtype=torch.float32, generator=generator)
    upstream = torch.randn(13, 7, device=device, dtype=torch.float32, generator=generator)

    def activation(inputs: torch.Tensor) -> torch.Tensor:
        gate, linear = inputs.chunk(2, dim=-1)
        return squared_relu(gate) * linear

    def evaluate(chunked: bool) -> tuple[torch.Tensor, ...]:
        inputs = values.clone().requires_grad_()
        probs = weights.clone().requires_grad_()
        with torch.autocast("cuda", dtype=dtype, enabled=enabled):
            if chunked:
                output = implementation["apply_weighted_activation"](activation, inputs, probs)
            else:
                output = _reference(activation(inputs), probs)
        # Change the chunked backward's ambient state; keep native backward outside AMP.
        with torch.autocast("cuda", dtype=dtype, enabled=not enabled if chunked else False):
            gradients = torch.autograd.grad(output, (inputs, probs), upstream.to(output.dtype))
        return output, *gradients

    for actual, expected in zip(evaluate(True), evaluate(False), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_partial_chunk_matches_native(monkeypatch, gated: bool, dtype: torch.dtype) -> None:
    """Compare CUDA outputs and gradients with a one-row tail chunk."""
    from megatron.core.ops.moe import weighted_activation

    monkeypatch.setattr(weighted_activation, "_MAX_TEMP_ELEMENTS", 256 * 1024)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    generator = torch.Generator(device=device).manual_seed(421)
    rows, features = 129, 4096
    values = torch.randn(
        rows, features * (2 if gated else 1), device=device, dtype=dtype, generator=generator
    )
    weights = torch.rand(rows, 1, device=device, dtype=torch.float32, generator=generator)
    upstream = torch.randn(rows, features, device=device, dtype=dtype, generator=generator)

    def activation(inputs: torch.Tensor) -> torch.Tensor:
        return _glu(inputs, clamp=0.75)

    def evaluate(chunked: bool) -> tuple[torch.Tensor, ...]:
        inputs = values.clone().requires_grad_()
        probs = weights.clone().requires_grad_()
        if gated and chunked:
            output = weighted_activation.apply_weighted_activation(activation, inputs, probs)
        elif chunked:
            output = weighted_activation.apply_router_probs(inputs, probs)
        else:
            output = _reference(activation(inputs) if gated else inputs, probs)
        return output, *torch.autograd.grad(output, (inputs, probs), upstream)

    actual_output, actual_inputs, actual_probs = evaluate(True)
    expected_output, expected_inputs, expected_probs = evaluate(False)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_inputs, expected_inputs, rtol=0, atol=0)
    # CUDA reductions can use a different FP32 summation tree for the tail row.
    torch.testing.assert_close(actual_probs, expected_probs)
    activations = activation(values) if gated else values
    precise_probs = (
        (upstream.float() * activations).sum(dim=-1, keepdim=True, dtype=torch.float64).float()
    )
    torch.testing.assert_close(actual_probs, precise_probs)
    torch.testing.assert_close(expected_probs, precise_probs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("requires_grad", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("clamp", [None, 0.75])
def test_weighted_glu_outputs_and_gradients_match(
    implementation: dict[str, Any],
    dtype: torch.dtype,
    requires_grad: tuple[bool, bool],
    clamp: float | None,
) -> None:
    generator = torch.Generator().manual_seed(512)
    values = torch.randn(13, 14, generator=generator, dtype=dtype)
    weights = torch.rand(13, 1, generator=generator, dtype=torch.float32)
    upstream = torch.randn(13, 7, generator=generator, dtype=dtype)

    def activation(inputs: torch.Tensor) -> torch.Tensor:
        return _glu(inputs, clamp)

    def evaluate(chunked: bool) -> tuple[torch.Tensor, ...]:
        inputs = values.clone().requires_grad_(requires_grad[0])
        probs = weights.clone().requires_grad_(requires_grad[1])
        if chunked:
            output = implementation["apply_weighted_activation"](activation, inputs, probs)
        else:
            output = _reference(activation(inputs), probs)
        variables = [value for value in (inputs, probs) if value.requires_grad]
        return output, *torch.autograd.grad(output, variables, upstream)

    for actual, expected in zip(evaluate(True), evaluate(False), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_weighted_glu_promoted_intermediates_are_bounded(
    implementation: dict[str, Any], phase: str
) -> None:
    inputs = torch.randn(31, 14, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(31, 1, dtype=torch.float32, requires_grad=True)
    products = _LargeFloatIntermediates()
    if phase == "forward":
        with products:
            output = implementation["apply_weighted_activation"](_glu, inputs, probs)
    else:
        output = implementation["apply_weighted_activation"](_glu, inputs, probs)
        with products:
            torch.autograd.grad(output, (inputs, probs), torch.ones_like(output))
    assert output.shape == (inputs.shape[0], inputs.shape[1] // 2)
    assert 0 < products.largest <= implementation["_MAX_TEMP_ELEMENTS"]


def test_weighted_glu_retains_only_input_storage(implementation: dict[str, Any]) -> None:
    """The activation graph must not retain additional full-size intermediates."""
    inputs = torch.randn(31, 14, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(31, 1, dtype=torch.float32, requires_grad=True)

    def retained_bytes(chunked: bool) -> int:
        storages: dict[int, int] = {}

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            if chunked:
                output = implementation["apply_weighted_activation"](_glu, inputs, probs)
            else:
                output = _reference(_glu(inputs), probs)
        # Keep the graph alive while measuring its saved tensors.
        assert output.requires_grad
        return sum(storages.values())

    input_bytes = inputs.untyped_storage().nbytes() + probs.untyped_storage().nbytes()
    assert retained_bytes(True) <= input_bytes
    assert retained_bytes(False) > input_bytes


@pytest.mark.parametrize("reentrant", [True, False])
def test_weighted_glu_checkpoint_matches(implementation: dict[str, Any], reentrant: bool) -> None:
    from torch.utils.checkpoint import checkpoint

    values = torch.randn(13, 14, dtype=torch.bfloat16)
    weights = torch.rand(13, 1, dtype=torch.float32)

    def evaluate(chunked: bool) -> tuple[torch.Tensor, ...]:
        inputs = values.clone().requires_grad_()
        probs = weights.clone().requires_grad_()

        def layer(inputs: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
            if chunked:
                return implementation["apply_weighted_activation"](_glu, inputs, probs)
            return _reference(_glu(inputs), probs)

        output = checkpoint(layer, inputs, probs, use_reentrant=reentrant)
        output.sum().backward()
        assert inputs.grad is not None
        assert probs.grad is not None
        return output, inputs.grad, probs.grad

    for actual, expected in zip(evaluate(True), evaluate(False), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 13])
def test_weighted_glu_boundary_sizes(implementation: dict[str, Any], rows: int) -> None:
    inputs = torch.randn(rows, 14, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(rows, 1, dtype=torch.float32, requires_grad=True)
    actual = implementation["apply_weighted_activation"](_glu, inputs, probs)
    expected = _reference(_glu(inputs), probs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual, (inputs, probs), torch.ones_like(actual))
    expected_grads = torch.autograd.grad(expected, (inputs, probs), torch.ones_like(expected))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_weighted_glu_noncontiguous_inputs_without_grad(implementation: dict[str, Any]) -> None:
    inputs = torch.randn(14, 13, dtype=torch.bfloat16).t()
    probs = torch.rand(13, 1, dtype=torch.float32)
    with torch.no_grad():
        actual = implementation["apply_weighted_activation"](_glu, inputs, probs)
        expected = _reference(_glu(inputs), probs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

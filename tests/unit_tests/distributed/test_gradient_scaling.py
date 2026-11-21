# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_tokens", [10, 2**25, 2**26, 3 * 2**25, 3 * 2**25 - 13])
@pytest.mark.parametrize("use_graph", [False, True])
def test_scale_gradients_preserves_factor_precision(dtype, num_tokens, use_graph):
    """Token scaling matches FP32 multiplication followed by one destination cast."""
    original = torch.tensor(
        [-32768, -130, -1, -(2**-10), 0, 2**-10, 1, 130, 32768], device="cuda", dtype=dtype
    )
    original_extra = [original.float(), torch.tensor(0.125, device="cuda")]
    buffer = SimpleNamespace(
        grad_data=original.clone(), extra_main_grads=[grad.clone() for grad in original_extra]
    )
    pointers = [grad.data_ptr() for grad in [buffer.grad_data, *buffer.extra_main_grads]]
    scale = torch.tensor(1.0 / num_tokens, device="cuda", dtype=torch.float32)
    graph = None
    if use_graph:
        _ParamAndGradBuffer.scale_gradients(buffer, scale)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _ParamAndGradBuffer.scale_gradients(buffer, scale)

    # Replay identical inputs, then change the factor to check that it remains dynamic.
    for denominator in (num_tokens, num_tokens, num_tokens + 1, num_tokens + 1):
        scale.fill_(1.0 / denominator)
        buffer.grad_data.copy_(original)
        for grad, initial in zip(buffer.extra_main_grads, original_extra):
            grad.copy_(initial)
        if graph is None:
            _ParamAndGradBuffer.scale_gradients(buffer, scale)
        else:
            graph.replay()

        expected = (original.float() * scale).to(dtype)
        torch.testing.assert_close(buffer.grad_data, expected, rtol=0, atol=0)
        for grad, initial in zip(buffer.extra_main_grads, original_extra):
            torch.testing.assert_close(grad, initial * scale, rtol=0, atol=0)
        assert scale.ndim == 0
        assert pointers == [
            grad.data_ptr() for grad in [buffer.grad_data, *buffer.extra_main_grads]
        ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("factor_kind", ["cpu_scalar", "cuda_vector"])
def test_scale_gradients_other_tensor_factors(dtype, factor_kind):
    """CPU-scalar and dimensioned CUDA tensor factors remain supported."""
    if factor_kind == "cpu_scalar":
        factor = torch.tensor(0.1, dtype=torch.float32)
    else:
        factor = torch.tensor([0.1], device="cuda", dtype=torch.float32)
    original = torch.tensor([-130, 0, 130], device="cuda", dtype=dtype)
    buffer = SimpleNamespace(grad_data=original.clone(), extra_main_grads=[original.float()])
    expected = original * factor
    expected_extra = original.float() * factor
    _ParamAndGradBuffer.scale_gradients(buffer, factor)
    torch.testing.assert_close(buffer.grad_data, expected.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(buffer.extra_main_grads[0], expected_extra, rtol=0, atol=0)


def _bits(tensor):
    return tensor.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[tensor.element_size()])


def _scale_gradients_unfused(buffer, scaling_factor):
    """`_ParamAndGradBuffer.scale_gradients` with `*=` for every tensor."""
    buffer_scale = scaling_factor
    if buffer_scale.is_cuda and buffer_scale.ndim == 0:
        buffer_scale = buffer_scale.reshape(1)
    buffer.grad_data *= buffer_scale
    for grad in buffer.extra_main_grads:
        grad *= scaling_factor


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("factor_dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("factor_kind", ["cuda_scalar", "cpu_scalar", "cuda_vector"])
def test_scale_gradients_matches_unfused_multiplication(dtype, factor_dtype, factor_kind):
    """Scaling same-dtype tensors with `_foreach_mul_` leaves every result bitwise unchanged."""
    torch.manual_seed(0)
    special_values = [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), 1e-40, 3.0e38]
    values = torch.randn(2**20 + 3, device="cuda") * 1000
    values[: len(special_values)] = torch.tensor(special_values)
    extra_main_grads = [values[:4099].clone(), torch.tensor([0.125], device="cuda")]
    factor = torch.tensor(1.0 / 123457, dtype=factor_dtype)
    if factor_kind != "cpu_scalar":
        factor = factor.cuda()
    if factor_kind == "cuda_vector":
        factor = factor.reshape(1)
    fused, unfused = (
        SimpleNamespace(
            grad_data=values.to(dtype, copy=True),
            extra_main_grads=[grad.clone() for grad in extra_main_grads],
        )
        for _ in range(2)
    )

    _ParamAndGradBuffer.scale_gradients(fused, factor)
    _scale_gradients_unfused(unfused, factor)

    for actual, expected in zip(
        [fused.grad_data, *fused.extra_main_grads], [unfused.grad_data, *unfused.extra_main_grads]
    ):
        assert torch.equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_scale_gradients_fuses_same_dtype_tensors(dtype):
    """FP32 tensors share one fused kernel; a BF16 buffer keeps the promoted multiplication."""
    buffer = SimpleNamespace(
        grad_data=torch.ones(2**20, device="cuda", dtype=dtype),
        extra_main_grads=[torch.ones(4096, device="cuda"), torch.ones(77, device="cuda")],
    )
    scale = torch.tensor(0.5, device="cuda")
    _ParamAndGradBuffer.scale_gradients(buffer, scale)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        _ParamAndGradBuffer.scale_gradients(buffer, scale)
        torch.cuda.synchronize()

    kernels = [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert sum("multi_tensor_apply_kernel" in name for name in kernels) == 1, kernels
    if dtype == torch.float32:
        assert len(kernels) == 1, kernels

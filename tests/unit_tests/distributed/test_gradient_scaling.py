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
@pytest.mark.parametrize("factor_kind", ["python", "cpu_scalar", "cuda_vector"])
def test_scale_gradients_other_factor_types(dtype, factor_kind):
    """Existing Python, CPU-scalar and dimensioned CUDA factors remain supported."""
    factor = 0.1
    if factor_kind == "cpu_scalar":
        factor = torch.tensor(factor, dtype=torch.float32)
    elif factor_kind == "cuda_vector":
        factor = torch.tensor([factor], device="cuda", dtype=torch.float32)
    original = torch.tensor([-130, 0, 130], device="cuda", dtype=dtype)
    buffer = SimpleNamespace(grad_data=original.clone(), extra_main_grads=[original.float()])
    expected = original * factor
    expected_extra = original.float() * factor
    _ParamAndGradBuffer.scale_gradients(buffer, factor)
    torch.testing.assert_close(buffer.grad_data, expected.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(buffer.extra_main_grads[0], expected_extra, rtol=0, atol=0)

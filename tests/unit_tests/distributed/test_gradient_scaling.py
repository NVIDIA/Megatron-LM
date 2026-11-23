# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import TestModel, Utils


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


@pytest.mark.parametrize("grad_reduce_in_fp32", [False, True])
def test_scale_gradients_on_padded_default_layout(grad_reduce_in_fp32):
    """Token scaling of a real buffer whose default layout pads each param start.

    Without the distributed optimizer every param starts on a 64-element boundary, so 3x3
    weights and 3-element biases leave padding between the main_grad views. Scaling matches
    FP32 multiplication followed by one cast in every view and leaves the padding at zero.
    """
    Utils.initialize_model_parallel()
    try:
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            use_distributed_optimizer=False,
            overlap_grad_reduce=True,
            average_in_collective=False,
        )
        module = TestModel(
            input_dim=3, output_dim=3, num_layers=4, bias=True, shared_embedding=False
        ).bfloat16()
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            ddp_config=ddp_config,
            module=module,
        )
        assert len(model.buffers) == 1
        buffer = model.buffers[0]
        in_a_view = torch.zeros_like(buffer.grad_data, dtype=torch.bool)
        originals = []
        for index, param in enumerate(model.module.parameters()):
            start, end, _ = buffer.param_index_map[param]
            in_a_view[start:end] = True
            values = torch.arange(param.numel(), device=param.main_grad.device) - 4 + index
            param.main_grad.copy_((values * 32.5).view(param.shape))
            originals.append((param, param.main_grad.clone()))
        assert not in_a_view.all(), "the default layout should leave padding between params"

        scale = torch.tensor(1.0 / 3, device="cuda", dtype=torch.float32)
        buffer.scale_gradients(scale)

        for param, original in originals:
            expected = (original.float() * scale).to(original.dtype)
            torch.testing.assert_close(param.main_grad, expected, rtol=0, atol=0)
        assert torch.count_nonzero(buffer.grad_data[~in_a_view]) == 0
    finally:
        Utils.destroy_model_parallel()

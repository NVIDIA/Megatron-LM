import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import fp8_quantize
from megatron.core.fusions.fused_bias_swiglu import (
    bias_swiglu_impl,
    weighted_bias_swiglu_impl,
    weighted_swiglu,
    weighted_swiglu_back,
)
from megatron.core.fusions.fused_weighted_swiglu_quant import (
    fused_weighted_swiglu_quant,
    fused_weighted_swiglu_quant_back,
)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_weighted_bias_swiglu(input_dtype):
    if input_dtype == torch.float32:
        tols = dict(rtol=1.0e-6, atol=1.0e-6)
    elif input_dtype == torch.bfloat16:
        tols = dict(rtol=2.0e-2, atol=1.0e-3)
    else:
        raise ValueError(f"Invalid input dtype: {input_dtype}")

    x = torch.randn(16, 64, dtype=input_dtype, device="cuda")
    x.requires_grad = True
    weights = torch.randn(16, 1, dtype=torch.float32, device="cuda")
    weights.requires_grad = True
    bwd_input = torch.randn(16, 32, dtype=input_dtype, device="cuda")

    y = bias_swiglu_impl(x, None) * weights
    y = y.to(input_dtype)
    y.backward(bwd_input)

    x_2 = x.detach()
    x_2.requires_grad = True
    weights_2 = weights.detach()
    weights_2.requires_grad = True
    bwd_input_2 = bwd_input.detach()

    y_2 = weighted_bias_swiglu_impl(x_2, None, weights_2)
    y_2.backward(bwd_input_2)

    assert y_2.dtype == y.dtype
    assert torch.allclose(y, y_2, **tols)
    assert x_2.grad.dtype == x.grad.dtype
    assert torch.allclose(x.grad, x_2.grad, **tols)
    assert weights_2.grad.dtype == weights.grad.dtype
    if input_dtype == torch.float32:
        assert torch.allclose(weights.grad, weights_2.grad, **tols)


def _test_fused_weighted_bias_swiglu_quant(
    num_tokens, topK, MOE_INTERMEDIATE_SIZE, BENCHMARK=False
):
    num_out_tokens = num_tokens * topK
    x = torch.rand((num_out_tokens, 2 * MOE_INTERMEDIATE_SIZE), dtype=torch.float16).cuda()
    weights = torch.rand((num_out_tokens, 1), dtype=torch.float32).cuda()
    grad_output = torch.rand((num_out_tokens, MOE_INTERMEDIATE_SIZE), dtype=x.dtype).cuda()
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    # Forward: fused kernel vs non-fused weighted_swiglu + TE blockwise quant.
    fused_data, fused_scale = fused_weighted_swiglu_quant(x, weights)

    ref_out = weighted_swiglu(x, weights)
    ref_data, ref_scale = fp8_quantize(Fp8Recipe.blockwise, ref_out)

    torch.testing.assert_close(fused_data.view(torch.uint8), ref_data, rtol=0, atol=1)
    torch.testing.assert_close(fused_scale, ref_scale)

    # Backward: fused kernel vs non-fused weighted_swiglu_back + TE blockwise quant.
    fused_dgrad_data, fused_dgrad_scale, fused_wgrad = fused_weighted_swiglu_quant_back(
        grad_output, x, weights
    )

    ref_dgrad, ref_wgrad = weighted_swiglu_back(grad_output, x, weights)
    ref_dgrad_data, ref_dgrad_scale = fp8_quantize(Fp8Recipe.blockwise, ref_dgrad)

    torch.testing.assert_close(fused_dgrad_data.view(torch.uint8), ref_dgrad_data, rtol=0, atol=1)
    torch.testing.assert_close(fused_dgrad_scale, ref_dgrad_scale)
    torch.testing.assert_close(fused_wgrad, ref_wgrad)

    if BENCHMARK:

        def _run_fused_fwd():
            _ = fused_weighted_swiglu_quant(x, weights)

        def _run_ref_fwd():
            ref_out_ = weighted_swiglu(x, weights)
            _ = fp8_quantize(Fp8Recipe.blockwise, ref_out_)

        def _run_fused_bwd():
            _ = fused_weighted_swiglu_quant_back(grad_output, x, weights)

        def _run_ref_bwd():
            ref_dgrad_, _ = weighted_swiglu_back(grad_output, x, weights)
            _ = fp8_quantize(Fp8Recipe.blockwise, ref_dgrad_)

        def _benchmark(fn, warmup=20, iters=100):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters

        fused_fwd_ms = _benchmark(_run_fused_fwd)
        ref_fwd_ms = _benchmark(_run_ref_fwd)
        fused_bwd_ms = _benchmark(_run_fused_bwd)
        ref_bwd_ms = _benchmark(_run_ref_bwd)

        print(
            "[perf] fused_weighted_swiglu_quant "
            f"fwd: fused={fused_fwd_ms:.3f} ms, ref={ref_fwd_ms:.3f} ms, speedup={ref_fwd_ms/fused_fwd_ms:.3f}"
            f"bwd: fused={fused_bwd_ms:.3f} ms, ref={ref_bwd_ms:.3f} ms, speedup={ref_bwd_ms/fused_bwd_ms:.3f}"
        )
        assert fused_fwd_ms > 0.0 and ref_fwd_ms > 0.0
        assert fused_bwd_ms > 0.0 and ref_bwd_ms > 0.0


@pytest.mark.parametrize(
    "num_tokens, topK, MOE_INTERMEDIATE_SIZE", [(4096, 7, 256), (4096, 6, 1408)]
)
def test_fused_weighted_bias_swiglu_quant(num_tokens, topK, MOE_INTERMEDIATE_SIZE):
    BENCHMARK = False

    _test_fused_weighted_bias_swiglu_quant(
        num_tokens=num_tokens,
        topK=topK,
        MOE_INTERMEDIATE_SIZE=MOE_INTERMEDIATE_SIZE,
        BENCHMARK=BENCHMARK,
    )

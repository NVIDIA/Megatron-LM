from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from megatron.lite.model.deepseek_v4.vllm.primitive.moe import grouped as vllm_grouped_moe
from megatron.lite.primitive.modules.experts import swiglu_with_probs


def _reference(
    hidden: torch.Tensor,
    counts: tuple[int, ...],
    limit: float,
    w13: tuple[torch.Tensor, ...],
    w2: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    outputs = []
    offset = 0
    for count, fc1, fc2 in zip(counts, w13, w2, strict=True):
        selected = hidden[offset : offset + count]
        gate_up = F.linear(selected, fc1)
        outputs.append(F.linear(swiglu_with_probs(gate_up, None, limit), fc2))
        offset += count
    return torch.cat(outputs)


@pytest.mark.parametrize(
    ("scale_format_name", "expected_quantizer", "expected_use_ue8m0"),
    [
        ("FLOAT32", "float32", False),
        ("FLOAT32_CEIL_UE8M0", "float32", True),
        ("UE8M0", "packed", True),
    ],
)
def test_contiguous_input_quant_matches_vllm_scale_format(
    monkeypatch,
    scale_format_name: str,
    expected_quantizer: str,
    expected_use_ue8m0: bool,
) -> None:
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def float32_quant(value, group_size, **kwargs):
        calls.append(("float32", group_size, kwargs))
        return value, torch.ones(1)

    def packed_quant(value, group_size, **kwargs):
        calls.append(("packed", group_size, kwargs))
        return value, torch.ones(1)

    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: getattr(DeepGemmQuantScaleFMT, scale_format_name)),
    )
    monkeypatch.setattr(fp8_utils, "per_token_group_quant_fp8", float32_quant)
    monkeypatch.setattr(
        fp8_utils,
        "per_token_group_quant_fp8_packed_for_deepgemm",
        packed_quant,
    )

    value = torch.randn(2, 128, dtype=torch.bfloat16)
    vllm_grouped_moe._vllm_quantize_contiguous_input(value)

    assert len(calls) == 1
    quantizer, group_size, kwargs = calls[0]
    assert quantizer == expected_quantizer
    assert group_size == 128
    assert kwargs["use_ue8m0"] is expected_use_ue8m0


def test_vllm_visible_silu_quant_has_no_layout_dependent_fallback(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append((value, kwargs))
        return kwargs["output_q"], torch.ones(1)

    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: DeepGemmQuantScaleFMT.FLOAT32),
    )
    monkeypatch.setattr(fp8_utils, "fused_silu_mul_per_token_group_quant_fp8", fused)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", raising=False)
    value = torch.randn(2, 256, dtype=torch.bfloat16)
    output = torch.empty(2, 128, dtype=torch.float8_e4m3fn)

    quantized, _scales = vllm_grouped_moe._vllm_silu_mul_quant(
        value, output=output, swiglu_limit=0.0
    )

    assert quantized is output
    assert len(calls) == 1
    assert calls[0][1]["masked_m"] is None


def test_vllm_visible_silu_quant_preserves_ds4_clamp(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append((value, kwargs))
        return kwargs["output_q"], torch.ones(1)

    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: DeepGemmQuantScaleFMT.UE8M0),
    )
    monkeypatch.setattr(fp8_utils, "fused_silu_mul_per_token_group_quant_fp8", fused)
    value = torch.randn(2, 256, dtype=torch.bfloat16)
    output = torch.empty(2, 128, dtype=torch.float8_e4m3fn)

    quantized, _scales = vllm_grouped_moe._vllm_silu_mul_quant(
        value, output=output, swiglu_limit=10.0
    )

    assert quantized is output
    assert len(calls) == 1
    assert calls[0][1]["clamp_limit"] == 10.0
    assert calls[0][1]["masked_m"] is None


def test_grouped_moe_preserves_clamped_forward_and_bf16_master_vjp(
    monkeypatch,
) -> None:
    torch.manual_seed(7)
    counts = (2, 1)
    limit = 10.0
    hidden = (torch.randn(3, 4) * 8).requires_grad_(True)
    w13 = tuple((torch.randn(6, 4) * 3).requires_grad_(True) for _ in counts)
    w2 = tuple(torch.randn(4, 3).requires_grad_(True) for _ in counts)
    tokens_per_expert = torch.tensor(counts, dtype=torch.int32)
    monkeypatch.setattr(vllm_grouped_moe, "_vllm_grouped_forward", _reference)
    output = vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden,
        tokens_per_expert,
        limit,
        *w13,
        *w2,
    )
    expected = _reference(hidden, counts, limit, w13, w2)
    unclamped = _reference(hidden, counts, 0.0, w13, w2)
    assert torch.equal(output, expected)
    assert not torch.allclose(output, unclamped)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    actual_grads = (hidden.grad, *(weight.grad for weight in w13 + w2))

    ref_hidden = hidden.detach().requires_grad_(True)
    ref_w13 = tuple(weight.detach().requires_grad_(True) for weight in w13)
    ref_w2 = tuple(weight.detach().requires_grad_(True) for weight in w2)
    ref_output = _reference(ref_hidden, counts, limit, ref_w13, ref_w2)
    expected_grads = torch.autograd.grad(
        ref_output,
        (ref_hidden, *ref_w13, *ref_w2),
        grad_output,
    )
    for actual, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected_grad)


@pytest.mark.gpus(1)
@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("vllm") is None,
    reason="requires CUDA and vLLM grouped DeepGEMM",
)
def test_real_grouped_deepgemm_forward_has_bf16_master_vjp() -> None:
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    DeepGemmQuantScaleFMT.init_oracle_cache()
    torch.manual_seed(11)
    counts = (2, 1)
    limit = 10.0
    hidden = ((torch.randn(3, 128, device="cuda", dtype=torch.bfloat16) * 8)).requires_grad_(True)
    w13 = tuple(
        torch.nn.Parameter(
            torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 3
        )
        for _ in counts
    )
    w2 = tuple(
        torch.nn.Parameter(
            torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        )
        for _ in counts
    )
    tokens_per_expert = torch.tensor(counts, device="cuda", dtype=torch.int32)
    output = vllm_grouped_moe.VLLMGroupedMoEWithBF16Backward.apply(
        hidden,
        tokens_per_expert,
        limit,
        *w13,
        *w2,
    )
    assert torch.isfinite(output).all()

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    actual_grads = (hidden.grad, *(weight.grad for weight in w13 + w2))

    ref_hidden = hidden.detach().requires_grad_(True)
    ref_w13 = tuple(weight.detach().requires_grad_(True) for weight in w13)
    ref_w2 = tuple(weight.detach().requires_grad_(True) for weight in w2)
    ref_output = _reference(ref_hidden, counts, limit, ref_w13, ref_w2)
    expected_grads = torch.autograd.grad(
        ref_output,
        (ref_hidden, *ref_w13, *ref_w2),
        grad_output,
    )
    for actual, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected_grad, rtol=0, atol=0)

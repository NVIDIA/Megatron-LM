from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.lite.model.deepseek_v4 import deployment_block_fp8 as fp8


def _weight(device: str = "cpu") -> nn.Parameter:
    return nn.Parameter(torch.randn(128, 128, dtype=torch.bfloat16, device=device))


def test_deployment_linear_calls_all_vllm_boundaries(monkeypatch) -> None:
    calls: list[str] = []

    def weight_quant(x, block_size, use_ue8m0):
        calls.append("weight_quant")
        return x.to(torch.float8_e4m3fn), torch.ones(1, 1)

    def post_process(*, wq, ws, quant_block_shape, use_e8m0):
        calls.append("weight_postprocess")
        return wq.contiguous(), ws.to(torch.int32)

    def activation_quant(x, group_size, use_ue8m0):
        calls.append("activation_quant")
        return x.to(torch.float8_e4m3fn), torch.ones(x.shape[0], 1, dtype=torch.int32)

    def gemm(activation, weight, out, *, is_deep_gemm_e8m0_used):
        del activation, weight
        calls.append("gemm")
        assert is_deep_gemm_e8m0_used is True
        out.fill_(5)

    entries = {
        ("vllm.utils.deep_gemm", "per_block_cast_to_fp8"): weight_quant,
        ("vllm.utils.deep_gemm", "DeepGemmQuantScaleFMT"): SimpleNamespace(
            from_oracle=lambda: SimpleNamespace(name="UE8M0")
        ),
        ("vllm.utils.deep_gemm", "fp8_gemm_nt"): gemm,
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        ): post_process,
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "per_token_group_quant_fp8_packed_for_deepgemm",
        ): activation_quant,
    }
    monkeypatch.setattr(fp8, "_entry", lambda module, name: entries[(module, name)])

    output = fp8.DeploymentBlockFP8Adapter()(
        torch.zeros(3, 128, dtype=torch.bfloat16), _weight()
    )
    assert calls == ["weight_quant", "weight_postprocess", "activation_quant", "gemm"]
    assert torch.equal(output, torch.full_like(output, 5))


def test_linear_contract_fails_before_vllm_for_invalid_master(monkeypatch) -> None:
    lookup = Mock()
    monkeypatch.setattr(fp8, "_entry", lookup)
    with pytest.raises(TypeError, match="BF16"):
        fp8.DeploymentBlockFP8Adapter()(
            torch.zeros(2, 128, dtype=torch.bfloat16),
            nn.Parameter(torch.zeros(128, 128)),
        )
    lookup.assert_not_called()


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires official vLLM FP8 and DeepGEMM compiled kernels",
)
def test_official_vllm_fp8_linear_is_bitwise_through_adapter() -> None:
    torch.manual_seed(17)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    master = _weight("cuda")

    reference_packed = fp8.pack_block_fp8_weight(master)
    reference = fp8.fp8_gemm_nt(x, reference_packed)
    candidate = fp8.DeploymentBlockFP8Adapter()(x, master)

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

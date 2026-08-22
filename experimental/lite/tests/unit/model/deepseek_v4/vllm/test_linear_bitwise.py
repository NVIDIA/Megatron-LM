from __future__ import annotations

import importlib.util
import pytest
import torch
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.primitive import deployment_fp8 as fp8


def _weight(device: str = "cpu") -> nn.Parameter:
    return nn.Parameter(torch.randn(128, 128, dtype=torch.bfloat16, device=device))


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

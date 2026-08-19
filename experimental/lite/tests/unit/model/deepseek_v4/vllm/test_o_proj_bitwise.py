from __future__ import annotations

import importlib.util
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.lite.primitive.kernels import vllm_ds4
from megatron.lite.primitive.kernels.vllm_ds4 import OProjectionAdapter
from megatron.lite.primitive.quantization import deployment_block_fp8


def test_o_projection_cpu_contract_calls_all_official_boundaries(monkeypatch) -> None:
    calls: list[str] = []

    def cast(weight, **kwargs):
        calls.append("cast")
        return weight.to(torch.float8_e4m3fn), torch.ones(
            weight.shape[0] // 128,
            weight.shape[1] // 128,
            dtype=torch.float32,
        )

    def post_process(**kwargs):
        calls.append("post")
        return kwargs["wq"], kwargs["ws"].to(torch.int32)

    official = Mock(return_value=torch.ones(2, 128, dtype=torch.bfloat16))
    entries = {
        ("vllm.utils.deep_gemm", "per_block_cast_to_fp8"): cast,
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "deepgemm_post_process_fp8_weight_block",
        ): post_process,
        (
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "per_token_group_quant_fp8",
        ): lambda value, *args, **kwargs: (
            value.to(torch.float8_e4m3fn),
            torch.ones(value.shape[0], 1, dtype=torch.float32),
        ),
        ("vllm.envs", "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES"): False,
        (
            "vllm.models.deepseek_v4.nvidia.ops.o_proj",
            "deep_gemm_fp8_o_proj",
        ): official,
        (
            "vllm.models.deepseek_v4.nvidia.ops.o_proj",
            "compute_fp8_einsum_recipe",
        ): lambda: ((1, 128, 128), False),
    }
    monkeypatch.setattr(vllm_ds4, "_symbol", lambda module, name: entries[(module, name)])
    monkeypatch.setattr(
        deployment_block_fp8,
        "_entry",
        lambda module, name: entries[(module, name)],
    )
    result = OProjectionAdapter()(
        torch.zeros(2, 2, 128, dtype=torch.bfloat16),
        torch.arange(2, dtype=torch.int64),
        torch.ones(8, 64, dtype=torch.float32),
        nn.Parameter(torch.zeros(128, 256, dtype=torch.bfloat16)),
        nn.Parameter(torch.zeros(128, 128, dtype=torch.bfloat16)),
        n_groups=1,
        heads_per_group=2,
        nope_dim=64,
        rope_dim=64,
        o_lora_rank=128,
    )
    assert torch.equal(result, official.return_value)
    assert calls == ["cast", "post", "cast", "post"]
    official.assert_called_once()


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="requires official vLLM inverse-RoPE, FP8 einsum, and DeepGEMM kernels",
)
def test_official_vllm_o_projection_is_bitwise_through_candidate_callable() -> None:
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
        compute_fp8_einsum_recipe,
        deep_gemm_fp8_o_proj,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8,
    )
    from vllm import envs
    from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8

    torch.manual_seed(29)
    device = "cuda"
    o = torch.randn(2, 2, 128, dtype=torch.bfloat16, device=device)
    positions = torch.tensor([0, 1], dtype=torch.int64, device=device)
    cos = torch.ones(8, 64, dtype=torch.float32, device=device)
    master_a = torch.randn(128, 256, dtype=torch.bfloat16, device=device)
    qweight, scales = per_block_cast_to_fp8(
        master_a, block_size=[128, 128], use_ue8m0=False
    )
    qweight, scales = deepgemm_post_process_fp8_weight_block(
        wq=qweight,
        ws=scales,
        quant_block_shape=(128, 128),
        use_e8m0=True,
        is_bmm=True,
        bmm_batch_size=1,
    )
    wo_a = type("_PackedGroupedWeight", (), {})()
    wo_a.weight, wo_a.weight_scale = qweight, scales
    master_b = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
    wb_q, wb_s = per_block_cast_to_fp8(
        master_b, block_size=[128, 128], use_ue8m0=False
    )
    wb_q, wb_s = deepgemm_post_process_fp8_weight_block(
        wq=wb_q,
        ws=wb_s,
        quant_block_shape=(128, 128),
        use_e8m0=True,
    )

    def wo_b(value):
        aq, a_s = per_token_group_quant_fp8(
            value,
            128,
            use_ue8m0=True,
            column_major_scales=True,
            tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES,
        )
        output = torch.empty(value.shape[0], 128, dtype=torch.bfloat16, device=device)
        fp8_gemm_nt(
            (aq, a_s),
            (wb_q, wb_s),
            output,
            is_deep_gemm_e8m0_used=True,
        )
        return output

    recipe, aligned = compute_fp8_einsum_recipe()
    kwargs = dict(
        n_groups=1,
        heads_per_group=2,
        nope_dim=64,
        rope_dim=64,
        o_lora_rank=128,
        einsum_recipe=recipe,
        tma_aligned_scales=aligned,
    )

    reference = deep_gemm_fp8_o_proj(o, positions, cos, wo_a, wo_b, **kwargs)

    candidate_value = OProjectionAdapter()(
        o,
        positions,
        cos,
        nn.Parameter(master_a),
        nn.Parameter(master_b),
        n_groups=1,
        heads_per_group=2,
        nope_dim=64,
        rope_dim=64,
        o_lora_rank=128,
    )
    torch.testing.assert_close(candidate_value, reference, rtol=0, atol=0)

from __future__ import annotations

import importlib.util
import pytest
import torch
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.primitive.kernels import o_projection_visible


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

    candidate_value = o_projection_visible(
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

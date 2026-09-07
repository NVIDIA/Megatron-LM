from __future__ import annotations

import importlib.util
import pytest
import torch
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
    _inverse_rope,
    _o_projection,
    o_projection_visible,
)


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


@pytest.mark.gpus(1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_o_projection_explicit_grouped_gemm_vjp_matches_functional_gradients() -> None:
    torch.manual_seed(31)
    device = "cuda"
    tokens = 7
    n_groups = 2
    heads_per_group = 2
    nope_dim = 64
    rope_dim = 64
    head_dim = nope_dim + rope_dim
    o_lora_rank = 128
    hidden_size = 256

    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    cos_sin = torch.randn(32, rope_dim, dtype=torch.float32, device=device)
    o = torch.randn(
        tokens,
        n_groups * heads_per_group,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    wo_a = nn.Parameter(
        torch.randn(
            n_groups * o_lora_rank,
            heads_per_group * head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
    )
    wo_b = nn.Parameter(
        torch.randn(
            hidden_size,
            n_groups * o_lora_rank,
            dtype=torch.bfloat16,
            device=device,
        )
    )

    def functional(o_, wa_, wb_):
        inverse = _inverse_rope(o_, positions, cos_sin, nope_dim, rope_dim)
        grouped = inverse.reshape(tokens, n_groups, -1)
        wa = wa_.reshape(n_groups, o_lora_rank, -1)
        z = torch.einsum("tgd,grd->tgr", grouped, wa)
        return z.flatten(1) @ wb_.T

    def visible_with_intermediate(o_, wa_, wb_):
        inverse = _inverse_rope(o_, positions, cos_sin, nope_dim, rope_dim)
        grouped = inverse.reshape(tokens, n_groups, -1)
        wa = wa_.reshape(n_groups, o_lora_rank, -1)
        z = torch.einsum("tgd,grd->tgr", grouped, wa).flatten(1)
        return z @ wb_.T, z

    reference_o = o.detach().clone().requires_grad_(True)
    reference_a = wo_a.detach().clone().requires_grad_(True)
    reference_b = wo_b.detach().clone().requires_grad_(True)
    reference = functional(reference_o, reference_a, reference_b)

    candidate = _o_projection(
        visible_with_intermediate,
        o,
        wo_a,
        wo_b,
        positions=positions,
        cos_sin_cache=cos_sin,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        o_lora_rank=o_lora_rank,
    )
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    grad_output = torch.randn_like(candidate)
    reference.backward(grad_output)
    candidate.backward(grad_output)
    for actual, expected in (
        (o.grad, reference_o.grad),
        (wo_a.grad, reference_a.grad),
        (wo_b.grad, reference_b.grad),
    ):
        assert actual is not None
        assert expected is not None
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

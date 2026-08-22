"""vLLM-visible Q/K cache insertion and output projection."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from vllm import envs
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.utils.deep_gemm import fp8_gemm_nt

from megatron.lite.model.deepseek_v4.vllm.primitive.deployment_fp8 import (
    quantize_block_fp8_weight,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import visible_functional_vjp


def insert_qkv(
    q: Tensor,
    kv: Tensor,
    cache: Tensor,
    slot_mapping: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    *,
    eps: float,
    block_size: int,
    padded_heads: int,
) -> Tensor:
    return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
    )


def o_projection_visible(
    o: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    wo_a: Tensor,
    wo_b: Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> Tensor:
    with torch.no_grad():
        canonical_wa = quantize_block_fp8_weight(wo_a)
        wa_q, wa_s = deepgemm_post_process_fp8_weight_block(
            wq=canonical_wa.qweight,
            ws=canonical_wa.scales,
            quant_block_shape=(128, 128),
            use_e8m0=True,
            is_bmm=True,
            bmm_batch_size=n_groups,
        )
        packed_wa = type("_PackedGroupedWeight", (), {})()
        packed_wa.weight = wa_q
        packed_wa.weight_scale = wa_s

        canonical_wb = quantize_block_fp8_weight(wo_b)
        wb_q, wb_s = deepgemm_post_process_fp8_weight_block(
            wq=canonical_wb.qweight,
            ws=canonical_wb.scales,
            quant_block_shape=(128, 128),
            use_e8m0=True,
        )

    def packed_wb(value: Tensor) -> Tensor:
        aligned = bool(envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES)
        aq, a_s = per_token_group_quant_fp8(
            value,
            128,
            use_ue8m0=True,
            column_major_scales=True,
            tma_aligned_scales=aligned,
        )
        output = torch.empty(
            value.shape[0], wb_q.shape[0], dtype=torch.bfloat16, device=value.device
        )
        fp8_gemm_nt(
            (aq, a_s),
            (wb_q, wb_s),
            output,
            is_deep_gemm_e8m0_used=True,
        )
        return output

    with torch.no_grad():
        recipe, aligned = compute_fp8_einsum_recipe()
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            packed_wa,
            packed_wb,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
            einsum_recipe=recipe,
            tma_aligned_scales=aligned,
        )


def _inverse_rope(o, positions, cache, nope_dim, rope_dim):
    prefix, rope = o[..., :nope_dim], o[..., nope_dim : nope_dim + rope_dim]
    selected = cache.index_select(0, positions.long()).float()
    cos = selected[..., : rope_dim // 2].unsqueeze(-2)
    sin = selected[..., rope_dim // 2 : rope_dim].unsqueeze(-2)
    even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
    rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
    return torch.cat((prefix.float(), rotated.flatten(-2)), dim=-1)


def o_projection(
    visible_op: Callable,
    o: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    *,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
):
    def functional(o_, wa_, wb_):
        inverse = _inverse_rope(o_, positions, cos_sin_cache, nope_dim, rope_dim)
        grouped = inverse.reshape(inverse.shape[0], n_groups, -1)
        wa = wa_.float().reshape(n_groups, o_lora_rank, -1)
        z = torch.einsum("tgd,grd->tgr", grouped, wa)
        return F.linear(z.flatten(1), wb_.float()).to(o_.dtype)

    return visible_functional_vjp(
        visible_op, functional, (o, wo_a, wo_b), version_indices=(1, 2)
    )

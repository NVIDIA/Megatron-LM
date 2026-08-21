"""Thin calls into the vLLM DeepSeek-V4 kernels used by mLite training."""

from __future__ import annotations

import torch
from torch import Tensor
from vllm import envs
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.utils.deep_gemm import fp8_gemm_nt
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    quantize_block_fp8_weight,
)


_MHC_ENTRIES = {
    "pre": mhc_pre_tilelang,
    "pre_broadcast": mhc_pre_broadcast_tilelang,
    "post": mhc_post_tilelang,
    "post_pre": mhc_fused_post_pre_tilelang,
    "head": hc_head_fused_kernel_tilelang,
}


def mhc_kernel(name: str, *args, **kwargs):
    return _MHC_ENTRIES[name](*args, **kwargs)


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
    padded_heads: int | None = None,
    q_out: Tensor | None = None,
    **_unused,
) -> Tensor:
    if padded_heads is None:
        raise ValueError("padded_heads is required")
    if q_out is not None:
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(
            q,
            kv,
            q_out,
            cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            padded_heads,
            eps,
            block_size,
        )
        return q_out
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


def sparse_attention(
        q: Tensor,
        kv: Tensor,
        indices: Tensor,
        *,
        sm_scale: float,
        attn_sink: Tensor | None = None,
        topk_length: Tensor | None = None,
        out: Tensor | None = None,
    ):
    return flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
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

    with torch.no_grad():
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
                value.shape[0],
                wb_q.shape[0],
                dtype=torch.bfloat16,
                device=value.device,
            )
            fp8_gemm_nt(
                (aq, a_s),
                (wb_q, wb_s),
                output,
                is_deep_gemm_e8m0_used=True,
            )
            return output

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

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer spec helpers for Qwen3.5-VL vision encoder and language decoder.

Provides ModuleSpec builders that define the transformer layer composition.
Both the standalone and MIMO training paths import from here.
"""

from typing import Optional

from examples.multimodal_dev.models.base import _NO_CP_GROUP
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


def _apply_rope_fp32(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
    """Apply rotary positional embedding in fp32, then cast back to original dtype.

    Mirrors ``Qwen3VLSelfAttention.apply_rotary_pos_emb_absolute`` in Megatron-Bridge
    with ``apply_rotary_pos_emb_in_fp32=True``.
    """
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    orig_dtype = t.dtype
    if (
        cu_seqlens is not None
        and getattr(config, "apply_rope_fusion", False)
        and getattr(config, "mrope_section", None) is not None
        and getattr(config, "rotary_interleaved", False) is False
        and getattr(config, "multi_latent_attention", False) is False
        and mscale == 1.0
        and t.dim() == 3
        and freqs.dim() == 4
        and freqs.shape[0] == 3
        and cp_group is not None
        and rope_utils.fused_apply_mrope_thd is not None
        and rope_utils.get_fused_mrope_thd_unavailable_reason is not None
    ):
        unavailable_reason = rope_utils.get_fused_mrope_thd_unavailable_reason(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            cp_size=cp_group.size(),
            cp_rank=cp_group.rank(),
        )
        if unavailable_reason is None:
            return rope_utils.fused_apply_mrope_thd(
                t,
                cu_seqlens,
                freqs,
                config.mrope_section,
                interleaved_mrope=config.mrope_interleaved,
                rotary_interleaved=config.rotary_interleaved,
                cp_size=cp_group.size(),
                cp_rank=cp_group.rank(),
                fp32_compute=True,
            )

    t_fp32 = t.float()
    out = apply_rotary_pos_emb(
        t_fp32,
        freqs,
        config=config,
        cu_seqlens=cu_seqlens,
        mscale=mscale,
        cp_group=cp_group,
        mla_rotary_interleaved=getattr(config, 'multi_latent_attention', False),
    )
    return out.to(orig_dtype)


def _apply_rope_fp32_no_cp(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
    """Same as ``_apply_rope_fp32`` but forces CP-size=1.

    The vision encoder uses THD packed sequences for variable-resolution
    images.  When the language model uses CP>1, the global CP group would
    incorrectly split the vision seqlens.  This wrapper substitutes a
    trivial group so the vision RoPE sees the full packed sequence.
    """
    range_name = "qwen35_vl.vision_encoder.rope_apply"
    nvtx_range_push(range_name)
    try:
        return _apply_rope_fp32(
            t,
            freqs,
            config,
            cu_seqlens,
            mscale,
            cp_group=_NO_CP_GROUP,
        )
    finally:
        nvtx_range_pop(range_name)


class Qwen35VLVisionSelfAttention(SelfAttention):
    """ViT self-attention with RoPE applied in fp32.

    Matches Bridge's ``Qwen3VLSelfAttention`` behaviour when
    ``apply_rotary_pos_emb_in_fp32=True``:  query and key are cast to float32
    before the rotary multiply and cast back to bf16 afterwards.  The
    monkey-patch approach avoids duplicating the 300-line ``SelfAttention.forward``
    while keeping the change local to this class.
    """

    def forward(self, *args, **kwargs):
        import megatron.core.transformer.attention as _attn_mod

        _orig = _attn_mod.apply_rotary_pos_emb
        _attn_mod.apply_rotary_pos_emb = _apply_rope_fp32_no_cp
        try:
            return super().forward(*args, **kwargs)
        finally:
            _attn_mod.apply_rotary_pos_emb = _orig


def get_qwen35_vl_language_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Transformer block spec for the Qwen3.5-VL language decoder.

    Uses the experimental attention variant infrastructure to build hybrid
    GatedDeltaNet + full-attention layers with optional MoE interleaving.

    Args:
        config: Language decoder TransformerConfig.
        vp_stage: Virtual pipeline stage.
        pp_rank: Pipeline parallel rank.

    Returns:
        TransformerBlockSubmodules with per-layer specs.
    """
    return get_transformer_block_with_experimental_attention_variant_spec(
        config=config,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )


def get_qwen35_vl_vision_spec() -> ModuleSpec:
    """ModuleSpec for vision encoder transformer layers.

    Uses ``TEDotProductAttention`` which supports packed-sequence (THD)
    attention via ``PackedSeqParams`` for variable-length images.

    ``Qwen35VLVisionSelfAttention`` replaces the default ``SelfAttention`` so
    that RoPE is applied in fp32, matching Bridge's
    ``apply_rotary_pos_emb_in_fp32=True`` behaviour.
    """
    spec = get_vit_layer_with_transformer_engine_spec()
    spec.submodules.self_attention.module = Qwen35VLVisionSelfAttention
    return spec

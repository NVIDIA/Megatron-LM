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


def _apply_rope_fp32(
    t,
    freqs,
    config,
    cu_seqlens=None,
    mscale=1.0,
    cp_group=None,
    mla_rotary_interleaved=None,
    inverse=False,
    mla_output_remove_interleaving=False,
    max_seqlen=None,
):
    """Apply rotary positional embedding in fp32, then cast back to original dtype.

    Mirrors ``Qwen3VLSelfAttention.apply_rotary_pos_emb_absolute`` in Megatron-Bridge
    with ``apply_rotary_pos_emb_in_fp32=True``.

    The signature matches ``rope_utils.apply_rotary_pos_emb`` parameter-for-parameter so the
    wrapper can substitute it positionally. One deliberate divergence: the
    ``mla_rotary_interleaved`` *default* is ``None`` (config-derived, matching this wrapper's
    legacy behaviour) where upstream's signature default is ``False``; explicit arguments and
    explicit ``None`` behave identically to upstream.
    """
    from megatron.core.models.common.embeddings import rope_utils
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    orig_dtype = t.dtype
    if mla_rotary_interleaved is None:
        mla_rotary_interleaved = getattr(config, 'multi_latent_attention', False)
    if (
        cu_seqlens is not None
        and getattr(config, "apply_rope_fusion", False)
        and getattr(config, "mrope_section", None) is not None
        and getattr(config, "rotary_interleaved", False) is False
        and getattr(config, "multi_latent_attention", False) is False
        # The fused mRoPE kernel implements neither the inverse rotation nor MLA output
        # de-interleaving, and takes no mla_rotary_interleaved argument. Taking this path
        # with any of them requested would silently compute the wrong rotation, so fall
        # through to the unfused wrapper instead.
        and not inverse
        and not mla_output_remove_interleaving
        and mla_rotary_interleaved is False
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
        mla_rotary_interleaved=mla_rotary_interleaved,
        inverse=inverse,
        mla_output_remove_interleaving=mla_output_remove_interleaving,
        max_seqlen=max_seqlen,
    )
    return out.to(orig_dtype)


def _apply_rope_fp32_no_cp(
    t,
    freqs,
    config,
    cu_seqlens=None,
    mscale=1.0,
    cp_group=None,
    mla_rotary_interleaved=None,
    inverse=False,
    mla_output_remove_interleaving=False,
    max_seqlen=None,
):
    """Same as ``_apply_rope_fp32`` but forces CP-size=1.

    The vision encoder uses THD packed sequences for variable-resolution
    images.  When the language model uses CP>1, the global CP group would
    incorrectly split the vision seqlens.  This wrapper substitutes a
    trivial group so the vision RoPE sees the full packed sequence; the
    caller's ``cp_group`` argument is intentionally ignored.
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
            mla_rotary_interleaved=mla_rotary_interleaved,
            inverse=inverse,
            mla_output_remove_interleaving=mla_output_remove_interleaving,
            max_seqlen=max_seqlen,
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

        # The fp32 cast rides on the module-level UNFUSED RoPE entry point. The fused-QKV
        # entry (config.fused_single_qkv_rope) cannot bypass it here: vision always runs THD
        # packed sequences, and packed_seq_params forces split_qkv=True in SelfAttention.
        # The cos/sin and flash-decode entries are inference-context-only and likewise
        # unreachable during vision training.
        # Save/restore of a module global: exception-safe and reentrancy-safe, but not
        # thread-safe — concurrent forwards from separate Python threads would race.
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

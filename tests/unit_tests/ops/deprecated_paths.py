# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated -> canonical module table for the SSM / sparse-attention move.

Generated alongside the forwarder modules; keep the two in sync.
"""

FORWARDED = {
    "megatron.core.ssm.causal_conv1d": ("megatron.core.ops.ssm.common.causal_conv1d_cp",),
    "megatron.core.ssm.gated_delta_product": ("megatron.core.ops.ssm.gdp.mixer",),
    "megatron.core.ssm.gdn_layer_config": ("megatron.core.transformer.gdn_layer_config",),
    "megatron.core.ssm.gdp_context_parallel": ("megatron.core.ops.ssm.gdp.context_parallel",),
    "megatron.core.ssm.mamba_block": ("megatron.core.models.hybrid.hybrid_block",),
    "megatron.core.ssm.mamba_context_parallel": ("megatron.core.ops.ssm.mamba2.context_parallel",),
    "megatron.core.ssm.mamba_hybrid_layer_allocation": (
        "megatron.core.models.hybrid.hybrid_layer_allocation",
    ),
    "megatron.core.ssm.mamba_layer": ("megatron.core.transformer.mamba_layer",),
    "megatron.core.ssm.mamba_layer_config": ("megatron.core.transformer.mamba_layer_config",),
    "megatron.core.ssm.mamba_mixer": ("megatron.core.ops.ssm.mamba2.mixer",),
    "megatron.core.ssm.mlp_layer": ("megatron.core.transformer.mlp_layer",),
    "megatron.core.ssm.mlp_layer_config": ("megatron.core.transformer.mlp_layer_config",),
    "megatron.core.ssm.packed_seq_helpers": ("megatron.core.ops.ssm.common.packed_seq",),
    "megatron.core.ssm.ssm_inference": (
        "megatron.core.ops.ssm.common.inference",
        "megatron.core.inference.ssm_config",
    ),
    "megatron.core.ssm.triton_cache_manager": ("megatron.core.ops.ssm.triton_cache_manager",),
    "megatron.core.ssm.utils": ("megatron.core.ops.ssm.common.checkpointing",),
    "megatron.core.ssm.context_parallel": ("megatron.core.ops.ssm.context_parallel",),
    "megatron.core.ssm.context_parallel.chunkwise": (
        "megatron.core.ops.ssm.context_parallel.chunkwise",
    ),
    "megatron.core.ssm.context_parallel.gdp": ("megatron.core.ops.ssm.context_parallel.gdp",),
    "megatron.core.ssm.context_parallel.gdp_common": (
        "megatron.core.ops.ssm.context_parallel.gdp_common",
    ),
    "megatron.core.ssm.context_parallel.gdp_cutedsl": (
        "megatron.core.ops.ssm.context_parallel.gdp_cutedsl",
    ),
    "megatron.core.ssm.gated_delta_net": (
        "megatron.core.ops.ssm.gated_delta.modules",
        "megatron.core.ops.ssm.gated_delta.gdn",
        "megatron.core.ops.ssm.gated_delta.gdn2",
        "megatron.core.ops.ssm.gated_delta.common",
    ),
    "megatron.core.ssm.gated_delta_net.common": ("megatron.core.ops.ssm.gated_delta.common",),
    "megatron.core.ssm.gated_delta_net.gdn": ("megatron.core.ops.ssm.gated_delta.gdn",),
    "megatron.core.ssm.gated_delta_net.gdn2": ("megatron.core.ops.ssm.gated_delta.gdn2",),
    "megatron.core.ssm.ops": (
        "megatron.core.ops.ssm.mamba2.ssd_combined",
        "megatron.core.ops.ssm.common.causal_conv1d_varlen",
    ),
    "megatron.core.ssm.ops.common": ("megatron.core.ops.ssm.common",),
    "megatron.core.ssm.ops.common.causal_conv1d_triton": (
        "megatron.core.ops.ssm.common.causal_conv1d_triton",
    ),
    "megatron.core.ssm.ops.common.causal_conv1d_varlen": (
        "megatron.core.ops.ssm.common.causal_conv1d_varlen",
    ),
    "megatron.core.ssm.ops.common.determinism": ("megatron.core.ops.ssm.common.determinism",),
    "megatron.core.ssm.ops.common.intermediate_extraction": (
        "megatron.core.ops.ssm.common.intermediate_extraction",
    ),
    "megatron.core.ssm.ops.mamba2": ("megatron.core.ops.ssm.mamba2",),
    "megatron.core.ssm.ops.mamba2.batch_invariant_decode": (
        "megatron.core.ops.ssm.mamba2.batch_invariant_decode",
    ),
    "megatron.core.ssm.ops.mamba2.mamba_ssm": ("megatron.core.ops.ssm.mamba2.mamba_ssm",),
    "megatron.core.ssm.ops.mamba2.ssd_bmm": ("megatron.core.ops.ssm.mamba2.ssd_bmm",),
    "megatron.core.ssm.ops.mamba2.ssd_chunk_scan": ("megatron.core.ops.ssm.mamba2.ssd_chunk_scan",),
    "megatron.core.ssm.ops.mamba2.ssd_chunk_state": (
        "megatron.core.ops.ssm.mamba2.ssd_chunk_state",
    ),
    "megatron.core.ssm.ops.mamba2.ssd_combined": ("megatron.core.ops.ssm.mamba2.ssd_combined",),
    "megatron.core.ssm.ops.mamba2.ssd_state_passing": (
        "megatron.core.ops.ssm.mamba2.ssd_state_passing",
    ),
    "megatron.core.ssm.ops.gdp": ("megatron.core.ops.ssm.gdp",),
    "megatron.core.ssm.ops.gdp.chunk": ("megatron.core.ops.ssm.gdp.chunk",),
    "megatron.core.ssm.ops.gdp.chunk_h": ("megatron.core.ops.ssm.gdp.chunk_h",),
    "megatron.core.ssm.ops.gdp.chunk_o": ("megatron.core.ops.ssm.gdp.chunk_o",),
    "megatron.core.ssm.ops.gdp.common": ("megatron.core.ops.ssm.gdp.common",),
    "megatron.core.ssm.ops.gdp.cumsum": ("megatron.core.ops.ssm.gdp.cumsum",),
    "megatron.core.ssm.ops.gdp.decode_prepare": ("megatron.core.ops.ssm.gdp.decode_prepare",),
    "megatron.core.ssm.ops.gdp.fused_recurrent": ("megatron.core.ops.ssm.gdp.fused_recurrent",),
    "megatron.core.ssm.ops.gdp.metadata": ("megatron.core.ops.ssm.gdp.metadata",),
    "megatron.core.ssm.ops.gdp.scaled_dot_kkt": ("megatron.core.ops.ssm.gdp.scaled_dot_kkt",),
    "megatron.core.ssm.ops.gdp.solve_tril": ("megatron.core.ops.ssm.gdp.solve_tril",),
    "megatron.core.ssm.ops.gdp.wy_fast": ("megatron.core.ops.ssm.gdp.wy_fast",),
    "megatron.core.transformer.experimental_attention_variant.absorbed_mla": (
        "megatron.core.ops.attention.mla",
    ),
    "megatron.core.transformer.experimental_attention_variant.csa": (
        "megatron.core.ops.attention.csa.modules",
    ),
    "megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention": (
        "megatron.core.ops.attention.dsv4",
    ),
    "megatron.core.transformer.experimental_attention_variant."
    "deepseek_v4_hybrid_attention_module_specs": (
        "megatron.core.models.gpt.deepseek_v4_hybrid_attention_module_specs",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa": (
        "megatron.core.ops.attention.dsa.modules",
        "megatron.core.transformer.dsa_loss",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa_layer_config": (
        "megatron.core.transformer.dsa_layer_config",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels": (
        "megatron.core.ops.attention.dsa.dsa_cudnn_kernels",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa_indexer_loss": (
        "megatron.core.ops.attention.dsa.dsa_indexer_loss",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa_kernels": (
        "megatron.core.ops.attention.dsa.dsa_kernels",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa_layout": (
        "megatron.core.ops.attention.dsa.dsa_layout",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa_masking": (
        "megatron.core.ops.attention.dsa.dsa_masking",
    ),
    "megatron.core.transformer.experimental_attention_variant.dsa_tilelang_kernels": (
        "megatron.core.ops.attention.dsa.dsa_tilelang_kernels",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops": (
        "megatron.core.ops.attention.dsa.kernels",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.indexer": (
        "megatron.core.ops.attention.dsa.kernels.indexer",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.sparse_mla": (
        "megatron.core.ops.attention.dsa.kernels.sparse_mla",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.tilelang_dsa": (
        "megatron.core.ops.attention.dsa.kernels.tilelang_dsa",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.tilelang_indexer_bwd": (
        "megatron.core.ops.attention.dsa.kernels.tilelang_indexer_bwd",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.tilelang_indexer_fwd": (
        "megatron.core.ops.attention.dsa.kernels.tilelang_indexer_fwd",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.tilelang_indexer_loss": (
        "megatron.core.ops.attention.dsa.kernels.tilelang_indexer_loss",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.tilelang_sparse_mla_bwd": (
        "megatron.core.ops.attention.dsa.kernels.tilelang_sparse_mla_bwd",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.tilelang_sparse_mla_fwd": (
        "megatron.core.ops.attention.dsa.kernels.tilelang_sparse_mla_fwd",
    ),
    "megatron.core.transformer.experimental_attention_variant.ops.tilelang_utils": (
        "megatron.core.ops.attention.dsa.kernels.tilelang_utils",
    ),
}

PACKAGE_MARKERS = ("megatron.core.ssm", "megatron.core.transformer.experimental_attention_variant",)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deprecated -> canonical module table for the SSM / sparse-attention move.

Generated alongside the forwarder modules; keep the two in sync.
"""

FORWARDED = {
    "megatron.core.ssm.causal_conv1d": ("megatron.core.ops.ssm.common.causal_conv1d_cp",),
    "megatron.core.ssm.utils": ("megatron.core.ops.ssm.common.checkpointing",),
    "megatron.core.ssm.ssm_inference": ("megatron.core.ops.ssm.common.inference",),
    "megatron.core.ssm.packed_seq_helpers": ("megatron.core.ops.ssm.common.packed_seq",),
    "megatron.core.ssm.triton_cache_manager": ("megatron.core.ops.ssm.triton_cache_manager",),
    "megatron.core.ssm.gated_delta_product": ("megatron.core.ops.ssm.gdp.mixer",),
    "megatron.core.ssm.gdp_context_parallel": ("megatron.core.ops.ssm.gdp.context_parallel",),
    "megatron.core.ssm.mamba_context_parallel": ("megatron.core.ops.ssm.mamba2.context_parallel",),
    "megatron.core.ssm.mamba_mixer": ("megatron.core.ops.ssm.mamba2.mixer",),
    "megatron.core.ssm.ops": ("megatron.core.ops.ssm",),
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
    "megatron.core.ssm.gated_delta_net": ("megatron.core.ops.ssm.gated_delta",),
    "megatron.core.ssm.gated_delta_net.common": ("megatron.core.ops.ssm.gated_delta.common",),
    "megatron.core.ssm.gated_delta_net.gdn": ("megatron.core.ops.ssm.gated_delta.gdn",),
    "megatron.core.ssm.gated_delta_net.gdn2": ("megatron.core.ops.ssm.gated_delta.gdn2",),
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
}

# Names each historical package re-exported, and the module that defines each. The
# canonical packages export nothing; these stay available on the deprecated paths only.
EXPORTS = {
    "megatron.core.ssm.ops": {
        "mamba_chunk_scan_combined_varlen": "megatron.core.ops.ssm.mamba2.ssd_combined",
        "causal_conv1d_varlen_fn": "megatron.core.ops.ssm.common.causal_conv1d_varlen",
    },
    "megatron.core.ssm.ops.gdp": {
        "chunk_gated_delta_product_varlen": "megatron.core.ops.ssm.gdp.chunk",
        "fused_recurrent_gated_delta_rule_update": "megatron.core.ops.ssm.gdp.fused_recurrent",
        "gdp_decode_prepare": "megatron.core.ops.ssm.gdp.decode_prepare",
        "build_gdp_chunk_descriptors": "megatron.core.ops.ssm.gdp.metadata",
        "max_gdp_chunk_counts": "megatron.core.ops.ssm.gdp.metadata",
    },
    "megatron.core.ssm.gated_delta_net": {
        "HAVE_FLA": "megatron.core.ops.ssm.gated_delta.common",
        "HAVE_FLA_GDN2": "megatron.core.ops.ssm.gated_delta.gdn2",
        "HAVE_TE_GDN": "megatron.core.extensions.transformer_engine",
        "GatedDeltaNet": "megatron.core.ops.ssm.gated_delta.gdn",
        "GatedDeltaNet2": "megatron.core.ops.ssm.gated_delta.gdn2",
        "GatedDeltaNetSubmodules": "megatron.core.ops.ssm.gated_delta.common",
        "causal_conv1d": "megatron.core.ops.ssm.gated_delta.common",
        "chunk_gated_delta_rule": "megatron.core.ops.ssm.gated_delta.common",
        "chunk_gdn2": "megatron.core.ops.ssm.gated_delta.gdn2",
        "get_parameter_local_cp": "megatron.core.ops.ssm.gated_delta.common",
        "l2norm": "megatron.core.ops.ssm.gated_delta.common",
        "tensor_a2a_cp2hp": "megatron.core.ops.ssm.gated_delta.common",
        "tensor_a2a_hp2cp": "megatron.core.ops.ssm.gated_delta.common",
        "torch_chunk_gated_delta_rule": "megatron.core.ops.ssm.gated_delta.gdn",
        "torch_chunk_gdn2": "megatron.core.ops.ssm.gated_delta.gdn2",
    },
}

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 configuration checks through the existing DSv4 path."""

from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsv4_hybrid_module_spec_for_backend,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.training.argument_utils import _resolve_dsa_kernel_backend_cli_default
from megatron.training.arguments import _add_network_size_args


def _make_config(
    *, params_dtype: torch.dtype = torch.float32, **overrides: Any
) -> MLATransformerConfig:
    """Create six layers covering SWA, Full2, Reuse2, Full1, Reindex1, and Reuse1."""
    values = dict(
        experimental_attention_variant="dsv4_hybrid",
        dsv4_version="v4.1",
        transformer_impl="local",
        use_cpu_initialization=True,
        params_dtype=params_dtype,
        bf16=params_dtype == torch.bfloat16,
        num_layers=6,
        hidden_size=32,
        num_attention_heads=4,
        q_lora_rank=16,
        v_head_dim=16,
        qk_pos_emb_head_dim=8,
        o_groups=2,
        o_lora_rank=8,
        qk_layernorm=True,
        layernorm_epsilon=1e-20,
        normalization="RMSNorm",
        rotary_base=10000,
        csa_compress_rotary_base=160000,
        original_max_position_embeddings=65536,
        rotary_scaling_factor=16,
        beta_fast=32,
        beta_slow=1,
        mscale=0,
        mscale_all_dim=0,
        csa_window_size=4,
        csa_compress_ratios=[0, 2, 2, 1, 1, 1],
        csa2_kv_source_layers=[1, 3],
        csa2_index_source_layers=[1, 3, 4],
        csa2_candidate_source_layer=3,
        csa2_candidate_topk_blocks=2,
        csa2_candidate_block_size=2,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=4,
        dsa_indexer_rotate_activation=False,
        enable_hyper_connections=True,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        mhc_epsilon=1e-6,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=48,
        moe_shared_expert_intermediate_size=48,
        moe_router_score_function="sqrtsoftplus",
        moe_router_dtype="fp32",
        moe_router_topk_scaling_factor=1.5,
        moe_router_enable_expert_bias=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        activation_func_clamp_value=10,
        add_bias_linear=False,
        hidden_dropout=0,
        attention_dropout=0,
    )
    values.update(overrides)
    return MLATransformerConfig(**values)


def test_tiny_config_accepts_source_reuse_and_ratio_transition():
    config = _make_config()
    assert config.csa_compress_ratios == [0, 2, 2, 1, 1, 1]
    assert config.csa2_kv_source_layers == [1, 3]
    assert config.csa2_index_source_layers == [1, 3, 4]
    assert config.hidden_size == 32
    assert config.params_dtype == torch.float32 and not config.bf16


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"csa_compress_ratios": [0, 2]}, "exactly num_layers"),
        ({"csa_compress_ratios": [0, 4, 4, 1, 1, 1]}, "only 0, 1, or 2"),
        ({"csa2_kv_source_layers": [1, 1, 3]}, "strictly increasing"),
        ({"csa2_index_source_layers": [1, 3, 6]}, "zero-based"),
        ({"csa2_index_source_layers": [0, 1, 3, 4]}, "SWA-only"),
        ({"csa2_index_source_layers": [1, 4]}, "must also be an index source"),
        ({"csa2_kv_source_layers": [3]}, "no preceding Full"),
        ({"csa_compress_ratios": [0, 2, 1, 1, 1, 1]}, "compression ratio"),
        ({"csa2_candidate_source_layer": 4}, "must be a Full"),
        ({"csa2_candidate_source_layer": 1}, "cannot cross a later KV source"),
        ({"csa2_candidate_topk_blocks": 1}, "capacity"),
        ({"csa2_candidate_source_layer": None}, "Disabled CSA2 candidates"),
        ({"dsa_kernel_backend": "cudnn"}, "dsa_kernel_backend='none'"),
        ({"gradient_accumulation_fusion": True}, "gradient_accumulation_fusion=False"),
        ({"use_fused_mhc": True}, "use_fused_mhc=False"),
        ({"dsa_indexer_rotate_activation": True}, "Hadamard"),
        ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size=1"),
        ({"qk_pos_emb_head_dim": 7}, "rotary dimension"),
        ({"dsa_indexer_head_dim": 4}, "rotary dimension"),
        ({"o_groups": 3}, "divisible by o_groups"),
        ({"q_lora_rank": None}, "positive integer q_lora_rank"),
        ({"rotary_interleaved": True}, "rotary_interleaved does not work"),
        ({"mscale": 1}, "amplitude scaling"),
        ({"layernorm_epsilon": float("nan")}, "positive finite"),
    ],
)
def test_reject_invalid_v41_configs(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_config(**overrides)


def test_candidates_can_be_disabled():
    config = _make_config(
        csa2_candidate_source_layer=None, csa2_candidate_topk_blocks=0, csa2_candidate_block_size=0
    )
    assert config.csa2_candidate_source_layer is None


def test_v41_cannot_build_legacy_attention_spec():
    with pytest.raises(NotImplementedError, match="CSA2 attention"):
        get_dsv4_hybrid_module_spec_for_backend(_make_config())


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_v4_config_remains_unchanged(ratio):
    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        experimental_attention_variant="dsv4_hybrid",
        csa_compress_ratios=[ratio],
    )
    assert config.dsv4_version == "v4"
    assert config.csa_compress_ratios == [ratio]
    assert config.csa_compress_rotary_base == 40000
    assert config.dsa_kernel_backend == "none"
    assert not config.rotary_interleaved


def test_v41_requires_explicit_version_and_mla_config():
    with pytest.raises(ValueError, match="requires experimental_attention_variant"):
        TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4, dsv4_version="v4.1")
    with pytest.raises(ValueError, match="requires MLATransformerConfig"):
        TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            experimental_attention_variant="dsv4_hybrid",
            multi_latent_attention=True,
            dsv4_version="v4.1",
        )
    with pytest.raises(ValueError, match="requires dsv4_version"):
        TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, csa2_kv_source_layers=[0]
        )


def test_v41_cli_fields():
    parser = ArgumentParser()
    _add_network_size_args(parser)
    args = parser.parse_args(
        [
            "--dsv4-version",
            "v4.1",
            "--csa2-kv-source-layers",
            "1",
            "3",
            "--csa2-index-source-layers",
            "1",
            "3",
            "4",
            "--csa2-candidate-source-layer",
            "3",
            "--csa2-candidate-topk-blocks",
            "2",
            "--csa2-candidate-block-size",
            "2",
            "--mhc-epsilon",
            "1e-6",
        ]
    )
    assert args.dsv4_version == "v4.1"
    assert args.csa2_kv_source_layers == [1, 3]
    assert args.csa2_index_source_layers == [1, 3, 4]
    assert (
        args.csa2_candidate_source_layer,
        args.csa2_candidate_topk_blocks,
        args.csa2_candidate_block_size,
    ) == (3, 2, 2)
    assert args.mhc_epsilon == 1e-6


@pytest.mark.parametrize("version, expected", [(None, "cudnn"), ("v4", "cudnn"), ("v4.1", "none")])
@pytest.mark.parametrize("explicit_backend", [None, "none"])
def test_cli_backend_defaults_preserve_v4(version, expected, explicit_backend):
    kwargs = {
        "experimental_attention_variant": "dsv4_hybrid",
        "dsa_kernel_backend": explicit_backend,
    }
    if version is not None:
        kwargs["dsv4_version"] = version
    _resolve_dsa_kernel_backend_cli_default(SimpleNamespace(), kwargs)
    assert kwargs["dsa_kernel_backend"] == (explicit_backend or expected)

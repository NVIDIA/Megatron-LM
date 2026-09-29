# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 configuration checks through the existing DSv4 path."""

from argparse import ArgumentParser
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

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
        transformer_impl="transformer_engine",
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
        # This recipe opts in explicitly; model-version selection does not enable it.
        mhc_single_pass=overrides.get("enable_hyper_connections", True),
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
        ({"dsa_kernel_backend": "tilelang"}, "dsa_kernel_backend='none' or 'cudnn'"),
        ({"dsa_indexer_precision": "mxfp8"}, "MXFP8 indexers require"),
        ({"mla_down_proj_fusion": True}, "mla_down_proj_fusion=False"),
        ({"masked_softmax_fusion": True}, "masked_softmax_fusion=False"),
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


@pytest.mark.parametrize("method", ["uniform", "block"])
@pytest.mark.parametrize("num_layers", [0, -1, 1.5, True])
def test_v41_full_recompute_requires_positive_integer_chunk_size(method, num_layers):
    with pytest.raises(ValueError, match="positive integer recompute_num_layers"):
        _make_config(
            recompute_granularity="full", recompute_method=method, recompute_num_layers=num_layers
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"cuda_graph_impl": "local"}, "require cuda_graph_impl='transformer_engine'"),
        ({"cuda_graph_impl": "full_iteration"}, "require cuda_graph_impl='transformer_engine'"),
        ({"mhc_single_pass": False}, "require single-pass mHC"),
        (
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            },
            "do not yet support full recompute",
        ),
    ],
)
def test_v41_cuda_graph_rejects_unimplemented_state_lifetimes(options, message):
    values = dict(cuda_graph_impl="transformer_engine", cuda_graph_modules=["attn"])
    values.update(options)
    with pytest.raises(ValueError, match=message):
        _make_config(**values)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"params_dtype": torch.float32}, "BF16 parameters"),
        ({"v_head_dim": 16}, "v_head_dim=512"),
        ({}, "SM90"),
    ],
)
def test_v41_fused_attention_validates_kernel_requirements(monkeypatch, overrides, message):
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config._validate_dsa_kernel_backend_dependencies",
        lambda backend: None,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    values = dict(params_dtype=torch.bfloat16, v_head_dim=512, dsa_kernel_backend="cudnn")
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        _make_config(**values)


@pytest.mark.parametrize(
    "overrides, sm, message",
    [
        ({"dsa_indexer_n_heads": 16}, (10, 0), "n_heads=32 or 64"),
        ({"dsa_indexer_head_dim": 64}, (10, 0), "head_dim=128"),
        ({"dsa_indexer_precision": "mxfp8"}, (9, 0), "SM100"),
        (
            {"dsa_indexer_precision": "mxfp8", "attention_backend": "unfused"},
            (10, 0),
            "fused attention",
        ),
    ],
)
def test_v41_fused_indexer_validates_requirements(monkeypatch, overrides, sm, message):
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config._validate_dsa_kernel_backend_dependencies",
        lambda backend: None,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: sm)
    values = dict(
        params_dtype=torch.bfloat16,
        v_head_dim=512,
        dsa_kernel_backend="cudnn",
        dsa_indexer_n_heads=32,
        dsa_indexer_head_dim=128,
    )
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        _make_config(**values)


def test_v41_does_not_implicitly_enable_single_pass():
    """Omitting the independent switch preserves legacy mHC even with CSA2."""
    recipe = _make_config()
    values = {
        field.name: getattr(recipe, field.name)
        for field in fields(recipe)
        if field.init and field.name != "mhc_single_pass"
    }
    config = MLATransformerConfig(**values)
    assert config.dsv4_version == "v4.1"
    assert config.enable_hyper_connections
    assert not config.mhc_single_pass


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
            "--mhc-single-pass",
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
    assert args.mhc_single_pass


@pytest.mark.parametrize("version", [None, "v4", "v4.1"])
@pytest.mark.parametrize("explicit_backend", [None, "none", "cudnn"])
def test_cli_backend_defaults_to_cudnn_for_all_dsv4_versions(version, explicit_backend):
    kwargs = {
        "experimental_attention_variant": "dsv4_hybrid",
        "dsa_kernel_backend": explicit_backend,
    }
    if version is not None:
        kwargs["dsv4_version"] = version
    _resolve_dsa_kernel_backend_cli_default(SimpleNamespace(), kwargs)
    assert kwargs["dsa_kernel_backend"] == (explicit_backend or "cudnn")

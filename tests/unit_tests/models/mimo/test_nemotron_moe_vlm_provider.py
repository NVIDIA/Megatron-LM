# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the Nemotron6-MoE VLM model provider.

Covers the post-parse derived knobs and the config parity gate: the from-args
language config must reproduce the reference Nemotron architecture
field-for-field, except the two fields that
``core_transformer_config_from_args`` correctly supplies (documented below).
"""

import argparse
import sys

import pytest

from examples.mimo.model_providers.nemotron_moe_vlm import (
    NEMOTRON_MODEL_PROVIDER,
    NEMOTRON_VISION_ENCODER_KEY,
    add_model_provider_args,
)

# (num_layers, hybrid_layer_pattern) is the ONLY architecture delta between the
# 20L and 54L Nemotron presets; every other field is shared. num_layers follows
# the pattern length (get_hybrid_total_layer_count): 20 and 54 layer-tokens.
_PRESET_20L = (20, "MEMEM*EMEMEM*EMEMEM*")
_PRESET_54L = (54, "MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME")

# Shared Nemotron6-MoE architecture (the reference fixture): the exact values the
# run script passes as stock CLI flags.
_NEMOTRON_ARCH = dict(
    hidden_size=2688,
    num_attention_heads=32,
    num_query_groups=8,
    ffn_hidden_size=1856,
    kv_channels=128,
    num_moe_experts=128,
    moe_router_topk=6,
    moe_grouped_gemm=True,
    moe_ffn_hidden_size=1856,
    moe_router_score_function="sigmoid",
    moe_router_topk_scaling_factor=2.5,
    moe_router_enable_expert_bias=True,
    moe_router_dtype="fp32",
    moe_router_load_balancing_type="seq_aux_loss",
    moe_router_fusion=True,
    moe_aux_loss_coeff=1.0e-4,
    moe_shared_expert_intermediate_size=3712,
    moe_shared_expert_overlap=True,
    moe_token_dispatcher_type="alltoall",
    moe_flex_dispatcher_backend="deepep",
    moe_permute_fusion=True,
    use_fused_weighted_squared_relu=True,
    mamba_num_heads=64,
    mamba_head_dim=64,
    mamba_num_groups=8,
    mamba_state_dim=128,
    linear_conv_kernel_dim=4,
    normalization="RMSNorm",
    init_method_std=0.0173,
    add_bias_linear=False,
    gated_linear_unit=False,
    calculate_per_token_loss=True,
    cross_entropy_loss_fusion=True,
)


def _parse(argv):
    """Parse provider args then backfill stock-arg defaults (simulating stock parse)."""
    parser = argparse.ArgumentParser()
    add_model_provider_args(parser)
    args = parser.parse_args(argv)
    for key, value in dict(hidden_size=None, num_layers=None, fp16=False).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def test_dynamic_resolution_defaults_on():
    args = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER])
    assert args.dynamic_resolution is True  # default-on; --no-dynamic-resolution disables
    off = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER, "--no-dynamic-resolution"])
    assert off.dynamic_resolution is False


def test_freeze_flags_drive_tower_freezing():
    # The freeze interface is the --freeze-* flags.
    args = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER, "--freeze-vit", "--freeze-lm"])
    assert args.freeze_vit is True
    assert args.freeze_lm is True
    assert args.freeze_projection is False


# --- Config parity gate (requires torch; runs in CI) ----------------------

pytest.importorskip("torch")


def _build_argv(num_layers, hybrid_pattern):
    """Full stock + provider CLI for the Nemotron preset (mirrors the run script)."""
    return [
        "--model-provider", NEMOTRON_MODEL_PROVIDER,
        "--pixel-shuffle",
        "--disable-vision-class-token",
        "--num-layers", str(num_layers),
        "--hybrid-layer-pattern", hybrid_pattern,
        "--hidden-size", "2688",
        "--num-attention-heads", "32",
        "--group-query-attention", "--num-query-groups", "8",
        "--ffn-hidden-size", "1856",
        "--kv-channels", "128",
        "--squared-relu",
        "--disable-bias-linear",
        "--normalization", "RMSNorm",
        "--init-method-std", "0.0173",
        "--num-experts", "128",
        "--moe-router-topk", "6",
        "--moe-grouped-gemm",
        "--moe-ffn-hidden-size", "1856",
        "--moe-router-score-function", "sigmoid",
        "--moe-router-topk-scaling-factor", "2.5",
        "--moe-router-enable-expert-bias",
        "--moe-router-dtype", "fp32",
        "--moe-router-load-balancing-type", "seq_aux_loss",
        "--moe-router-fusion",
        "--moe-aux-loss-coeff", "1e-4",
        "--moe-shared-expert-intermediate-size", "3712",
        "--moe-shared-expert-overlap",
        "--moe-token-dispatcher-type", "alltoall",
        "--moe-flex-dispatcher-backend", "deepep",
        "--moe-permute-fusion",
        "--use-fused-weighted-squared-relu",
        "--mamba-num-heads", "64",
        "--mamba-head-dim", "64",
        "--mamba-num-groups", "8",
        "--mamba-state-dim", "128",
        "--linear-conv-kernel-dim", "4",
        "--position-embedding-type", "none",
        "--calculate-per-token-loss",
        "--cross-entropy-loss-fusion",
        "--seq-length", "8192",
        "--max-position-embeddings", "8192",
        "--micro-batch-size", "1",
        "--vocab-size", "131072",
        "--tokenizer-type", "NullTokenizer",
        "--bf16",
    ]


def _parse_validate(argv):
    """Build args via the production pipeline so validate_args-derived fields
    (params_dtype, padded_vocab_size, ...) resolve exactly as in a real run.

    Mirrors examples/mimo/pretrain_mimo.py: parse_args -> validate_args. Runs at
    world_size=1, tp=pp=cp=1 so validate_args' divisibility checks pass with no
    distributed/mpu init.
    """
    from megatron.training.arguments import parse_args, validate_args

    saved = sys.argv
    sys.argv = ["pytest"] + argv
    try:
        args = parse_args(add_model_provider_args, ignore_unknown_args=True)
    finally:
        sys.argv = saved
    validate_args(args)
    return args


@pytest.mark.parametrize("num_layers,hybrid_pattern", [_PRESET_20L, _PRESET_54L])
def test_language_config_parity(num_layers, hybrid_pattern):
    """from-args language config == reference arch, modulo 2 documented fields.

    ``deallocate_pipeline_outputs`` and ``inference_sampling_seed`` are supplied
    by ``core_transformer_config_from_args`` and intentionally differ from a raw
    hardcoded config: deallocate=True is the stock-correct value (inert at PP=1,
    matches pretrain_gpt/vlm) and inference_sampling_seed tracks --seed. We assert
    those took the from-args values and exclude them from the field compare.
    """
    from examples.mimo.model_providers.nemotron_moe_vlm import nemotron_language_config

    args = _parse_validate(_build_argv(num_layers, hybrid_pattern))

    config = nemotron_language_config(args, tp_size=1, pp_size=1, ep_size=1, expt_tp_size=1)

    assert config.num_layers == num_layers
    assert config.is_hybrid_model is True
    for field, expected in _NEMOTRON_ARCH.items():
        assert getattr(config, field) == expected, field

    # The two documented from-args fields.
    assert config.deallocate_pipeline_outputs is True
    assert config.inference_sampling_seed == args.seed

    # Code-only overrides. (seq_length / max_position_embeddings are NOT
    # TransformerConfig fields; the seq-length contract is covered by
    # test_language_model_spec_builds_mamba via max_sequence_length.)
    assert config.position_embedding_type == "none"
    assert config.tensor_model_parallel_size == 1


def test_language_model_spec_builds_mamba():
    """language_model_spec returns a MambaModel spec carrying the preset config."""
    from examples.mimo.model_providers.nemotron_moe_vlm import language_model_spec
    from megatron.core.models.mamba.mamba_model import MambaModel

    args = _parse_validate(_build_argv(*_PRESET_20L))
    spec = language_model_spec(args, pg_collection=None, llm_grid=None)
    assert spec.module is MambaModel
    assert spec.params["config"].num_layers == 20
    assert spec.params["max_sequence_length"] == args.seq_length


def test_vision_submodules_spec_wires_radio_encoder():
    """vision_submodules_spec wires the RADIO encoder + affine projector, and the
    preset's pixel-shuffle / class-token-drop knobs reach the wrapper params."""
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        NEMOTRON_VISION_ENCODER_KEY,
        vision_submodules_spec,
    )
    from examples.mimo.model_providers.radio_encoder import RADIOEncoderWrapper

    args = _parse_validate(_build_argv(*_PRESET_20L))
    spec = vision_submodules_spec(args, pg_collection=None, encoder_grid=None)

    encoder = spec.submodules["encoders"][NEMOTRON_VISION_ENCODER_KEY]
    assert encoder.module is RADIOEncoderWrapper
    assert encoder.params["apply_pixel_shuffle"] is True
    assert encoder.params["drop_class_token"] is True

    projection = spec.submodules["input_projections"][0]
    assert projection.params["projector_type"] == "affine"

# A full model instantiation (constructing MambaModel / RADIOEncoderWrapper) needs
# TE + a distributed init and is left to the cog functional check.

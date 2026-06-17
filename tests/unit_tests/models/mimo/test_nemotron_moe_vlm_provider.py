# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the Nemotron6-MoE VLM model provider.

Covers the post-parse preset (vision/data-path knobs + stage freezing) and the
config parity gate: the from-args language config must reproduce the reference
Nemotron architecture field-for-field, except the two fields that
``core_transformer_config_from_args`` correctly supplies (documented below).
"""

import argparse
import sys

import pytest

from examples.mimo.model_providers.nemotron_moe_vlm import (
    NEMOTRON_MODEL_PROVIDER,
    add_model_provider_args,
    apply_model_provider_defaults,
    apply_training_stage,
    validate_model_provider_args,
)

# (num_layers, hybrid_layer_pattern) is the ONLY architecture delta between the
# 20L and 52L Nemotron presets; every other field is shared.
_PRESET_20L = (20, "MEMEM*EMEMEM*EMEMEM*")
_PRESET_52L = (52, "MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME")

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


def test_preset_sets_vision_and_datapath_knobs():
    args = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER, "--num-image-tiles", "12"])
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    assert args.pixel_shuffle is True
    assert args.disable_vision_class_token is True
    assert args.image_seq_length == 256 * 12
    assert args.dynamic_resolution is True
    assert args.use_tiling is False
    assert args.use_thumbnail is False
    # Default stage freezes only the vision tower.
    assert args.training_stage == "stage2"
    assert args.freeze_vit is True
    assert getattr(args, "freeze_lm", False) is False


def test_stage1_freezes_both_towers():
    args = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER, "--training-stage", "stage1"])
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    assert args.freeze_vit is True
    assert args.freeze_lm is True


def test_preset_skips_non_nemotron_provider():
    args = _parse(["--model-provider", "mock"])
    apply_model_provider_defaults(args)
    assert getattr(args, "image_seq_length", None) is None


def test_validate_rejects_out_of_range_image_token():
    args = _parse(["--model-provider", NEMOTRON_MODEL_PROVIDER])
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    args.padded_vocab_size = 131072
    args.image_token_id = 131072  # == vocab size -> out of range
    with pytest.raises(ValueError):
        validate_model_provider_args(args)


# --- Config parity gate (requires torch; runs in CI) ----------------------

pytest.importorskip("torch")


def _build_argv(num_layers, hybrid_pattern):
    """Full stock + provider CLI for the Nemotron preset (mirrors the run script)."""
    return [
        "--model-provider", NEMOTRON_MODEL_PROVIDER,
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

    Mirrors examples/mimo/pretrain_mimo.py::_parse_and_validate: parse_args ->
    prepare_model_provider_args (preset, before validate) -> validate_args. Runs
    at world_size=1, tp=pp=cp=1 so validate_args' divisibility checks pass with
    no distributed/mpu init.
    """
    from examples.mimo.model_providers.nemotron_moe_vlm import prepare_model_provider_args
    from megatron.training.arguments import parse_args, validate_args

    saved = sys.argv
    sys.argv = ["pytest"] + argv
    try:
        args = parse_args(add_model_provider_args, ignore_unknown_args=True)
    finally:
        sys.argv = saved
    prepare_model_provider_args(args)
    validate_args(args)
    return args


@pytest.mark.parametrize("num_layers,hybrid_pattern", [_PRESET_20L, _PRESET_52L])
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

    # Code-only overrides.
    assert config.position_embedding_type == "none"
    assert config.seq_length == 8192
    assert config.tensor_model_parallel_size == 1

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the Nemotron6-MoE VLM model provider (PR-E1).

These tests exercise the post-parse arg-reconciliation policy (the crux of the
PR): the provider must apply the ``nemotron-moe-vlm-20l`` preset onto the STOCK
argument namespace without re-declaring stock language args, force-overriding the
runtime-critical fields and only filling architecture fields the user left at the
stock default. No distributed init / GPUs required.
"""

import argparse

import pytest

from examples.mimo.model_providers.nemotron_moe_vlm import (
    NEMOTRON_20L_MODEL_PROVIDER,
    add_model_provider_args,
    apply_model_provider_defaults,
    apply_training_stage,
    validate_model_provider_args,
)

# Stock arguments.py defaults for the fields the preset reconciles against.
# (num_layers/hidden_size/... are required stock args -> default None;
#  moe_router_topk default 2, moe_grouped_gemm default False, num_experts None.)
_STOCK_DEFAULTS = dict(
    num_layers=None,
    hidden_size=None,
    num_attention_heads=None,
    num_query_groups=None,
    ffn_hidden_size=None,
    kv_channels=None,
    num_experts=None,
    moe_router_topk=2,
    moe_grouped_gemm=False,
    moe_shared_expert_intermediate_size=None,
    seq_length=None,
    max_position_embeddings=None,
    hybrid_layer_pattern=None,
    pixel_shuffle=False,
    disable_vision_class_token=False,
    use_tiling=False,
    use_thumbnail=False,
    fp16=False,
    fp32=False,
)


def _parse(argv):
    """Parse provider args then backfill stock-arg defaults (simulating stock parse)."""
    parser = argparse.ArgumentParser()
    add_model_provider_args(parser)
    args = parser.parse_args(argv)
    for key, value in _STOCK_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def test_preset_force_overrides_and_fills_20l():
    args = _parse(["--model-provider", NEMOTRON_20L_MODEL_PROVIDER, "--num-image-tiles", "12"])
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    # Force-overrides.
    assert args.num_layers == 20
    assert args.hybrid_layer_pattern == "MEMEM*EMEMEM*EMEMEM*"
    assert args.image_seq_length == 256 * 12
    assert args.pixel_shuffle is True
    assert args.disable_vision_class_token is True
    assert args.dynamic_resolution is True
    # Fill-if-default architecture sizes.
    assert args.hidden_size == 2688
    assert args.num_attention_heads == 32
    assert args.num_experts == 128
    assert args.moe_router_topk == 6
    assert args.moe_grouped_gemm is True
    assert args.seq_length == 8192
    # Stage-derived freezing.
    assert args.training_stage == "stage2"
    assert args.freeze_vit is True
    assert getattr(args, "freeze_lm", False) is False


def test_preset_respects_explicit_user_override():
    args = _parse(["--model-provider", NEMOTRON_20L_MODEL_PROVIDER, "--num-image-tiles", "12"])
    args.moe_router_topk = 4  # explicit user value (differs from stock default 2)
    apply_model_provider_defaults(args)
    assert args.moe_router_topk == 4  # not clobbered by the preset


def test_stage1_freezes_both_towers():
    args = _parse(
        ["--model-provider", NEMOTRON_20L_MODEL_PROVIDER, "--training-stage", "stage1"]
    )
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    assert args.freeze_vit is True
    assert args.freeze_lm is True


def test_validate_rejects_out_of_range_image_token():
    args = _parse(["--model-provider", NEMOTRON_20L_MODEL_PROVIDER])
    apply_model_provider_defaults(args)
    apply_training_stage(args)
    args.padded_vocab_size = 131072
    args.image_token_id = 131072  # == vocab size -> out of range
    with pytest.raises(ValueError):
        validate_model_provider_args(args)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-args (no-GPU) tests for the hetero grid arg group + validation."""

from __future__ import annotations

import argparse

import pytest

from examples.mimo.training.args import (
    add_hetero_grid_args,
    build_module_grid_specs,
    validate_hetero_grid_args,
)
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

WORLD_SIZE_8 = 8


def _parse(argv):
    """Parse only the hetero grid args from a token list."""
    parser = argparse.ArgumentParser()
    add_hetero_grid_args(parser)
    return parser.parse_args(argv)


def _layout_8gpu_20l(**overrides):
    """Canonical 8-GPU layout: encoder 0-3 (tp2/pp1/dp2), llm 4-7 (tp2/pp1/dp2/ep4)."""
    argv = (
        "--encoder-tp 2 --encoder-pp 1 --encoder-dp 2 "
        "--llm-offset 4 --llm-tp 2 --llm-pp 1 --llm-dp 2 --llm-ep 4"
    ).split()
    args = _parse(argv)
    # Stock args the validator reads but the grid parser does not own.
    args.micro_batch_size = 1
    args.num_microbatches = 2
    args.global_batch_size = None
    args.train_samples = None
    args.num_experts = 128
    args.vision_encoder_key = "radio_encoder"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_canonical_layout_validates_and_maps_specs():
    args = _layout_8gpu_20l()
    encoder_size, llm_size = validate_hetero_grid_args(args, WORLD_SIZE_8)
    assert (encoder_size, llm_size) == (4, 4)

    encoder_grid_spec, language_grid_spec = build_module_grid_specs(args, WORLD_SIZE_8)
    assert encoder_grid_spec.name == "radio_encoder"
    assert encoder_grid_spec.num_ranks == 4
    assert encoder_grid_spec.rank_offset == 0  # encoder span always starts at rank 0
    assert encoder_grid_spec.dp == 2  # derived: 4 // (tp2 * cp1 * pp1)
    assert language_grid_spec.name == MIMO_LANGUAGE_MODULE_KEY
    assert language_grid_spec.num_ranks == 4
    assert language_grid_spec.rank_offset == 4
    assert language_grid_spec.dp == 2
    # expt_tp defaults to 1 when --llm-expt-tp unset (ep=4 over 4 ranks needs expt_tp=1).
    assert language_grid_spec.expt_tp == 1


def test_overlapping_spans_raise():
    # llm-offset 2 makes llm ranks {2,3,4,5} overlap encoder ranks {0,1,2,3}.
    args = _layout_8gpu_20l(llm_offset=2)
    with pytest.raises(ValueError, match="disjoint"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_non_covering_spans_raise():
    # encoder 0-3 + llm 4-7 cover only 8 ranks; declare world_size 10 -> gap.
    args = _layout_8gpu_20l()
    with pytest.raises(ValueError, match="cover every torchrun rank"):
        validate_hetero_grid_args(args, 10)


def test_fanout_divisibility_raises():
    # mbs(1) * llm_dp(2) = 2 not divisible by encoder_dp(3).
    args = _layout_8gpu_20l(encoder_dp=3, micro_batch_size=1, llm_dp=2)
    with pytest.raises(ValueError, match="divisible by --encoder-dp"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_ep_divisibility_raises():
    # num_experts 128 not divisible by llm_ep 3.
    args = _layout_8gpu_20l(llm_ep=3, num_experts=128)
    with pytest.raises(ValueError, match="divisible by --llm-ep"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_cp_must_be_one():
    args = _layout_8gpu_20l(llm_cp=2)
    with pytest.raises(ValueError, match="CP=1 only"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_llm_only_requires_offset_zero():
    args = _layout_8gpu_20l(llm_only=True, llm_offset=4)
    with pytest.raises(ValueError, match="--llm-only requires --llm-offset 0"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_llm_only_covers_world():
    # llm tp2/pp1/dp2 = 4 ranks at offset 0; world_size 4 -> covers exactly, no encoder spec.
    args = _layout_8gpu_20l(llm_only=True, llm_offset=0, llm_ep=2, num_experts=128)
    encoder_size, llm_size = validate_hetero_grid_args(args, 4)
    assert (encoder_size, llm_size) == (0, 4)
    specs = build_module_grid_specs(args, 4)
    assert len(specs) == 1
    assert specs[0].name == MIMO_LANGUAGE_MODULE_KEY


def test_train_samples_resolves_iters():
    # gbs = mbs(1) * num_microbatches(2) * llm_dp(2) = 4; 17 samples -> ceil(17/4)=5.
    args = _layout_8gpu_20l(train_samples=17, train_iters=999)
    validate_hetero_grid_args(args, WORLD_SIZE_8)
    assert args.train_iters == 5

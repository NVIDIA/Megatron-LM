# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-args (no-GPU) tests for the hetero grid arg group + validation.

Covers:
  * the canonical 8-GPU 20L layout parses and validates OK
    (encoder tp2/pp1/dp2 = ranks 0-3, llm offset4 tp2/pp1/dp2/ep4 = ranks 4-7);
  * disjoint / covering violations raise;
  * fan-out divisibility + EP-divisibility checks raise on bad inputs;
  * build_module_grid_specs yields the correct num_ranks / rank_offset.

These run without torch.distributed -- they only exercise argparse + the
validation/spec-mapping helpers, so they are safe in the CPU CI lane.
"""

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
    argv = [
        "--encoder-offset", "0",
        "--encoder-tp", "2",
        "--encoder-pp", "1",
        "--encoder-dp", "2",
        "--llm-offset", "4",
        "--llm-tp", "2",
        "--llm-pp", "1",
        "--llm-dp", "2",
        "--llm-ep", "4",
    ]
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


def test_canonical_8gpu_20l_layout_validates():
    args = _layout_8gpu_20l()
    encoder_size, llm_size = validate_hetero_grid_args(args, WORLD_SIZE_8)
    assert encoder_size == 4  # tp2 * cp1 * pp1 * dp2
    assert llm_size == 4  # tp2 * cp1 * pp1 * dp2


def test_module_grid_specs_mapping():
    args = _layout_8gpu_20l()
    specs = build_module_grid_specs(args, WORLD_SIZE_8)
    assert len(specs) == 2
    encoder_spec, language_spec = specs

    assert encoder_spec.name == "radio_encoder"
    assert encoder_spec.num_ranks == 4
    assert encoder_spec.rank_offset == 0
    assert encoder_spec.tp == 2
    assert encoder_spec.dp == 2  # derived in __post_init__: 4 // (2*1*1)

    assert language_spec.name == MIMO_LANGUAGE_MODULE_KEY
    assert language_spec.num_ranks == 4
    assert language_spec.rank_offset == 4
    assert language_spec.tp == 2
    assert language_spec.ep == 4
    assert language_spec.dp == 2  # 4 // (2*1*1)
    # expt_tp defaults to 1 when --llm-expt-tp unset (the 20L script passes --llm-expt-tp 1);
    # ep=4 over 4 ranks requires expt_tp=1 so expt_tp*ep*pp divides num_ranks.
    assert language_spec.expt_tp == 1


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
    # mbs(1) * llm_dp(2) = 2 not divisible by encoder_dp(3); also keep spans valid
    # by widening encoder to tp1/dp3 isn't tiling-clean, so just trigger the
    # fan-out check before the span check by an indivisible encoder_dp.
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
    # llm tp2/pp1/dp2 = 4 ranks at offset 0; world_size 4 -> covers exactly.
    args = _layout_8gpu_20l(llm_only=True, llm_offset=0, llm_ep=2, num_experts=128)
    encoder_size, llm_size = validate_hetero_grid_args(args, 4)
    assert encoder_size == 0
    assert llm_size == 4
    specs = build_module_grid_specs(args, 4)
    assert len(specs) == 1
    assert specs[0].name == MIMO_LANGUAGE_MODULE_KEY


def test_train_samples_resolves_iters():
    # gbs = mbs(1) * num_microbatches(2) * llm_dp(2) = 4; 17 samples -> ceil(17/4)=5.
    args = _layout_8gpu_20l(train_samples=17, train_iters=999)
    validate_hetero_grid_args(args, WORLD_SIZE_8)
    assert args.train_iters == 5

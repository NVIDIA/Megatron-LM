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
OLD_GRID_FLAGS = (
    "--encoder-tp",
    "--encoder-dp",
    "--encoder-ddp-overlap",
    "--llm-offset",
    "--llm-tp",
    "--llm-cp",
    "--llm-pp",
    "--llm-dp",
    "--llm-ep",
    "--llm-expt-tp",
    "--llm-only",
)


def _parse(argv):
    """Parse only the hetero grid args from a token list."""
    parser = argparse.ArgumentParser()
    add_hetero_grid_args(parser)
    return parser.parse_args(argv)


def _layout_8gpu_20l(**overrides):
    """Canonical 8-GPU layout: encoder 0-3 (tp2/dp2), llm 4-7 (tp2/pp1/dp2/ep4)."""
    argv = (
        "--mimo-encoder-tp 2 --mimo-encoder-dp 2 "
        "--mimo-llm-offset 4 --mimo-llm-tp 2 --mimo-llm-pp 1 "
        "--mimo-llm-dp 2 --mimo-llm-ep 4"
    ).split()
    args = _parse(argv)
    # Stock args the validator reads but the grid parser does not own.
    args.micro_batch_size = 1
    args.num_experts = 128
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_parser_uses_mimo_prefixed_destinations():
    args = _parse(
        "--mimo-encoder-tp 3 --mimo-encoder-dp 4 --mimo-llm-offset 5 "
        "--mimo-llm-tp 6 --mimo-llm-cp 1 --mimo-llm-pp 7 --mimo-llm-dp 8 "
        "--mimo-llm-ep 9 --mimo-llm-expt-tp 10 --mimo-llm-only "
        "--mimo-encoder-ddp-overlap".split()
    )

    assert vars(args) == {
        "mimo_encoder_tp": 3,
        "mimo_encoder_dp": 4,
        "mimo_llm_offset": 5,
        "mimo_llm_tp": 6,
        "mimo_llm_cp": 1,
        "mimo_llm_pp": 7,
        "mimo_llm_dp": 8,
        "mimo_llm_ep": 9,
        "mimo_llm_expt_tp": 10,
        "mimo_llm_only": True,
        "mimo_encoder_ddp_overlap": True,
    }


@pytest.mark.parametrize("flag", OLD_GRID_FLAGS)
def test_parser_rejects_unprefixed_grid_flags(flag):
    with pytest.raises(SystemExit):
        _parse([flag])


def test_canonical_layout_validates_and_maps_specs():
    args = _layout_8gpu_20l()
    encoder_size, llm_size = validate_hetero_grid_args(args, WORLD_SIZE_8)
    assert (encoder_size, llm_size) == (4, 4)
    assert args.mimo_llm_world_size == llm_size

    encoder_grid_spec, language_grid_spec = build_module_grid_specs(
        args, WORLD_SIZE_8, encoder_module_name="radio_encoder"
    )
    assert encoder_grid_spec.name == "radio_encoder"
    assert encoder_grid_spec.num_ranks == 4
    assert encoder_grid_spec.rank_offset == 0  # encoder span always starts at rank 0
    assert encoder_grid_spec.cp == 1
    assert encoder_grid_spec.pp == 1
    assert encoder_grid_spec.dp == 2  # derived: 4 // tp2
    assert language_grid_spec.name == MIMO_LANGUAGE_MODULE_KEY
    assert language_grid_spec.num_ranks == 4
    assert language_grid_spec.rank_offset == 4
    assert language_grid_spec.dp == 2
    # expt_tp defaults to 1 when --mimo-llm-expt-tp is unset.
    assert language_grid_spec.expt_tp == 1


def test_gtp_layout_validates_and_maps_weight_shard_axes():
    args = _layout_8gpu_20l(
        mimo_llm_dp=1,
        mimo_llm_ep=2,
        tensor_parallel_num_weight_shards=4,
        expert_tensor_parallel_num_weight_shards=2,
    )

    assert validate_hetero_grid_args(args, WORLD_SIZE_8) == (4, 4)
    assert not hasattr(args, "gtp_weight_remat_size")
    assert not hasattr(args, "expert_gtp_weight_remat_size")

    _, language_grid_spec = build_module_grid_specs(
        args, WORLD_SIZE_8, encoder_module_name="radio_encoder"
    )
    assert language_grid_spec.gtp_remat == 2
    assert language_grid_spec.dp == 1
    assert language_grid_spec.expt_gtp_remat == 2
    assert language_grid_spec.expt_dp == 1


def test_weight_shards_must_divide_language_tp():
    args = _layout_8gpu_20l(tensor_parallel_num_weight_shards=3)
    with pytest.raises(ValueError, match="must be divisible"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_overlapping_spans_raise():
    # mimo_llm_offset 2 makes llm ranks {2,3,4,5} overlap encoder ranks {0,1,2,3}.
    args = _layout_8gpu_20l(mimo_llm_offset=2)
    with pytest.raises(ValueError, match="disjoint"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_non_covering_spans_raise():
    # encoder 0-3 + llm 4-7 cover only 8 ranks; declare world_size 10 -> gap.
    args = _layout_8gpu_20l()
    with pytest.raises(ValueError, match="cover every torchrun rank"):
        validate_hetero_grid_args(args, 10)


def test_fanout_divisibility_raises():
    # mbs(1) * mimo_llm_dp(2) = 2 not divisible by mimo_encoder_dp(3).
    args = _layout_8gpu_20l(mimo_encoder_dp=3, micro_batch_size=1, mimo_llm_dp=2)
    with pytest.raises(ValueError, match="divisible by --mimo-encoder-dp"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_ep_divisibility_raises():
    # num_experts 128 not divisible by mimo_llm_ep 3.
    args = _layout_8gpu_20l(mimo_llm_ep=3, num_experts=128)
    with pytest.raises(ValueError, match="divisible by --mimo-llm-ep"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_parser_does_not_expose_unsupported_grid_knobs():
    args = _parse([])
    assert not hasattr(args, "mimo_encoder_cp")
    assert not hasattr(args, "mimo_encoder_pp")
    assert not hasattr(args, "mimo_llm_expt_dp")


def test_llm_cp_must_be_one():
    args = _layout_8gpu_20l(mimo_llm_cp=2)
    with pytest.raises(ValueError, match="CP=1 only"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_encoder_overlap_requires_grad_reduce():
    args = _layout_8gpu_20l(mimo_encoder_ddp_overlap=True, overlap_grad_reduce=False)
    with pytest.raises(ValueError, match="requires --overlap-grad-reduce"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_encoder_overlap_accepts_uniform_participation_opt_in():
    args = _layout_8gpu_20l(mimo_encoder_ddp_overlap=True, overlap_grad_reduce=True)
    assert validate_hetero_grid_args(args, WORLD_SIZE_8) == (4, 4)


def test_llm_only_requires_offset_zero():
    args = _layout_8gpu_20l(mimo_llm_only=True, mimo_llm_offset=4)
    with pytest.raises(ValueError, match="--mimo-llm-only requires --mimo-llm-offset 0"):
        validate_hetero_grid_args(args, WORLD_SIZE_8)


def test_llm_only_rejects_encoder_overlap():
    args = _layout_8gpu_20l(
        mimo_llm_only=True,
        mimo_llm_offset=0,
        mimo_llm_ep=2,
        mimo_encoder_ddp_overlap=True,
        overlap_grad_reduce=True,
    )
    with pytest.raises(ValueError, match="cannot be used with --mimo-llm-only"):
        validate_hetero_grid_args(args, 4)


def test_llm_only_covers_world():
    # llm tp2/pp1/dp2 = 4 ranks at offset 0; world_size 4 -> covers exactly, no encoder spec.
    args = _layout_8gpu_20l(mimo_llm_only=True, mimo_llm_offset=0, mimo_llm_ep=2, num_experts=128)
    encoder_size, llm_size = validate_hetero_grid_args(args, 4)
    assert (encoder_size, llm_size) == (0, 4)
    assert args.mimo_llm_world_size == llm_size
    specs = build_module_grid_specs(args, 4, encoder_module_name="radio_encoder")
    assert len(specs) == 1
    assert specs[0].name == MIMO_LANGUAGE_MODULE_KEY

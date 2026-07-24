# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-args (no-GPU) tests for the hetero grid arg group + validation."""

from __future__ import annotations

import argparse

import pytest

from examples.mimo.training.args import (
    MIMO_LAYOUT_COLOCATED,
    add_hetero_grid_args,
    build_module_grid_specs,
    validate_colocated_runtime_args,
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
    """Canonical 8-GPU layout: encoder 0-3 (tp2/dp2), llm 4-7 (tp2/pp1/dp2/ep4)."""
    argv = (
        "--encoder-tp 2 --encoder-dp 2 "
        "--llm-offset 4 --llm-tp 2 --llm-pp 1 --llm-dp 2 --llm-ep 4"
    ).split()
    args = _parse(argv)
    # Stock args the validator reads but the grid parser does not own.
    vars(args).update(
        micro_batch_size=1,
        num_experts=128,
        recompute_granularity=None,
        async_save=False,
        rerun_mode="disabled",
        te_rng_tracker=False,
        log_params_norm=False,
        log_num_zeros_in_grad=False,
        fp16=False,
        use_megatron_fsdp=False,
        use_torch_fsdp2=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        moe_use_upcycling=False,
        init_model_with_meta_device=False,
        cuda_graph_impl="none",
        save=None,
        load=None,
        pretrained_checkpoint=None,
        data_parallel_random_init=False,
        ckpt_format="torch_dist",
        ckpt_fully_parallel_save=False,
        ckpt_fully_parallel_load=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.eval_micro_batch_size = args.micro_batch_size
    return args


def test_canonical_layout_validates_and_maps_specs():
    args = _layout_8gpu_20l()
    encoder_size, llm_size = validate_hetero_grid_args(args, WORLD_SIZE_8)
    assert (encoder_size, llm_size) == (4, 4)

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


def test_parser_does_not_expose_unsupported_grid_knobs():
    args = _parse([])
    assert not hasattr(args, "encoder_cp")
    assert not hasattr(args, "encoder_pp")
    assert not hasattr(args, "llm_expt_dp")


def test_llm_cp_must_be_one():
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
    specs = build_module_grid_specs(args, 4, encoder_module_name="radio_encoder")
    assert len(specs) == 1
    assert specs[0].name == MIMO_LANGUAGE_MODULE_KEY


def test_layout_defaults_to_non_colocated():
    assert _parse([]).mimo_layout == "non-colocated"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(encoder_tp=2, encoder_dp=4, llm_tp=4, llm_dp=2, micro_batch_size=2),
        dict(encoder_tp=4, encoder_dp=2, llm_tp=2, llm_dp=4, micro_batch_size=1),
    ],
)
def test_colocated_layout_maps_both_grids_to_the_world(overrides):
    args = _layout_8gpu_20l(mimo_layout=MIMO_LAYOUT_COLOCATED, llm_offset=0, **overrides)
    specs = build_module_grid_specs(args, WORLD_SIZE_8, "radio_encoder")
    assert [(spec.rank_offset, spec.num_ranks) for spec in specs] == [(0, 8), (0, 8)]


@pytest.mark.parametrize(
    "field",
    [
        "recompute_granularity",
        "async_save",
        "te_rng_tracker",
        "log_params_norm",
        "log_num_zeros_in_grad",
        "fp16",
        "overlap_grad_reduce",
    ],
)
def test_colocated_runtime_rejects_unsupported_modes(field):
    args = _layout_8gpu_20l(mimo_layout=MIMO_LAYOUT_COLOCATED, llm_offset=0)
    setattr(args, field, "full" if field == "recompute_granularity" else True)
    with pytest.raises(ValueError, match="does not support"):
        validate_colocated_runtime_args(args)


def test_colocated_checkpoint_requires_supported_rng_and_format():
    args = _layout_8gpu_20l(mimo_layout=MIMO_LAYOUT_COLOCATED, llm_offset=0)
    args.save = "/tmp/checkpoint"
    args.ckpt_format = "torch_dist"
    args.data_parallel_random_init = False
    with pytest.raises(ValueError, match="data-parallel-random-init"):
        validate_colocated_runtime_args(args)

    args.data_parallel_random_init = True
    validate_colocated_runtime_args(args)


def test_colocated_runtime_rerun_error_gives_disable_flag():
    args = _layout_8gpu_20l(
        mimo_layout=MIMO_LAYOUT_COLOCATED, llm_offset=0, rerun_mode="validate_results"
    )
    with pytest.raises(ValueError, match="--rerun-mode disabled"):
        validate_colocated_runtime_args(args)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the RL config section and the argument group generated from it."""

import argparse
import dataclasses
import importlib.util
import os
import sys
import types
import typing
from dataclasses import fields
from unittest.mock import patch

import pytest

from megatron.training.argument_utils import _default_config_from_args
from megatron.training.arguments import _add_rl_args, parse_args, validate_args
from megatron.training.config import RLConfig, rl_config
from megatron.training.config.container import ConfigContainerBase
from megatron.training.config.utils import sanitize_dataclass_config

INIT_FIELDS = [f for f in fields(RLConfig) if f.init]
GRPO_4x3 = dict(grpo_prompts_per_step=4, grpo_group_size=3)  # 12 samples per iteration


def _parse(argv):
    return _add_rl_args(argparse.ArgumentParser()).parse_args(argv)


def _cli_cases():
    """(argv, field, expected) per field, with a value that differs from the field default."""
    cases = []
    for f in INIT_FIELDS:
        flag = "--" + f.name.replace("_", "-")
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        tp = f.type
        if typing.get_origin(tp) in (typing.Union, types.UnionType):
            tp = [t for t in typing.get_args(tp) if t is not type(None)][0]
        if tp is bool:
            cases.append(([flag], f.name, True))
            if f.metadata.get("argparse_meta", {}).get("action") is argparse.BooleanOptionalAction:
                cases.append(([flag.replace("--", "--no-", 1)], f.name, False))
        elif typing.get_origin(tp) is typing.Literal:
            choice = [c for c in typing.get_args(tp) if c != default][0]
            cases.append(([flag, str(choice)], f.name, choice))
        elif typing.get_origin(tp) is list:
            cases += [([flag, "a", "b"], f.name, ["a", "b"]), ([flag], f.name, [])]
        elif tp in (int, float):
            value = tp(7) if default is None else default + tp(1)
            cases.append(([flag, str(value)], f.name, value))
        else:
            assert tp is str, f"unhandled field type {f.type} for {f.name}"
            cases.append(([flag, "some/path"], f.name, "some/path"))
    return cases


CLI_CASES = _cli_cases()


class TestGeneratedArgumentGroup:
    def test_group_mirrors_dataclass(self):
        parser = _add_rl_args(argparse.ArgumentParser())
        assert "rl" in [g.title for g in parser._action_groups]
        ns = parser.parse_args(["--grpo-prompts-per-step", "4", "--grpo-group-size", "3"])
        # Every init field is a flag with the dataclass default; the derived field is not a flag.
        assert set(vars(ns)) == {f.name for f in INIT_FIELDS}
        assert all(getattr(_parse([]), f.name) == getattr(RLConfig(), f.name) for f in INIT_FIELDS)
        # A stray derived value on the namespace is ignored when the section is rebuilt.
        ns.grpo_samples_per_iteration = 999
        assert _default_config_from_args(RLConfig, ns).grpo_samples_per_iteration == 12
        from megatron.rl import rollout_granularity

        for name in ("SubmissionGranularity", "ConsumptionGranularity"):
            assert typing.get_args(getattr(rl_config, name)) == typing.get_args(
                getattr(rollout_granularity, name)
            )

    @pytest.mark.parametrize(
        "argv, name, expected", CLI_CASES, ids=[" ".join(c[0]) for c in CLI_CASES]
    )
    def test_every_flag_round_trips(self, argv, name, expected):
        assert getattr(_parse(argv), name) == expected
        assert getattr(RLConfig(**{name: expected}), name) == expected


class TestPostInit:
    @pytest.fixture(autouse=True)
    def _without_torch_memory_saver(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_memory_saver", None)

    @pytest.mark.parametrize(
        "kwargs, lag",
        [
            ({}, 0.0),
            (dict(rl_partial_rollouts=True), None),
            (dict(rl_partial_rollouts=True, rl_generation_lag=2.5), 2.5),
            (dict(rl_max_inflight_requests=6), -0.5),
            (dict(rl_partial_rollouts=True, rl_max_inflight_requests=24), 1.0),
            # UVM satisfies the weight-offload requirement without torch_memory_saver.
            (
                dict(
                    rl_offload_inference_model_weights=True,
                    rl_inference_model_unified_memory_level=1,
                ),
                0.0,
            ),
            # Inactive section: inputs are kept as given, only the sample count is derived.
            (dict(perform_rl_step=False, rl_max_inflight_requests=6), None),
        ],
    )
    def test_derivations(self, kwargs, lag):
        cfg = RLConfig(**{"perform_rl_step": True, **GRPO_4x3, **kwargs})
        assert (cfg.grpo_samples_per_iteration, cfg.rl_generation_lag) == (12, lag)

    @pytest.mark.parametrize(
        "kwargs, match, rl_only",
        [
            (dict(rl_generation_lag=1.0), "requires --rl-partial-rollouts", True),
            (dict(rl_generation_lag=-2.0, rl_partial_rollouts=True), "must be >= -1", True),
            (dict(rl_max_inflight_requests=0), "must be >= 1", True),
            (dict(rl_max_inflight_requests=65), "requires --rl-partial-rollouts", True),
            (dict(rl_max_inflight_requests=64, rl_generation_lag=0.5), "mutually exclusive", True),
            (dict(rl_submission_granularity="R"), "requires --rl-partial-rollouts", True),
            (dict(rl_consumption_granularity="G"), "not supported", True),
            (
                dict(rl_consumption_granularity="R", rl_partial_rollouts=True),
                "not currently supported",
                True,
            ),
            (dict(refit_method="nccl_m2n"), "unsupported by the built-in RL loop", True),
            (dict(rl_offload_inference_model_weights=True), "torch_memory_saver", True),
            (dict(refit_execution_batch_bytes=0), "positive integer", False),
        ],
    )
    def test_rejections(self, kwargs, match, rl_only):
        with pytest.raises(AssertionError, match=match):
            RLConfig(perform_rl_step=True, **kwargs)
        if rl_only:
            RLConfig(perform_rl_step=False, **kwargs)  # accepted while the RL loop is off

    @pytest.mark.parametrize(
        "rebuild",
        [
            pytest.param(
                lambda cfg: RLConfig(**{f.name: getattr(cfg, f.name) for f in INIT_FIELDS}),
                id="init-fields",
            ),
            pytest.param(
                lambda cfg: _instantiate(ConfigContainerBase._convert_value_to_dict(cfg)),
                id="serialized",
                marks=pytest.mark.skipif(
                    importlib.util.find_spec("omegaconf") is None, reason="omegaconf not installed"
                ),
            ),
        ],
    )
    def test_normalized_section_is_a_fixed_point(self, rebuild):
        cfg = RLConfig(
            perform_rl_step=True,
            rl_partial_rollouts=True,
            rl_max_inflight_requests=24,
            rl_inference_parsers=["deepseek-r1-reasoning"],
            **GRPO_4x3,
        )
        assert cfg.rl_generation_lag == 1.0
        assert rebuild(cfg) == cfg


def _instantiate(as_dict):
    from megatron.training.config.instantiate_utils import instantiate

    assert as_dict["_target_"] == "megatron.training.config.rl_config.RLConfig"
    assert as_dict["grpo_samples_per_iteration"] == 12  # serialized, then stripped on load
    return instantiate(sanitize_dataclass_config(as_dict))


RUN = dict(
    cuda_graph_impl="local",
    micro_batch_size=1,
    global_batch_size=12,
    skip_train=False,
    load_optim=True,
    save_interval=None,
    exit_interval=None,
)


@pytest.mark.parametrize(
    "rl_kwargs, run_overrides, match",
    [
        (dict(rl_training_cuda_graphs=True), dict(cuda_graph_impl="none"), "no CUDA graphs"),
        (dict(rl_use_sequence_packing=True), dict(micro_batch_size=4), "must be 1"),
        (dict(grpo_iterations=1), dict(global_batch_size=8), "divisible by global_batch_size"),
        (dict(), dict(exit_interval=3), "exit_interval should be divisible"),
        (dict(), dict(save_interval=5), "save_interval should be divisible"),
        (
            dict(rl_offload_optimizer_during_inference=True),
            dict(skip_train=True, load_optim=False),
            "no optimizer to offload",
        ),
        (dict(), dict(global_batch_size=None), "must be resolved"),
    ],
)
def test_validate_run_rejects_runs_inconsistent_with_the_section(rl_kwargs, run_overrides, match):
    # 4 x 3 samples per iteration over 2 GRPO iterations: 24 samples, 2 batches of 12.
    cfg = RLConfig(perform_rl_step=True, **GRPO_4x3, **rl_kwargs)
    with pytest.raises(ValueError, match=match):
        cfg.validate_run(**{**RUN, **run_overrides})


def test_validate_run_at_every_call_site():
    from megatron.core.optimizer import OptimizerConfig
    from megatron.training.config import (
        CheckpointConfig,
        LoggerConfig,
        PretrainConfigContainer,
        SchedulerConfig,
        TrainingConfig,
    )

    # Direct: a consistent run passes, the CUDA-graph rule is skipped without a model config, and
    # nothing is checked unless the run is an RL run.
    cfg = RLConfig(
        perform_rl_step=True, rl_training_cuda_graphs=True, rl_use_sequence_packing=True, **GRPO_4x3
    )
    cfg.validate_run(**{**RUN, "save_interval": 4, "exit_interval": 2})
    cfg.validate_run(**{**RUN, "cuda_graph_impl": None})
    RLConfig(rl_use_sequence_packing=True).validate_run(**{**RUN, "micro_batch_size": 4})

    # Through the container: the sections supply the inputs; no model skips the CUDA-graph rule.
    def container(**rl_kwargs):
        return PretrainConfigContainer(
            train=TrainingConfig(micro_batch_size=4, global_batch_size=12),
            model=None,
            optimizer=OptimizerConfig(),
            scheduler=SchedulerConfig(),
            logger=LoggerConfig(),
            checkpoint=CheckpointConfig(),
            rl=RLConfig(perform_rl_step=True, **GRPO_4x3, **rl_kwargs),
        )

    container(rl_training_cuda_graphs=True).validate()
    with pytest.raises(ValueError, match="must be 1"):
        container(rl_use_sequence_packing=True).validate()

    # Through validate_args: CLI runs still fail at parse time.
    dp = int(os.environ.get("WORLD_SIZE", "1"))
    argv = (
        "test_rl_config.py --num-layers 2 --hidden-size 128 --num-attention-heads 8 "
        f"--micro-batch-size 2 --global-batch-size {2 * dp} --seq-length 32 "
        f"--max-position-embeddings 32 --perform-rl-step --grpo-prompts-per-step {dp} "
        "--grpo-group-size 2 --rl-use-sequence-packing"
    ).split()
    with patch("sys.argv", argv), pytest.raises(ValueError, match="must be 1"):
        validate_args(parse_args())


def test_validate_args_leaves_derived_values_on_the_section():
    dp = int(os.environ.get("WORLD_SIZE", "1"))
    samples = 12 * dp
    argv = (
        "test_rl_config.py --num-layers 2 --hidden-size 128 --num-attention-heads 8 "
        f"--micro-batch-size 1 --global-batch-size {samples} --seq-length 32 "
        f"--max-position-embeddings 32 --perform-rl-step --grpo-prompts-per-step {4 * dp} "
        f"--grpo-group-size 3 --rl-partial-rollouts --rl-max-inflight-requests {2 * samples}"
    ).split()
    with patch("sys.argv", argv):
        args = validate_args(parse_args())
    cfg = _default_config_from_args(RLConfig, args)
    # validate_args checks the section but does not mirror its derived values onto args.
    assert not hasattr(args, "grpo_samples_per_iteration")
    assert args.rl_generation_lag is None
    assert (cfg.grpo_samples_per_iteration, cfg.rl_generation_lag) == (samples, 1.0)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for the RL config section, its container, and the argument group generated from it."""

import argparse
import dataclasses
import importlib.util
import os
import sys
import types
import typing
from dataclasses import fields
from unittest.mock import Mock, patch

import pytest

from megatron.rl.config import (
    add_rl_args,
    finalize_rl_run,
    get_rl_run_config,
    rollout_inference_container_from_args,
    rollout_model_is_separate,
    set_rl_run_config,
    validate_rl_run,
)
from megatron.training.argument_utils import (
    _default_config_from_args,
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import (
    InferenceConfigContainer,
    InferenceSetupConfig,
    RLConfig,
    RunConfigRecord,
    rl_config,
)
from megatron.training.config.container import ConfigContainerBase
from megatron.training.config.utils import sanitize_dataclass_config

INIT_FIELDS = [f for f in fields(RLConfig) if f.init]
GRPO_4x3 = dict(grpo_prompts_per_step=4, grpo_group_size=3)  # 12 samples per iteration


def _parse(argv):
    return add_rl_args(argparse.ArgumentParser()).parse_args(argv)


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
        parser = add_rl_args(argparse.ArgumentParser())
        assert "rl" in [g.title for g in parser._action_groups]
        ns = parser.parse_args(["--grpo-prompts-per-step", "4", "--grpo-group-size", "3"])
        # Every init field is a flag with the dataclass default; the derived field is not a flag;
        # the rollout engine's deployment flags ride along in their own group.
        deployment_dests = {
            "rl_inference_tensor_model_parallel_size",
            "rl_inference_pipeline_model_parallel_size",
            "rl_inference_expert_model_parallel_size",
            "rl_inference_expert_tensor_model_parallel_size",
        }
        assert set(vars(ns)) == {f.name for f in INIT_FIELDS} | deployment_dests
        assert all(getattr(_parse([]), f.name) == getattr(RLConfig(), f.name) for f in INIT_FIELDS)
        # A stray derived value on the namespace is ignored when the section is rebuilt.
        ns.grpo_samples_per_iteration = 999
        assert _default_config_from_args(RLConfig, ns).grpo_samples_per_iteration == 12
        # The section spells the granularity choices itself (it must not import megatron.rl).
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
    train_iters=10,
    train_samples=None,
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
            dict(skip_train=True, train_iters=None),
            "nothing to offload",
        ),
        (
            dict(rl_offload_optimizer_during_inference=True),
            dict(global_batch_size=None, skip_train=True, train_iters=None),
            "nothing to offload",
        ),
        (
            dict(rl_offload_optimizer_during_inference=True, rl_rollout_only=True),
            dict(load_optim=False),
            "nothing to offload",
        ),
        (dict(rl_rollout_only=True), dict(train_iters=None), "needs --train-iters"),
        (dict(rl_rollout_only=True), dict(train_iters=0), "needs --train-iters"),
        (dict(), dict(skip_train=True), "--rl-rollout-only collects"),
        (dict(rl_rollout_only=True), dict(skip_train=True), "--rl-rollout-only collects"),
    ],
)
def test_validate_run_rejects_runs_inconsistent_with_the_section(rl_kwargs, run_overrides, match):
    # 4 x 3 samples per iteration over 2 GRPO iterations: 24 samples, 2 batches of 12.
    cfg = RLConfig(perform_rl_step=True, **GRPO_4x3, **rl_kwargs)
    with pytest.raises(ValueError, match=match):
        cfg.validate_run(**{**RUN, **run_overrides})


MINIMAL_ARGV = (
    "test_rl_config.py --num-layers 2 --hidden-size 128 --num-attention-heads 8 "
    "--seq-length 32 --max-position-embeddings 32"
).split()


def _pretrain_container(micro_batch_size=4, model=None):
    from megatron.core.optimizer import OptimizerConfig
    from megatron.training.config import (
        CheckpointConfig,
        LoggerConfig,
        PretrainConfigContainer,
        SchedulerConfig,
        TrainingConfig,
    )

    return PretrainConfigContainer(
        train=TrainingConfig(micro_batch_size=micro_batch_size, global_batch_size=12),
        model=model,
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        logger=LoggerConfig(),
        checkpoint=CheckpointConfig(),
    )


def _inference_container():
    """A rollout deployment sharing the training model, as the record of a colocated run holds."""
    return InferenceConfigContainer(model=None, inference=InferenceSetupConfig())


def _model_with_layout(**layout):
    """A model config whose transformer section carries only the parallel layout."""
    from megatron.training.models.base import ModelConfig

    defaults = dict(
        cuda_graph_impl="none",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )

    @dataclasses.dataclass
    class LayoutModelConfig(ModelConfig):
        builder: typing.ClassVar[str] = "tests.layout"
        transformer: types.SimpleNamespace = dataclasses.field(
            default_factory=lambda: types.SimpleNamespace(**{**defaults, **layout})
        )

    return LayoutModelConfig()


def _rl_run(pretrain, inference=None, **rl_kwargs):
    return RunConfigRecord(
        pretrain=pretrain,
        inference=_inference_container() if inference is None else inference,
        rl=RLConfig(perform_rl_step=True, **GRPO_4x3, **rl_kwargs),
    )


def _record_from_args(args):
    """What the RL entry point composes from its parsed arguments."""
    pretrain = pretrain_cfg_container_from_args(args, gpt_config_from_args(args))
    return RunConfigRecord(
        pretrain=pretrain,
        inference=rollout_inference_container_from_args(args, pretrain),
        rl=_default_config_from_args(RLConfig, args),
    )


def test_validate_run_from_the_container_and_the_entry_point():
    # Direct: a consistent run passes, the CUDA-graph rule is skipped without a model config, and
    # nothing is checked unless the run is an RL run.
    cfg = RLConfig(
        perform_rl_step=True, rl_training_cuda_graphs=True, rl_use_sequence_packing=True, **GRPO_4x3
    )
    cfg.validate_run(**{**RUN, "save_interval": 4, "exit_interval": 2})
    cfg.validate_run(**{**RUN, "cuda_graph_impl": None})
    RLConfig(rl_use_sequence_packing=True).validate_run(**{**RUN, "micro_batch_size": 4})

    cfg.validate_run(**{**RUN, "global_batch_size": None})
    # The RL modes: rollout-only with or without the optimizer (either horizon form), and
    # eval-only (--skip-train with no horizon, including the --train-iters 0 spelling).
    rollout_only = RLConfig(perform_rl_step=True, rl_rollout_only=True, **GRPO_4x3)
    rollout_only.validate_run(**RUN)
    rollout_only.validate_run(**{**RUN, "load_optim": False})
    rollout_only.validate_run(**{**RUN, "train_iters": None, "train_samples": 120})
    # Offloading the optimizer needs one: accepted while it stays resident, rejected above when
    # --no-load-optim leaves it out.
    RLConfig(
        perform_rl_step=True,
        rl_rollout_only=True,
        rl_offload_optimizer_during_inference=True,
        **GRPO_4x3,
    ).validate_run(**RUN)
    cfg.validate_run(**{**RUN, "skip_train": True, "train_iters": None})
    cfg.validate_run(**{**RUN, "skip_train": True, "train_iters": 0})
    RLConfig(rl_rollout_only=True).validate_run(**RUN)  # accepted while the RL loop is off

    # Through the RL container, against the pretrain container it drives; the pretrain
    # container itself carries no RL section. The CUDA-graph rule reads the model's transformer
    # section and is skipped for a model without one.
    from megatron.training.models.base import ModelConfig

    @dataclasses.dataclass
    class TopologyOnlyModelConfig(ModelConfig):
        builder: typing.ClassVar[str] = "tests.topology_only"

    @dataclasses.dataclass
    class TransformerModelConfig(ModelConfig):
        builder: typing.ClassVar[str] = "tests.transformer"
        transformer: types.SimpleNamespace = dataclasses.field(
            default_factory=lambda: types.SimpleNamespace(
                cuda_graph_impl="none", context_parallel_size=1
            )
        )

    assert not hasattr(_pretrain_container(), "rl")

    pretrain = _pretrain_container()
    offload_run = _rl_run(pretrain, rl_offload_optimizer_during_inference=True)
    validate_rl_run(offload_run)
    assert not pretrain.ddp.grad_buffer_offload
    offload_run.rl.rl_offload_optimizer_during_inference = False
    validate_rl_run(offload_run)
    assert not pretrain.ddp.grad_buffer_offload
    offload_run.rl.rl_offload_optimizer_during_inference = True
    pretrain.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    with pytest.raises(ValueError, match="parameters share the gradient-buffer storage"):
        validate_rl_run(offload_run)

    pretrain = _pretrain_container()
    cuda_graph_run = _rl_run(
        pretrain, rl_offload_optimizer_during_inference=True, rl_training_cuda_graphs=True
    )
    validate_rl_run(cuda_graph_run)
    assert not pretrain.ddp.grad_buffer_offload
    finalize_rl_run(cuda_graph_run)
    assert not pretrain.ddp.grad_buffer_offload
    pretrain = _pretrain_container()
    pretrain.ddp.grad_buffer_offload = True
    explicit_layout_run = RunConfigRecord(
        pretrain=pretrain, inference=_inference_container(), rl=RLConfig()
    )
    validate_rl_run(explicit_layout_run)
    assert pretrain.ddp.grad_buffer_offload
    pretrain.ddp.fp8_param_gather = True
    pretrain.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    with pytest.raises(ValueError, match="parameters share the gradient-buffer storage"):
        validate_rl_run(explicit_layout_run)
    validate_rl_run(_rl_run(_pretrain_container(), rl_training_cuda_graphs=True))
    validate_rl_run(
        _rl_run(_pretrain_container(model=TopologyOnlyModelConfig()), rl_training_cuda_graphs=True)
    )
    validate_rl_run(
        RunConfigRecord(
            pretrain=_pretrain_container(model=TopologyOnlyModelConfig()),
            inference=_inference_container(),
            rl=RLConfig(),
        )
    )
    with pytest.raises(ValueError, match="no CUDA graphs"):
        validate_rl_run(
            _rl_run(
                _pretrain_container(model=TransformerModelConfig()), rl_training_cuda_graphs=True
            )
        )
    with pytest.raises(ValueError, match="must be 1"):
        validate_rl_run(_rl_run(_pretrain_container(), rl_use_sequence_packing=True))

    # Through the entry point: parsing with the RL group and validate_args both pass; the
    # containers built from args reject the run.
    dp = int(os.environ.get("WORLD_SIZE", "1"))
    argv = (
        MINIMAL_ARGV
        + (
            f"--micro-batch-size 2 --global-batch-size {2 * dp} --perform-rl-step "
            f"--grpo-prompts-per-step {dp} --grpo-group-size 2 --rl-use-sequence-packing"
        ).split()
    )
    with patch("sys.argv", argv):
        args = validate_args(parse_args(extra_args_provider=add_rl_args))
    with pytest.raises(ValueError, match="must be 1"):
        validate_rl_run(_record_from_args(args))
    # The retired inference-only spelling points at the new flag, which does not combine with
    # --skip-train; the variant without an optimizer is --rl-rollout-only --no-load-optim.
    argv = (
        "test_rl_config.py --num-layers 2 --hidden-size 128 --num-attention-heads 8 "
        f"--micro-batch-size 1 --global-batch-size {2 * dp} --seq-length 32 "
        f"--max-position-embeddings 32 --perform-rl-step --grpo-prompts-per-step {dp} "
        "--grpo-group-size 2 --train-iters 10"
    ).split()
    for retired in (["--skip-train"], ["--skip-train", "--rl-rollout-only"]):
        with patch("sys.argv", argv + retired):
            args = validate_args(parse_args(extra_args_provider=add_rl_args))
        assert args.no_load_optim  # --skip-train still coerces --no-load-optim with the RL group
        with pytest.raises(ValueError, match="--rl-rollout-only collects"):
            validate_rl_run(_record_from_args(args))
    with patch("sys.argv", argv + ["--rl-rollout-only", "--no-load-optim"]):
        args = validate_args(parse_args(extra_args_provider=add_rl_args))
    validate_rl_run(_record_from_args(args))
    assert args.rl_rollout_only and args.no_load_optim


def test_entry_point_parse_leaves_the_section_to_the_container():
    dp = int(os.environ.get("WORLD_SIZE", "1"))
    samples = 12 * dp
    argv = (
        MINIMAL_ARGV
        + (
            f"--micro-batch-size 1 --global-batch-size {samples} --perform-rl-step "
            f"--grpo-prompts-per-step {4 * dp} --grpo-group-size 3 --rl-partial-rollouts "
            f"--rl-max-inflight-requests {2 * samples}"
        ).split()
    )
    with patch("sys.argv", argv):
        args = validate_args(parse_args(extra_args_provider=add_rl_args))
    cfg = _record_from_args(args).rl
    # validate_args neither derives nor mirrors RL values; the container does.
    assert not hasattr(args, "grpo_samples_per_iteration")
    assert args.rl_generation_lag is None
    assert (cfg.grpo_samples_per_iteration, cfg.rl_generation_lag) == (samples, 1.0)


@pytest.mark.parametrize("perform_rl_step", [False, True])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("cuda_graphs", [False, True])
@pytest.mark.parametrize("explicit_layout", [False, True])
def test_ddp_layout_validation_and_finalization(
    perform_rl_step, offload, cuda_graphs, explicit_layout
):
    from megatron.core.distributed import DistributedDataParallelConfig

    cfg = RLConfig(
        perform_rl_step=perform_rl_step,
        rl_offload_optimizer_during_inference=offload,
        rl_training_cuda_graphs=cuda_graphs,
    )
    ddp = DistributedDataParallelConfig(grad_buffer_offload=explicit_layout)
    before = dataclasses.asdict(ddp)
    for _ in range(2):
        cfg.validate_ddp_config(ddp)
        assert dataclasses.asdict(ddp) == before
    cfg.finalize_ddp_config(ddp)
    assert ddp.grad_buffer_offload == (
        explicit_layout or (perform_rl_step and offload and not cuda_graphs)
    )
    assert ConfigContainerBase._convert_value_to_dict(ddp)["grad_buffer_offload"] == (
        ddp.grad_buffer_offload
    )


def test_ddp_finalization_rejects_aliasing_before_mutation():
    from megatron.core.distributed import DistributedDataParallelConfig

    cfg = RLConfig(perform_rl_step=True, rl_offload_optimizer_during_inference=True)
    ddp = DistributedDataParallelConfig(
        fp8_param_gather=True, reuse_grad_buf_for_mxfp8_param_ag=True
    )
    before = dataclasses.asdict(ddp)
    with pytest.raises(ValueError, match="parameters share the gradient-buffer storage"):
        cfg.finalize_ddp_config(ddp)
    assert dataclasses.asdict(ddp) == before


def test_ddp_finalization_preserves_subclass_fields():
    from megatron.core.distributed import TorchFullyShardedDataParallelConfig

    cfg = RLConfig(perform_rl_step=True, rl_offload_optimizer_during_inference=True)
    ddp = TorchFullyShardedDataParallelConfig(reshard_after_forward=False)
    before = dataclasses.asdict(ddp)
    cfg.validate_ddp_config(ddp)
    cfg.finalize_ddp_config(ddp)
    assert type(ddp) is TorchFullyShardedDataParallelConfig
    assert dataclasses.asdict(ddp) == {**before, "grad_buffer_offload": True}


@pytest.mark.parametrize("native_builder", [False, True])
@pytest.mark.parametrize("perform_rl_step", [False, True])
def test_model_setup_uses_owned_ddp_config(
    monkeypatch, run_config, native_builder, perform_rl_step
):
    from megatron.training import arguments, global_vars, training

    parser = argparse.ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    # The RL container, not legacy args, owns the RL policy.
    args.perform_rl_step = not perform_rl_step
    args.batch_parallel_size = 1
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    rl_run = RunConfigRecord(
        pretrain=run_config,
        inference=_inference_container(),
        rl=RLConfig(perform_rl_step=perform_rl_step, rl_offload_optimizer_during_inference=True),
    )
    ddp = run_config.ddp
    finalize = Mock(wraps=rl_run.rl.finalize_ddp_config)
    monkeypatch.setattr(rl_run.rl, "finalize_ddp_config", finalize)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
    monkeypatch.setattr(training, "_add_model_freeze_pre_wrap_hook", Mock())
    monkeypatch.setattr("megatron.training.utils.start_memory_history_recording", Mock())

    class ReachedConstruction(Exception):
        pass

    def construct(*unused_args, **kwargs):
        assert kwargs["ddp_config"] is ddp
        assert ddp.grad_buffer_offload is perform_rl_step
        assert run_config.ddp is ddp
        if perform_rl_step:
            finalize.assert_called_once_with(ddp)
        else:
            finalize.assert_not_called()
        raise ReachedConstruction

    if native_builder:
        builder = types.SimpleNamespace(build_distributed_models=construct)
        run_config.model = types.SimpleNamespace(get_builder_cls=lambda: lambda cfg: builder)
    else:
        run_config.model = None
        monkeypatch.setattr(training, "get_model", construct)
    # The RL entry point finalizes the layout before runtime initialization; model setup only
    # consumes its result.
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 8
    finalize_rl_run(rl_run)
    run_config.finalize()
    with pytest.raises(ReachedConstruction):
        training.setup_model_and_optimizer(
            Mock(), model_provider_func=Mock(), cfg_container=run_config
        )


def test_global_parser_has_no_rl_flags():
    """RL flags come from add_rl_args; the global parser rejects them, and its --skip-train
    normalization has no RL term."""
    dp = int(os.environ.get("WORLD_SIZE", "1"))
    argv = MINIMAL_ARGV + f"--micro-batch-size 1 --global-batch-size {dp} --skip-train".split()
    with patch("sys.argv", argv + ["--perform-rl-step"]), pytest.raises(SystemExit):
        parse_args()
    with patch("sys.argv", argv):
        assert validate_args(parse_args()).no_load_optim is True


@pytest.mark.parametrize("offload", [False, True])
def test_weight_offload_needs_a_separate_rollout_model(offload):
    """The RL section's weight offload is checked against the deployment: serving the training
    model itself leaves nothing to offload; a different layout or disaggregated shards mean a
    separately built model."""
    pretrain = _pretrain_container(model=_model_with_layout())
    rl = dict(rl_offload_inference_model_weights=offload, rl_inference_model_unified_memory_level=1)
    shared = InferenceConfigContainer(model=pretrain.model, inference=InferenceSetupConfig())
    assert not rollout_model_is_separate(shared, pretrain.model)
    if offload:
        with pytest.raises(ValueError, match="requires a separate inference model"):
            validate_rl_run(_rl_run(pretrain, inference=shared, **rl))
    else:
        validate_rl_run(_rl_run(pretrain, inference=shared, **rl))
    refit = InferenceConfigContainer(
        model=_model_with_layout(tensor_model_parallel_size=2), inference=InferenceSetupConfig()
    )
    shards = InferenceConfigContainer(
        model=pretrain.model,
        inference=InferenceSetupConfig(inference_shards="tp=1,role=prefill+tp=1,role=decode"),
    )
    for deployment in (refit, shards):
        assert rollout_model_is_separate(deployment, pretrain.model)
        validate_rl_run(_rl_run(pretrain, inference=deployment, **rl))


def test_rl_run_config_is_a_singleton():
    with pytest.raises(AssertionError, match="not initialized"):
        get_rl_run_config()
    record = RunConfigRecord(
        pretrain=_pretrain_container(), inference=_inference_container(), rl=RLConfig()
    )
    set_rl_run_config(record)
    assert get_rl_run_config() is record
    with pytest.raises(AssertionError, match="already initialized"):
        set_rl_run_config(record)
    with pytest.raises(AssertionError, match="inference and rl members"):
        set_rl_run_config(RunConfigRecord(pretrain=_pretrain_container()))


def test_the_record_is_one_type_for_every_run(monkeypatch, tmp_path):
    """Every process writes a RunConfigRecord whose members are its containers: a training run
    records `pretrain`, an inference run `inference`, an RL run adds `inference` and `rl`; absent
    members stay out of the document; any record plus the missing members is the larger run's
    record; and a legacy flat record still yields its sections."""
    import yaml

    from megatron.training import checkpointing, global_vars
    from megatron.training.utils.checkpoint_utils import run_config_sections

    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG_RECORD", None)
    pretrain = _pretrain_container()
    global_vars.set_run_config(pretrain)
    assert global_vars.get_run_config_record() is None
    # A plain training run records its run config under `pretrain`, an inference run under
    # `inference`; the writer wraps what the process registered.
    plain = checkpointing._run_config_record()
    assert isinstance(plain, RunConfigRecord) and plain.pretrain is pretrain
    assert set(plain.to_dict()) == {"_target_", "pretrain"}
    inference_record = RunConfigRecord.wrap(_inference_container()).to_dict()
    assert set(inference_record) == {"_target_", "inference"}

    record = RunConfigRecord(
        pretrain=pretrain, inference=_inference_container(), rl=RLConfig(**GRPO_4x3)
    )
    set_rl_run_config(record)
    assert global_vars.get_run_config() is pretrain  # the loop keeps the pretrain container
    assert checkpointing._run_config_record() is record

    path = str(tmp_path / "run_config.yaml")
    record.to_yaml(path)
    rebuilt = RunConfigRecord.from_yaml(path)
    # The pretrain member's own YAML round trip turns one tuple field into a list (its own
    # serialization); the document and the RL-side members round-trip exactly.
    assert rebuilt.to_dict() == record.to_dict()
    assert (rebuilt.rl, rebuilt.inference) == (record.rl, record.inference)
    with open(path) as f:
        document = yaml.safe_load(f)
    assert document["_target_"] == "megatron.training.config.container.RunConfigRecord"
    assert set(document) == {"_target_", "pretrain", "inference", "rl"}
    assert document["pretrain"]["_target_"].endswith(".PretrainConfigContainer")
    assert document["rl"]["_target_"] == "megatron.training.config.rl_config.RLConfig"
    # Symmetry: an inference run's record plus the training and RL members is the RL record.
    as_dict = record.to_dict()
    grown = RunConfigRecord.from_dict(
        {**inference_record, "pretrain": as_dict["pretrain"], "rl": as_dict["rl"]}
    )
    assert grown.to_dict() == as_dict
    # Readers that look sections up by name get the pretrain member of a record, nothing from a
    # record without one, and a legacy flat record as it is.
    assert run_config_sections(document) is document["pretrain"]
    assert run_config_sections(inference_record) is None
    assert run_config_sections(document["pretrain"]) is document["pretrain"]
    assert run_config_sections(None) is None

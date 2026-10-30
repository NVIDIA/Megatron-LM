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
from unittest.mock import Mock, patch

import pytest

from megatron.core.transformer import TransformerConfig
from megatron.training.argument_utils import _default_config_from_args
from megatron.training.arguments import _add_rl_args, parse_args, validate_args
from megatron.training.config import RLConfig, rl_config
from megatron.training.config.utils import sanitize_dataclass_config, serialize_value

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
                lambda cfg: _instantiate(serialize_value(cfg)),
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
    separate_inference_model=False,
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
        (
            dict(
                rl_offload_inference_model_weights=True, rl_inference_model_unified_memory_level=1
            ),
            dict(),
            "requires a separate inference model",
        ),
    ],
)
def test_validate_run_rejects_runs_inconsistent_with_the_section(rl_kwargs, run_overrides, match):
    # 4 x 3 samples per iteration over 2 GRPO iterations: 24 samples, 2 batches of 12.
    cfg = RLConfig(perform_rl_step=True, **GRPO_4x3, **rl_kwargs)
    with pytest.raises(ValueError, match=match):
        cfg.validate_run(**{**RUN, **run_overrides})


@pytest.mark.parametrize(
    "rl_kwargs, context_parallel_size, inference_shards, separate",
    [
        (dict(), 1, None, False),
        # Shards without roles do not by themselves build a separate model.
        (dict(), 1, "tp=1+tp=1", False),
        (dict(), 1, "tp=1,role=prefill+tp=1,role=decode", True),
        (dict(), 2, None, True),
        (dict(rl_inference_tensor_model_parallel_size=2), 1, None, True),
        (dict(rl_inference_pipeline_model_parallel_size=2), 1, None, True),
        (dict(rl_inference_expert_model_parallel_size=2), 1, None, True),
        (dict(rl_inference_expert_tensor_model_parallel_size=2), 1, None, True),
    ],
)
def test_uses_separate_inference_model(
    rl_kwargs, context_parallel_size, inference_shards, separate
):
    assert (
        RLConfig(**rl_kwargs).uses_separate_inference_model(
            context_parallel_size=context_parallel_size, inference_shards=inference_shards
        )
        is separate
    )


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
    # The weight-offload rule needs the deployment: it passes with a separate model and is skipped
    # when the caller cannot see the inference shards.
    offload_weights = RLConfig(
        perform_rl_step=True,
        rl_offload_inference_model_weights=True,
        rl_inference_model_unified_memory_level=1,
        **GRPO_4x3,
    )
    offload_weights.validate_run(**{**RUN, "separate_inference_model": True})
    offload_weights.validate_run(**{**RUN, "separate_inference_model": None})

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

    # Through the container: the sections supply the inputs.
    # Training containers require a transformer section for the CUDA-graph rule.
    from megatron.training.models.base import ModelConfig

    @dataclasses.dataclass
    class TopologyOnlyModelConfig(ModelConfig):
        builder: typing.ClassVar[str] = "tests.topology_only"

    @dataclasses.dataclass
    class TransformerModelConfig(ModelConfig):
        builder: typing.ClassVar[str] = "tests.transformer"
        transformer: TransformerConfig = dataclasses.field(
            default_factory=lambda: TransformerConfig(
                num_layers=2, hidden_size=32, num_attention_heads=4
            )
        )

    def container(model=None, rl=None, **rl_kwargs):
        return PretrainConfigContainer(
            train=TrainingConfig(micro_batch_size=4, global_batch_size=12),
            model=TransformerModelConfig() if model is None else model,
            optimizer=OptimizerConfig(),
            scheduler=SchedulerConfig(),
            logger=LoggerConfig(),
            checkpoint=CheckpointConfig(),
            rl=rl if rl is not None else RLConfig(perform_rl_step=True, **GRPO_4x3, **rl_kwargs),
        )

    offload_container = container(rl_offload_optimizer_during_inference=True)
    offload_container.validate()
    assert not offload_container.ddp.grad_buffer_offload
    offload_container.rl.rl_offload_optimizer_during_inference = False
    offload_container.validate()
    assert not offload_container.ddp.grad_buffer_offload
    offload_container.rl.rl_offload_optimizer_during_inference = True
    offload_container.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    with pytest.raises(ValueError, match="parameters share the gradient-buffer storage"):
        offload_container.validate()

    cuda_graph_container = container(
        rl_offload_optimizer_during_inference=True, rl_training_cuda_graphs=True
    )
    cuda_graph_container.model.transformer.cuda_graph_impl = "local"
    cuda_graph_container.validate()
    assert not cuda_graph_container.ddp.grad_buffer_offload
    cuda_graph_container.rl.finalize_ddp_config(cuda_graph_container.ddp)
    assert not cuda_graph_container.ddp.grad_buffer_offload
    explicit_layout_container = container(rl=RLConfig())
    explicit_layout_container.ddp.grad_buffer_offload = True
    explicit_layout_container.validate()
    assert explicit_layout_container.ddp.grad_buffer_offload
    explicit_layout_container.ddp.fp8_param_gather = True
    explicit_layout_container.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    with pytest.raises(ValueError, match="parameters share the gradient-buffer storage"):
        explicit_layout_container.validate()
    with pytest.raises(ValueError, match="Training requires model.transformer"):
        container(model=TopologyOnlyModelConfig(), rl_training_cuda_graphs=True).validate()
    with pytest.raises(ValueError, match="Training requires model.transformer"):
        container(model=TopologyOnlyModelConfig(), rl=RLConfig()).validate()
    # The container has no inference section, so it leaves the weight-offload rule to validate_args.
    container(
        rl_offload_inference_model_weights=True, rl_inference_model_unified_memory_level=1
    ).validate()
    with pytest.raises(ValueError, match="no CUDA graphs"):
        container(model=TransformerModelConfig(), rl_training_cuda_graphs=True).validate()
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
    argv = (
        "test_rl_config.py --num-layers 2 --hidden-size 128 --num-attention-heads 8 "
        f"--micro-batch-size 1 --global-batch-size {2 * dp} --seq-length 32 "
        f"--max-position-embeddings 32 --perform-rl-step --grpo-prompts-per-step {dp} "
        "--grpo-group-size 2 --rl-offload-inference-model-weights "
        "--rl-inference-model-unified-memory-level 1"
    ).split()
    with (
        patch("sys.argv", argv),
        pytest.raises(ValueError, match="requires a separate inference model"),
    ):
        validate_args(parse_args())
    with patch("sys.argv", argv + ["--rl-inference-tensor-model-parallel-size", "1"]):
        validate_args(parse_args())
    # The retired inference-only spelling points at the new flag, which does not combine with
    # --skip-train; the variant without an optimizer is --rl-rollout-only --no-load-optim.
    argv = (
        "test_rl_config.py --num-layers 2 --hidden-size 128 --num-attention-heads 8 "
        f"--micro-batch-size 1 --global-batch-size {2 * dp} --seq-length 32 "
        f"--max-position-embeddings 32 --perform-rl-step --grpo-prompts-per-step {dp} "
        "--grpo-group-size 2 --train-iters 10"
    ).split()
    for retired in (["--skip-train"], ["--skip-train", "--rl-rollout-only"]):
        with (
            patch("sys.argv", argv + retired),
            pytest.raises(ValueError, match="--rl-rollout-only collects"),
        ):
            validate_args(parse_args())
    with patch("sys.argv", argv + ["--rl-rollout-only", "--no-load-optim"]):
        args = validate_args(parse_args())
    assert args.rl_rollout_only and args.no_load_optim


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
    assert serialize_value(ddp)["grad_buffer_offload"] == (ddp.grad_buffer_offload)


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
    from megatron.training.models import GPTModelConfig
    from megatron.training.state import TrainState

    parser = argparse.ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    # The container, not legacy args, owns the RL policy.
    args.perform_rl_step = not perform_rl_step
    args.batch_parallel_size = 1
    args.world_size = 1
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    monkeypatch.setattr(global_vars, "_GLOBAL_TRAIN_STATE", TrainState())
    run_config.rl = RLConfig(
        perform_rl_step=perform_rl_step, rl_offload_optimizer_during_inference=True
    )
    ddp = run_config.ddp
    finalize = Mock(wraps=run_config.rl.finalize_ddp_config)
    monkeypatch.setattr(run_config.rl, "finalize_ddp_config", finalize)
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
        run_config.model = GPTModelConfig(
            transformer=run_config.model.transformer, vocab_size=128, seq_length=16
        )
        monkeypatch.setattr(run_config.model, "get_builder_cls", lambda: lambda cfg: builder)
    else:
        monkeypatch.setattr(training, "get_model", construct)
    # Runtime initialization owns finalization; model setup only consumes its result.
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 8
    run_config.finalize()
    with pytest.raises(ReachedConstruction):
        training.setup_model_and_optimizer(
            Mock(), model_provider_func=None if native_builder else Mock(), cfg_container=run_config
        )

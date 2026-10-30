# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scheduler settings are owned by SchedulerConfig; consumers do not read them from args."""

from argparse import ArgumentParser
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.training import arguments, global_vars, training
from megatron.training.argument_utils import _default_config_from_args
from megatron.training.config import SchedulerConfig, TrainingConfig, container


def cli_config(*options):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(options)
    return args, _default_config_from_args(SchedulerConfig, args)


def remove_scheduler_args(args):
    for field in fields(SchedulerConfig):
        if field.init and hasattr(args, field.name):
            delattr(args, field.name)


def test_cli_and_native_settings_match_without_aliasing():
    args, config = cli_config(
        "--lr-decay-style",
        "cosine",
        "--lr-decay-iters",
        "100",
        "--lr-warmup-iters",
        "10",
        "--lr-warmup-init",
        "1e-6",
        "--weight-decay-incr-style",
        "linear",
        "--start-weight-decay",
        "0.01",
        "--end-weight-decay",
        "0.1",
        "--override-opt-param-scheduler",
    )
    native = SchedulerConfig(
        lr_decay_style="cosine",
        lr_decay_iters=100,
        lr_warmup_iters=10,
        lr_warmup_init=1e-6,
        weight_decay_incr_style="linear",
        start_weight_decay=0.01,
        end_weight_decay=0.1,
        override_opt_param_scheduler=True,
    )
    assert asdict(config) == asdict(native)
    # The config is a snapshot: later edits to args must not leak into it.
    args.lr_decay_iters = 1
    args.lr_decay_style = "constant"
    args.start_weight_decay = 5.0
    assert config.lr_decay_iters == 100
    assert config.lr_decay_style == "cosine"
    assert config.start_weight_decay == 0.01


def test_deprecated_no_weight_decay_cond_type_is_not_a_cli_argument():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-weight-decay-cond-type", "apply_wd_to_qk_layernorm"])
    assert "no_weight_decay_cond_type" not in {f.name for f in fields(SchedulerConfig)}


@pytest.fixture
def build_scheduler(monkeypatch, run_config):
    """Resolve the scheduler steps at finalize, then run the real get_optimizer_param_scheduler."""
    scheduler_cls = Mock()
    monkeypatch.setattr(training, "OptimizerParamScheduler", scheduler_cls)
    monkeypatch.setattr(container, "update_train_iters", Mock())

    def build(config, **train):
        args, _ = cli_config()
        # Everything the scheduler reads from SchedulerConfig must come from the config owner.
        remove_scheduler_args(args)
        args.lr, args.min_lr = 1e-3, 1e-5
        run_config.train = TrainingConfig(global_batch_size=4, **train)
        run_config.scheduler = config
        run_config._resolve_scheduler_step_fields()
        monkeypatch.setattr(training, "get_args", lambda: args)
        training.get_optimizer_param_scheduler(Mock())
        return scheduler_cls.call_args.kwargs

    return build


def test_iteration_based_scheduler_uses_config(build_scheduler):
    config = SchedulerConfig(
        lr_decay_style="cosine",
        lr_decay_iters=50,
        lr_warmup_iters=5,
        lr_warmup_init=1e-6,
        lr_wsd_decay_iters=7,
        lr_wsd_decay_style="linear",
        start_weight_decay=0.0,
        end_weight_decay=0.1,
        weight_decay_incr_style="linear",
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False,
    )
    kwargs = build_scheduler(config, train_iters=100)
    assert kwargs["init_lr"] == 1e-6
    assert kwargs["lr_decay_style"] == "cosine"
    assert kwargs["lr_decay_steps"] == 50 * 4
    assert kwargs["lr_warmup_steps"] == 5 * 4
    assert kwargs["wsd_decay_steps"] == 7 * 4
    assert kwargs["lr_wsd_decay_style"] == "linear"
    assert kwargs["wd_incr_steps"] == 100 * 4
    assert kwargs["wd_incr_style"] == "linear"
    assert kwargs["start_wd"] == 0.0 and kwargs["end_wd"] == 0.1
    assert kwargs["use_checkpoint_opt_param_scheduler"] is True
    assert kwargs["override_opt_param_scheduler"] is False


def test_sample_based_scheduler_uses_config(build_scheduler):
    config = SchedulerConfig(
        lr_decay_samples=400,
        lr_warmup_fraction=0.25,
        lr_wsd_decay_samples=40,
        start_weight_decay=0.01,
        end_weight_decay=0.01,
    )
    kwargs = build_scheduler(config, train_samples=1000)
    assert kwargs["lr_decay_steps"] == 400
    assert kwargs["lr_warmup_steps"] == 0.25 * 400
    assert kwargs["wsd_decay_steps"] == 40
    assert kwargs["wd_incr_steps"] == 1000


@pytest.mark.parametrize("iteration_based", [True, False])
def test_unset_decay_defaults_to_training_length_in_config(build_scheduler, iteration_based):
    config = SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0)
    run = {"train_iters": 100} if iteration_based else {"train_samples": 1000}
    build_scheduler(config, **run)
    if iteration_based:
        assert config.lr_decay_iters == 100
    else:
        assert config.lr_decay_samples == 1000


def test_scheduler_requires_a_training_length(build_scheduler):
    with pytest.raises(Exception, match="train-iters or train-samples"):
        build_scheduler(SchedulerConfig())


@pytest.mark.parametrize(
    "train,scheduler",
    [
        # Iteration-based training rejects sample-based options.
        ({"train_iters": 10, "train_samples": 10}, {}),
        ({"train_iters": 10}, {"lr_decay_samples": 10}),
        ({"train_iters": 10}, {"lr_warmup_samples": 10}),
        ({"train_iters": 10}, {"lr_warmup_fraction": 0.1, "lr_warmup_iters": 5}),
        # Sample-based training rejects iteration-based options.
        ({"train_samples": 10}, {"lr_decay_iters": 10}),
        ({"train_samples": 10}, {"lr_warmup_iters": 10}),
        ({"train_samples": 10}, {"lr_warmup_fraction": 0.1, "lr_warmup_samples": 5}),
    ],
)
def test_mixed_iteration_and_sample_options_are_rejected(run_config, train, scheduler):
    run_config.train = TrainingConfig(**train)
    run_config.scheduler = SchedulerConfig(
        start_weight_decay=0.0, end_weight_decay=0.0, **scheduler
    )
    with pytest.raises(AssertionError):
        run_config.validate()


@pytest.mark.parametrize(
    "train,scheduler",
    [
        ({"train_iters": 10}, {"lr_decay_iters": 5, "lr_warmup_iters": 2}),
        ({"train_iters": 10}, {"lr_warmup_fraction": 0.1}),
        ({"train_samples": 10}, {"lr_decay_samples": 5, "lr_warmup_samples": 2}),
        ({"train_samples": 10}, {"lr_warmup_fraction": 0.1}),
    ],
)
def test_consistent_options_are_accepted(run_config, train, scheduler):
    run_config.train = TrainingConfig(**train)
    run_config.scheduler = SchedulerConfig(
        start_weight_decay=0.0, end_weight_decay=0.0, **scheduler
    )
    run_config.optimizer.weight_decay = 0.0
    run_config.validate()


def test_skip_train_bypasses_lr_validation(run_config):
    run_config.train = TrainingConfig(train_iters=10, train_samples=10)
    run_config.validation.skip_train = True
    run_config.scheduler = SchedulerConfig(
        lr_decay_samples=10, start_weight_decay=0.0, end_weight_decay=0.0
    )
    run_config.optimizer.weight_decay = 0.0
    run_config.validate()


def test_cli_options_are_validated_by_the_scheduler_config():
    args, config = cli_config(
        "--train-iters", "10", "--lr-decay-samples", "5", "--micro-batch-size", "1"
    )
    train = _default_config_from_args(TrainingConfig, args)
    # CLI argument validation no longer owns these checks, so it does not reject them.
    assert config.lr_decay_samples == 5 and train.train_iters == 10
    with pytest.raises(AssertionError, match="expected iteration-based learning rate decay"):
        config.validate_run(
            train_iters=train.train_iters,
            train_samples=train.train_samples,
            skip_train=False,
            weight_decay=0.0,
        )


@pytest.fixture
def validate_at_pretrain(monkeypatch, run_config):
    """Exercise validation through pretrain without initializing runtime services."""

    class BeforeInitialization(Exception):
        pass

    monkeypatch.setattr(training, "initialize_megatron", Mock(side_effect=BeforeInitialization))
    monkeypatch.setattr(training.ft_integration, "setup", Mock())

    def validate(scheduler, **train):
        run_config.train = TrainingConfig(**train)
        run_config.scheduler = scheduler
        with pytest.raises(BeforeInitialization):
            training.pretrain(run_config, Mock(), Mock())

    return validate


def test_pretrain_validates_scheduler_before_initialization(validate_at_pretrain, run_config):
    run_config.optimizer.weight_decay = 0.0
    scheduler = SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0)
    validate_at_pretrain(scheduler, train_iters=10)
    run_config.scheduler = SchedulerConfig(
        lr_decay_samples=1, start_weight_decay=0.0, end_weight_decay=0.0
    )
    run_config.optimizer.weight_decay = 0.0
    run_config.train = TrainingConfig(train_iters=10)
    with pytest.raises(AssertionError):
        training.pretrain(run_config, Mock(), Mock())


@pytest.fixture
def finalizable(monkeypatch, run_config):
    """Provide the args and micro batch size that finalization still needs."""
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", SimpleNamespace(batch_parallel_size=1))
    run_config.train.micro_batch_size = 1


def test_constant_weight_decay_is_derived_from_optimizer_at_finalize(finalizable, run_config):
    run_config.optimizer.weight_decay = 0.05
    run_config.scheduler = SchedulerConfig(weight_decay_incr_style="constant")
    assert run_config.scheduler.start_weight_decay is None
    assert run_config.scheduler.end_weight_decay is None
    run_config.validate()
    # Validation never derives values.
    assert run_config.scheduler.start_weight_decay is None
    run_config.finalize()
    assert run_config.scheduler.start_weight_decay == 0.05
    assert run_config.scheduler.end_weight_decay == 0.05


def test_validation_is_repeatable_after_finalize(finalizable, run_config):
    run_config.optimizer.weight_decay = 0.05
    run_config.train = TrainingConfig(train_iters=10, micro_batch_size=1)
    run_config.scheduler = SchedulerConfig(weight_decay_incr_style="constant")
    run_config.finalize()
    # An in-process restart re-enters pretrain, which validates the finalized config again.
    run_config.validate()
    run_config.finalize()
    assert run_config.scheduler.start_weight_decay == 0.05


def test_constant_weight_decay_must_match_optimizer(run_config):
    run_config.optimizer.weight_decay = 0.05
    run_config.scheduler = SchedulerConfig(
        weight_decay_incr_style="constant", start_weight_decay=0.1, end_weight_decay=0.1
    )
    with pytest.raises(AssertionError):
        run_config.validate()


@pytest.mark.parametrize("style", ["linear", "cosine"])
@pytest.mark.parametrize("missing", ["start_weight_decay", "end_weight_decay"])
def test_scheduled_weight_decay_requires_both_bounds(run_config, style, missing):
    bounds = {"start_weight_decay": 0.0, "end_weight_decay": 0.1}
    bounds[missing] = None
    run_config.scheduler = SchedulerConfig(weight_decay_incr_style=style, **bounds)
    with pytest.raises(AssertionError):
        run_config.validate()


@pytest.mark.parametrize("style", ["linear", "cosine"])
def test_scheduled_weight_decay_is_not_overwritten_at_finalize(finalizable, run_config, style):
    run_config.optimizer.weight_decay = 0.05
    run_config.scheduler = SchedulerConfig(
        weight_decay_incr_style=style, start_weight_decay=0.0, end_weight_decay=0.1
    )
    run_config.finalize()
    assert run_config.scheduler.start_weight_decay == 0.0
    assert run_config.scheduler.end_weight_decay == 0.1

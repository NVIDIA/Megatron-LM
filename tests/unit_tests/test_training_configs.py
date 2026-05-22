# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import signal

from megatron.training.config import (
    CheckpointConfig,
    DistributedInitConfig,
    LoggerConfig,
    ProfilingConfig,
    RNGConfig,
    RerunStateMachineConfig,
    SchedulerConfig,
    StragglerDetectionConfig,
    TrainingConfig,
    ValidationConfig,
)


def _field_names(config_cls):
    return {field.name for field in dataclasses.fields(config_cls)}


def test_training_config_defaults_and_field_metadata():
    config = TrainingConfig(micro_batch_size=2, global_batch_size=8)

    assert config.micro_batch_size == 2
    assert config.global_batch_size == 8
    assert config.exit_signal == signal.SIGTERM
    assert config.iterations_to_skip == []
    assert "rampup_batch_size" in _field_names(TrainingConfig)


def test_validation_and_scheduler_config_defaults():
    validation = ValidationConfig()
    scheduler = SchedulerConfig()

    assert validation.eval_iters == 100
    assert not validation.skip_train
    assert scheduler.lr_decay_style == "linear"
    assert scheduler.lr_warmup_iters == 0
    assert scheduler.override_opt_param_scheduler is False
    assert "lr_decay_steps" in _field_names(SchedulerConfig)


def test_logger_and_checkpoint_config_defaults():
    logger = LoggerConfig()
    checkpoint = CheckpointConfig()

    assert logger.log_interval == 100
    assert logger.timing_log_option == "minmax"
    assert checkpoint.save is None
    assert checkpoint.load is None
    assert "async_save" in _field_names(CheckpointConfig)


def test_common_config_defaults(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "5")

    rng = RNGConfig()
    profiling = ProfilingConfig()
    distributed = DistributedInitConfig()

    assert rng.seed == 1234
    assert not rng.te_rng_tracker
    assert profiling.profile_step_start == 10
    assert profiling.profile_step_end == 12
    assert distributed.distributed_backend == "nccl"
    assert distributed.local_rank == 5
    assert distributed.use_gloo_process_groups


def test_resilience_config_defaults():
    rerun = RerunStateMachineConfig()
    straggler = StragglerDetectionConfig()

    assert rerun.rerun_mode == "validate_results"
    assert rerun.error_injection_type == "transient_error"
    assert rerun.check_for_nan_in_loss
    assert not straggler.log_straggler
    assert straggler.straggler_ctrlr_port == 65535

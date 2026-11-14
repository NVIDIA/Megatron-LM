"""Shared configuration ownership, native flags, and saved driver settings."""

from argparse import ArgumentParser
from dataclasses import dataclass

import pytest

from megatron.training import arguments
from megatron.training.config import DistributedInitConfig, SchedulerConfig, TrainingConfig
from megatron.training.config.container import ConfigContainerBase
from megatron.training.config.instantiate_utils import target_allowlist


@dataclass(kw_only=True)
class DriverConfig(ConfigContainerBase):
    train: TrainingConfig
    scheduler: SchedulerConfig
    dist: DistributedInitConfig


def test_driver_settings_survive_nested_serialization():
    config = DriverConfig(
        train=TrainingConfig(
            check_optimizer_step_success=False, skip_sync_grad_norm_across_mp=True
        ),
        scheduler=SchedulerConfig(max_steps=1000),
        dist=DistributedInitConfig(external_gpu_device_mapping=True),
    )
    saved = config.to_dict()
    target_allowlist.add_exact(saved["_target_"])
    try:
        loaded = DriverConfig.from_dict(saved)
    finally:
        target_allowlist.remove_exact(saved["_target_"])
    assert not loaded.train.check_optimizer_step_success
    assert loaded.train.skip_sync_grad_norm_across_mp
    assert loaded.scheduler.max_steps == 1000
    assert loaded.dist.external_gpu_device_mapping is True
    loaded.dist.lazy_init = True
    assert loaded.dist.lazy_mpu_init is True


def test_native_cli_owns_training_flags_and_accepts_disabled_timer_reports():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    flags = parser._option_string_actions
    assert "--num-epochs" not in flags
    assert "--max-steps" not in flags
    assert "--also-save-hf-checkpoint" not in flags
    assert "--external-gpu-device-mapping" not in flags
    args = parser.parse_args(
        [
            "--eval-iters",
            "7",
            "--timing-log-level",
            "-1",
            "--no-check-optimizer-step-success",
            "--skip-sync-grad-norm-across-mp",
        ]
    )
    assert args.eval_iters == 7
    assert args.timing_log_level == -1
    assert not args.check_optimizer_step_success
    assert args.skip_sync_grad_norm_across_mp
    defaults = parser.parse_args([])
    assert defaults.check_optimizer_step_success
    assert not defaults.skip_sync_grad_norm_across_mp
    assert defaults.timing_log_level == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--timing-log-level", "-2"])

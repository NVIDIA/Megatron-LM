"""Shared schemas preserve driver settings without claiming native CLI features."""

from argparse import ArgumentParser
from dataclasses import dataclass

import pytest

from megatron.training import arguments
from megatron.training.config import CheckpointConfig, DistributedInitConfig, TrainingConfig
from megatron.training.config.container import ConfigContainerBase
from megatron.training.config.instantiate_utils import target_allowlist


@dataclass(kw_only=True)
class DriverConfig(ConfigContainerBase):
    train: TrainingConfig
    checkpoint: CheckpointConfig
    dist: DistributedInitConfig


def test_driver_settings_survive_nested_serialization():
    config = DriverConfig(
        train=TrainingConfig(num_epochs=1.5, global_batch_size=8),
        checkpoint=CheckpointConfig(
            also_save_hf_checkpoint=True,
            hf_source_path="org/source",
            hf_distributed_save=True,
            stage_precision_aware_optimizer_state_on_cpu=True,
        ),
        dist=DistributedInitConfig(external_gpu_device_mapping=True),
    )
    saved = config.to_dict()
    target_allowlist.add_exact(saved["_target_"])
    try:
        loaded = DriverConfig.from_dict(saved)
    finally:
        target_allowlist.remove_exact(saved["_target_"])
    assert loaded.train.num_epochs == 1.5
    assert loaded.checkpoint.hf_source_path == "org/source"
    assert loaded.checkpoint.hf_distributed_save
    assert loaded.checkpoint.stage_precision_aware_optimizer_state_on_cpu
    assert loaded.dist.external_gpu_device_mapping is True
    loaded.dist.lazy_init = True
    assert loaded.dist.lazy_mpu_init is True


def test_native_cli_does_not_advertise_bridge_runtime_features():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    flags = parser._option_string_actions
    assert "--num-epochs" not in flags
    assert "--max-steps" not in flags
    assert "--also-save-hf-checkpoint" not in flags
    assert "--external-gpu-device-mapping" not in flags
    # Existing validation flags remain owned by ValidationConfig.
    args = parser.parse_args(["--eval-iters", "7", "--timing-log-level", "2"])
    assert args.eval_iters == 7
    assert args.timing_log_level == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["--timing-log-level", "-1"])

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline KD constraints against the training container's sections."""

import copy

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.training.config import (
    AlgorithmConfigContainer,
    LogitsLoadConfig,
    LogitsSaveConfig,
    OfflineDistillationConfig,
    RootConfigContainer,
)
from megatron.training.models.distillation import DistillationConfig


@pytest.fixture
def offline_run_config(run_config):
    run_config.train.micro_batch_size = 1
    run_config.train.global_batch_size = 8
    run_config.data.dataset = GPTDatasetConfig(
        sequence_length=16,
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    run_config.checkpoint.async_save = True
    run_config.checkpoint.use_persistent_ckpt_worker = True
    return run_config


def _phase_config(phase):
    if phase == "teacher_save":
        return OfflineDistillationConfig(teacher_save=LogitsSaveConfig(directory="/cache"))
    return OfflineDistillationConfig(student_load=LogitsLoadConfig(directory="/cache"))


def _offline_run(run_config, phase, distillation=None):
    """A run's root container: `run_config` and one offline KD phase."""
    return RootConfigContainer(
        training=run_config,
        algorithm=AlgorithmConfigContainer(
            distillation=distillation, offline_distillation=_phase_config(phase)
        ),
    )


@pytest.mark.parametrize("phase", ["teacher_save", "student_load"])
def test_valid_offline_run_validation_is_read_only(offline_run_config, phase):
    root = _offline_run(offline_run_config, phase)
    before = copy.deepcopy(root.to_dict())
    for _ in range(2):
        root.validate()
        assert root.to_dict() == before
        assert not root._finalized and not offline_run_config._finalized


@pytest.mark.parametrize("phase", ["teacher_save", "student_load"])
@pytest.mark.parametrize("schedule", ["rampup_batch_size", "step_batch_size_schedule"])
def test_batch_schedules_are_rejected(offline_run_config, phase, schedule):
    cfg = offline_run_config
    setattr(cfg.train, schedule, [8, 8, 1000] if schedule == "rampup_batch_size" else "0:8 1K:16")
    with pytest.raises(ValueError, match="fixed global batch"):
        _offline_run(cfg, phase).validate()


@pytest.mark.parametrize("async_save,persistent", [(False, False), (True, False), (False, True)])
def test_saving_requires_the_async_checkpoint_worker(offline_run_config, async_save, persistent):
    cfg = offline_run_config
    cfg.checkpoint.async_save = async_save
    cfg.checkpoint.use_persistent_ckpt_worker = persistent
    # CheckpointConfig also rejects an async distributed save without a worker.
    with pytest.raises((ValueError, AssertionError), match="persistent.ckpt.worker"):
        _offline_run(cfg, "teacher_save").validate()


def test_cached_and_online_student_losses_are_exclusive(offline_run_config):
    cfg = offline_run_config
    cfg.model.transformer.cross_entropy_fusion_impl = "native"
    distillation = DistillationConfig(
        teacher_load="/teacher", teacher_model=copy.deepcopy(cfg.model)
    )
    with pytest.raises(ValueError, match="online ModelOpt"):
        _offline_run(cfg, "student_load", distillation).validate()


def test_offline_run_requires_a_dataset_for_cache_identity(offline_run_config):
    cfg = offline_run_config
    cfg.data.dataset = None
    with pytest.raises(ValueError, match="cfg.data.dataset"):
        _offline_run(cfg, "student_load").validate()

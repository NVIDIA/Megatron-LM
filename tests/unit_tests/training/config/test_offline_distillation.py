# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline KD constraints involving other training-run sections."""

import copy

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.training import global_vars
from megatron.training.config import LogitsLoadConfig, LogitsSaveConfig, OfflineDistillationConfig
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
    run_config.checkpoint.save = "/save"
    run_config.checkpoint.async_save = True
    run_config.checkpoint.use_persistent_ckpt_worker = True
    return run_config


def _phase_config(phase):
    if phase == "teacher_save":
        return OfflineDistillationConfig(teacher_save=LogitsSaveConfig(directory="/cache"))
    return OfflineDistillationConfig(student_load=LogitsLoadConfig(directory="/cache"))


@pytest.mark.parametrize("phase", ["teacher_save", "student_load"])
def test_valid_offline_run_validation_is_read_only(offline_run_config, phase):
    cfg = offline_run_config
    cfg.offline_distillation = _phase_config(phase)
    before = copy.deepcopy(cfg.to_dict())
    for _ in range(2):
        cfg.validate()
        assert cfg.to_dict() == before
        assert not cfg._finalized


@pytest.mark.parametrize("phase", ["teacher_save", "student_load"])
@pytest.mark.parametrize(
    "schedule,value",
    [
        ("rampup_batch_size", [8, 8, 1000]),
        ("step_batch_size_schedule", "0:8 1K:16"),
        ("decrease_batch_size_if_needed", True),
    ],
)
def test_batch_schedules_are_rejected(offline_run_config, phase, schedule, value):
    cfg = offline_run_config
    cfg.offline_distillation = _phase_config(phase)
    setattr(cfg.train, schedule, value)
    with pytest.raises(ValueError, match="fixed global batch"):
        cfg.validate()


@pytest.mark.parametrize("async_save,persistent", [(False, False), (True, False), (False, True)])
def test_saving_requires_the_async_checkpoint_worker(offline_run_config, async_save, persistent):
    cfg = offline_run_config
    cfg.offline_distillation = _phase_config("teacher_save")
    cfg.checkpoint.async_save = async_save
    cfg.checkpoint.use_persistent_ckpt_worker = persistent
    # CheckpointConfig also rejects an async distributed save without a worker.
    with pytest.raises((ValueError, AssertionError), match="persistent.ckpt.worker"):
        cfg.validate()


def test_cached_and_online_student_losses_are_exclusive(offline_run_config):
    cfg = offline_run_config
    cfg.offline_distillation = _phase_config("student_load")
    cfg.model.transformer.cross_entropy_fusion_impl = "native"
    cfg.distillation = DistillationConfig(
        teacher_load="/teacher", teacher_model=copy.deepcopy(cfg.model)
    )
    with pytest.raises(ValueError, match="online ModelOpt"):
        cfg.validate()


def test_offline_run_requires_a_dataset_for_cache_identity(offline_run_config):
    cfg = offline_run_config
    cfg.offline_distillation = _phase_config("student_load")
    cfg.data.dataset = None
    with pytest.raises(ValueError, match="cfg.data.dataset"):
        cfg.validate()


def test_saving_requires_a_checkpoint_save_directory(offline_run_config):
    cfg = offline_run_config
    cfg.offline_distillation = _phase_config("teacher_save")
    cfg.checkpoint.save = None
    with pytest.raises(ValueError, match="checkpoint.save"):
        cfg.validate()


@pytest.mark.parametrize("frozen", [False, True])
def test_saving_from_a_trainable_model_warns_at_finalize(
    monkeypatch, offline_run_config, caplog, frozen
):
    cfg = offline_run_config
    cfg.offline_distillation = _phase_config("teacher_save")
    cfg.train.freeze_all_layers = frozen
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    with caplog.at_level("WARNING", logger="megatron.training.config.container"):
        cfg.finalize()
    assert ("without train.freeze_all_layers" in caplog.text) is not frozen

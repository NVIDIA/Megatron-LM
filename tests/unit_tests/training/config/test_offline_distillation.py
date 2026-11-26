# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline KD field validation, CLI/serialization compatibility and run constraints."""

import copy
from argparse import ArgumentParser

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.training import global_vars
from megatron.training.argument_utils import offline_distillation_config_from_args
from megatron.training.arguments import _add_logits_distillation_args
from megatron.training.config import LogitsLoadConfig, LogitsSaveConfig, OfflineDistillationConfig
from megatron.training.config.instantiate_utils import instantiate
from megatron.training.config.utils import serialize_dataclass
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


def _cli_config(*flags):
    parser = _add_logits_distillation_args(ArgumentParser())
    return offline_distillation_config_from_args(parser.parse_args(flags))


@pytest.mark.parametrize("phase", ["teacher_save", "student_load"])
def test_cli_and_native_config_round_trip(phase):
    """The old CLI and a saved config select the same behavior, including non-default knobs."""
    if phase == "teacher_save":
        native = OfflineDistillationConfig(
            teacher_save=LogitsSaveConfig(
                directory="/cache", top_k=32, top_p=0.9, min_k=2, dtype="bf16"
            )
        )
        cli = _cli_config(
            "--logits-save-dir",
            "/cache",
            "--logits-save-top-k",
            "32",
            "--logits-save-top-p",
            "0.9",
            "--logits-save-top-p-min-k",
            "2",
            "--logits-save-dtype",
            "bf16",
        )
    else:
        native = OfflineDistillationConfig(
            student_load=LogitsLoadConfig(
                directory="/cache",
                loss_alpha=0.7,
                decode_threads=2,
                msc_prefetch_depth=0,
                ignore_errors=True,
                ignore_hash=True,
            )
        )
        cli = _cli_config(
            "--logits-load-dir",
            "/cache",
            "--logits-load-kd-loss-alpha",
            "0.7",
            "--logits-load-decode-threads",
            "2",
            "--logits-load-msc-prefetch-depth",
            "0",
            "--logits-load-ignore-errors",
            "--logits-load-ignore-hash",
        )
    restored = instantiate(serialize_dataclass(native))
    restored.validate()
    assert restored == cli == native


@pytest.mark.parametrize("both", [False, True])
def test_exactly_one_phase_is_required(both):
    cfg = OfflineDistillationConfig(
        teacher_save=LogitsSaveConfig(directory="/out") if both else None,
        student_load=LogitsLoadConfig(directory="/in") if both else None,
    )
    with pytest.raises(ValueError, match="exactly one"):
        cfg.validate()


def test_cli_rejects_both_phases_and_disables_kd_without_either():
    assert _cli_config() is None
    with pytest.raises(ValueError, match="exactly one"):
        _cli_config("--logits-save-dir", "/out", "--logits-load-dir", "/in")


@pytest.mark.parametrize(
    "config,message",
    [
        (LogitsSaveConfig(directory=""), "directory"),
        (LogitsSaveConfig(), "directory"),
        (LogitsLoadConfig(), "directory"),
        (LogitsSaveConfig(directory="/cache", top_k=0), "top_k"),
        (LogitsSaveConfig(directory="/cache", top_p=0), "top_p"),
        (LogitsSaveConfig(directory="/cache", top_k=2, min_k=3), "min_k"),
        (LogitsSaveConfig(directory="/cache", dtype="int8"), "dtype"),
        (LogitsLoadConfig(directory="/cache", loss_alpha=1.1), "loss_alpha"),
        (LogitsLoadConfig(directory="/cache", decode_threads=0), "decode_threads"),
        (LogitsLoadConfig(directory="/cache", msc_prefetch_depth=-1), "prefetch"),
    ],
)
def test_invalid_cache_settings_fail_before_runtime(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()

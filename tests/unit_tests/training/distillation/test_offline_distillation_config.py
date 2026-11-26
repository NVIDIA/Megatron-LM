# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline KD configuration, CLI compatibility and cache-identity contracts."""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.training.argument_utils import offline_distillation_config_from_args
from megatron.training.arguments import _add_logits_distillation_args
from megatron.training.config import (
    LogitsLoadConfig,
    LogitsSaveConfig,
    OfflineDistillationConfig,
    TrainingConfig,
)
from megatron.training.config.dataset_config import DataConfig, dataset_type_from_family
from megatron.training.config.instantiate_utils import instantiate
from megatron.training.config.utils import serialize_dataclass
from megatron.training.distillation import utils


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


def test_dataset_identity_preserves_the_existing_cache_schema():
    cfg = SimpleNamespace(
        train=TrainingConfig(train_iters=5, micro_batch_size=1, global_batch_size=8),
        rng=SimpleNamespace(seed=1234),
        data=SimpleNamespace(
            dataset=SimpleNamespace(
                blend=(["/new/mount/a", "/new/mount/b"], [0.25, 0.75]), blend_per_split=None
            )
        ),
    )
    digest, identifiers = utils.compute_dataset_hash(cfg)
    assert identifiers == {
        "seed": 1234,
        "train_samples": 40,
        "blend": {"kind": "blend", "blend": [[0.25, "a"], [0.75, "b"]]},
    }
    # The identity stored in pre-config-refactor cache metadata for this sample stream.
    assert digest == "c80444069b8dba1182ad29ea78127ba9"


@pytest.mark.parametrize("sft,inter_document", [(True, False), (False, True), (False, False)])
def test_packing_comes_from_the_dataset_config(sft, inter_document):
    data = DataConfig(
        dataset=GPTDatasetConfig(
            sequence_length=16,
            random_seed=1234,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            inter_document_masking=inter_document,
            create_attention_mask=not inter_document,
        ),
        dataset_type=dataset_type_from_family("sft" if sft else "gpt", mock=True),
    )
    assert utils.is_packed_sequence_run(data) is (sft or inter_document)


def test_sample_position_follows_live_train_state(monkeypatch):
    state = SimpleNamespace(consumed_train_samples=16)
    monkeypatch.setattr(utils, "get_train_state", lambda: state)
    assert utils.get_consumed_train_samples() == 16
    state.consumed_train_samples = 24
    assert utils.get_consumed_train_samples() == 24

# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for --freeze-all-layers auto-resume iteration reading.

``read_frozen_resume_iteration`` reads the progress tracker
(``latest_checkpointed_iteration.txt``) that a frozen (--freeze-all-layers) run writes
to its --save dir, so an identical resubmitted job continues where it stopped instead
of restarting. Its main use today is offline-KD teacher-logit dumps. Pure file I/O
(no CUDA, no distributed init), so it runs on CPU.
"""

from argparse import Namespace

import pytest

from megatron.training import checkpointing
from megatron.training.checkpointing import (
    get_checkpoint_tracker_filename,
    read_frozen_resume_iteration,
)
from megatron.training.models import ModelConfig


def _write_tracker(dir_path, content):
    with open(get_checkpoint_tracker_filename(str(dir_path)), "w") as f:
        f.write(content)


def test_missing_tracker_is_fresh_dump(tmp_path):
    """No tracker in the load dir -> start at iteration 0 (first run, --finetune-like)."""
    assert read_frozen_resume_iteration(str(tmp_path)) == 0


def test_none_load_dir_is_zero():
    """A None load dir (nothing to resume from) -> 0."""
    assert read_frozen_resume_iteration(None) == 0


def test_reads_recorded_iteration(tmp_path):
    """A tracker written by a prior dump is read back as the resume iteration."""
    _write_tracker(tmp_path, "1500")
    assert read_frozen_resume_iteration(str(tmp_path)) == 1500


def test_tolerates_trailing_whitespace(tmp_path):
    """A trailing newline in the tracker is stripped (read_metadata semantics)."""
    _write_tracker(tmp_path, "42\n")
    assert read_frozen_resume_iteration(str(tmp_path)) == 42


def test_release_tracker_is_zero(tmp_path):
    """A 'release' tracker maps to iteration 0 (start from the beginning)."""
    _write_tracker(tmp_path, "release")
    assert read_frozen_resume_iteration(str(tmp_path)) == 0


def test_advancing_progress_reads_latest(tmp_path):
    """Overwriting the tracker (progress advancing across runs) reads the newest value."""
    _write_tracker(tmp_path, "100")
    assert read_frozen_resume_iteration(str(tmp_path)) == 100
    _write_tracker(tmp_path, "250")
    assert read_frozen_resume_iteration(str(tmp_path)) == 250


@pytest.mark.parametrize("frozen", [False, True])
def test_frozen_resume_policy_uses_model_config(monkeypatch, tmp_path, run_config, frozen):
    """Frozen-model resume follows the model config even with contradictory legacy args."""
    _write_tracker(tmp_path, "42")
    run_config.model = ModelConfig(freeze_all_layers=frozen)
    args = Namespace(
        load="teacher",
        save=str(tmp_path),
        pretrained_checkpoint=None,
        freeze_all_layers=not frozen,
        finetune=False,
        override_ckpt_iteration=None,
    )
    monkeypatch.setattr(checkpointing, "get_args", lambda: args)

    class ReachedModelLoad(Exception):
        pass

    def stop_at_model_load(model):
        raise ReachedModelLoad

    monkeypatch.setattr(checkpointing, "unwrap_model", stop_at_model_load)
    with pytest.raises(ReachedModelLoad):
        checkpointing.load_checkpoint([], None, None, restore_training_state=True)
    assert args.finetune is frozen
    assert args.override_ckpt_iteration == (42 if frozen else None)

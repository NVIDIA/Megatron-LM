# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A checkpoint requested by the rerun state machine is skipped, with a warning, without --save.

After a reproducible unexpected result the rerun state machine asks the training loop to save a
checkpoint and exit. Without --save, saving used to fail in get_checkpoint_name with a TypeError,
which hid the state machine's exit code.
"""

from megatron.training import training


def _capture_rank_0_messages(monkeypatch):
    # print_rank_0 prints only on rank 0; capture its calls so the test holds on every rank.
    messages = []
    monkeypatch.setattr(training, "print_rank_0", messages.append)
    return messages


def test_requested_checkpoint_is_saved_with_save_dir(monkeypatch):
    messages = _capture_rank_0_messages(monkeypatch)
    assert training._rerun_checkpoint_can_be_saved("/checkpoints") is True
    assert messages == []


def test_requested_checkpoint_is_skipped_without_save_dir(monkeypatch):
    messages = _capture_rank_0_messages(monkeypatch)
    assert training._rerun_checkpoint_can_be_saved(None) is False
    assert len(messages) == 1
    assert "requested a checkpoint, but --save is not set" in messages[0]

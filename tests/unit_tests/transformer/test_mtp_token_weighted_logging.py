# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for token-weighted MTP logging across dynamic microbatches."""

import pytest
import torch

from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper


@pytest.fixture(autouse=True)
def tracker(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
    return MTPLossLoggingHelper.tracker


def save(loss, tokens, group=None):
    MTPLossLoggingHelper.save_metrics_to_tracker(
        torch.tensor(float(loss)),
        torch.tensor(1.0 if tokens else 0.0),
        torch.tensor(float(tokens or 0)),
        0,
        1,
        avg_group=group,
        num_tokens=torch.tensor(float(tokens)) if tokens is not None else None,
    )


def test_global_token_weighted_loss_and_acceptance(monkeypatch, tracker):
    group = object()
    save(2, 1, group)
    save(36, 9, group)
    save(0, 0, group)
    # A remote rank contributes 42 loss over 6 tokens. Means of rank/microbatch
    # means would differ from the global reference (2 + 36 + 42) / (1 + 9 + 6).
    remote = iter([42.0, 6.0, 3.0, 6.0])

    def reduce(value, group=None, op=None):
        assert group is expected_group
        assert op == torch.distributed.ReduceOp.SUM
        value.add_(next(remote))

    expected_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    losses = {}
    MTPLossLoggingHelper.track_mtp_metrics(1 / 3, 1, None, total_loss_dict=losses)
    torch.testing.assert_close(losses["mtp_1 loss"], torch.tensor(5.0))
    torch.testing.assert_close(tracker["cumulative_correct_values"], torch.tensor([5.0]))
    torch.testing.assert_close(tracker["cumulative_total_values"], torch.tensor([16.0]))
    for key in ("loss_values", "loss_sums", "loss_token_counts", "correct_values", "total_values"):
        assert tracker[key].count_nonzero() == 0


def test_empty_tokens_and_next_logging_window(tracker):
    save(0, 0)
    losses = {}
    MTPLossLoggingHelper.track_mtp_metrics(0.25, 1, None, total_loss_dict=losses)
    assert losses["mtp_1 loss"].item() == 0
    save(18, 3)
    MTPLossLoggingHelper.track_mtp_metrics(0.5, 2, None, total_loss_dict=losses)
    assert losses["mtp_1 loss"].item() == 6
    assert tracker["cumulative_total_values"].item() == 3


def test_legacy_normalized_loss_still_uses_microbatch_scale(monkeypatch):
    group = object()
    save(2, None, group)
    save(4, None, group)
    operations = []

    def reduce(value, group=None, op=None):
        operations.append(op)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    losses = {}
    MTPLossLoggingHelper.track_mtp_metrics(0.5, 1, None, total_loss_dict=losses)
    assert losses["mtp_1 loss"].item() == 3
    assert operations == [
        torch.distributed.ReduceOp.AVG,
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.SUM,
    ]


def test_rejects_mixed_normalization_modes():
    save(2, 1)
    with pytest.raises(ValueError, match="Cannot mix"):
        save(2, None)

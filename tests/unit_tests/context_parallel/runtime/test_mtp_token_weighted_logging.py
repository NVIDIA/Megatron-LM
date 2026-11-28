# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regression tests for token-weighted MTP logging across dynamic microbatches."""

import pytest
import torch

from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(autouse=True)
def tracker(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(MTPLossLoggingHelper, "tracker", {})
    return MTPLossLoggingHelper.tracker


def save(loss, tokens, group=None):
    """Record one microbatch; ``tokens=None`` selects microbatch-normalized logging.

    The dev logging helper receives the raw local loss sum and token count in both
    modes and normalizes legacy losses itself, so a legacy ``loss`` uses one token.
    """
    MTPLossLoggingHelper.save_loss_to_tracker(
        torch.tensor(float(loss)),
        torch.tensor(float(1 if tokens is None else tokens)),
        0,
        1,
        correct=torch.tensor(1.0 if tokens else 0.0),
        total=torch.tensor(float(tokens or 0)),
        avg_group=group,
        calculate_per_token_loss=tokens is not None,
    )


def test_global_token_weighted_loss_and_acceptance(monkeypatch, tracker):
    group = object()
    save(2, 1, group)
    save(36, 9, group)
    save(0, 0, group)
    # A remote rank contributes 42 loss over 6 tokens and 3 of 6 accepted tokens.
    # Means of rank/microbatch means would differ from the global reference
    # (2 + 36 + 42) / (1 + 9 + 6). Loss sums/token counts and correct/total
    # acceptance counts are each reduced as one packed tensor.
    remote = iter([torch.tensor([42.0, 6.0]), torch.tensor([[3.0], [6.0]])])

    def reduce(value, group=None, op=torch.distributed.ReduceOp.SUM):
        assert group is expected_group
        assert op == torch.distributed.ReduceOp.SUM
        value.add_(next(remote))

    expected_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    losses = {}
    # Per-token training logs MTP losses with a unit loss scale.
    MTPLossLoggingHelper.track_mtp_metrics(1.0, 1, None, total_loss_dict=losses)
    torch.testing.assert_close(losses["mtp_1 loss"], torch.tensor(5.0))
    torch.testing.assert_close(
        tracker["cumulative_acceptance_counts"], torch.tensor([[5.0], [16.0]])
    )
    for key in ("loss_sums", "num_tokens", "acceptance_counts"):
        assert tracker[key].count_nonzero() == 0


def test_empty_tokens_and_next_logging_window(tracker):
    save(0, 0)
    losses = {}
    MTPLossLoggingHelper.track_mtp_metrics(1.0, 1, None, total_loss_dict=losses)
    assert losses["mtp_1 loss"].item() == 0
    save(18, 3)
    MTPLossLoggingHelper.track_mtp_metrics(1.0, 2, None, total_loss_dict=losses)
    assert losses["mtp_1 loss"].item() == 6
    assert tracker["cumulative_acceptance_counts"][1].item() == 3


def test_legacy_normalized_loss_still_uses_microbatch_scale(monkeypatch):
    group = object()
    save(2, None, group)
    save(4, None, group)
    operations = []

    def reduce(value, group=None, op=torch.distributed.ReduceOp.SUM):
        operations.append(op)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    losses = {}
    MTPLossLoggingHelper.track_mtp_metrics(0.5, 1, None, total_loss_dict=losses)
    assert losses["mtp_1 loss"].item() == 3
    # Normalized losses are averaged; packed acceptance counts are summed once.
    assert operations == [torch.distributed.ReduceOp.AVG, torch.distributed.ReduceOp.SUM]


def test_rejects_mixed_normalization_modes():
    save(2, 1)
    with pytest.raises(AssertionError, match="cannot mix"):
        save(2, None)


@pytest.mark.parametrize("first_per_token", [False, True])
@pytest.mark.parametrize("cleanup", ["clean", "log"])
def test_mode_change_requires_tracker_reset(first_per_token, cleanup):
    save(4, 2 if first_per_token else None)
    if cleanup == "clean":
        MTPLossLoggingHelper.clean_metrics_in_tracker()
    else:
        MTPLossLoggingHelper.track_mtp_metrics(0.5, 1, None)
    # Clearing a logging window keeps the accumulators (which CUDA graphs may
    # reference) and their mode; switching modes requires reset_metrics_tracker().
    with pytest.raises(AssertionError, match="cannot mix"):
        save(18, None if first_per_token else 3)
    MTPLossLoggingHelper.reset_metrics_tracker()
    save(18, None if first_per_token else 3)
    losses = {}
    loss_scale = 0.5 if first_per_token else 1.0
    MTPLossLoggingHelper.track_mtp_metrics(loss_scale, 2, None, total_loss_dict=losses)
    assert MTPLossLoggingHelper.tracker["calculate_per_token_loss"] is not first_per_token
    assert losses["mtp_1 loss"].item() == (9 if first_per_token else 6)


def test_reset_discards_tracker_mode_and_storage(tracker):
    save(4, 2)
    MTPLossLoggingHelper.reset_metrics_tracker()
    assert MTPLossLoggingHelper.tracker == {}
    assert MTPLossLoggingHelper.tracker is not tracker
    save(6, None)
    assert MTPLossLoggingHelper.tracker["calculate_per_token_loss"] is False

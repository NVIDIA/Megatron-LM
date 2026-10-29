# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-token replay and recomputation use the same selected experts and gradients."""

import pytest
import torch

from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


@pytest.fixture(autouse=True)
def restore_router_registry():
    """Keep these isolated tests from changing other models' router registries."""
    instances = list(RouterReplay.global_router_replay_instances)
    try:
        yield
    finally:
        RouterReplay.global_router_replay_instances[:] = instances


def topk(scores, k, num_groups=None, group_topk=None):
    return scores.topk(k, dim=-1)


@pytest.mark.parametrize("mask", [None, [True, False, True], [False, False, False]])
def test_masked_replay_preserves_scores_and_gradients(mask):
    scores = torch.tensor([[0.1, 0.8, 0.4], [0.9, 0.2, 0.5], [0.3, 0.7, 0.6]], requires_grad=True)
    target = torch.tensor([[0, 2], [1, 2], [0, 1]], dtype=torch.int16)
    replay = RouterReplay()
    replay.set_target_indices(target, replay_mask=None if mask is None else torch.tensor(mask))
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    values, indices = replay.get_replay_topk(scores, 2, default_compute_topk=topk)
    expected = target.long()
    if mask is not None:
        expected = torch.where(torch.tensor(mask).unsqueeze(-1), expected, scores.topk(2).indices)
    assert torch.equal(indices, expected)
    values.sum().backward()
    expected_grad = torch.zeros_like(scores).scatter_(1, expected, 1)
    assert torch.equal(scores.grad, expected_grad)
    replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    recomputed, recomputed_indices = replay.get_replay_topk(scores, 2, default_compute_topk=topk)
    assert torch.equal(recomputed_indices, indices)
    assert torch.equal(recomputed, values)
    with pytest.raises(RuntimeError, match="no recorded microbatch"):
        replay.get_replay_topk(scores, 2, default_compute_topk=topk)
    replay.clear_indices()
    assert replay.target_replay_mask is None
    assert not replay.replay_backward_mask_list


def test_recompute_consumes_each_microbatch_mask_in_order():
    replay = RouterReplay()
    scores = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    first = torch.tensor([[0], [2]])
    second = torch.tensor([[1], [1]])
    replay.set_target_indices(first, torch.tensor([True, False]))
    replay.set_target_indices(second, torch.tensor([False, True]))
    replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    _, actual_first = replay.get_replay_topk(scores, 1, default_compute_topk=topk)
    _, actual_second = replay.get_replay_topk(scores, 1, default_compute_topk=topk)
    assert actual_first.tolist() == [[0], [0]]
    assert actual_second.tolist() == [[2], [1]]


def test_replay_errors_do_not_silently_use_native_routes():
    replay = RouterReplay()
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    with pytest.raises(RuntimeError, match="requires target"):
        replay.get_replay_topk(torch.randn(2, 3), 1, default_compute_topk=topk)
    replay.set_target_indices(torch.zeros(2, 1, dtype=torch.long), torch.ones(3))
    with pytest.raises(ValueError, match="one entry per token"):
        replay.get_replay_topk(torch.randn(2, 3), 1, default_compute_topk=topk)


def test_backward_reuses_native_rows_selected_by_forward():
    """Unmasked forward choices must survive a different recomputation score order."""
    replay = RouterReplay()
    scores = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    replay.set_target_indices(torch.tensor([[0], [2]]), torch.tensor([True, False]))
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _, selected = replay.get_replay_topk(scores, 1, default_compute_topk=topk)
    replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    def forbidden(*args, **kwargs):
        raise AssertionError("Backward must reuse resolved forward routes.")

    values, recomputed = replay.get_replay_topk(-scores, 1, default_compute_topk=forbidden)
    assert torch.equal(selected, recomputed)
    assert values.tolist() == [[-1.0], [-3.0]]

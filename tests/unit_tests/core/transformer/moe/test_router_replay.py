# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest
import torch

from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction


def setup_function():
    RouterReplay.global_router_replay_instances.clear()


def teardown_function():
    RouterReplay.global_router_replay_instances.clear()


def test_record_mode_with_topk_routing_softmax_post():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.RECORD)
    logits = torch.randn(4, 6)
    probs, routing_map = topk_routing_with_score_function(
        logits=logits, topk=2, use_pre_softmax=False, router_replay=rr, score_function="softmax"
    )
    recorded = rr.get_recorded_indices()
    expected_idx = torch.topk(logits, k=2, dim=1).indices
    assert recorded is not None
    assert torch.equal(recorded, expected_idx)
    assert probs.shape == (4, 6)
    assert routing_map.shape == (4, 6)
    assert routing_map.sum(dim=1).eq(2).all()


def test_replay_forward_with_topk_routing_softmax_pre():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    logits = torch.randn(3, 5)
    target = torch.tensor([[1, 2], [0, 3], [2, 4]], dtype=torch.long)
    rr.set_target_indices(target)
    probs, routing_map = topk_routing_with_score_function(
        logits=logits, topk=2, use_pre_softmax=True, router_replay=rr, score_function="softmax"
    )
    assert routing_map.sum(dim=1).eq(2).all()
    scores = torch.softmax(logits, dim=-1)
    assert torch.equal(probs.gather(1, target), scores.gather(1, target))


def test_replay_forward_with_topk_routing_softmax_post():
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    logits = torch.randn(3, 6)
    target = torch.tensor([[1, 2], [0, 5], [3, 4]], dtype=torch.long)
    rr.set_target_indices(target)
    probs, routing_map = topk_routing_with_score_function(
        logits=logits, topk=2, use_pre_softmax=False, router_replay=rr, score_function="softmax"
    )
    selected = torch.softmax(logits.gather(1, target), dim=-1)
    assert torch.equal(probs.gather(1, target), selected)
    assert routing_map.sum(dim=1).eq(2).all()


def test_global_set_get_clear_indices():
    r1 = RouterReplay()
    r2 = RouterReplay()
    t1 = torch.tensor([[0, 1]], dtype=torch.long)
    t2 = torch.tensor([[1, 0]], dtype=torch.long)
    RouterReplay.set_replay_data([t1, t2])
    assert torch.equal(r1.target_topk_idx, t1)
    assert torch.equal(r2.target_topk_idx, t2)
    r1.record_indices(t1)
    r2.record_indices(t2)
    rec = RouterReplay.get_recorded_data()
    assert len(rec) == 2
    assert torch.equal(rec[0], t1)
    assert torch.equal(rec[1], t2)
    RouterReplay.clear_global_indices()
    assert r1.target_topk_idx is None and r2.target_topk_idx is None
    assert r1.get_recorded_indices() is None and r2.get_recorded_indices() is None


def test_global_action_set_and_clear():
    r1 = RouterReplay()
    r2 = RouterReplay()
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    assert r1.router_replay_action == RouterReplayAction.REPLAY_FORWARD
    assert r2.router_replay_action == RouterReplayAction.REPLAY_FORWARD
    RouterReplay.clear_global_router_replay_action()
    assert r1.router_replay_action is None and r2.router_replay_action is None


def test_set_replay_data_length_mismatch():
    _ = RouterReplay()
    with pytest.raises(ValueError):
        RouterReplay.set_replay_data(
            [torch.tensor([[0, 1]], dtype=torch.long), torch.tensor([[1, 0]], dtype=torch.long)]
        )

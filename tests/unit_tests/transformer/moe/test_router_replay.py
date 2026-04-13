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


def test_set_rollout_routing_data_activates_replay_forward():
    """set_rollout_routing_data with activate_replay=True should set REPLAY_FORWARD on all layers."""
    r1 = RouterReplay()
    r2 = RouterReplay()
    t1 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    t2 = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)

    RouterReplay.set_rollout_routing_data([t1, t2])

    assert torch.equal(r1.target_topk_idx, t1)
    assert torch.equal(r2.target_topk_idx, t2)
    assert r1.router_replay_action == RouterReplayAction.REPLAY_FORWARD
    assert r2.router_replay_action == RouterReplayAction.REPLAY_FORWARD


def test_set_rollout_routing_data_no_activate():
    """set_rollout_routing_data with activate_replay=False should only set indices, not action."""
    r1 = RouterReplay()
    t1 = torch.tensor([[0, 1]], dtype=torch.long)

    RouterReplay.set_rollout_routing_data([t1], activate_replay=False)

    assert torch.equal(r1.target_topk_idx, t1)
    assert r1.router_replay_action is None  # action should NOT be set


def test_set_rollout_routing_data_end_to_end():
    """Full R3 training-side workflow: inject rollout indices then run routing with replay."""
    num_tokens, num_experts, topk = 4, 8, 2

    # Simulate two MoE layers
    r1 = RouterReplay()
    r2 = RouterReplay()

    # Rollout indices captured during inference (e.g., from SGLang/vLLM)
    rollout_idx_l1 = torch.tensor([[0, 3], [1, 5], [2, 7], [4, 6]], dtype=torch.long)
    rollout_idx_l2 = torch.tensor([[1, 4], [0, 6], [3, 7], [2, 5]], dtype=torch.long)

    # Inject into training via the public API
    RouterReplay.set_rollout_routing_data([rollout_idx_l1, rollout_idx_l2])

    # Simulate training forward pass for layer 1
    logits_l1 = torch.randn(num_tokens, num_experts)
    probs_l1, routing_map_l1 = topk_routing_with_score_function(
        logits=logits_l1,
        topk=topk,
        use_pre_softmax=False,
        router_replay=r1,
        score_function="softmax",
    )
    # Expert selection must exactly match rollout indices
    replayed_idx_l1 = routing_map_l1.nonzero(as_tuple=False)[:, 1].view(num_tokens, topk)
    assert torch.equal(replayed_idx_l1.sort(dim=1).values, rollout_idx_l1.sort(dim=1).values)
    # Router logits gradient is preserved (probs are non-zero)
    assert (probs_l1 > 0).any()

    # Simulate training forward pass for layer 2
    logits_l2 = torch.randn(num_tokens, num_experts)
    probs_l2, routing_map_l2 = topk_routing_with_score_function(
        logits=logits_l2,
        topk=topk,
        use_pre_softmax=False,
        router_replay=r2,
        score_function="softmax",
    )
    replayed_idx_l2 = routing_map_l2.nonzero(as_tuple=False)[:, 1].view(num_tokens, topk)
    assert torch.equal(replayed_idx_l2.sort(dim=1).values, rollout_idx_l2.sort(dim=1).values)

    # Clean up (as a training framework would do after the step)
    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()

    assert r1.target_topk_idx is None
    assert r2.target_topk_idx is None
    assert r1.router_replay_action is None
    assert r2.router_replay_action is None


def test_set_rollout_routing_data_length_mismatch():
    """set_rollout_routing_data with wrong number of layers should raise ValueError."""
    _ = RouterReplay()
    with pytest.raises(ValueError):
        RouterReplay.set_rollout_routing_data(
            [torch.tensor([[0, 1]], dtype=torch.long), torch.tensor([[1, 0]], dtype=torch.long)]
        )


def test_rollout_replay_preserves_gradient_path():
    """Gradient from the replayed gating probs must flow back through the router logits.

    With post-softmax scoring and dense_output=True, the computation graph is:
      probs = softmax(logits.gather(1, rollout_idx))
    so d(loss)/d(logits) is non-zero at the replayed expert columns.
    """
    num_tokens, num_experts, topk = 3, 6, 2

    rr = RouterReplay()
    rollout_idx = torch.tensor([[0, 2], [1, 4], [3, 5]], dtype=torch.long)
    RouterReplay.set_rollout_routing_data([rollout_idx])

    logits = torch.randn(num_tokens, num_experts, requires_grad=True)

    # Use dense_output=True so probs is [N, topk] and directly connected to
    # logits via gather+softmax (no scatter intermediary).
    probs, top_indices = topk_routing_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        router_replay=rr,
        score_function="softmax",
        dense_output=True,
    )

    # Verify that probs is a function of logits by checking require_grad
    assert probs.requires_grad, "probs tensor must require grad (must be on the autograd graph)"

    # The probs should equal softmax(logits gathered at rollout_idx)
    expected_probs = torch.softmax(logits.detach().gather(1, rollout_idx).float(), dim=-1).to(
        logits.dtype
    )
    assert torch.allclose(
        probs.detach(), expected_probs, atol=1e-5
    ), "Replayed probs do not match softmax(logits.gather(rollout_idx))"

    # top_indices must match rollout indices
    assert torch.equal(top_indices, rollout_idx), "top_indices must equal rollout_idx"

    # Clean up
    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()


# ---------------------------------------------------------------------------
# REPLAY_BACKWARD: activation recompute during backward pass
# ---------------------------------------------------------------------------


def test_replay_backward_pops_in_order():
    """REPLAY_BACKWARD should pop indices from replay_backward_list in FIFO order."""
    rr = RouterReplay()
    idx_fwd = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    idx_bwd = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)

    # set_target_indices appends to replay_backward_list
    rr.set_target_indices(idx_fwd)
    rr.set_target_indices(idx_bwd)

    logits = torch.randn(2, 6)
    rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    probs1, top1 = topk_routing_with_score_function(
        logits=logits,
        topk=2,
        use_pre_softmax=False,
        router_replay=rr,
        score_function="softmax",
        dense_output=True,
    )
    assert torch.equal(top1, idx_fwd), "First REPLAY_BACKWARD call should use idx_fwd"

    probs2, top2 = topk_routing_with_score_function(
        logits=logits,
        topk=2,
        use_pre_softmax=False,
        router_replay=rr,
        score_function="softmax",
        dense_output=True,
    )
    assert torch.equal(top2, idx_bwd), "Second REPLAY_BACKWARD call should use idx_bwd"


def test_replay_backward_list_cleared_by_clear_indices():
    """clear_indices must also clear replay_backward_list."""
    rr = RouterReplay()
    rr.set_target_indices(torch.tensor([[0, 1]], dtype=torch.long))
    assert len(rr.replay_backward_list) == 1
    rr.clear_indices()
    assert len(rr.replay_backward_list) == 0


# ---------------------------------------------------------------------------
# Sigmoid score function with replay
# ---------------------------------------------------------------------------


def test_replay_forward_sigmoid_score_function():
    """REPLAY_FORWARD should work correctly with sigmoid score function."""
    rr = RouterReplay()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    logits = torch.randn(4, 8)
    rollout_idx = torch.tensor([[0, 3], [1, 5], [2, 7], [4, 6]], dtype=torch.long)
    rr.set_target_indices(rollout_idx)

    probs, top_indices = topk_routing_with_score_function(
        logits=logits,
        topk=2,
        use_pre_softmax=False,
        router_replay=rr,
        score_function="sigmoid",
        dense_output=True,
    )
    assert torch.equal(top_indices, rollout_idx), "top_indices must match rollout indices"
    # sigmoid path: scores = sigmoid(logits), then gather, then normalize
    sig_scores = torch.sigmoid(logits.float()).to(logits.dtype).gather(1, rollout_idx)
    expected_probs = sig_scores / (sig_scores.sum(dim=-1, keepdim=True) + 1e-20)
    assert torch.allclose(probs, expected_probs, atol=1e-5)


# ---------------------------------------------------------------------------
# static_buffer: CUDA-graph-compatible recording
# ---------------------------------------------------------------------------


def test_static_buffer_recording():
    """When a static_buffer is set, record_indices copies into it instead of storing a reference."""
    rr = RouterReplay()
    max_tokens, topk = 10, 2
    static_buf = torch.zeros(max_tokens, topk, dtype=torch.long)
    rr.set_static_buffer(static_buf)

    rr.set_router_replay_action(RouterReplayAction.RECORD)
    logits = torch.randn(4, 6)
    _, _ = topk_routing_with_score_function(
        logits=logits, topk=topk, use_pre_softmax=False, router_replay=rr, score_function="softmax"
    )

    recorded = rr.get_recorded_indices()
    expected_idx = torch.topk(logits, k=topk, dim=1).indices
    # The slice [0:4] of the static buffer should contain the recorded indices
    assert torch.equal(recorded, expected_idx)
    assert torch.equal(static_buf[:4], expected_idx)

    rr.clear_static_buffer()
    assert rr.static_buffer is None


def test_set_global_static_buffers_distributes_correctly():
    """set_global_static_buffers must distribute per-layer slices correctly."""
    r1 = RouterReplay()
    r2 = RouterReplay()
    max_tokens, topk, num_layers = 8, 2, 2
    combined = torch.zeros(max_tokens, num_layers, topk, dtype=torch.long)

    RouterReplay.set_global_static_buffers(combined)

    # Each instance should hold a [max_tokens, topk] view
    assert r1.static_buffer is not None and r1.static_buffer.shape == (max_tokens, topk)
    assert r2.static_buffer is not None and r2.static_buffer.shape == (max_tokens, topk)

    # They should be views into the combined buffer, not copies
    r1.static_buffer[0, 0] = 99
    assert combined[0, 0, 0] == 99

    RouterReplay.clear_global_static_buffers()
    assert r1.static_buffer is None and r2.static_buffer is None


# ---------------------------------------------------------------------------
# clear_global_router_replay_instances
# ---------------------------------------------------------------------------


def test_clear_global_router_replay_instances():
    """clear_global_router_replay_instances must empty the global list."""
    _ = RouterReplay()
    _ = RouterReplay()
    assert len(RouterReplay.global_router_replay_instances) == 2
    RouterReplay.clear_global_router_replay_instances()
    assert len(RouterReplay.global_router_replay_instances) == 0

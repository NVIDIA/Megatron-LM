# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GPU checks for pack-directed feature transport and its explicit adjoint.

Run under torch.distributed.run with four or eight GPUs. These tests deliberately
use real P2P and changing CP groups, without a global feature all-gather.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.models.mimo.comm.pack_bridge import (
    FeatureSlice,
    PackFeatureBridge,
    PackFeaturePlan,
)
from megatron.core.models.mimo.planning import (
    MediaTask,
    assign_media_to_producers,
    build_round_plans,
)


@pytest.fixture(scope="module")
def bridge_groups():
    """Four-rank transport domains, compatible with the standard eight-GPU runner."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    world = dist.get_world_size()
    if world % 4:
        pytest.skip("Pack bridge tests require a multiple of four ranks")
    rank = dist.get_rank()
    owned, created = {}, []
    for base in range(0, world, 4):
        for ranks in (
            tuple(range(base, base + 4)),
            (base, base + 1),
            (base + 2, base + 3),
            *((i,) for i in range(base, base + 4)),
        ):
            group = dist.new_group(ranks=list(ranks), backend="nccl")
            if rank in ranks:
                owned[ranks] = group
                created.append(group)
    yield owned
    for group in reversed(created):
        dist.destroy_process_group(group)


def _source(rank, base, counts, hidden=4):
    local_rank = rank - base
    return (
        torch.arange(counts[local_rank] * hidden, device="cuda", dtype=torch.float64)
        .reshape(counts[local_rank], hidden)
        .add_(100 * local_rank + 1)
    )


def _round_plans(base):
    """Uneven producers, reordered/repeated rows, empty packs and CP1/2/4."""
    a = FeatureSlice(base, 1, 3)
    b = FeatureSlice(base + 2, 0, 2)
    c = FeatureSlice(base + 3, 2, 5)
    d = FeatureSlice(base + 3, 0, 2)
    return (
        (
            PackFeaturePlan((base, base + 1), (c, a, b)),
            PackFeaturePlan((base + 2, base + 3), (b, a, a, d)),
        ),
        (
            PackFeaturePlan((base, base + 1), (b,)),
            PackFeaturePlan((base + 2,), (c, a)),
            PackFeaturePlan((base + 3,), ()),
        ),
        (PackFeaturePlan(tuple(range(base, base + 4)), (d, a, c, b, a)),),
        tuple(PackFeaturePlan((rank,), ()) for rank in range(base, base + 4)),
    )


def _cotangent(plan, consumer, base, round_index):
    """Rank 1 has no local visual-input gradient but must still participate."""
    gradient = torch.arange(plan.num_rows * 4, device="cuda", dtype=torch.float64).reshape(
        plan.num_rows, 4
    )
    if consumer == base + 1:
        return torch.zeros_like(gradient)
    return gradient.mul_(0.125).add_(1 + consumer - base + round_index)


def test_directed_pack_forward_and_explicit_backward(bridge_groups):
    rank = dist.get_rank()
    base = rank // 4 * 4
    ranks = tuple(range(base, base + 4))
    transport = bridge_groups[ranks]
    counts = (5, 0, 3, 7)
    source_input = _source(rank, base, counts)
    encoder_weight = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
    source_output = source_input * encoder_weight
    source_output.retain_grad()
    bridge = PackFeatureBridge(transport)

    states, expected_gradients = [], []
    for round_index, plans in enumerate(_round_plans(base)):
        local_plan = next(plan for plan in plans if rank in plan.cp_ranks)
        state = bridge.forward(source_output, plans, bridge_groups[local_plan.cp_ranks])
        assert state.features.is_leaf and state.features.requires_grad
        assert state.local_features is source_output
        expected = (
            torch.cat(
                [
                    _source(feature.producer_rank, base, counts).narrow(
                        0, feature.offset, feature.length
                    )
                    * 2
                    for feature in local_plan.features
                ]
            )
            if local_plan.features
            else source_output.new_empty((0, 4))
        )
        torch.testing.assert_close(state.features, expected, rtol=0, atol=0)
        local_cotangent = _cotangent(local_plan, rank, base, round_index)
        if rank != base + 1 and local_plan.num_rows:
            (state.features * local_cotangent).sum().backward()
        assert source_output.grad is None  # Even local routing is an explicit boundary.
        assert encoder_weight.grad is None

        expected_gradient = torch.zeros_like(source_output)
        for plan in plans:
            offset = 0
            for feature in plan.features:
                if feature.producer_rank == rank:
                    for consumer in plan.cp_ranks:
                        expected_gradient.narrow(0, feature.offset, feature.length).add_(
                            _cotangent(plan, consumer, base, round_index).narrow(
                                0, offset, feature.length
                            )
                        )
                offset += feature.length
        states.append(state)
        expected_gradients.append(expected_gradient)

    accumulated = torch.zeros_like(source_output)
    # Return older rounds after newer CP groups have already run.
    for index in (2, 0, 3, 1):
        state, expected_gradient = states[index], expected_gradients[index]
        gradient = bridge.backward(state, gradient_dtype=torch.float64)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
        accumulated.add_(gradient)
        with pytest.raises(RuntimeError, match="only be returned once"):
            bridge.backward(state)
    source_output.backward(accumulated)
    torch.testing.assert_close(source_output.grad, accumulated, rtol=0, atol=0)
    torch.testing.assert_close(
        encoder_weight.grad, (source_input * accumulated).sum(), rtol=0, atol=0
    )

    # Evaluation uses the same transport without creating an autograd leaf.
    plans = _round_plans(base)[0]
    local_plan = next(plan for plan in plans if rank in plan.cp_ranks)
    evaluated = bridge.forward(
        source_output, plans, bridge_groups[local_plan.cp_ranks], forward_only=True
    )
    assert not evaluated.features.requires_grad
    torch.testing.assert_close(evaluated.features, states[0].features, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="forward-only"):
        bridge.backward(evaluated)


def test_feature_transport_adjoint_identity(bridge_groups):
    rank = dist.get_rank()
    base = rank // 4 * 4
    transport = bridge_groups[tuple(range(base, base + 4))]
    source = _source(rank, base, (5, 0, 3, 7))
    plans = _round_plans(base)[0]
    local_plan = next(plan for plan in plans if rank in plan.cp_ranks)
    bridge = PackFeatureBridge(transport)
    state = bridge.forward(source, plans, bridge_groups[local_plan.cp_ranks])
    cotangent = _cotangent(local_plan, rank, base, 0)
    source_gradient = bridge.backward(state, cotangent, gradient_dtype=torch.float64)
    products = torch.stack(
        [(state.features.detach() * cotangent).sum(), (source * source_gradient).sum()]
    )
    dist.all_reduce(products, group=transport)
    torch.testing.assert_close(products[0], products[1], rtol=0, atol=0)


def test_plan_validation(bridge_groups):
    rank = dist.get_rank()
    base = rank // 4 * 4
    ranks = tuple(range(base, base + 4))
    bridge = PackFeatureBridge(bridge_groups[ranks])
    source = _source(rank, base, (5, 0, 3, 7))
    with pytest.raises(ValueError, match="partition"):
        bridge.forward(source, [PackFeaturePlan((rank,), ())], None)
    with pytest.raises(ValueError, match="explicit CP"):
        bridge.forward(source, [PackFeaturePlan(ranks, ())], None)
    with pytest.raises(ValueError, match="group ranks"):
        bridge.forward(source, [PackFeaturePlan(ranks, ())], bridge_groups[(rank,)])


def test_media_planning_preserves_stable_identities_and_reference_order():
    """Costs are caller policy; neither sample nor media IDs are local offsets."""
    tasks = [MediaTask('image-b', 3, 1), MediaTask(41, 2, 4), MediaTask(('audio', 9), 5, 2)]
    assigned, slices = assign_media_to_producers(tasks, (8, 3, 11, 5))
    assert assigned == {8: [tasks[1]], 3: [tasks[2]], 11: [tasks[0]], 5: []}
    assert slices == {
        41: FeatureSlice(8, 0, 2),
        ('audio', 9): FeatureSlice(3, 0, 5),
        'image-b': FeatureSlice(11, 0, 3),
    }
    samples = {
        'sample-A': {'media_ids': ['image-b', 41]},
        9001: {'media_ids': [('audio', 9), 'image-b']},
        ('text', 2): {'media_ids': []},
    }
    plans = build_round_plans(
        [['sample-A', 9001], ['sample-A', 9001], [('text', 2)], [('text', 2)]],
        samples,
        slices,
        (8, 3, 11, 5),
    )
    assert plans == (
        PackFeaturePlan(
            (8, 3), (slices['image-b'], slices[41], slices[('audio', 9)], slices['image-b'])
        ),
        PackFeaturePlan((11, 5), ()),
    )
    # On one producer, restore caller order after cost-sorted assignment.
    assigned, slices = assign_media_to_producers(tasks, (3,))
    assert assigned[3] == tasks
    assert [slices[task.media_id].offset for task in tasks] == [0, 3, 5]
    assert assign_media_to_producers([], (8, 3)) == ({8: [], 3: []}, {})
    with pytest.raises(ValueError, match="unique media ID"):
        assign_media_to_producers([tasks[0], tasks[0]], (3,))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        assign_media_to_producers([MediaTask('invalid', 2, float('nan'))], (3,))
    with pytest.raises(ValueError, match="producer feature slice"):
        build_round_plans([['sample-A']], samples, {}, (3,))

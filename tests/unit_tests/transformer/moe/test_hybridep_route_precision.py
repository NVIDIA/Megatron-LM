# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layout-independent combine and dgrad against an independent FP32 sum."""

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("masked", [False, True])
def test_route_precision_forward_backward_across_layouts(distributed_world, masked):
    group = distributed_world
    if group.size() < 4:
        pytest.skip("Route placement comparison needs at least four ranks.")
    num_experts = 8 * group.size()
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(98 + group.rank())
    values = torch.randn(512, 256, device=device, dtype=torch.bfloat16, generator=generator)
    upstream = torch.randn(512, 256, device=device, dtype=torch.bfloat16, generator=generator)
    weights = torch.tensor([0.13, 0.17, 0.11, 0.23, 0.19, 0.17], device=device)
    expected_output = (
        torch.stack([(values.float() * weight).bfloat16().float() for weight in weights])
        .sum(0)
        .bfloat16()
    )
    expected_dgrad = (
        torch.stack([(upstream.float() * weight).bfloat16().float() for weight in weights])
        .sum(0)
        .bfloat16()
    )
    if masked:
        expected_output[-64:] = 0
        expected_dgrad[-64:] = 0
    score_grad = None
    try:
        for ids in ([0, 1, 2, 3, 4, 5], [0, 1, 8, 9, 16, 24], [0, 8, 16, 17, 24, 25]):
            hidden = values.detach().clone().requires_grad_()
            indices = torch.tensor(ids, device=device, dtype=torch.int16).expand(512, -1).clone()
            probs = torch.zeros(512, num_experts, device=device)
            probs[:, ids] = weights
            if masked:
                indices[-64:] = -1
            probs.requires_grad_()
            dispatched, scores, _, counts, handle = fused_a2a.hybrid_ep_dispatch(
                x=hidden,
                routing_map=None,
                probs=probs,
                group=group,
                num_local_experts=8,
                num_sms_dispatch_api=4,
                num_sms_combine_api=4,
                topk_idx=indices,
                num_of_experts=num_experts,
                pad_multiple=32,
                preserve_route_precision=True,
            )
            # A deliberately simple expert isolates dispatch/combine from GEMM/optimizer math.
            expert_output = (dispatched.float() * scores.reshape(-1, 1)).bfloat16()
            actual = fused_a2a.hybrid_ep_combine(expert_output, handle, counts.sum(), 32)
            torch.testing.assert_close(actual, expected_output, atol=0, rtol=0)
            actual.backward(upstream)
            torch.testing.assert_close(hidden.grad, expected_dgrad, atol=0, rtol=0)
            selected = probs.grad[:, ids]
            if score_grad is None:
                score_grad = selected.clone()
            else:
                torch.testing.assert_close(selected, score_grad, atol=0, rtol=0)
            if masked:
                assert torch.count_nonzero(probs.grad[-64:]) == 0
            dist.barrier(group=group)
    finally:
        fused_a2a.reset_hybrid_ep_buffer()


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_token_dispatcher_type": "alltoall"},
        {"moe_flex_dispatcher_backend": "deepep"},
        {"moe_hybridep_routing_map_mode": "bool"},
        {"moe_pad_expert_input_to_capacity": True, "moe_expert_capacity_factor": 1.0},
    ],
)
def test_route_precision_rejects_incompatible_config(overrides):
    args = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=32,
        moe_router_topk=6,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_hybridep_preserve_route_precision=True,
    )
    args.update(overrides)
    with pytest.raises(ValueError, match="moe_hybridep_preserve_route_precision requires"):
        TransformerConfig(**args)


@pytest.mark.parametrize("topk", [6, 10])
def test_route_reduction_is_invariant_to_route_order(distributed_world, topk):
    device = torch.device("cuda", torch.cuda.current_device())
    # The two tiny terms together cross a BF16 rounding midpoint. Summing them
    # separately after the leading1 loses them in FP32, reproducing the old bug.
    values = torch.tensor(
        [1.0, 2**-8, 2**-24, 2**-24] + [0.0] * (topk - 4), dtype=torch.bfloat16, device=device
    )
    routes = values.reshape(1, topk, 1).expand(32, -1, 64).clone()
    expected = routes.double().sum(1).bfloat16()
    generator = torch.Generator(device=device).manual_seed(317)
    upstream = torch.randn(32, 64, device=device, dtype=torch.bfloat16, generator=generator)
    for _ in range(8):
        permutation = torch.randperm(topk, device=device, generator=generator)
        permuted = routes[:, permutation].detach().requires_grad_()
        actual = fused_a2a._HybridEPRouteReduce.apply(permuted)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.backward(upstream)
        torch.testing.assert_close(
            permuted.grad, upstream[:, None].expand_as(routes), atol=0, rtol=0
        )
        hidden = torch.zeros(32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        expanded = fused_a2a._HybridEPRouteExpand.apply(hidden, topk)
        expanded.backward(permuted.detach().reshape_as(expanded))
        torch.testing.assert_close(hidden.grad, expected, atol=0, rtol=0)


def test_route_reduction_does_not_retain_sort_indices(distributed_world):
    saved = []
    device = torch.device("cuda", torch.cuda.current_device())
    routes = torch.randn(32, 10, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor
    ):
        output = fused_a2a._HybridEPRouteReduce.apply(routes)
    assert saved == []
    output.sum().backward()
    torch.testing.assert_close(routes.grad, torch.ones_like(routes), atol=0, rtol=0)

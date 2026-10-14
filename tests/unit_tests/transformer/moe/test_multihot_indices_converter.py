# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import random

import pytest
import torch

from megatron.core import config
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.transformer.moe.megakernel.mok.route_adapter import routing_map_to_mok_inputs
from megatron.core.transformer.moe.moe_utils import drop_padding_routes


class PytorchIndicesToMultihot:
    def __init__(self, num_instances):
        self.num_instances = num_instances

    def _indices_to_multihot(self, indices, probs):
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_instances), dtype=torch.long, device=indices.device
        )
        multihot_probs = torch.zeros(
            (batch_size, self.num_instances), dtype=torch.float, device=indices.device
        )
        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs


class TestIndicesToMultihot:

    def setup_method(self, method):
        # enable experimental feature
        if config.ENABLE_EXPERIMENTAL is False:
            config.ENABLE_EXPERIMENTAL = True

    def teardown_method(self, method):
        # disable experimental feature
        if config.ENABLE_EXPERIMENTAL is True:
            config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.experimental
    @pytest.mark.parametrize("num_of_token", [3, 5, 8, 128, 512])
    @pytest.mark.parametrize("topk", [2, 4, 6, 7, 8])
    @pytest.mark.parametrize("num_of_local_experts", [4, 7, 8, 12, 20, 30, 31, 32])
    def test_indices_to_multihot(self, num_of_token, topk, num_of_local_experts):
        # construct the indices and probs_indices
        indices = torch.full((num_of_token, topk), -1, dtype=torch.int32, device='cuda')
        probs_indices = torch.full((num_of_token, topk), 0, dtype=torch.float32, device='cuda')
        # Fill the indices with random values
        # There are 2 non-ordinary values in each row
        for i in range(num_of_token):
            positions = random.sample(range(indices.shape[1]), 2)
            values = random.sample(range(num_of_local_experts), 2)
            indices[i, positions[0]] = values[0]
            indices[i, positions[1]] = values[1]
        mask = indices != -1
        probs_indices[mask] = torch.rand(mask.sum(), device=indices.device)
        probs_indices.requires_grad = True

        indices_pytorch = copy.deepcopy(indices)
        probs_indices_pytorch = copy.deepcopy(probs_indices)

        # test forward
        multihot_indices, probs_in_multihot = fused_indices_to_multihot(
            indices, probs_indices, num_of_local_experts
        )
        pytorch_class = PytorchIndicesToMultihot(num_of_local_experts)
        multihot_indices_pytorch, probs_in_multihot_pytorch = pytorch_class._indices_to_multihot(
            indices_pytorch, probs_indices_pytorch
        )
        assert torch.allclose(multihot_indices, multihot_indices_pytorch)
        assert torch.allclose(probs_in_multihot, probs_in_multihot_pytorch)

        # test backward
        loss = (probs_in_multihot @ torch.transpose(probs_in_multihot, 0, 1)).sum() / 2
        loss.backward()
        loss_pytorch = (
            probs_in_multihot_pytorch @ torch.transpose(probs_in_multihot_pytorch, 0, 1)
        ).sum() / 2
        loss_pytorch.backward()
        assert torch.allclose(probs_indices.grad, probs_indices_pytorch.grad)


class TestRoutingMapToIndices:

    @pytest.mark.experimental
    @pytest.mark.parametrize(
        ("num_tokens", "num_experts", "topk"), [(1, 7, 3), (17, 8, 4), (33, 31, 6), (4096, 384, 6)]
    )
    def test_routing_map_to_indices(self, num_tokens, num_experts, topk):
        generator = torch.Generator(device="cpu").manual_seed(1234)
        routing_map_cpu = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
        expected_indices_cpu = torch.full((num_tokens, topk), -1, dtype=torch.int32)
        for token_idx in range(num_tokens):
            num_routes = token_idx % (topk + 1)
            selected = torch.randperm(num_experts, generator=generator)[:num_routes]
            selected = selected.sort().values
            routing_map_cpu[token_idx, selected] = True
            expected_indices_cpu[token_idx, :num_routes] = selected

        routing_map = routing_map_cpu.cuda()
        expected_indices = expected_indices_cpu.cuda()
        probs = torch.rand(
            (num_tokens, num_experts), device="cuda", dtype=torch.float32, requires_grad=True
        )
        probs_reference = probs.detach().clone().requires_grad_(True)

        weights, indices = routing_map_to_mok_inputs(probs, routing_map, topk)

        gather_indices = expected_indices.clamp_min(0).to(torch.int64)
        expected_weights = torch.gather(probs_reference, dim=1, index=gather_indices)
        expected_weights = expected_weights.masked_fill(expected_indices < 0, 0.0)

        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)

        grad = torch.randn_like(weights)
        (weights * grad).sum().backward()
        (expected_weights * grad).sum().backward()
        torch.testing.assert_close(probs.grad, probs_reference.grad, rtol=0, atol=0)

    @pytest.mark.experimental
    @pytest.mark.parametrize(
        ("seq_length", "batch_size", "num_experts", "topk", "mask_layout"),
        [
            (4096, 1, 64, 10, "bool"),
            (512, 4, 256, 8, "bool"),
            (37, 3, 31, 6, "int"),
            (64, 2, 8, 2, "strided"),
        ],
    )
    def test_padding_mask_matches_drop_padding_routes(
        self, seq_length, batch_size, num_experts, topk, mask_layout
    ):
        generator = torch.Generator(device="cuda").manual_seed(1234)
        num_tokens = seq_length * batch_size
        scores = torch.rand((num_tokens, num_experts), device="cuda", generator=generator)
        routing_map = torch.zeros_like(scores, dtype=torch.bool)
        routing_map.scatter_(1, scores.topk(topk, dim=1).indices, True)
        routing_map[::3] &= scores[::3] > 0.5  # Some tokens keep fewer than topk routes.
        probs = torch.rand(
            (num_tokens, num_experts), device="cuda", generator=generator, requires_grad=True
        )
        probs_reference = probs.detach().clone().requires_grad_(True)
        padding_mask = (
            torch.rand((batch_size, seq_length), device="cuda", generator=generator) < 0.6
        )
        mask_argument = {
            "bool": padding_mask,
            "int": padding_mask.int(),
            "strided": padding_mask.t().contiguous().t(),
        }[mask_layout]

        weights, indices = routing_map_to_mok_inputs(probs, routing_map, topk, mask_argument)
        expected_weights, expected_indices = routing_map_to_mok_inputs(
            *drop_padding_routes(probs_reference, routing_map, padding_mask), topk
        )

        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
        # Tokens flatten sequence-first: token s * batch_size + b is padding[b, s].
        padding_rows = padding_mask.t().reshape(-1)
        assert padding_rows.any() and (~padding_rows).any()
        assert (indices[padding_rows] == -1).all() and (weights[padding_rows] == 0).all()
        grad = torch.randn_like(weights)
        torch.autograd.backward(weights, grad)
        torch.autograd.backward(expected_weights, grad)
        torch.testing.assert_close(probs.grad, probs_reference.grad, rtol=0, atol=0)

    @pytest.mark.experimental
    def test_padding_mask_must_cover_every_token(self):
        probs = torch.rand((12, 8), device="cuda")
        routing_map = probs > 0.5
        with pytest.raises(ValueError, match="one entry per token"):
            routing_map_to_mok_inputs(probs, routing_map, 2, torch.zeros((2, 5), dtype=torch.bool))

    @pytest.mark.experimental
    def test_backward_does_not_materialize_the_indices_gradient(self):
        num_tokens, num_experts, topk = 256, 16, 4
        probs = torch.rand((num_tokens, num_experts), device="cuda", requires_grad=True)
        routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device="cuda")
        routing_map[:, :topk] = True
        weights, _ = routing_map_to_mok_inputs(probs, routing_map, topk)
        grad = torch.randn_like(weights)

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            torch.autograd.backward(weights, grad)

        # The integer indices output has no gradient, so autograd creates no zeros for it.
        names = {event.name for event in profiler.events()}
        assert not names & {"aten::zeros", "aten::zeros_like"}, sorted(names)
        torch.testing.assert_close(probs.grad[:, :topk], grad, rtol=0, atol=0)
        assert not probs.grad[:, topk:].any()

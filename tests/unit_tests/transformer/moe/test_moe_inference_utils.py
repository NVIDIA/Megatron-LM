# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for megatron.core.transformer.moe.moe_inference_utils.

Tests the compute_local_tokens_per_expert triton kernel against a PyTorch
reference implementation across various routing configurations.
"""

import pytest
import torch

from megatron.core.transformer.moe.moe_inference_utils import (
    compute_expert_offsets,
    compute_local_tokens_per_expert,
)


def _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts):
    """PyTorch reference: count (token, k) pairs routed to each local expert."""
    flat = routing_map.flatten()
    local_mask = (flat >= local_expert_start) & (flat < local_expert_start + num_local_experts)
    local_ids = flat[local_mask] - local_expert_start
    ref = torch.zeros(num_local_experts, dtype=torch.int32, device=routing_map.device)
    ref.scatter_add_(0, local_ids.long(), torch.ones_like(local_ids, dtype=torch.int32))
    return ref


class TestComputeLocalTokensPerExpert:
    """Tests for the _count_local_tokens_kernel triton kernel."""

    @pytest.mark.internal
    def test_random_routing(self):
        """Random routing_map should match PyTorch reference."""
        torch.manual_seed(42)
        num_tokens, topk = 128, 8
        num_total_experts, num_local_experts = 64, 8
        local_expert_start = 16

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        ref = _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        assert torch.equal(result, ref)

    @pytest.mark.internal
    def test_no_local_experts(self):
        """All tokens routed to non-local experts should give all zeros."""
        num_tokens, topk = 64, 4
        num_local_experts = 8
        local_expert_start = 16

        # Expert 0 is not in [16..23]
        routing_map = torch.zeros(num_tokens, topk, dtype=torch.int32, device="cuda")

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        expected = torch.zeros(num_local_experts, dtype=torch.int32, device="cuda")
        assert torch.equal(result, expected)

    @pytest.mark.internal
    def test_all_to_single_expert(self):
        """All tokens routed to one local expert."""
        num_tokens, topk = 64, 4
        num_local_experts = 8
        local_expert_start = 16
        target_local_idx = 3

        routing_map = torch.full(
            (num_tokens, topk),
            local_expert_start + target_local_idx,
            dtype=torch.int32,
            device="cuda",
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        expected = torch.zeros(num_local_experts, dtype=torch.int32, device="cuda")
        expected[target_local_idx] = num_tokens * topk
        assert torch.equal(result, expected)

    @pytest.mark.internal
    def test_single_token(self):
        """Minimal case: 1 token, topk=1."""
        num_local_experts = 8
        local_expert_start = 16

        routing_map = torch.tensor(
            [[local_expert_start]], dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        expected = torch.zeros(num_local_experts, dtype=torch.int32, device="cuda")
        expected[0] = 1
        assert torch.equal(result, expected)

    @pytest.mark.internal
    def test_uniform_distribution(self):
        """Each token routes to all local experts exactly once (topk == num_local_experts)."""
        num_tokens = 32
        num_local_experts = 8
        local_expert_start = 0

        # Each row is [0, 1, 2, ..., 7] — one hit per local expert per token
        routing_map = (
            torch.arange(num_local_experts, device="cuda", dtype=torch.int32)
            .unsqueeze(0)
            .expand(num_tokens, -1)
            .contiguous()
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        expected = torch.full(
            (num_local_experts,), num_tokens, dtype=torch.int32, device="cuda"
        )
        assert torch.equal(result, expected)

    @pytest.mark.internal
    def test_local_expert_start_at_zero(self):
        """Local experts starting at global index 0."""
        torch.manual_seed(123)
        num_tokens, topk = 256, 4
        num_total_experts = 32
        num_local_experts = 4
        local_expert_start = 0

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        ref = _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        assert torch.equal(result, ref)

    @pytest.mark.internal
    def test_local_experts_at_end(self):
        """Local experts at the tail end of the global expert range."""
        torch.manual_seed(456)
        num_tokens, topk = 256, 4
        num_total_experts = 32
        num_local_experts = 4
        local_expert_start = 28  # experts 28..31

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        ref = _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        assert torch.equal(result, ref)

    @pytest.mark.internal
    def test_large_batch(self):
        """Larger batch to exercise multi-block histogram kernel."""
        torch.manual_seed(789)
        num_tokens, topk = 2048, 8
        num_total_experts = 128
        num_local_experts = 16
        local_expert_start = 48

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        ref = _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        assert torch.equal(result, ref)

    @pytest.mark.internal
    def test_topk_one(self):
        """topk=1: each token assigned to exactly one expert."""
        torch.manual_seed(101)
        num_tokens = 512
        num_total_experts = 64
        num_local_experts = 8
        local_expert_start = 8

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, 1), dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        ref = _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        assert torch.equal(result, ref)

    @pytest.mark.internal
    def test_non_power_of_two_tokens(self):
        """Non-power-of-2 num_tokens to check masking at block boundaries."""
        torch.manual_seed(202)
        num_tokens, topk = 137, 5
        num_total_experts = 32
        num_local_experts = 4
        local_expert_start = 12

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        ref = _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        assert torch.equal(result, ref)


class TestComputeExpertOffsets:
    """Tests for the _prefix_sum_kernel triton kernel."""

    @pytest.mark.internal
    def test_basic_prefix_sum(self):
        """Simple known input: [3, 1, 4, 1, 5] -> offsets [0, 3, 4, 8, 9]."""
        tokens_per_expert = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int32, device="cuda")
        offsets, counters = compute_expert_offsets(tokens_per_expert)

        expected = torch.tensor([0, 3, 4, 8, 9], dtype=torch.int32, device="cuda")
        assert torch.equal(offsets, expected)
        assert torch.equal(counters, expected)

    @pytest.mark.internal
    def test_single_expert(self):
        """Single expert: offset is always 0."""
        tokens_per_expert = torch.tensor([42], dtype=torch.int32, device="cuda")
        offsets, counters = compute_expert_offsets(tokens_per_expert)

        expected = torch.tensor([0], dtype=torch.int32, device="cuda")
        assert torch.equal(offsets, expected)
        assert torch.equal(counters, expected)

    @pytest.mark.internal
    def test_all_zeros(self):
        """All experts have zero tokens: offsets are all 0."""
        tokens_per_expert = torch.zeros(8, dtype=torch.int32, device="cuda")
        offsets, counters = compute_expert_offsets(tokens_per_expert)

        expected = torch.zeros(8, dtype=torch.int32, device="cuda")
        assert torch.equal(offsets, expected)

    @pytest.mark.internal
    def test_one_hot(self):
        """Only one expert has tokens."""
        tokens_per_expert = torch.tensor([0, 0, 10, 0, 0], dtype=torch.int32, device="cuda")
        offsets, counters = compute_expert_offsets(tokens_per_expert)

        expected = torch.tensor([0, 0, 0, 10, 10], dtype=torch.int32, device="cuda")
        assert torch.equal(offsets, expected)

    @pytest.mark.internal
    def test_counters_are_independent_copy(self):
        """Mutating counters should not affect offsets."""
        tokens_per_expert = torch.tensor([2, 3, 5], dtype=torch.int32, device="cuda")
        offsets, counters = compute_expert_offsets(tokens_per_expert)

        # Simulate what the permute kernel does
        counters[0] += 1
        counters[1] += 2

        expected_offsets = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
        assert torch.equal(offsets, expected_offsets), "offsets should be unmodified"

    @pytest.mark.internal
    def test_end_to_end_with_histogram(self):
        """Chain histogram -> prefix sum and verify consistency."""
        torch.manual_seed(99)
        num_tokens, topk = 128, 8
        num_total_experts = 64
        num_local_experts = 8
        local_expert_start = 16

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )

        tokens_per_expert = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        offsets, counters = compute_expert_offsets(tokens_per_expert)

        # Verify: offsets[i] == sum(tokens_per_expert[0:i])
        ref_offsets = torch.zeros_like(tokens_per_expert)
        ref_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], dim=0)
        assert torch.equal(offsets, ref_offsets)

        # Verify: last offset + last count == total
        assert offsets[-1] + tokens_per_expert[-1] == tokens_per_expert.sum()


class TestComputeLocalTokensPerExpertSweep:
    """Exhaustive parametrized sweep for _count_local_tokens_kernel."""

    @pytest.mark.internal
    @pytest.mark.parametrize("num_tokens", [1, 7, 32, 137, 256, 512, 2048])
    @pytest.mark.parametrize("topk", [1, 2, 3, 4, 6, 8])
    @pytest.mark.parametrize("num_local_experts", [1, 3, 4, 8, 16])
    @pytest.mark.parametrize("num_total_experts", [8, 32, 64, 128])
    def test_sweep(self, num_tokens, topk, num_local_experts, num_total_experts):
        """Sweep across token counts, topk, and expert configurations."""
        if num_local_experts > num_total_experts:
            pytest.skip("num_local_experts > num_total_experts")

        torch.manual_seed(num_tokens * 1000 + topk * 100 + num_local_experts)
        local_expert_start = (num_total_experts - num_local_experts) // 2

        routing_map = torch.randint(
            0, num_total_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
        )

        result = compute_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        ref = _reference_local_tokens_per_expert(routing_map, local_expert_start, num_local_experts)
        assert torch.equal(result, ref), (
            f"Mismatch for num_tokens={num_tokens}, topk={topk}, "
            f"num_local_experts={num_local_experts}, num_total_experts={num_total_experts}"
        )


class TestComputeExpertOffsetsSweep:
    """Exhaustive parametrized sweep for _prefix_sum_kernel."""

    @pytest.mark.internal
    @pytest.mark.parametrize("num_local_experts", [1, 2, 3, 4, 5, 7, 8, 13, 16, 32, 64])
    @pytest.mark.parametrize("max_count", [0, 1, 10, 100, 1000])
    def test_sweep(self, num_local_experts, max_count):
        """Sweep across expert counts and token magnitudes."""
        torch.manual_seed(num_local_experts * 100 + max_count)

        if max_count == 0:
            tokens_per_expert = torch.zeros(num_local_experts, dtype=torch.int32, device="cuda")
        else:
            tokens_per_expert = torch.randint(
                0, max_count + 1, (num_local_experts,), dtype=torch.int32, device="cuda"
            )

        offsets, counters = compute_expert_offsets(tokens_per_expert)

        # Reference: exclusive prefix sum via torch.cumsum
        ref_offsets = torch.zeros_like(tokens_per_expert)
        if num_local_experts > 1:
            ref_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], dim=0)

        assert torch.equal(offsets, ref_offsets), (
            f"Mismatch for num_local_experts={num_local_experts}, max_count={max_count}\n"
            f"  tokens_per_expert={tokens_per_expert.tolist()}\n"
            f"  expected={ref_offsets.tolist()}\n"
            f"  got={offsets.tolist()}"
        )
        assert torch.equal(counters, ref_offsets), "counters should equal offsets initially"

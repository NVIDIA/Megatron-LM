# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.core.inference.moe.permute.

Tests cover:
- compute_local_tokens_per_expert: token counting against PyTorch reference
- compute_expert_offsets: prefix sums with and without alignment
- permute_tokens: expert grouping, data integrity, alignment padding
- unpermute_tokens: weighted scatter-back, fp32 accumulation
- permute -> unpermute roundtrip
"""

import pytest
import torch


def _ref_tokens_per_expert(routing_map, local_expert_start, num_local_experts):
    """PyTorch reference for compute_local_tokens_per_expert."""
    counts = torch.zeros(num_local_experts, dtype=torch.int32, device=routing_map.device)
    for eid in routing_map.flatten():
        lid = eid.item() - local_expert_start
        if 0 <= lid < num_local_experts:
            counts[lid] += 1
    return counts


def _ref_expert_offsets(tokens_per_expert, alignment):
    """PyTorch reference for compute_expert_offsets."""
    aligned = tokens_per_expert.clone().to(torch.int32)
    for i in range(len(aligned)):
        if aligned[i] > 0 and alignment > 1:
            aligned[i] = ((aligned[i] + alignment - 1) // alignment) * alignment
    inc = torch.cumsum(aligned, dim=0)
    exc = inc - aligned
    return exc.to(torch.int32), inc.to(torch.int32)


def _make_inputs(num_tokens, hidden_dim, topk, num_experts, seed=42):
    """Create random hidden states, probs, and routing_map."""
    torch.manual_seed(seed)
    hidden = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
    routing_map = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")
    return hidden, probs, routing_map

@pytest.mark.internal
class TestComputeLocalTokensPerExpert:

    @pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 128, 256, 512])
    @pytest.mark.parametrize("topk", [1, 2, 4, 6, 8])
    @pytest.mark.parametrize("num_experts,num_local,start", [
        (4, 4, 0),      # all local, small expert count
        (8, 8, 0),      # all local (EP=1)
        (8, 4, 0),      # first half local (EP=2, rank 0)
        (8, 4, 4),      # second half local (EP=2, rank 1)
        (8, 2, 2),      # middle slice (EP=4, rank 1)
        (8, 1, 7),      # single expert local (EP=8, last rank)
        (32, 8, 0),     # 32 experts, first 8 local
        (32, 8, 24),    # 32 experts, last 8 local
        (128, 32, 0),   # 128 experts, first 32 local (EP=4, rank 0)
        (128, 32, 96),  # 128 experts, last 32 local (EP=4, rank 3)
    ])
    def test_matches_reference(self, num_tokens, topk, num_experts, num_local, start):
        from megatron.core.inference.moe.permute import compute_local_tokens_per_expert

        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")
        result = compute_local_tokens_per_expert(routing_map, start, num_local)
        expected = _ref_tokens_per_expert(routing_map, start, num_local)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_no_local_tokens(self):
        """All tokens routed to non-local experts -> all zeros."""
        from megatron.core.inference.moe.permute import compute_local_tokens_per_expert

        routing_map = torch.full((16, 4), 99, dtype=torch.int64, device="cuda")
        result = compute_local_tokens_per_expert(routing_map, 0, 8)
        assert result.sum().item() == 0

    def test_single_expert_all_tokens(self):
        """All token-topk pairs route to a single local expert."""
        from megatron.core.inference.moe.permute import compute_local_tokens_per_expert

        num_tokens, topk, num_local = 32, 4, 8
        routing_map = torch.full((num_tokens, topk), 3, dtype=torch.int64, device="cuda")
        result = compute_local_tokens_per_expert(routing_map, 0, num_local)
        assert result[3].item() == num_tokens * topk
        assert result.sum().item() == num_tokens * topk

    @pytest.mark.parametrize("seed", [0, 7, 42, 123, 999])
    def test_total_count_equals_local_pairs(self, seed):
        """Sum of tokens_per_expert equals total routing pairs hitting local experts."""
        from megatron.core.inference.moe.permute import compute_local_tokens_per_expert

        torch.manual_seed(seed)
        num_tokens, topk, num_experts = 64, 6, 16
        local_start, num_local = 4, 4
        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")
        result = compute_local_tokens_per_expert(routing_map, local_start, num_local)
        local_mask = (routing_map >= local_start) & (routing_map < local_start + num_local)
        assert result.sum().item() == local_mask.sum().item()


@pytest.mark.internal
class TestComputeExpertOffsets:

    @pytest.mark.parametrize("alignment", [1, 8, 16, 32, 64, 128])
    @pytest.mark.parametrize("tpe_values", [
        [5, 0, 12, 3, 0, 7, 1, 20],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [100, 0, 0, 50],
        [1],
        [33, 33, 33, 33, 33, 33, 33, 33],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [127, 0, 129, 0, 1, 0, 255, 0],
    ])
    def test_matches_reference(self, alignment, tpe_values):
        from megatron.core.inference.moe.permute import compute_expert_offsets

        tpe = torch.tensor(tpe_values, dtype=torch.int32, device="cuda")
        exc, inc = compute_expert_offsets(tpe, alignment=alignment)
        ref_exc, ref_inc = _ref_expert_offsets(tpe, alignment)
        torch.testing.assert_close(exc, ref_exc, atol=0, rtol=0)
        torch.testing.assert_close(inc, ref_inc, atol=0, rtol=0)

    @pytest.mark.parametrize("n_experts", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_exclusive_starts_at_zero(self, n_experts):
        from megatron.core.inference.moe.permute import compute_expert_offsets

        tpe = torch.randint(1, 50, (n_experts,), dtype=torch.int32, device="cuda")
        exc, inc = compute_expert_offsets(tpe, alignment=1)
        assert exc[0].item() == 0
        assert inc[-1].item() == tpe.sum().item()

    def test_zero_experts_skipped(self):
        """Experts with 0 tokens should not consume any aligned space."""
        from megatron.core.inference.moe.permute import compute_expert_offsets

        tpe = torch.tensor([0, 5, 0, 3], dtype=torch.int32, device="cuda")
        exc, inc = compute_expert_offsets(tpe, alignment=32)
        # Expert 0: 0 tokens -> 0 aligned -> exc=0, inc=0
        assert exc[0].item() == 0
        assert inc[0].item() == 0
        # Expert 1: 5 tokens -> 32 aligned -> exc=0, inc=32
        assert exc[1].item() == 0
        assert inc[1].item() == 32
        # Expert 2: 0 tokens -> exc=32, inc=32
        assert exc[2].item() == 32
        assert inc[2].item() == 32

    @pytest.mark.parametrize("alignment", [16, 32, 128])
    def test_all_offsets_aligned(self, alignment):
        """Every inclusive offset should be a multiple of alignment."""
        from megatron.core.inference.moe.permute import compute_expert_offsets

        tpe = torch.tensor([3, 7, 0, 15, 1, 0, 50, 2], dtype=torch.int32, device="cuda")
        exc, inc = compute_expert_offsets(tpe, alignment=alignment)
        for i in range(len(tpe)):
            assert inc[i].item() % alignment == 0, (
                f"inc[{i}]={inc[i].item()} not aligned to {alignment}"
            )
            assert exc[i].item() % alignment == 0, (
                f"exc[{i}]={exc[i].item()} not aligned to {alignment}"
            )



class TestPermuteTokens:

    @pytest.mark.parametrize("num_tokens,hidden_dim,topk,num_experts", [
        (1, 64, 1, 4),
        (1, 128, 8, 8),
        (4, 64, 2, 4),
        (16, 128, 2, 8),
        (32, 64, 4, 8),
        (64, 256, 6, 8),
        (128, 128, 8, 128),
        (256, 64, 2, 32),
        (512, 128, 6, 16),
    ])
    def test_data_integrity(self, num_tokens, hidden_dim, topk, num_experts):
        """Every permuted row matches the original token's hidden state."""
        from megatron.core.inference.moe.permute import permute_tokens

        hidden, probs, routing_map = _make_inputs(
            num_tokens, hidden_dim, topk, num_experts
        )
        perm_h, perm_p, perm_map, offs = permute_tokens(
            hidden, probs, routing_map, 0, num_experts, alignment=1,
        )

        # Check every non-padding row
        for i in range(perm_map.shape[0]):
            src = perm_map[i].item()
            if src < 0:
                continue
            torch.testing.assert_close(
                perm_h[i], hidden[src],
                msg=f"Row {i} (src={src}) hidden mismatch",
            )

    @pytest.mark.parametrize("alignment", [1, 16, 32, 64, 128])
    @pytest.mark.parametrize("num_tokens,topk,num_experts", [
        (16, 2, 4),
        (64, 4, 8),
        (128, 8, 32),
    ])
    def test_offsets_are_aligned(self, alignment, num_tokens, topk, num_experts):
        """Inclusive offsets are multiples of alignment."""
        from megatron.core.inference.moe.permute import permute_tokens

        hidden, probs, routing_map = _make_inputs(num_tokens, 128, topk, num_experts)
        _, _, _, offs = permute_tokens(
            hidden, probs, routing_map, 0, num_experts, alignment=alignment,
        )
        if alignment > 1:
            for i in range(offs.shape[0]):
                assert offs[i].item() % alignment == 0, (
                    f"Offset {i}={offs[i].item()} not aligned to {alignment}"
                )

    @pytest.mark.parametrize("num_tokens,topk,num_experts,alignment", [
        (8, 2, 4, 128),
        (32, 2, 4, 128),
        (16, 4, 8, 64),
        (64, 6, 8, 32),
    ])
    def test_padding_rows_have_neg1(self, num_tokens, topk, num_experts, alignment):
        """Padding rows in permutation_map are -1."""
        from megatron.core.inference.moe.permute import permute_tokens

        hidden, probs, routing_map = _make_inputs(num_tokens, 64, topk, num_experts)
        _, _, perm_map, _ = permute_tokens(
            hidden, probs, routing_map, 0, num_experts, alignment=alignment,
        )
        padding_mask = perm_map == -1
        real_mask = perm_map >= 0
        assert padding_mask.sum() > 0, "Expected some padding rows with large alignment"
        assert real_mask.sum() > 0, "Expected some real rows"

    @pytest.mark.parametrize("num_tokens,topk,num_experts", [
        (16, 2, 4),
        (32, 4, 8),
        (64, 6, 16),
        (128, 8, 128),
    ])
    @pytest.mark.parametrize("alignment", [1, 32, 128])
    def test_total_real_rows_equals_routed_pairs(self, num_tokens, topk, num_experts, alignment):
        """Number of non-padding rows equals total (token, topk) pairs routed locally."""
        from megatron.core.inference.moe.permute import permute_tokens

        hidden, probs, routing_map = _make_inputs(num_tokens, 64, topk, num_experts)
        _, _, perm_map, _ = permute_tokens(
            hidden, probs, routing_map, 0, num_experts, alignment=alignment,
        )
        real_count = (perm_map >= 0).sum().item()
        # All experts are local, so every pair should appear
        assert real_count == num_tokens * topk

    @pytest.mark.parametrize("num_tokens,topk,num_experts,local_start,num_local", [
        (64, 4, 8, 2, 3),     # experts 2, 3, 4
        (64, 4, 8, 0, 1),     # only expert 0
        (64, 4, 8, 7, 1),     # only expert 7
        (128, 6, 16, 4, 8),   # experts 4-11
        (32, 2, 32, 16, 16),  # second half of 32
        (256, 8, 128, 0, 32), # first 32 of 128
    ])
    def test_expert_subset(self, num_tokens, topk, num_experts, local_start, num_local):
        """Only tokens routed to local experts appear in output."""
        from megatron.core.inference.moe.permute import permute_tokens

        hidden, probs, routing_map = _make_inputs(num_tokens, 64, topk, num_experts)
        _, _, perm_map, _ = permute_tokens(
            hidden, probs, routing_map, local_start, num_local, alignment=1,
        )
        real_count = (perm_map >= 0).sum().item()
        local_mask = (routing_map >= local_start) & (routing_map < local_start + num_local)
        expected_count = local_mask.sum().item()
        assert real_count == expected_count

    @pytest.mark.parametrize("hidden_dim", [32, 64, 128, 256, 512, 1024, 2688])
    def test_various_hidden_dims(self, hidden_dim):
        """Permute works across various hidden dimensions including non-power-of-2."""
        from megatron.core.inference.moe.permute import permute_tokens

        hidden, probs, routing_map = _make_inputs(32, hidden_dim, 4, 8)
        perm_h, _, perm_map, _ = permute_tokens(
            hidden, probs, routing_map, 0, 8, alignment=1,
        )
        # Spot-check first real row
        for i in range(perm_map.shape[0]):
            src = perm_map[i].item()
            if src >= 0:
                torch.testing.assert_close(perm_h[i], hidden[src])
                break

@pytest.mark.internal
class TestUnpermuteTokens:

    def test_weighted_scatter(self):
        """Unpermute correctly accumulates prob-weighted expert outputs."""
        from megatron.core.inference.moe.permute import unpermute_tokens

        num_tokens, hidden_dim = 4, 8
        # Two entries map to token 0, one to token 2
        expert_output = torch.ones(3, hidden_dim, device="cuda", dtype=torch.bfloat16)
        permuted_probs = torch.tensor([0.5, 0.3, 0.7], device="cuda", dtype=torch.float32)
        perm_map = torch.tensor([0, 0, 2], dtype=torch.int32, device="cuda")

        result = unpermute_tokens(expert_output, permuted_probs, perm_map, num_tokens)

        assert result.dtype == torch.float32
        # Token 0: 0.5 * 1.0 + 0.3 * 1.0 = 0.8
        torch.testing.assert_close(
            result[0], torch.full((hidden_dim,), 0.8, device="cuda"),
            atol=1e-5, rtol=1e-5,
        )
        # Token 1: untouched -> 0
        torch.testing.assert_close(
            result[1], torch.zeros(hidden_dim, device="cuda"),
        )
        # Token 2: 0.7 * 1.0 = 0.7
        torch.testing.assert_close(
            result[2], torch.full((hidden_dim,), 0.7, device="cuda"),
            atol=1e-5, rtol=1e-5,
        )

    def test_padding_rows_ignored(self):
        """Rows with permutation_map == -1 are skipped."""
        from megatron.core.inference.moe.permute import unpermute_tokens

        expert_output = torch.ones(4, 8, device="cuda", dtype=torch.bfloat16)
        permuted_probs = torch.ones(4, device="cuda", dtype=torch.float32)
        perm_map = torch.tensor([0, -1, -1, 1], dtype=torch.int32, device="cuda")

        result = unpermute_tokens(expert_output, permuted_probs, perm_map, 3)
        # Only tokens 0 and 1 get values
        assert result[0].sum().item() != 0
        assert result[1].sum().item() != 0
        assert result[2].sum().item() == 0

    @pytest.mark.parametrize("hidden_dim", [8, 64, 128, 256, 512, 2688])
    def test_various_hidden_dims(self, hidden_dim):
        """Unpermute works across various hidden dimensions."""
        from megatron.core.inference.moe.permute import unpermute_tokens

        num_tokens = 8
        expert_output = torch.randn(4, hidden_dim, device="cuda", dtype=torch.bfloat16)
        permuted_probs = torch.tensor([1.0, 1.0, 1.0, 1.0], device="cuda", dtype=torch.float32)
        perm_map = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

        result = unpermute_tokens(expert_output, permuted_probs, perm_map, num_tokens)
        assert result.shape == (num_tokens, hidden_dim)
        # First 4 tokens should have values, rest should be zero
        for t in range(4):
            torch.testing.assert_close(
                result[t], expert_output[t].float(), atol=1e-5, rtol=1e-5,
            )
        for t in range(4, num_tokens):
            assert result[t].sum().item() == 0

    @pytest.mark.parametrize("topk", [1, 2, 4, 6, 8])
    def test_multiple_topk_accumulation(self, topk):
        """Multiple topk entries for the same token are summed correctly."""
        from megatron.core.inference.moe.permute import unpermute_tokens

        hidden_dim = 32
        # All topk entries point to token 0
        expert_output = torch.ones(topk, hidden_dim, device="cuda", dtype=torch.bfloat16)
        probs = torch.full((topk,), 0.1, device="cuda", dtype=torch.float32)
        perm_map = torch.zeros(topk, dtype=torch.int32, device="cuda")

        result = unpermute_tokens(expert_output, probs, perm_map, 1)
        expected_val = 0.1 * topk
        torch.testing.assert_close(
            result[0], torch.full((hidden_dim,), expected_val, device="cuda"),
            atol=1e-4, rtol=1e-4,
        )

@pytest.mark.internal
class TestPermuteUnpermuteRoundtrip:

    @pytest.mark.parametrize("num_tokens,hidden_dim,topk,num_experts,alignment", [
        (1, 64, 1, 4, 1),
        (1, 128, 1, 4, 128),
        (8, 64, 1, 4, 1),
        (16, 64, 2, 4, 1),
        (16, 64, 2, 4, 32),
        (32, 128, 4, 8, 32),
        (32, 128, 4, 8, 128),
        (64, 256, 6, 8, 1),
        (64, 256, 6, 8, 128),
        (128, 128, 8, 32, 1),
        (128, 128, 8, 32, 128),
        (256, 64, 2, 128, 32),
        (64, 2688, 8, 128, 128),  # nanov3-like hidden_dim
    ])
    def test_roundtrip_identity(
        self, num_tokens, hidden_dim, topk, num_experts, alignment
    ):
        """permute -> (identity transform) -> unpermute recovers weighted sum of inputs."""
        from megatron.core.inference.moe.permute import permute_tokens, unpermute_tokens

        torch.manual_seed(42)
        hidden = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")

        perm_h, perm_p, perm_map, _ = permute_tokens(
            hidden, probs, routing_map, 0, num_experts, alignment=alignment,
        )
        # Pass permuted hidden directly through (identity expert)
        result = unpermute_tokens(perm_h, perm_p, perm_map, num_tokens)

        # Build reference: for each token, sum prob[k] * hidden[token] over topk
        ref = torch.zeros(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)
        for t in range(num_tokens):
            prob_sum = probs[t].sum()
            ref[t] = hidden[t].float() * prob_sum

        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("local_start,num_local,num_experts", [
        (0, 4, 8),
        (4, 4, 8),
        (0, 1, 8),
        (0, 8, 8),
        (0, 32, 128),
        (96, 32, 128),
    ])
    def test_roundtrip_with_expert_subset(self, local_start, num_local, num_experts):
        """Roundtrip works when only a subset of experts are local."""
        from megatron.core.inference.moe.permute import permute_tokens, unpermute_tokens

        torch.manual_seed(42)
        num_tokens, hidden_dim, topk = 64, 128, 4
        hidden = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
        probs = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (num_tokens, topk), device="cuda")

        perm_h, perm_p, perm_map, _ = permute_tokens(
            hidden, probs, routing_map, local_start, num_local, alignment=32,
        )
        result = unpermute_tokens(perm_h, perm_p, perm_map, num_tokens)

        # Reference: only accumulate probs for local experts
        ref = torch.zeros(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)
        for t in range(num_tokens):
            local_prob_sum = 0.0
            for k in range(topk):
                eid = routing_map[t, k].item()
                if local_start <= eid < local_start + num_local:
                    local_prob_sum += probs[t, k].item()
            ref[t] = hidden[t].float() * local_prob_sum

        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)

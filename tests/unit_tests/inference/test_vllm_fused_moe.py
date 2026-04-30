# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.core.inference.moe.vllm_fused_moe.

Tests cover:
- _select_block_size_m: BLOCK_SIZE_M selection based on token count
- _moe_align_block_size_cuda_graphable: indirection table construction
- _moe_sum: fused topk reduction with routing weight application
- vllm_fused_moe: end-to-end correctness vs sequential reference
"""

import os
import tempfile

# Redirect Triton cache to /tmp BEFORE triton is imported (at module level) so
# compiled kernels don't accumulate in ~/.triton/ across test runs.
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton_test_cache"))

import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def _single_autotune_config():
    """Replace the 25-entry autotune config list with a single config.

    Each unique (N, K, BLOCK_SIZE_M) combo triggers a full autotune pass that
    compiles ALL configs. Tests only need correctness, not peak throughput, so
    one config is sufficient and cuts compiled-kernel count by ~25x.
    """
    from megatron.core.inference.moe.vllm_fused_moe import _fused_moe_kernel

    orig = list(_fused_moe_kernel.configs)
    _fused_moe_kernel.configs = [orig[0]]
    yield
    _fused_moe_kernel.configs = orig


def _vt(n):
    """Create a valid_tokens scalar int32 CUDA tensor."""
    return torch.tensor(n, dtype=torch.int32, device="cuda")


def _ref_sequential_moe(
    hidden_states,
    probs,
    fc1_weight,
    fc2_weight,
    routing_map,
    num_local_experts,
    local_expert_start,
    valid_tokens,
):
    """PyTorch reference: sequential per-token MoE computation.

    For each valid token, for each topk slot routed to a local expert:
        intermediate = squared_relu(hidden @ fc1_weight[expert].T)
        output += prob * (intermediate @ fc2_weight[expert].T)
    """
    vt = valid_tokens if isinstance(valid_tokens, int) else valid_tokens.item()
    max_tokens, topk = routing_map.shape
    hidden_size = hidden_states.shape[1]

    out = torch.zeros(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)

    for t in range(vt):
        acc = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
        for k in range(topk):
            eid = routing_map[t, k].item()
            lid = eid - local_expert_start
            if 0 <= lid < num_local_experts:
                h = hidden_states[t].float()
                fc1_out = h @ fc1_weight[lid].float().T
                activated = torch.clamp(fc1_out, min=0.0) ** 2
                fc2_out = activated @ fc2_weight[lid].float().T
                acc += probs[t, k].item() * fc2_out
        out[t] = acc.bfloat16()

    return out


def _make_moe_inputs(
    max_tokens, hidden_size, ffn_hidden, topk, num_experts, valid_tokens=None, seed=42
):
    """Create random inputs for fused MoE testing."""
    torch.manual_seed(seed)
    if valid_tokens is None:
        valid_tokens = max_tokens
    hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
    routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
    fc1_weight = (
        torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    fc2_weight = (
        torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    return hidden, probs, routing_map, fc1_weight, fc2_weight


# ──────────────────────────────────────────────────────────────────────
# _select_block_size_m
# ──────────────────────────────────────────────────────────────────────


class TestSelectBlockSizeM:

    @pytest.mark.parametrize(
        "max_tokens,expected",
        [
            (1, 16),
            (16, 16),
            (32, 16),
            (33, 32),
            (64, 32),
            (96, 32),
            (97, 64),
            (256, 64),
            (512, 64),
            (513, 128),
            (1024, 128),
            (4096, 128),
        ],
    )
    def test_returns_expected(self, max_tokens, expected):
        from megatron.core.inference.moe.vllm_fused_moe import _select_block_size_m

        assert _select_block_size_m(max_tokens) == expected

    def test_minimum_is_16(self):
        from megatron.core.inference.moe.vllm_fused_moe import _select_block_size_m

        assert _select_block_size_m(1) >= 16

    def test_monotonically_nondecreasing(self):
        from megatron.core.inference.moe.vllm_fused_moe import _select_block_size_m

        prev = _select_block_size_m(1)
        for n in range(2, 2048):
            cur = _select_block_size_m(n)
            assert cur >= prev, f"Decreased at n={n}: {prev} -> {cur}"
            prev = cur


# ──────────────────────────────────────────────────────────────────────
# _moe_align_block_size_cuda_graphable
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestMoeAlignBlockSize:

    @pytest.mark.parametrize(
        "max_tokens,topk,num_experts,block_size",
        [
            (4, 1, 4, 16),
            (8, 2, 4, 16),
            (16, 2, 8, 32),
            (32, 4, 8, 64),
            (64, 6, 8, 128),
            (128, 8, 16, 64),
            (1, 1, 4, 16),
        ],
    )
    def test_output_shapes(self, max_tokens, topk, num_experts, block_size):
        from megatron.core.inference.moe.vllm_fused_moe import (
            _ceil_div,
            _moe_align_block_size_cuda_graphable,
        )

        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        sorted_ids, expert_ids, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )

        max_sorted = max_tokens * topk + block_size * (num_experts + 1)
        max_blocks = _ceil_div(max_sorted, block_size)

        assert sorted_ids.shape == (max_sorted,)
        assert sorted_ids.dtype == torch.int32
        assert expert_ids.shape == (max_blocks,)
        assert expert_ids.dtype == torch.int32
        assert num_post_padded.shape == (1,)
        assert num_post_padded.dtype == torch.int32

    @pytest.mark.parametrize(
        "max_tokens,topk,num_experts", [(16, 2, 4), (32, 4, 8), (64, 6, 8), (128, 8, 16)]
    )
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_num_post_padded_is_aligned(self, max_tokens, topk, num_experts, block_size):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        _, _, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )
        npp = num_post_padded.item()
        assert npp % block_size == 0, f"num_post_padded={npp} not aligned to {block_size}"

    @pytest.mark.parametrize("max_tokens,topk,num_experts", [(8, 2, 4), (16, 4, 8), (32, 6, 8)])
    def test_all_local_tokens_present(self, max_tokens, topk, num_experts, block_size=16):
        """Every (token, topk) pair routed to a local expert appears in sorted_token_ids."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        torch.manual_seed(42)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        sentinel = max_tokens * topk
        sorted_ids, _, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )

        valid_sorted = sorted_ids[: num_post_padded.item()]
        real_ids = valid_sorted[valid_sorted < sentinel]

        expected_pairs = set()
        for t in range(max_tokens):
            for k in range(topk):
                expected_pairs.add(t * topk + k)
        actual_pairs = set(real_ids.cpu().tolist())
        assert actual_pairs == expected_pairs

    @pytest.mark.parametrize(
        "max_tokens,topk,num_experts,local_start,num_local",
        [(32, 4, 8, 0, 4), (32, 4, 8, 4, 4), (64, 6, 16, 4, 8), (64, 6, 16, 0, 1)],
    )
    def test_expert_subset_only_local_tokens(
        self, max_tokens, topk, num_experts, local_start, num_local
    ):
        """Only tokens routed to local experts appear in sorted_token_ids."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        torch.manual_seed(42)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        sentinel = max_tokens * topk
        sorted_ids, _, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, 16, num_local, local_start, _vt(max_tokens)
        )

        valid_sorted = sorted_ids[: num_post_padded.item()]
        real_ids = valid_sorted[valid_sorted < sentinel]

        expected_pairs = set()
        rm_flat = routing_map.cpu()
        for t in range(max_tokens):
            for k in range(topk):
                eid = rm_flat[t, k].item()
                if local_start <= eid < local_start + num_local:
                    expected_pairs.add(t * topk + k)

        actual_pairs = set(real_ids.cpu().tolist())
        assert actual_pairs == expected_pairs

    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    def test_expert_ids_cover_all_blocks(self, block_size):
        """expert_ids has a valid expert index for every block in [0, num_post_padded/block_size)."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        max_tokens, topk, num_experts = 32, 4, 8
        torch.manual_seed(42)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        _, expert_ids, num_post_padded = _moe_align_block_size_cuda_graphable(
            routing_map, block_size, num_experts, 0, _vt(max_tokens)
        )

        n_blocks = num_post_padded.item() // block_size
        active_eids = expert_ids[:n_blocks].cpu()
        assert (active_eids >= 0).all(), "Found negative expert_id in active range"
        assert (active_eids < num_experts).all(), "Found expert_id >= num_experts"


# ──────────────────────────────────────────────────────────────────────
# _moe_sum
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestMoeSum:

    def _ref_moe_sum(
        self,
        input,
        topk_weights,
        max_tokens,
        topk,
        K,
        routing_map,
        local_expert_start,
        num_local_experts,
    ):
        """PyTorch reference for _moe_sum: reduce topk with local-expert filtering."""
        out = torch.zeros(max_tokens, K, device="cuda", dtype=torch.bfloat16)
        for t in range(max_tokens):
            acc = torch.zeros(K, device="cuda", dtype=torch.float32)
            for k in range(topk):
                eid = routing_map[t, k].item()
                lid = eid - local_expert_start
                if 0 <= lid < num_local_experts:
                    v = input[t * topk + k].float()
                    w = topk_weights[t, k].item()
                    acc += v * w
            out[t] = acc.bfloat16()
        return out

    @pytest.mark.parametrize(
        "max_tokens,topk,K,num_experts",
        [(4, 1, 64, 4), (8, 2, 64, 4), (16, 4, 128, 8), (32, 6, 128, 8), (64, 8, 256, 16)],
    )
    def test_matches_reference_all_local(self, max_tokens, topk, K, num_experts):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        result = _moe_sum(
            input, topk_weights, max_tokens, topk, K, _vt(max_tokens), routing_map, 0, num_experts
        )
        expected = self._ref_moe_sum(
            input, topk_weights, max_tokens, topk, K, routing_map, 0, num_experts
        )

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "local_start,num_local,num_experts", [(0, 4, 8), (4, 4, 8), (0, 1, 8), (2, 3, 8)]
    )
    def test_matches_reference_expert_subset(self, local_start, num_local, num_experts):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K = 32, 4, 128
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        result = _moe_sum(
            input,
            topk_weights,
            max_tokens,
            topk,
            K,
            _vt(max_tokens),
            routing_map,
            local_start,
            num_local,
        )
        expected = self._ref_moe_sum(
            input, topk_weights, max_tokens, topk, K, routing_map, local_start, num_local
        )

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("valid_tokens", [0, 1, 8, 15])
    def test_partial_valid_tokens(self, valid_tokens):
        """Rows beyond valid_tokens are zeroed."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K, num_experts = 16, 2, 64, 4
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        result = _moe_sum(
            input, topk_weights, max_tokens, topk, K, _vt(valid_tokens), routing_map, 0, num_experts
        )

        if valid_tokens < max_tokens:
            zeros = result[valid_tokens:]
            assert (zeros == 0).all(), "Rows beyond valid_tokens should be zero"

    def test_writes_to_provided_output_buffer(self):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K, num_experts = 8, 2, 64, 4
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        out_buf = torch.empty(max_tokens, K, dtype=torch.bfloat16, device="cuda")
        result = _moe_sum(
            input,
            topk_weights,
            max_tokens,
            topk,
            K,
            _vt(max_tokens),
            routing_map,
            0,
            num_experts,
            out=out_buf,
        )

        assert result.data_ptr() == out_buf.data_ptr()

    @pytest.mark.parametrize("K", [32, 64, 128, 256, 512, 1024, 2688])
    def test_various_hidden_dims(self, K):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, num_experts = 8, 4, 4
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        result = _moe_sum(
            input, topk_weights, max_tokens, topk, K, _vt(max_tokens), routing_map, 0, num_experts
        )
        expected = self._ref_moe_sum(
            input, topk_weights, max_tokens, topk, K, routing_map, 0, num_experts
        )

        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# vllm_fused_moe (end-to-end)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestVllmFusedMoe:

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [
            (1, 64, 64, 1, 4),
            (4, 64, 64, 2, 4),
            (8, 128, 128, 2, 8),
            (16, 128, 128, 4, 8),
            (32, 128, 128, 6, 8),
            (64, 128, 256, 4, 8),
            (128, 64, 128, 8, 16),
        ],
    )
    def test_matches_reference(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """vllm_fused_moe output matches sequential per-token reference."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        assert result.shape == (max_tokens, hidden_size)
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "local_start,num_local,num_experts", [(0, 4, 8), (4, 4, 8), (0, 2, 8), (6, 2, 8)]
    )
    def test_expert_subset(self, local_start, num_local, num_experts):
        """Correct output when only a subset of experts are local."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk = 32, 128, 128, 4
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        fc1_local = fc1_weight[local_start : local_start + num_local].contiguous()
        fc2_local = fc2_weight[local_start : local_start + num_local].contiguous()

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_local,
            fc2_local,
            ActivationType.SQUARED_RELU,
            num_local,
            local_start,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_local, local_start, max_tokens
        )

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("valid_tokens", [1, 4, 8, 15])
    def test_partial_valid_tokens(self, valid_tokens):
        """Only the first valid_tokens rows have meaningful output."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(valid_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, valid_tokens
        )

        torch.testing.assert_close(
            result[:valid_tokens], expected[:valid_tokens], atol=5e-2, rtol=5e-2
        )

    def test_output_buffer_reuse(self):
        """Output is written to provided buffer when out= is specified."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 8, 128, 128, 2, 4
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        out_buf = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
            out=out_buf,
        )

        assert result.data_ptr() == out_buf.data_ptr()

    def test_rejects_non_bf16_input(self):
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 4, 64, 64, 2, 4
        _, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        hidden_fp32 = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.float32)

        with pytest.raises(AssertionError, match="bf16"):
            vllm_fused_moe(
                hidden_fp32,
                probs,
                fc1_weight,
                fc2_weight,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                _vt(max_tokens),
                routing_map,
            )

    @pytest.mark.parametrize("seed", [0, 7, 42, 123, 999])
    def test_deterministic_across_seeds(self, seed):
        """Same inputs produce the same output regardless of seed."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=seed
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    def test_num_tokens_hint(self):
        """num_tokens_hint selects a potentially different block size but result is still correct."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
            num_tokens_hint=4,
        )
        expected = _ref_sequential_moe(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)


# ──────────────────────────────────────────────────────────────────────
# CUDA graph capture + replay
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestVllmFusedMoeCudaGraph:

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(8, 128, 128, 2, 4), (16, 128, 128, 4, 8), (32, 128, 128, 6, 8), (64, 128, 256, 4, 8)],
    )
    def test_capture_and_replay(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """vllm_fused_moe can be captured in a CUDA graph and replayed correctly."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

        # Warmup to trigger Triton autotuning
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        graph.replay()

        expected = _ref_sequential_moe(
            static_hidden, static_probs, fc1, fc2, static_routing_map, num_experts, 0, max_tokens
        )
        torch.testing.assert_close(static_out, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "max_tokens,valid_tokens_list",
        [(16, [16, 8, 1, 4, 16]), (32, [32, 1, 16, 8, 32]), (64, [64, 32, 4, 1, 48])],
    )
    def test_replay_with_varying_valid_tokens(self, max_tokens, valid_tokens_list):
        """Replaying with different valid_tokens produces correct results each time.

        This is the core decode use case: the buffer is max-sized but only a
        varying prefix is valid each iteration. The graph is captured once and
        replayed with updated valid_tokens.
        """
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden_size, ffn_hidden, topk, num_experts = 128, 128, 4, 8

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        for vt in valid_tokens_list:
            static_vt.fill_(vt)
            graph.replay()

            expected = _ref_sequential_moe(
                static_hidden, static_probs, fc1, fc2, static_routing_map, num_experts, 0, vt
            )
            torch.testing.assert_close(
                static_out[:vt],
                expected[:vt],
                atol=5e-2,
                rtol=5e-2,
                msg=f"Mismatch with valid_tokens={vt}",
            )

    def test_replay_with_new_inputs(self):
        """Replaying after mutating hidden/probs/routing produces correct results.

        After graph capture, we overwrite the static input buffers with entirely
        new data and replay. The graph re-reads from the same device pointers,
        so it picks up the new values.
        """
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        for seed in [7, 123, 999]:
            torch.manual_seed(seed)
            static_hidden.copy_(
                torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
            )
            static_probs.copy_(torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32))
            static_routing_map.copy_(
                torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
            )

            graph.replay()

            expected = _ref_sequential_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                static_routing_map,
                num_experts,
                0,
                max_tokens,
            )
            torch.testing.assert_close(
                static_out,
                expected,
                atol=5e-2,
                rtol=5e-2,
                msg=f"Mismatch after replaying with seed={seed}",
            )

    def test_replay_matches_eager(self):
        """Graph replay produces the same output as eager execution on identical inputs."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 32, 128, 128, 4, 8

        torch.manual_seed(42)
        static_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        static_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        static_routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
        static_vt = _vt(max_tokens)
        fc1 = (
            torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        fc2 = (
            torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                vllm_fused_moe(
                    static_hidden,
                    static_probs,
                    fc1,
                    fc2,
                    ActivationType.SQUARED_RELU,
                    num_experts,
                    0,
                    static_vt,
                    static_routing_map,
                    out=static_out,
                )
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vllm_fused_moe(
                static_hidden,
                static_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                static_vt,
                static_routing_map,
                out=static_out,
            )

        for seed in [0, 7, 42]:
            torch.manual_seed(seed)
            new_hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
            new_probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
            new_routing = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
            new_vt = torch.randint(1, max_tokens + 1, (1,)).item()

            static_hidden.copy_(new_hidden)
            static_probs.copy_(new_probs)
            static_routing_map.copy_(new_routing)
            static_vt.fill_(new_vt)

            graph.replay()
            graph_result = static_out[:new_vt].clone()

            eager_out = torch.empty(max_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
            vllm_fused_moe(
                new_hidden,
                new_probs,
                fc1,
                fc2,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                _vt(new_vt),
                new_routing,
                out=eager_out,
            )
            eager_result = eager_out[:new_vt]

            torch.testing.assert_close(
                graph_result,
                eager_result,
                atol=0,
                rtol=0,
                msg=f"Graph/eager mismatch at seed={seed}, valid_tokens={new_vt}",
            )


# ──────────────────────────────────────────────────────────────────────
# Cross-backend: vllm vs mcore_fused_moe (BF16)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestVllmVsMcoreFusedMoe:

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(4, 64, 64, 2, 4), (16, 128, 128, 4, 8), (32, 128, 128, 6, 8), (64, 128, 256, 4, 8)],
    )
    def test_vllm_matches_mcore(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """vllm_fused_moe and mcore_fused_moe produce equivalent results on BF16."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        vt = _vt(max_tokens)

        vllm_out = vllm_fused_moe(
            hidden.clone(),
            probs.clone(),
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            vt,
            routing_map.clone(),
        )
        mcore_out = mcore_fused_moe(
            hidden.clone(),
            probs.clone(),
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            vt,
            routing_map.clone(),
        )

        torch.testing.assert_close(vllm_out, mcore_out, atol=5e-2, rtol=5e-2)

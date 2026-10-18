# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for megatron.core.inference.moe.vllm_fused_moe.

Tests cover:
- _get_default_config: full launch config (tile sizes, warps, stages) from M/E/top_k
- _moe_align_block_size_cuda_graphable / _moe_align_block_size_parallel: indirection tables
- _moe_sum: fused topk reduction with routing weight application
- vllm_fused_moe: end-to-end correctness vs sequential reference
"""

import importlib
import os
import tempfile

# Redirect Triton cache to /tmp BEFORE triton is imported (at module level) so
# compiled kernels don't accumulate in ~/.triton/ across test runs.
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "triton_test_cache"))

import pytest
import torch


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

    out = torch.zeros(max_tokens, hidden_size, device="cuda", dtype=torch.float32)

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
        out[t] = acc

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


def _make_swiglu_inputs(max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=42):
    """Create random inputs for SwiGLU fused MoE: FC1 output is 2*ffn_hidden (gate|up)."""
    torch.manual_seed(seed)
    hidden = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
    routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")
    fc1_weight = (
        torch.randn(num_experts, 2 * ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    fc2_weight = (
        torch.randn(num_experts, hidden_size, ffn_hidden, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    return hidden, probs, routing_map, fc1_weight, fc2_weight


def _ref_sequential_moe_swiglu(
    hidden_states,
    probs,
    fc1_weight,
    fc2_weight,
    routing_map,
    num_local_experts,
    local_expert_start,
    valid_tokens,
):
    """PyTorch reference for SwiGLU MoE: gate|up split -> SiLU(gate)*up -> FC2."""
    vt = valid_tokens if isinstance(valid_tokens, int) else valid_tokens.item()
    max_tokens, topk = routing_map.shape
    hidden_size = hidden_states.shape[1]
    n_half = fc1_weight.shape[1] // 2
    out = torch.zeros(max_tokens, hidden_size, device="cuda", dtype=torch.float32)
    for t in range(vt):
        acc = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
        for k in range(topk):
            eid = routing_map[t, k].item()
            lid = eid - local_expert_start
            if 0 <= lid < num_local_experts:
                h = hidden_states[t].float()
                fc1_out = h @ fc1_weight[lid].float().T  # [2*ffn_hidden]
                gate, up = fc1_out[:n_half], fc1_out[n_half:]
                activated = torch.nn.functional.silu(gate) * up
                fc2_out = activated @ fc2_weight[lid].float().T
                acc += probs[t, k].item() * fc2_out
        out[t] = acc
    return out


# ──────────────────────────────────────────────────────────────────────
# _get_default_config (mirrors vLLM's get_default_config)
# ──────────────────────────────────────────────────────────────────────


class TestGetDefaultConfig:

    @pytest.mark.parametrize(
        "M,expected_block_m",
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
    def test_block_m_thresholds(self, M, expected_block_m):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        cfg = _get_default_config(M=M, E=8, top_k=2)
        assert cfg['BLOCK_SIZE_M'] == expected_block_m

    def test_block_n_and_k_split_at_64(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        small = _get_default_config(M=64, E=8, top_k=2)
        large = _get_default_config(M=65, E=8, top_k=2)
        assert small['BLOCK_SIZE_N'] == 64 and small['BLOCK_SIZE_K'] == 128
        assert large['BLOCK_SIZE_N'] == 128 and large['BLOCK_SIZE_K'] == 64

    def test_group_size_m_uses_tokens_per_expert(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        # 1024 / 8 = 128 → group_m = 1 (boundary not strict-greater).
        assert _get_default_config(M=1024, E=8, top_k=2)['GROUP_SIZE_M'] == 1
        # 2048 / 8 = 256 > 128 → group_m = 16.
        assert _get_default_config(M=2048, E=8, top_k=2)['GROUP_SIZE_M'] == 16
        # Many experts, few tokens-per-expert → group_m = 1.
        assert _get_default_config(M=512, E=64, top_k=2)['GROUP_SIZE_M'] == 1

    def test_num_warps_split_at_128(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        assert _get_default_config(M=128, E=8, top_k=2)['num_warps'] == 4
        assert _get_default_config(M=129, E=8, top_k=2)['num_warps'] == 8

    def test_num_stages_split_at_32(self):
        from megatron.core.inference.moe.vllm_fused_moe import _get_default_config

        assert _get_default_config(M=32, E=8, top_k=2)['num_stages'] == 4
        assert _get_default_config(M=33, E=8, top_k=2)['num_stages'] == 3


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

    @pytest.mark.parametrize("num_local", [1, 7, 128])
    def test_graph_replay_starts_counting_from_zero(self, num_local):
        """The table initializer re-zeroes the expert counters on every replay."""
        from megatron.core.inference.moe.vllm_fused_moe import _moe_align_block_size_cuda_graphable

        max_tokens, topk, block_size, local_start = 35, 4, 16, 3
        torch.manual_seed(42)
        routing_map = torch.randint(0, num_local + 6, (max_tokens, topk), device="cuda")
        valid_tokens = _vt(max_tokens)

        def align():
            return _moe_align_block_size_cuda_graphable(
                routing_map, block_size, num_local, local_start, valid_tokens
            )

        def routed_pairs(sorted_ids, expert_ids, num_post_padded):
            # Atomics order pairs within an expert's segment, so compare (pair, expert) sets.
            ids = sorted_ids[:num_post_padded].view(-1, block_size).long()
            experts = expert_ids[: num_post_padded // block_size, None].long()
            return (ids * num_local + experts).flatten().sort().values

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            align()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_outputs = align()
        torch.cuda.current_stream().wait_stream(stream)

        for n, routes in [
            (max_tokens, "mixed"),
            (0, "mixed"),
            (1, "local"),
            (17, "nonlocal"),
            (max_tokens, "local"),
            (max_tokens, "local"),
        ]:
            valid_tokens.fill_(n)
            if routes == "local":
                routing_map.random_(local_start, local_start + num_local)
            elif routes == "nonlocal":
                routing_map.zero_()
            else:
                routing_map.random_(0, num_local + 6)
            graph.replay()
            replayed = [t.clone() for t in graph_outputs]
            eager = align()

            local_ids = routing_map[:n].flatten() - local_start
            local_ids = local_ids[(local_ids >= 0) & (local_ids < num_local)]
            counts = torch.bincount(local_ids, minlength=num_local)
            aligned = (counts + block_size - 1) // block_size * block_size
            num_post_padded = replayed[2].item()
            assert num_post_padded == aligned.sum().item()
            assert torch.equal(replayed[1], eager[1])
            assert torch.equal(replayed[2], eager[2])
            assert torch.equal(
                routed_pairs(*replayed[:2], num_post_padded),
                routed_pairs(*eager[:2], num_post_padded),
            )
            assert torch.equal(replayed[0][num_post_padded:], eager[0][num_post_padded:])


def _routing(kind, max_tokens, topk, num_experts, seed=0):
    """Routing maps (distinct experts per token) for the indirection-table tests."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(max_tokens, num_experts, generator=gen, device="cuda")
    if kind == "skewed":
        scores = scores * torch.linspace(4.0, 1.0, num_experts, device="cuda")
    routing_map = scores.topk(topk, dim=1).indices
    if kind == "same_experts":
        routing_map = torch.arange(topk, device="cuda").repeat(max_tokens, 1)
    elif kind == "padding_rows":
        # CUDA-graph padding rows are masked to -1 by the dispatcher.
        routing_map[max_tokens // 2 :] = -1
    return routing_map.contiguous()


def _segments(sorted_ids, expert_ids, num_post_padded, block_size, num_local, sentinel):
    """Per local expert: (sorted real pair ids, padding slot count) over [0, num_post_padded)."""
    n = int(num_post_padded.item())
    ids = sorted_ids[:n].view(-1, block_size).cpu()
    owners = expert_ids[: n // block_size].cpu()
    result = {}
    for e in range(num_local):
        rows = ids[owners == e].flatten()
        real = rows[rows < sentinel]
        result[e] = (sorted(real.tolist()), int((rows == sentinel).sum()))
    return n, owners.tolist(), result


def _segment_orders(sorted_ids, expert_ids, num_post_padded, block_size, num_local, sentinel):
    """Per local expert: its real pair ids in slot order over [0, num_post_padded)."""
    n = int(num_post_padded.item())
    ids = sorted_ids[:n].view(-1, block_size).cpu()
    owners = expert_ids[: n // block_size].cpu()
    orders = {}
    for e in range(num_local):
        rows = ids[owners == e].flatten()
        orders[e] = rows[rows < sentinel].tolist()
    return orders


@pytest.mark.internal
class TestParallelAlignBlockSize:
    """The multi-CTA table builder matches the multi-kernel builder over the read range."""

    @pytest.mark.parametrize(
        "max_tokens,valid,topk,num_experts,local_start,num_local",
        [
            (64, 0, 6, 128, 0, 32),
            (64, 1, 6, 128, 0, 32),
            (64, 7, 6, 128, 32, 32),
            (64, 64, 6, 128, 96, 32),
            (1024, 600, 6, 128, 0, 32),
            (2048, 2000, 6, 128, 0, 32),  # beyond the multi-CTA threshold
            (128, 128, 8, 64, 16, 16),
            (32, 32, 2, 8, 0, 8),
            (32, 17, 1, 4, 3, 1),
            # Chunk boundaries: 31/32/33 and 63/64/65 pairs.
            (128, 31, 1, 64, 0, 64),
            (128, 32, 1, 64, 0, 64),
            (128, 33, 1, 64, 0, 64),
            (128, 63, 1, 64, 0, 64),
            (128, 64, 1, 64, 0, 64),
            (128, 65, 1, 64, 0, 64),
        ],
    )
    @pytest.mark.parametrize("kind", ["uniform", "skewed", "same_experts", "padding_rows"])
    @pytest.mark.parametrize("block_size", [16, 32, 64, 128])
    @pytest.mark.parametrize("hint", ["exact", "low", "high"])
    @pytest.mark.parametrize("num_sms", [None, 3])
    def test_matches_multi_kernel_tables(
        self,
        monkeypatch,
        max_tokens,
        valid,
        topk,
        num_experts,
        local_start,
        num_local,
        kind,
        block_size,
        hint,
        num_sms,
    ):
        module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
        if num_sms is not None:
            # A small grid gives each CTA several chunks.
            monkeypatch.setattr(module, "_get_num_sms", lambda device: num_sms)

        routing_map = _routing(kind, max_tokens, topk, num_experts)
        sentinel = max_tokens * topk
        args = (routing_map, block_size, num_local, local_start, _vt(valid))
        num_pairs_hint = {"exact": valid, "low": max(valid // 8, 1), "high": max_tokens}[
            hint
        ] * topk
        # Both builders share the persistent table buffers: read each before the next runs.
        expected = _segments(
            *module._moe_align_block_size_cuda_graphable(*args), block_size, num_local, sentinel
        )
        tables = module._moe_align_block_size_parallel(*args, num_pairs_hint)
        orders = _segment_orders(*tables, block_size, num_local, sentinel)
        assert _segments(*tables, block_size, num_local, sentinel) == expected
        # Deterministic placement: each segment lists its pairs in increasing order.
        for pairs in orders.values():
            assert pairs == sorted(pairs)

    @pytest.mark.parametrize("num_pairs_hint", [6, 48, 3072])
    def test_cuda_graph_replay_with_changing_routing_and_valid_tokens(self, num_pairs_hint):
        module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

        max_tokens, topk, num_experts, num_local, block_size = 2048, 6, 128, 32, 16
        sentinel = max_tokens * topk
        routing_map = torch.full((max_tokens, topk), -1, dtype=torch.int64, device="cuda")
        valid = _vt(0)
        args = (routing_map, block_size, num_local, 0, valid)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            module._moe_align_block_size_parallel(*args, num_pairs_hint)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_tables = module._moe_align_block_size_parallel(*args, num_pairs_hint)
        for step, (n, kind) in enumerate(
            [
                (500, "uniform"),
                (3, "skewed"),
                (0, "uniform"),
                (64, "same_experts"),
                (2048, "skewed"),
                (8, "uniform"),
                (1999, "padding_rows"),
                (1, "uniform"),
            ]
        ):
            routing_map.copy_(_routing(kind, max_tokens, topk, num_experts, seed=step))
            valid.fill_(n)
            graph.replay()
            got = _segments(*graph_tables, block_size, num_local, sentinel)
            ref = module._moe_align_block_size_cuda_graphable(*args)
            assert got == _segments(*ref, block_size, num_local, sentinel), (n, kind)

    @pytest.mark.parametrize("local_start,num_local", [(0, 32), (96, 32), (16, 16)])
    @pytest.mark.parametrize("block_size", [16, 64])
    @pytest.mark.parametrize("hint", [1, 8])
    def test_large_underestimated_hint_graph_matches_independent_reference(
        self, local_start, num_local, block_size, hint
    ):
        """An 8192-token replay must cover every pair even with a one-token hint."""
        module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
        max_tokens, topk, num_experts = 8192, 6, 128
        sentinel = max_tokens * topk
        routing_map = torch.full((max_tokens, topk), -1, dtype=torch.int64, device="cuda")
        valid = _vt(0)
        args = (routing_map, block_size, num_local, local_start, valid, hint * topk)
        module.VllmFusedMoeBuffers.allocate_buffers(max_tokens, topk, 1856, 2688, num_local)
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                module._moe_align_block_size_parallel(*args)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tables = module._moe_align_block_size_parallel(*args)
            for seed, (n, kind) in enumerate(
                [
                    (8192, "uniform"),
                    (65, "same_experts"),
                    (0, "uniform"),
                    (8192, "skewed"),
                    (31, "padding_rows"),
                    (4097, "uniform"),
                ]
            ):
                routing_map.copy_(_routing(kind, max_tokens, topk, num_experts, seed=seed))
                valid.fill_(n)
                graph.replay()
                got = _segments(*tables, block_size, num_local, sentinel)
                # Build the expected segments directly from valid routes on the CPU,
                # independently of both GPU indirection-table implementations.
                routes = routing_map[:n].flatten().cpu()
                expected, owners = {}, []
                for e in range(num_local):
                    pairs = (routes == local_start + e).nonzero().flatten().tolist()
                    padding = (-len(pairs)) % block_size
                    expected[e] = (pairs, padding)
                    owners.extend([e] * ((len(pairs) + padding) // block_size))
                assert got == (len(owners) * block_size, owners, expected), (n, kind)
                orders = _segment_orders(*tables, block_size, num_local, sentinel)
                assert orders == {e: pairs for e, (pairs, _) in expected.items()}
        finally:
            module.VllmFusedMoeBuffers._delete_buffers()


@pytest.mark.internal
class TestLaunchConfigs:

    def test_tuned_tables_are_consistent(self):
        from megatron.core.inference.moe.vllm_fused_moe import _TUNED_CONFIGS

        for key, entries in _TUNED_CONFIGS.items():
            bounds = [max_tokens for max_tokens, _, _ in entries]
            assert bounds == sorted(set(bounds)), key
            for _, fc1, fc2 in entries:
                assert fc1['BLOCK_SIZE_M'] == fc2['BLOCK_SIZE_M'], key
                for cfg in (fc1, fc2):
                    for field in ('BLOCK_SIZE_M', 'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'num_warps'):
                        assert cfg[field] & (cfg[field] - 1) == 0, (key, cfg)

    def test_lookup_matches_device_shape_and_hint(self, monkeypatch):
        from megatron.core.inference.moe.vllm_fused_moe import (
            _TUNED_CONFIGS,
            _get_default_config,
            _get_launch_configs,
        )

        # The package re-exports a function of the same name; use the module itself.
        vllm_module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
        key = ((10, 3), 32, 1856, 2688, 6)
        entries = _TUNED_CONFIGS[key]
        monkeypatch.setattr(vllm_module, "_get_device_capability", lambda device: (10, 3))
        assert _get_launch_configs(1, 32, 6, 1856, 2688, "cuda") == entries[0][1:]
        assert _get_launch_configs(9, 32, 6, 1856, 2688, "cuda") == entries[1][1:]
        last = entries[-1][0]
        assert _get_launch_configs(last, 32, 6, 1856, 2688, "cuda") == entries[-1][1:]
        default = _get_default_config(M=last + 1, E=32, top_k=6)
        assert _get_launch_configs(last + 1, 32, 6, 1856, 2688, "cuda") == (default, default)
        # EP=8 (16 local experts) has its own table.
        ep8 = _TUNED_CONFIGS[((10, 3), 16, 1856, 2688, 6)]
        assert _get_launch_configs(8, 16, 6, 1856, 2688, "cuda") == ep8[0][1:]
        assert _get_launch_configs(100, 16, 6, 1856, 2688, "cuda") == ep8[2][1:]
        # Other shapes and devices keep vLLM's heuristic.
        assert _get_launch_configs(8, 8, 6, 1856, 2688, "cuda")[0] == _get_default_config(
            M=8, E=8, top_k=6
        )
        monkeypatch.setattr(vllm_module, "_get_device_capability", lambda device: (9, 0))
        assert _get_launch_configs(8, 32, 6, 1856, 2688, "cuda")[1] == _get_default_config(
            M=8, E=32, top_k=6
        )


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
        out = torch.zeros(max_tokens, K, device="cuda", dtype=torch.float32)
        for t in range(max_tokens):
            acc = torch.zeros(K, device="cuda", dtype=torch.float32)
            for k in range(topk):
                eid = routing_map[t, k].item()
                lid = eid - local_expert_start
                if 0 <= lid < num_local_experts:
                    v = input[t * topk + k].float()
                    w = topk_weights[t, k].item()
                    acc += v * w
            out[t] = acc
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

    def test_writes_to_provided_output_buffer(self):
        from megatron.core.inference.moe.vllm_fused_moe import _moe_sum

        max_tokens, topk, K, num_experts = 8, 2, 64, 4
        torch.manual_seed(42)
        input = torch.randn(max_tokens * topk, K, device="cuda", dtype=torch.bfloat16)
        topk_weights = torch.rand(max_tokens, topk, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, num_experts, (max_tokens, topk), device="cuda")

        out_buf = torch.empty(max_tokens, K, dtype=torch.float32, device="cuda")
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
        assert result.dtype == torch.float32
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(4, 64, 64, 2, 4), (16, 128, 128, 4, 8), (32, 128, 256, 6, 8), (128, 64, 128, 8, 16)],
    )
    def test_matches_reference_swiglu(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        """SwiGLU (SiLU(gate)*up) output matches the sequential per-token reference."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_swiglu_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = vllm_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SWIGLU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
        )
        expected = _ref_sequential_moe_swiglu(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, 0, max_tokens
        )

        assert result.shape == (max_tokens, hidden_size)
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

        out_buf = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")
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

    def test_prefill_steps_reuse_compiled_kernels(self, monkeypatch):
        """After a large prefill step, smaller steps with the same tiles compile nothing.

        Triton specializes the GEMMs on the 16-byte alignment of the padded-count
        pointer, which differs between the two table builders. Switching builders
        within one tile family therefore compiled a second GEMM specialization (and
        the multi-CTA builder) in the middle of serving. Top-k 5 appears in no other
        test here, so this top-k's GEMMs are still uncompiled in this process.
        """
        import triton

        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        runtime = getattr(getattr(triton, "knobs", None), "runtime", None)
        if not hasattr(runtime, "jit_post_compile_hook"):
            pytest.skip("this Triton has no jit_post_compile_hook")
        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 4096, 256, 128, 5, 32
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        compiled = []
        monkeypatch.setattr(
            runtime, "jit_post_compile_hook", lambda *, fn, **_: compiled.append(fn.name)
        )

        def step(tokens):
            vllm_fused_moe(
                hidden,
                probs,
                fc1_weight,
                fc2_weight,
                ActivationType.SQUARED_RELU,
                num_experts,
                0,
                _vt(tokens),
                routing_map,
                num_tokens_hint=tokens,
            )

        # 4096 tokens x top-5 exceeds the multi-CTA builder's pair limit; 1025-2048
        # tokens fit within it. All of these steps use 128-row tiles.
        step(max_tokens)
        assert "_fused_moe_kernel" in compiled, "the hook saw no warmup compile"
        compiled.clear()
        for tokens in (1025, 1536, 2048):
            step(tokens)
        torch.cuda.synchronize()
        assert compiled == [], f"compiled after warmup: {compiled}"


# Nemotron 3.5 Lightning experts at EP=4: 32 of 128 local experts, top-6.
_MODEL_SHAPE = dict(hidden_size=2688, ffn_hidden=1856, topk=6, num_experts=128, num_local=32)


def _model_shape_inputs(max_tokens, valid, local_start, seed=0, num_local=None):
    s = dict(_MODEL_SHAPE, num_local=num_local or _MODEL_SHAPE["num_local"])
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(max_tokens, s["hidden_size"], generator=gen, device="cuda").to(
        torch.bfloat16
    )
    probs = torch.rand(max_tokens, s["topk"], generator=gen, device="cuda")
    routing_map = (
        torch.rand(max_tokens, s["num_experts"], generator=gen, device="cuda")
        .topk(s["topk"], dim=1)
        .indices
    )
    routing_map[valid:] = -1  # padding rows beyond the valid tokens
    fc1 = (
        torch.randn(
            s["num_local"], s["ffn_hidden"], s["hidden_size"], generator=gen, device="cuda"
        ).to(torch.bfloat16)
        * 0.02
    )
    fc2 = (
        torch.randn(
            s["num_local"], s["hidden_size"], s["ffn_hidden"], generator=gen, device="cuda"
        ).to(torch.bfloat16)
        * 0.02
    )
    return hidden, probs, routing_map.contiguous(), fc1, fc2


@pytest.mark.internal
class TestVllmFusedMoeModelShapes:
    """Real expert shapes, where measured launch configs and the fused table builder apply."""

    @pytest.mark.parametrize(
        "max_tokens,valid,hint,local_start",
        [
            (16, 1, 1, 0),
            (16, 8, 8, 32),
            (64, 33, 32, 96),  # hint below the valid count
            (64, 17, 64, 0),  # hint above the valid count
            (256, 200, 8, 0),  # large underestimate: GEMM CTAs loop over extra tiles
            (128, 128, 128, 64),
            (2048, 1025, 1024, 32),  # valid count crosses the tuned prefill boundary
            (2048, 1025, 1025, 0),
            (4096, 2049, 2048, 64),  # hint at the multi-CTA pair limit, one valid token over
        ],
    )
    @pytest.mark.parametrize("clamp", [None, 20.0])
    def test_matches_reference(self, max_tokens, valid, hint, local_start, clamp):
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1, fc2 = _model_shape_inputs(max_tokens, valid, local_start)
        out = torch.full((max_tokens, _MODEL_SHAPE["hidden_size"]), float("nan"), device="cuda")
        vllm_fused_moe(
            hidden,
            probs,
            fc1,
            fc2,
            ActivationType.SQUARED_RELU,
            _MODEL_SHAPE["num_local"],
            local_start,
            _vt(valid),
            routing_map,
            out=out,
            num_tokens_hint=hint,
            activation_clamp_scale=clamp,
        )
        if clamp is None:
            expected = _ref_sequential_moe(
                hidden, probs, fc1, fc2, routing_map, _MODEL_SHAPE["num_local"], local_start, valid
            )
            torch.testing.assert_close(out[:valid], expected[:valid], atol=5e-2, rtol=5e-2)
        assert torch.isfinite(out[:valid]).all()
        assert torch.isnan(out[valid:]).all(), "rows beyond valid_tokens must stay untouched"

    @pytest.mark.parametrize("num_local", [32, 16])  # EP=4 and EP=8
    @pytest.mark.parametrize(
        "hint", [1, 8, 16, 32, 64, 128, 256, 512, 1024, 1025, 2047, 2048, 2049, 4096, 8192]
    )
    def test_measured_configs_match_default_configs_bitwise(self, hint, num_local, monkeypatch):
        """On the measured device, the tuned configs reproduce the default configs' bits."""
        from megatron.core.inference.moe import fused_moe
        from megatron.core.inference.moe.vllm_fused_moe import _TUNED_CONFIGS, vllm_fused_moe

        capability = torch.cuda.get_device_capability()
        s = dict(_MODEL_SHAPE, num_local=num_local)
        key = (capability, s["num_local"], s["ffn_hidden"], s["hidden_size"], s["topk"])
        if key not in _TUNED_CONFIGS:
            pytest.skip(f"no measured launch configs for {key}")
        max_tokens = max(hint, 16)
        hidden, probs, routing_map, fc1, fc2 = _model_shape_inputs(
            max_tokens, hint, 0, num_local=num_local
        )
        # The package re-exports the vllm_fused_moe function under the module's name.
        module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
        outputs = []
        for tables in (_TUNED_CONFIGS, {}):
            monkeypatch.setattr(module, "_TUNED_CONFIGS", tables)
            out = torch.full((max_tokens, s["hidden_size"]), float("nan"), device="cuda")
            vllm_fused_moe(
                hidden,
                probs,
                fc1,
                fc2,
                fused_moe.ActivationType.SQUARED_RELU,
                s["num_local"],
                0,
                _vt(hint),
                routing_map,
                out=out,
                num_tokens_hint=hint,
            )
            outputs.append(out)
        assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))


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
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

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
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

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
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

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
        static_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")

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

            eager_out = torch.empty(max_tokens, hidden_size, dtype=torch.float32, device="cuda")
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


requires_torch_grouped_mm = pytest.mark.skipif(
    not hasattr(torch.nn.functional, "grouped_mm"),
    reason="Requires torch.nn.functional.grouped_mm (PyTorch 2.10+)",
)


@pytest.mark.internal
@requires_torch_grouped_mm
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


CLAMP_SCALE = 16.0


def _make_saturating_moe_inputs(max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=42):
    """Like _make_moe_inputs, but with FC1 weights large enough to saturate the tanh clamp.

    The default 0.01 weight scale keeps the FC1 pre-activation well inside the clamp's
    linear region, where a dropped clamp would still pass.
    """
    hidden, probs, routing_map, _, fc2_weight = _make_moe_inputs(
        max_tokens, hidden_size, ffn_hidden, topk, num_experts, seed=seed
    )
    fc1_weight = (
        torch.randn(num_experts, ffn_hidden, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.5
    )
    return hidden, probs, routing_map, fc1_weight, fc2_weight


def _ref_sequential_moe_clamped(
    hidden_states, probs, fc1_weight, fc2_weight, routing_map, num_local_experts, clamp_scale
):
    """Sequential per-token reference with the tanh soft-clamped squared-ReLU activation.

    Mirrors _ref_sequential_moe with ``squared_relu(s * tanh(x / s))`` in place of
    ``squared_relu(x)``, but also models the two BF16 roundings the kernel performs inside
    the activation. Unlike the unclamped reference — whose 0.01-scale weights keep every
    intermediate small enough for atol to swallow the rounding — the clamp only bites when
    the pre-activation is large, and at that scale the roundings move the result by more
    than the tolerance:

      1. the clamped pre-activation stays in FP32 through the square, matching training's
         fused ``weighted_clamped_squared_relu``;
      2. the squared activation is rounded to BF16 on its way into the FC1 output buffer,
         which is what FC2 actually reads.

    The FC2 output is likewise BF16 before ``_moe_sum`` applies the FP32 routing
    probability, so that rounding is modelled too.
    """
    max_tokens, topk = routing_map.shape
    hidden_size = hidden_states.shape[1]
    out = torch.zeros(max_tokens, hidden_size, device="cuda", dtype=torch.float32)
    for t in range(max_tokens):
        for k in range(topk):
            lid = routing_map[t, k].item()
            if not 0 <= lid < num_local_experts:
                continue
            fc1_out = hidden_states[t].float() @ fc1_weight[lid].float().T
            clamped = clamp_scale * torch.tanh(fc1_out / clamp_scale)
            activated = (torch.clamp(clamped, min=0.0) ** 2).to(torch.bfloat16)
            fc2_out = (activated.float() @ fc2_weight[lid].float().T).to(torch.bfloat16)
            out[t] += probs[t, k].item() * fc2_out.float()
    return out


def _assert_close_to_bf16_ulp(actual, expected):
    """Compare two MoE outputs to the precision a BF16 MoE can carry.

    Both the kernel and the sequential reference round the FC1 and FC2 outputs to BF16, but
    they reach those roundings through different fp32 accumulation orders: grouped_mm tiles
    the K loop while the reference does a per-token matmul. A value sitting within one fp32
    ULP of a BF16 rounding boundary rounds one way in the kernel and the other way in the
    reference, and that ``2 ** -8`` relative step then propagates through FC2. Exact
    agreement is therefore unattainable here, unlike in the activation-only tests where no
    GEMM is re-derived; the achievable bound is one BF16 ULP.

    Scaling ``atol`` by the reference's own magnitude keeps that bound meaningful across
    input regimes: saturating weights produce outputs orders of magnitude larger than
    small-weight inputs do, and a fixed absolute tolerance is either vacuous or
    accidentally tight depending on which regime it was chosen for. A dropped or
    mis-scaled activation clamp moves the result by orders of magnitude, well outside
    this bound.
    """
    bf16_ulp = 2.0**-8
    torch.testing.assert_close(
        actual, expected, atol=bf16_ulp * expected.abs().max().item(), rtol=bf16_ulp
    )


@pytest.mark.internal
class TestFusedMoeActivationClamp:
    """config.activation_func_tanh_clamp_scale fused into the squared-ReLU activation.

    The clamp preconditions the FC1 output with ``s * tanh(x / s)`` before the square,
    bounding the activation by ``s ** 2``. It rides the vLLM grouped-GEMM epilogue and the
    torch-backend activation kernels; these pin all of them to one PyTorch reference.
    """

    @pytest.mark.parametrize(
        "max_tokens,hidden_size,ffn_hidden,topk,num_experts",
        [(4, 64, 64, 2, 4), (16, 128, 128, 4, 8), (64, 128, 256, 4, 8)],
    )
    def test_vllm_matches_reference(self, max_tokens, hidden_size, ffn_hidden, topk, num_experts):
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_saturating_moe_inputs(
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
            activation_clamp_scale=CLAMP_SCALE,
        )
        expected = _ref_sequential_moe_clamped(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, CLAMP_SCALE
        )

        assert result.shape == (max_tokens, hidden_size)
        _assert_close_to_bf16_ulp(result, expected)

    def test_clamp_changes_the_result(self):
        """Guard against the clamp being silently dropped: it must alter the output."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_saturating_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        args = (
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
        unclamped = vllm_fused_moe(*args).clone()
        clamped = vllm_fused_moe(*args, activation_clamp_scale=CLAMP_SCALE).clone()
        assert not torch.allclose(unclamped, clamped, atol=1e-2, rtol=1e-2)

    def test_swiglu_with_clamp_is_rejected(self):
        """The gated (SiTU-GLU) form of the clamp has no inference kernel yet."""
        from megatron.core.inference.moe.fused_moe import ActivationType
        from megatron.core.inference.moe.vllm_fused_moe import vllm_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 4, 64, 64, 2, 4
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_swiglu_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )
        with pytest.raises(AssertionError, match="only implemented for squared ReLU"):
            vllm_fused_moe(
                hidden,
                probs,
                fc1_weight,
                fc2_weight,
                ActivationType.SWIGLU,
                num_experts,
                0,
                _vt(max_tokens),
                routing_map,
                activation_clamp_scale=CLAMP_SCALE,
            )

    @requires_torch_grouped_mm
    def test_mcore_matches_reference(self):
        """The torch grouped_mm backend applies the same clamp as the vLLM backend."""
        from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

        max_tokens, hidden_size, ffn_hidden, topk, num_experts = 16, 128, 128, 4, 8
        hidden, probs, routing_map, fc1_weight, fc2_weight = _make_saturating_moe_inputs(
            max_tokens, hidden_size, ffn_hidden, topk, num_experts
        )

        result = mcore_fused_moe(
            hidden,
            probs,
            fc1_weight,
            fc2_weight,
            ActivationType.SQUARED_RELU,
            num_experts,
            0,
            _vt(max_tokens),
            routing_map,
            activation_clamp_scale=CLAMP_SCALE,
        )
        expected = _ref_sequential_moe_clamped(
            hidden, probs, fc1_weight, fc2_weight, routing_map, num_experts, CLAMP_SCALE
        )
        _assert_close_to_bf16_ulp(result.float(), expected)

    def test_mxfp8_fused_quant_kernel_receives_clamp_scale(self):
        """The MXFP8 route binds the clamp onto the fused activation+quantize kernel.

        The kernel's numerics are covered in test_mxfp8_utils.py; this pins the dispatch,
        which is the only place the MXFP8 path could silently drop the clamp.
        """
        from functools import partial

        from megatron.core.inference.moe.activations import squared_relu_and_quantize_mxfp8
        from megatron.core.inference.moe.fused_moe import ActivationType, _get_activation_func

        bound = _get_activation_func(
            ActivationType.SQUARED_RELU,
            fused_quant=True,
            activation_kwargs={"clamp_scale": CLAMP_SCALE},
        )
        assert isinstance(bound, partial)
        assert bound.func is squared_relu_and_quantize_mxfp8
        assert bound.keywords == {"clamp_scale": CLAMP_SCALE}

        # An unset clamp must leave the kernel unwrapped so the default path is unchanged.
        assert (
            _get_activation_func(
                ActivationType.SQUARED_RELU,
                fused_quant=True,
                activation_kwargs={"clamp_scale": None},
            )
            is squared_relu_and_quantize_mxfp8
        )


@pytest.mark.internal
@pytest.mark.parametrize("bi_mode", [False, True])
def test_persistent_buffers_preserve_mode_allocation_order(monkeypatch, bi_mode):
    """The ordinary alignment scalar must not perturb BI's persistent allocations."""
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
    buffers = module.VllmFusedMoeBuffers
    monkeypatch.setattr(buffers, "_buffers", None)
    allocations = []
    empty = torch.empty

    def record_empty(size, *, dtype, device):
        assert buffers._buffers is None
        allocations.append((size, dtype))
        return empty(size, dtype=dtype, device=device)

    with set_batch_invariant_mode(bi_mode, backend="triton"):
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", record_empty)
            buffers.allocate_buffers(16, 3, 128, 256, 8)
    expected = [
        (16 * 3 * 128, torch.bfloat16),
        (16 * 3 * 256, torch.bfloat16),
        (16 * 3 + 128 * 9, torch.int32),
        (3 + 9, torch.int32),
    ]
    if not bi_mode:
        expected.append((1, torch.int32))
    expected.append((16 * 256, torch.float32))
    assert allocations == expected
    assert ("num_tokens_post_padded" in buffers._buffers) is not bi_mode


@pytest.mark.internal
@pytest.mark.parametrize("initial_bi_mode", [False, True])
def test_persistent_buffers_mode_switch_graph_replay(monkeypatch, initial_bi_mode):
    """Re-capture after mode switches, including ordinary's missing-scalar fallback."""
    from megatron.core.inference.moe.fused_moe import ActivationType
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")
    buffers = module.VllmFusedMoeBuffers
    monkeypatch.setattr(buffers, "_buffers", None)
    tokens, hidden, width, topk, experts = 16, 256, 128, 3, 8
    x, probs, routing, fc1, fc2 = _make_moe_inputs(tokens, hidden, width, topk, experts)
    valid = _vt(tokens)
    out = torch.empty((tokens, hidden), dtype=torch.float32, device="cuda")
    with set_batch_invariant_mode(initial_bi_mode, backend="triton"):
        buffers.allocate_buffers(tokens, topk, width, hidden, experts)
    pointers = {name: value.data_ptr() for name, value in buffers._buffers.items()}

    def forward():
        out.zero_()
        return module.vllm_fused_moe(
            x,
            probs,
            fc1,
            fc2,
            ActivationType.SQUARED_RELU,
            experts,
            0,
            valid,
            routing,
            out=out,
            num_tokens_hint=tokens,
        )

    for bi_mode in (initial_bi_mode, not initial_bi_mode):
        with set_batch_invariant_mode(bi_mode, backend="triton"):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            for n in (0, 1, tokens):
                valid.fill_(n)
                routing.random_(0, experts * 2)
                expected = forward().clone()
                graph.replay()
                assert torch.equal(out.view(torch.int32), expected.view(torch.int32))
            del graph
    assert {name: value.data_ptr() for name, value in buffers._buffers.items()} == pointers


@pytest.mark.internal
@pytest.mark.parametrize("activation", ["SQUARED_RELU", "SWIGLU"])
@pytest.mark.parametrize("hint", [8, 128, 2049])
def test_bi_keeps_original_expert_dispatch(monkeypatch, activation, hint):
    """BI must not enter ordinary tuning, parallel alignment, or the tiled reduction."""
    from megatron.core.inference.moe.fused_moe import ActivationType
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    module = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

    def forbidden(*args, **kwargs):
        raise AssertionError("BI reached an ordinary-only optimization")

    class ForbiddenKernel:
        def __getitem__(self, grid):
            return forbidden

    monkeypatch.setattr(module, "_get_launch_configs", forbidden)
    monkeypatch.setattr(module, "_moe_align_block_size_parallel", forbidden)
    monkeypatch.setattr(module, "_moe_sum_tiled_kernel", ForbiddenKernel())
    tokens, hidden, width, topk, experts = 16, 256, 128, 3, 8
    x, probs, routing, fc1, fc2 = _make_moe_inputs(tokens, hidden, width, topk, experts)
    if activation == "SWIGLU":
        fc1 = torch.cat((fc1, -fc1), dim=1)
    with set_batch_invariant_mode(True, backend="triton"):
        out = module.vllm_fused_moe(
            x,
            probs,
            fc1,
            fc2,
            getattr(ActivationType, activation),
            experts,
            0,
            _vt(tokens),
            routing,
            num_tokens_hint=hint,
        )
    assert torch.isfinite(out).all()

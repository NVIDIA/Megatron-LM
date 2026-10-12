# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_local_routing_metadata import (
    HAVE_TRITON,
    fused_local_routing_metadata,
)


def local_routing_metadata_reference(
    routing_map, probs, num_token_slots, num_permuted_tokens, pad_multiple, chunk_size
):
    """HybridEP's single-rank scan semantics in plain torch; dense_to_expert_map rows at or past
    the number of dense tokens are -1 here (the kernel leaves them unwritten)."""
    num_tokens, num_experts = routing_map.shape
    device = routing_map.device
    routed = torch.zeros((num_token_slots, num_experts), dtype=torch.bool, device=device)
    routed[:num_tokens] = routing_map
    counts_raw = routed.sum(dim=0, dtype=torch.int64)
    padded_raw = (counts_raw + pad_multiple - 1) // pad_multiple * pad_multiple
    starts = torch.cumsum(padded_raw, 0) - padded_raw
    ends_raw = starts + counts_raw
    counts = torch.where(
        ends_raw <= num_permuted_tokens,
        counts_raw,
        torch.where(
            starts < num_permuted_tokens, num_permuted_tokens - starts, torch.zeros_like(counts_raw)
        ),
    )
    padded = (counts + pad_multiple - 1) // pad_multiple * pad_multiple
    overflow_flag = (ends_raw[-1] > num_permuted_tokens).to(torch.int32).reshape(1)
    needed = routed.any(dim=1)
    dense = torch.cumsum(needed.to(torch.int64), 0) - needed.to(torch.int64)
    num_dense_tokens = int(needed.sum().item())
    rank = torch.cumsum(routed.to(torch.int64), 0) - routed.to(torch.int64)
    rows = starts[None, :] + rank
    kept = routed & (rows < num_permuted_tokens)
    dense_to_expert_map = torch.full(
        (num_token_slots, num_experts), -1, dtype=torch.int32, device=device
    )
    dense_to_expert_map[dense[needed]] = torch.where(kept, rows, -1).to(torch.int32)[needed]
    num_chunks = (num_token_slots - 1) // chunk_size + 1
    dense_chunk_layout = torch.empty((num_chunks,), dtype=torch.int32, device=device)
    dense_chunk_layout[:-1] = dense[torch.arange(1, num_chunks, device=device) * chunk_size]
    dense_chunk_layout[-1] = num_dense_tokens
    sparse_to_dense_map = torch.where(needed, dense, -1).to(torch.int32).reshape(-1, 1)
    row_to_token = torch.full((num_permuted_tokens,), -1, dtype=torch.int32, device=device)
    tokens = torch.arange(num_token_slots, device=device)[:, None].expand(-1, num_experts)
    row_to_token[rows[kept]] = tokens[kept].to(torch.int32)
    permuted_probs = None
    if probs is not None:
        probs_full = torch.zeros((num_token_slots, num_experts), dtype=torch.float32, device=device)
        probs_full[:num_tokens] = probs
        permuted_probs = torch.zeros((num_permuted_tokens,), dtype=torch.float32, device=device)
        permuted_probs[rows[kept]] = probs_full[kept]
    return dict(
        tokens_per_expert=counts.to(torch.int32),
        padded_tokens_per_expert=padded,
        dense_to_expert_map=dense_to_expert_map,
        dense_chunk_layout=dense_chunk_layout,
        sparse_to_dense_map=sparse_to_dense_map,
        rdma_to_attn_map=needed.reshape(-1, 1),
        num_dispatched_tokens=torch.tensor([num_dense_tokens], dtype=torch.int32, device=device),
        overflow_flag=overflow_flag,
        row_to_token=row_to_token,
        permuted_probs=permuted_probs,
    )


def make_routing(num_tokens, num_experts, topk, drop_fraction=0.0, skew=0.0, seed=0):
    """Top-k routing map and probabilities; drop_fraction clears slots (some tokens may end up
    routed nowhere), skew concentrates the picks on the low experts."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.exp(-skew * torch.arange(num_experts).float()).expand(num_tokens, -1)
    picks = torch.multinomial(weights, topk, replacement=False, generator=generator)
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool).scatter_(1, picks, True)
    if drop_fraction > 0:
        routing_map &= torch.rand(num_tokens, num_experts, generator=generator) >= drop_fraction
    probs = torch.rand(num_tokens, num_experts, generator=generator) * routing_map
    return routing_map.cuda(), probs.cuda()


def assert_same_metadata(metadata, reference):
    num_dense_tokens = reference["num_dispatched_tokens"].item()
    for name, expected in reference.items():
        actual = getattr(metadata, name)
        if expected is None:
            assert actual is None, name
            continue
        if name == "dense_to_expert_map":
            actual, expected = actual[:num_dense_tokens], expected[:num_dense_tokens]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available")
@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, pad_multiple, capacity_factor, drop_fraction, skew",
    [
        (1000, 32, 4, 128, 1.2, 0.0, 0.0),  # every token routed, no drop
        (1000, 32, 4, 128, 1.2, 0.05, 0.0),  # some tokens routed nowhere
        (1000, 32, 4, 256, 1.0, 0.0, 0.3),  # skewed: experts run past the budget
        (37, 6, 2, 128, 2.0, 0.1, 0.0),  # odd token count, non-power-of-2 experts
        (5, 8, 1, 128, 1.0, 0.0, 0.0),  # fewer tokens than one slot group of 16
    ],
)
def test_fused_local_routing_metadata_matches_reference(
    num_tokens, num_experts, topk, pad_multiple, capacity_factor, drop_fraction, skew
):
    routing_map, probs = make_routing(num_tokens, num_experts, topk, drop_fraction, skew)
    num_token_slots = (max(num_tokens, 16) + 15) // 16 * 16
    num_permuted_tokens = int(num_tokens * topk * capacity_factor)
    num_permuted_tokens += -num_permuted_tokens % pad_multiple
    chunk_size = 64

    metadata = fused_local_routing_metadata(
        routing_map, probs, num_token_slots, num_permuted_tokens, pad_multiple, chunk_size
    )
    reference = local_routing_metadata_reference(
        routing_map, probs, num_token_slots, num_permuted_tokens, pad_multiple, chunk_size
    )
    assert_same_metadata(metadata, reference)
    if skew > 0:
        assert metadata.overflow_flag.item() == 1
        assert metadata.tokens_per_expert.sum() < routing_map.sum()
    else:
        assert metadata.overflow_flag.item() == 0


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available")
def test_fused_local_routing_metadata_without_probs_accumulates_over_budget():
    routing_map, _ = make_routing(512, 16, 2, skew=0.5)
    num_permuted_tokens = 1024
    over_budget = torch.zeros(1, dtype=torch.bool, device="cuda")

    metadata = fused_local_routing_metadata(routing_map, None, 512, num_permuted_tokens, 128, 64)
    assert metadata.permuted_probs is None
    assert metadata.overflow_flag.item() == 1

    fused_local_routing_metadata(routing_map, None, 512, num_permuted_tokens, 128, 64, over_budget)
    assert over_budget.item()
    # A budget that holds every padded expert: no overflow, an earlier one stays accumulated.
    ample_budget = 512 * 2 + 16 * 128
    fresh = torch.zeros(1, dtype=torch.bool, device="cuda")
    metadata = fused_local_routing_metadata(routing_map, None, 512, ample_budget, 128, 64, fresh)
    assert metadata.overflow_flag.item() == 0 and not fresh.item()
    fused_local_routing_metadata(routing_map, None, 512, ample_budget, 128, 64, over_budget)
    assert over_budget.item()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TRITON, reason="Triton not available")
def test_fused_local_routing_metadata_cuda_graph_replay():
    routing_map, probs = make_routing(1024, 32, 4, drop_fraction=0.02)
    args = (routing_map, probs, 1024, 4096 + 1024, 128, 64)
    eager = fused_local_routing_metadata(*args)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_local_routing_metadata(*args)
    # Replay on another routing, then on the original one: the outputs follow the inputs.
    routing_map.copy_(make_routing(1024, 32, 4, drop_fraction=0.02, seed=1)[0])
    graph.replay()
    routing_map.copy_(make_routing(1024, 32, 4, drop_fraction=0.02)[0])
    graph.replay()
    torch.cuda.synchronize()

    num_dense_tokens = eager.num_dispatched_tokens.item()
    for name in eager._fields:
        expected, actual = getattr(eager, name), getattr(captured, name)
        if name == "dense_to_expert_map":
            expected, actual = expected[:num_dense_tokens], actual[:num_dense_tokens]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)

# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.

"""Bit-exact replay of the ``--moe-token-dedup`` kernels in
``megatron/core/transformer/moe/token_dedup.py``.

Covers the Triton send/receive index builders and the five autograd functions of the
deduplicating all-to-all dispatcher: payload pack, received-row expand, expert-output permute,
expert-output reduce, and the final restore. Every backward is a bounded gather (no
atomics), so forward and backward must replay byte-identically, also under stream contention.
"""

import pytest
import torch

from megatron.core.transformer.moe import token_dedup
from megatron.core.transformer.moe.token_dedup import (
    build_wire_recv_index,
    build_wire_send_index,
    expand_token_dedup_received,
    fused_index_build_available,
    pack_token_dedup_payload,
    permute_token_dedup_expert_output,
    reduce_token_dedup_expert_output,
    restore_token_dedup_output,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_index_build_available(torch.device("cuda")),
    reason="needs a GPU and triton",
)

NUM_TOKENS, HIDDEN, NUM_EXPERTS, TOPK, EP_SIZE = 4096, 1024, 32, 4, 4
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE


def _routing_map(num_tokens=NUM_TOKENS, num_experts=NUM_EXPERTS, topk=TOPK):
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    idx = logits.topk(topk, dim=-1).indices
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, idx, True)
    return routing_map


def _send_plan(routing_map):
    """Mirror MoEDeduplicatingAlltoAllTokenDispatcher.dispatch_preprocess (fused path)."""
    num_tokens = routing_map.shape[0]
    hits = routing_map.reshape(num_tokens, EP_SIZE, NUM_LOCAL_EXPERTS).any(dim=-1)
    dedup_send_counts = hits.sum(dim=0, dtype=torch.long)
    order = hits.t().contiguous().cumsum(dim=1).t().contiguous()
    base = dedup_send_counts.cumsum(dim=0) - dedup_send_counts
    max_wire_rows = num_tokens * min(TOPK, EP_SIZE)
    return hits, dedup_send_counts, order, base, max_wire_rows


def test_wire_send_index_replays():
    seeded()
    routing_map = _routing_map()
    _, dedup_send_counts, order, base, max_wire_rows = _send_plan(routing_map)
    # The wire tensors are sized by an upper bound and narrowed by the dispatcher once the
    # split counts reach the host; only the narrowed rows are defined (the tail is never written).
    num_wire_rows = int(dedup_send_counts.sum())

    def fn(routing_map, order, base):
        indices, slot_wire_rows, token_wire_rows, wire_tokens, wire_experts = build_wire_send_index(
            routing_map, order, base, NUM_LOCAL_EXPERTS, EP_SIZE, TOPK, max_wire_rows
        )
        return (
            indices,
            slot_wire_rows,
            token_wire_rows,
            wire_tokens[:num_wire_rows],
            wire_experts[:num_wire_rows],
        )

    assert_replays_bit_exact(
        fn, (routing_map, order, base), backward=False, contention=True, what="wire send index"
    )


def test_wire_recv_index_replays():
    seeded()
    routing_map = _routing_map()
    _, dedup_send_counts, order, base, max_wire_rows = _send_plan(routing_map)
    _, _, _, _, wire_experts = build_wire_send_index(
        routing_map, order, base, NUM_LOCAL_EXPERTS, EP_SIZE, TOPK, max_wire_rows
    )
    # Single process: act as EP rank 0, which receives the destination-0 block of wire rows.
    num_recv = int(dedup_send_counts[0])
    received_experts = wire_experts[:num_recv].contiguous()
    tokens_per_local_expert = routing_map[:, :NUM_LOCAL_EXPERTS].sum(dim=0, dtype=torch.long)
    num_assignments = int(tokens_per_local_expert.sum())

    def fn(received_experts, tokens_per_local_expert):
        return build_wire_recv_index(received_experts, tokens_per_local_expert, num_assignments)

    assert_replays_bit_exact(
        fn,
        (received_experts, tokens_per_local_expert),
        backward=False,
        contention=True,
        what="wire recv index",
    )


def _dedup_plan():
    """One complete single-rank dedup plan: send side for all ranks, receive side for rank 0."""
    routing_map = _routing_map()
    _, dedup_send_counts, order, base, max_wire_rows = _send_plan(routing_map)
    indices, slot_wire_rows, token_wire_rows, wire_tokens, wire_experts = build_wire_send_index(
        routing_map, order, base, NUM_LOCAL_EXPERTS, EP_SIZE, TOPK, max_wire_rows
    )
    num_wire_rows = int(dedup_send_counts.sum())
    wire_tokens = wire_tokens[:num_wire_rows]
    wire_experts = wire_experts[:num_wire_rows]
    num_recv = int(dedup_send_counts[0])
    received_experts = wire_experts[:num_recv].contiguous()
    tokens_per_local_expert = routing_map[:, :NUM_LOCAL_EXPERTS].sum(dim=0, dtype=torch.long)
    num_assignments = int(tokens_per_local_expert.sum())
    wire_to_expert_rows, expert_wire_rows = build_wire_recv_index(
        received_experts, tokens_per_local_expert, num_assignments
    )
    return dict(
        routing_map=routing_map,
        indices=indices,
        slot_wire_rows=slot_wire_rows,
        token_wire_rows=token_wire_rows,
        wire_tokens=wire_tokens,
        wire_experts=wire_experts,
        num_wire_rows=num_wire_rows,
        num_recv=num_recv,
        received_experts=received_experts,
        num_assignments=num_assignments,
        wire_to_expert_rows=wire_to_expert_rows,
        expert_wire_rows=expert_wire_rows,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pack_payload_replays(dtype):
    seeded()
    plan = _dedup_plan()
    hidden = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=dtype, requires_grad=True)
    probs = torch.rand(NUM_TOKENS, NUM_EXPERTS, device="cuda", dtype=dtype)
    slot_probs = probs.gather(1, plan["indices"]).detach().requires_grad_(True)
    assert_replays_bit_exact(
        pack_token_dedup_payload,
        (
            hidden,
            slot_probs,
            plan["wire_tokens"],
            plan["wire_experts"],
            plan["token_wire_rows"],
            plan["slot_wire_rows"],
        ),
        contention=True,
        what=f"pack payload[{dtype}]",
    )


def test_expand_received_replays():
    seeded()
    plan = _dedup_plan()
    received = torch.randn(
        plan["num_recv"], HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    assert_replays_bit_exact(
        expand_token_dedup_received,
        (received, plan["expert_wire_rows"], plan["wire_to_expert_rows"]),
        contention=True,
        what="expand received",
    )


def test_permute_expert_output_replays():
    seeded()
    n = 2048
    expert_output = torch.randn(n, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    perm = torch.randperm(n, device="cuda")
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n, device="cuda")
    assert_replays_bit_exact(
        permute_token_dedup_expert_output,
        (expert_output, perm, inverse),
        contention=True,
        what="permute expert output",
    )


def test_reduce_replays():
    seeded()
    plan = _dedup_plan()
    expert_output = torch.randn(
        plan["num_assignments"], HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    assert_replays_bit_exact(
        reduce_token_dedup_expert_output,
        (expert_output, plan["expert_wire_rows"], plan["wire_to_expert_rows"]),
        contention=True,
        what="reduce",
    )


def test_restore_output_replays():
    seeded()
    plan = _dedup_plan()
    returned = torch.randn(
        plan["num_wire_rows"], HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    assert_replays_bit_exact(
        restore_token_dedup_output,
        (returned, plan["token_wire_rows"], plan["wire_tokens"]),
        contention=True,
        what="restore output",
    )


def test_triton_and_torch_paths_agree():
    """The torch fallback (CPU / no triton) and the Triton kernels compute the same values."""
    seeded()
    plan = _dedup_plan()
    hidden = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", dtype=torch.float32)
    slot_probs = torch.rand(NUM_TOKENS, TOPK, device="cuda", dtype=torch.float32)
    args = (
        hidden,
        slot_probs,
        plan["wire_tokens"],
        plan["wire_experts"],
        plan["token_wire_rows"],
        plan["slot_wire_rows"],
    )
    packed_gpu = pack_token_dedup_payload(*args)
    packed_cpu = pack_token_dedup_payload(*(a.cpu() for a in args))
    hidden_cols = packed_gpu[:, : HIDDEN + 2 * TOPK]
    torch.testing.assert_close(hidden_cols.cpu(), packed_cpu[:, : HIDDEN + 2 * TOPK])
    assert token_dedup.triton is not None

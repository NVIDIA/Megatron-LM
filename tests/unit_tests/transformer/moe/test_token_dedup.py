# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
"""Tests for the MoE token-to-destination dedup plan and kernels."""

import dataclasses

import pytest
import torch

from megatron.core.transformer.moe.token_dedup import (
    expand_token_dedup_received,
    pack_token_dedup_payload,
    permute_token_dedup_expert_output,
    reduce_token_dedup_expert_output,
    restore_token_dedup_output,
)
from megatron.core.transformer.moe.token_dispatcher import (
    MoEDeduplicatingAlltoAllTokenDispatcher,
    _compact_mask_indices,
    _token_destination_hit_map,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer


def test_destination_hit_map_deduplicates_same_rank_experts():
    # Two experts per EP rank: {0, 1} -> rank 0 and {2, 3} -> rank 1.
    indices = torch.tensor(
        [
            [0, 1],  # one wire row to rank 0
            [0, 2],  # one wire row to each rank
            [3, 2],  # one wire row to rank 1
            [1, 1],  # repeated expert is still one wire row
        ],
        dtype=torch.long,
    )

    destinations, hits = _token_destination_hit_map(indices, num_local_experts=2, ep_size=2)

    assert torch.equal(destinations, torch.tensor([[0, 0], [0, 1], [1, 1], [0, 0]]))
    assert torch.equal(hits, torch.tensor([[True, True, False, True], [False, True, True, False]]))
    # Eight token/expert assignments become five token/destination wire rows.
    assert hits.sum().item() == 5


def test_destination_major_compaction_needs_no_sort():
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]])

    compact = _compact_mask_indices(mask, count=4)

    # Flattened destination-major positions: rank0 tokens 0/2, rank1 tokens 1/2.
    assert torch.equal(compact, torch.tensor([0, 2, 5, 6]))


def test_compaction_handles_empty_mask():
    mask = torch.zeros((2, 0), dtype=torch.bool)
    compact = _compact_mask_indices(mask, count=0)
    assert compact.dtype == torch.long
    assert compact.numel() == 0


def test_packed_metadata_uses_same_wire_rows_and_preserves_gradients():
    hidden = torch.arange(12, dtype=torch.float32).view(3, 4).requires_grad_()
    probs = torch.tensor([[0.25, 0.75], [0.4, 0.6], [0.125, 0.875]], requires_grad=True)
    wire_tokens = torch.tensor([0, 2, 0])
    wire_experts = torch.tensor([[1, -1], [0, 2], [-1, 3]])
    token_wire_rows = torch.tensor([[0, 2], [-1, -1], [1, -1]])
    slot_wire_rows = torch.tensor([[0, 2], [-1, -1], [1, 1]])

    packed = pack_token_dedup_payload(
        hidden, probs, wire_tokens, wire_experts, token_wire_rows, slot_wire_rows
    )

    assert packed.shape == (3, 8)
    assert torch.equal(packed[:, :4], hidden.detach().index_select(0, wire_tokens))
    assert torch.equal(packed[:, 4:6].long(), wire_experts)
    assert torch.equal(packed[:, 6:8], torch.tensor([[0.25, 0.0], [0.125, 0.875], [0.0, 0.75]]))

    packed.sum().backward()
    assert torch.equal(
        hidden.grad,
        torch.tensor([[2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    assert torch.equal(probs.grad, torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]))


def test_received_expansion_and_wire_permutation_preserve_gradients():
    received = torch.arange(12, dtype=torch.float32).view(3, 4).requires_grad_()
    wire_rows = torch.tensor([0, 2, 1, 0])
    wire_to_expert_rows = torch.tensor([[0, 3], [2, -1], [1, -1]])

    expanded = expand_token_dedup_received(received, wire_rows, wire_to_expert_rows)
    expanded.sum().backward()
    assert torch.equal(
        received.grad,
        torch.tensor([[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    )

    expert_output = torch.arange(16, dtype=torch.float32).view(4, 4)
    expert_output.requires_grad_()
    wire_to_expert_row = torch.tensor([0, 3, 2, 1])
    expert_row_to_wire_rank = torch.tensor([0, 3, 2, 1])
    wire_output = permute_token_dedup_expert_output(
        expert_output, wire_to_expert_row, expert_row_to_wire_rank
    )
    row_weights = torch.arange(1, 5, dtype=torch.float32).unsqueeze(1)
    (wire_output * row_weights).sum().backward()
    assert torch.equal(expert_output.grad[:, 0], torch.tensor([1.0, 4.0, 3.0, 2.0]))


def test_reduce_and_restore_sum_expert_rows():
    expert_output = torch.arange(8, dtype=torch.float32).view(4, 2).requires_grad_()
    # Wire 0 holds two experts of token 0; wires 1 and 2 hold token 1's experts.
    wire_to_expert_rows = torch.tensor([[0, 1], [2, -1], [-1, 3]])
    expert_wire_rows = torch.tensor([0, 0, 1, 2])
    token_wire_rows = torch.tensor([[0, -1], [1, 2]])
    wire_tokens = torch.tensor([0, 1, 1])

    partials = reduce_token_dedup_expert_output(
        expert_output, expert_wire_rows, wire_to_expert_rows
    )
    output = restore_token_dedup_output(partials, token_wire_rows, wire_tokens)
    e = expert_output.detach()
    assert torch.equal(output, torch.stack((e[0] + e[1], e[2] + e[3])))

    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    output.backward(grad)
    assert torch.equal(expert_output.grad, grad[[0, 0, 1, 1]])


# ---------------------------------------------------------------------------
# GPU: the deduplicating dispatcher inside a MoELayer (needs torch.distributed + EP > 1).
# ---------------------------------------------------------------------------


class _DedupContainer(MoEModelTestContainer):
    """MoEModelTestContainer whose MoE layer uses the deduplicating alltoall dispatcher."""

    def __init__(self, *args, moe_router_dtype="fp32", **kwargs):
        super().__init__(*args, **kwargs)
        self.config = dataclasses.replace(
            self.config, moe_token_dedup=True, moe_router_dtype=moe_router_dtype
        )
        self.moe_layer = self.new_moe_layer()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTokenDedupDispatcher:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("ep_size", [2, 8])
    def test_dropless_roundtrip(self, ep_size):
        """Dispatch -> expand -> combine -> restore must reproduce the input (and grad)."""
        if Utils.world_size % ep_size != 0:
            pytest.skip(f"needs a world size divisible by EP={ep_size}")
        container = _DedupContainer(
            tp_size=1,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_token_dispatcher_type="alltoall",
        )
        dispatcher = container.moe_layer.token_dispatcher
        assert isinstance(dispatcher, MoEDeduplicatingAlltoAllTokenDispatcher)
        container.dispatcher_dropless_test()

    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("ep_size,topk", [(2, 2), (8, 2), (8, 6)])
    def test_matches_stock_dispatcher(self, ep_size, topk):
        """Full MoELayer fwd+bwd: dedup and the stock alltoall dispatcher agree on identical weights.

        Only the summation order of a token's expert outputs differs, so the comparison is at
        fp32 rounding level; TF32 matmuls are disabled for the comparison because their
        reduced-precision accumulation is order-sensitive at the 1e-3 level, and the router
        runs in fp64 because its fp32 gate goes through TE's general GEMM, whose rounding
        amplifies last-bit differences in the probability gradient into the weight gradient.
        """
        if Utils.world_size % ep_size != 0:
            pytest.skip(f"needs a world size divisible by EP={ep_size}")
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_precision = torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        try:
            self._matches_stock_dispatcher(ep_size, topk)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            torch.set_float32_matmul_precision(prev_precision)

    def _matches_stock_dispatcher(self, ep_size, topk):
        container = _DedupContainer(
            tp_size=1,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=32,
            moe_router_topk=topk,
            moe_router_dtype="fp64",
        )
        dedup_layer = container.moe_layer
        stock_layer = container.new_moe_layer(moe_token_dedup=False)
        stock_layer.load_state_dict(dedup_layer.state_dict())
        assert isinstance(dedup_layer.token_dispatcher, MoEDeduplicatingAlltoAllTokenDispatcher)
        assert not isinstance(stock_layer.token_dispatcher, MoEDeduplicatingAlltoAllTokenDispatcher)

        torch.manual_seed(7 + torch.distributed.get_rank())
        x = torch.randn((16, 4, container.config.hidden_size), device="cuda")

        def run(layer):
            inp = x.clone().detach().requires_grad_(True)
            out, bias = layer(inp)
            assert bias is None
            out.sum().backward()
            grads = [p.grad.detach().clone() for p in layer.experts.parameters()]
            grads.append(layer.router.weight.grad.detach().clone())
            return out.detach(), inp.grad.detach(), grads

        out_ref, dx_ref, dw_ref = run(stock_layer)
        out_ded, dx_ded, dw_ded = run(dedup_layer)
        torch.testing.assert_close(out_ded, out_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(dx_ded, dx_ref, rtol=1e-4, atol=1e-4)
        assert len(dw_ref) == len(dw_ded) > 0
        for g_ref, g_ded in zip(dw_ref, dw_ded):
            torch.testing.assert_close(g_ded, g_ref, rtol=1e-4, atol=1e-4)

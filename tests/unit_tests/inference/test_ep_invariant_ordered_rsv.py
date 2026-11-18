# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""moe_ep_invariant_combine at inference: the ordered pairwise reduce-scatter-v over NVLS.

Kept in the inference test bucket: it allocates CUDA-backend symmetric memory, which fixes
the process-wide symmetric-memory backend and would break later NCCL-backend users (the
ncclep dispatcher tests) in the transformer/moe bucket.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe import ep_invariant_combine as eic
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_moe_ep_invariant_combine import (
    _bits,
    _case,
    _rank_inverse_map,
    _reference,
    needs_cuda,
)


@pytest.fixture
def ep_world():
    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


@needs_cuda
@pytest.mark.skipif(Utils.world_size < 2, reason="needs >= 2 GPUs")
def test_nvls_pairwise_reduce_scatter_equals_ep1(ep_world):
    from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
    from megatron.core.inference.moe import batch_invariant as inf_bi
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    if not is_device_nvls_capable(torch.device("cuda")):
        pytest.skip("needs NVLS")
    ep, rank = Utils.world_size, torch.distributed.get_rank()
    group = parallel_state.get_expert_model_parallel_group()
    experts = 8 * Utils.world_size
    idx, vals = _case(48 * Utils.world_size, experts, 6, 512, seed=11)
    tokens, hidden = idx.shape[0], vals.shape[-1]
    counts = [tokens // ep - 1] * (ep - 1) + [tokens - (tokens // ep - 1) * (ep - 1)]
    buf = SymmetricMemoryManager.get_buffer(
        "test_ep_inv", process_group=group, size_mb=-(-tokens * hidden * 4 // (1 << 20))
    )
    rsv = buf.maybe_get_tensor([tokens, hidden], dtype=torch.float32)
    inv, (t, k) = _rank_inverse_map(idx, rank, experts // ep)
    valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    y = vals[t, k].to(torch.bfloat16).contiguous()
    inf_bi.unpermute_tokens_in_expert_order(
        y, inv, valid, rsv["tensor"], eic.num_leaf_blocks(experts) // ep
    )
    out = torch.empty(counts[rank], hidden, dtype=torch.float32, device="cuda")
    inf_bi.ordered_reduce_scatter_v(
        out,
        rsv["tensor"],
        rsv["handle"],
        rank_token_offset=torch.tensor([sum(counts[:rank])], dtype=torch.int32, device="cuda"),
        ep_max_tokens=torch.tensor([max(counts)], dtype=torch.int32, device="cuda"),
        per_rank_max_tokens=max(counts),
        pairwise_tree=True,
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    ref = _reference(idx, vals, experts)
    off = sum(counts[:rank])
    assert torch.equal(_bits(out), _bits(ref[off : off + counts[rank]]))
    SymmetricMemoryManager.destroy("test_ep_inv")


@needs_cuda
@pytest.mark.skipif(
    Utils.world_size < 2 or Utils.world_size % 2, reason="needs an even number of GPUs"
)
def test_expert_tp_fold_adds_matching_rows_across_permutations():
    """mcore_fused_moe at expert TP 2: the ranks' expert-grouped row orders differ (the
    permutation claims rows with atomic counters), yet each (token, expert) row must be
    summed with its own expert-TP partial before the subtree, as training does."""
    from megatron.core.inference.moe.fused_moe import _rows_in_topk_order

    Utils.initialize_model_parallel(1, 1)
    rank = torch.distributed.get_rank()
    groups = [torch.distributed.new_group([r, r + 1]) for r in range(0, Utils.world_size, 2)]
    group = groups[rank // 2]
    try:
        tokens, topk, hidden, experts, num_local, start, leaves, valid = 64, 4, 256, 32, 8, 8, 4, 50
        gen = torch.Generator(device="cuda").manual_seed(5)  # same routing on both ranks
        routing = torch.stack(
            [torch.randperm(experts, device="cuda", generator=gen)[:topk] for _ in range(tokens)]
        )
        # Distinct partials per expert-TP rank, in top-k slot order.
        part_gen = torch.Generator(device="cuda").manual_seed(100 + group.rank())
        part = torch.randn(tokens * topk, hidden, device="cuda", generator=part_gen)
        part = part.to(torch.bfloat16)

        # Expert-grouped layout as the permutation produces it; the second rank stores each
        # expert's rows in reverse, with its inverse map to match.
        grouped, inverse_map = [], torch.full((tokens, num_local), -1, dtype=torch.int32)
        reversed_any = False
        for lid in range(num_local):
            pairs = [
                (t, k) for t in range(valid) for k in range(topk) if routing[t, k] == start + lid
            ]
            if group.rank() == 1:
                pairs.reverse()
                reversed_any |= len(pairs) > 1
            for t, k in pairs:
                inverse_map[t, lid] = len(grouped)
                grouped.append(part[t * topk + k])
        grouped = torch.stack(grouped + [torch.full((hidden,), 3.0, device="cuda")] * 5).bfloat16()
        assert group.rank() == 0 or reversed_any, "the test needs differing row orders"

        rows = _rows_in_topk_order(grouped, inverse_map.cuda(), routing, start, num_local)
        folded = eic.expert_tp_fold(rows, group)
        valid_t = torch.tensor([valid], dtype=torch.int32, device="cuda")
        out = torch.full((tokens, hidden), 7.0, device="cuda")
        eic.tree_combine_topk_forward(folded, routing, start, num_local, leaves, valid_t, out)

        # Reference: the two ranks' partials summed slot by slot, then the same subtree.
        local = ((routing >= start) & (routing < start + num_local)).flatten()
        mask = (local & (torch.arange(tokens * topk, device="cuda") // topk < valid))[:, None]
        mine = torch.where(mask, part, torch.zeros_like(part))
        parts = torch.empty((2, *mine.shape), dtype=mine.dtype, device="cuda")
        torch.distributed.all_gather_into_tensor(parts, mine, group=group)
        ref = torch.full((tokens, hidden), 7.0, device="cuda")
        eic.tree_combine_topk_forward(
            eic.fold_expert_tp_partials(parts), routing, start, num_local, leaves, valid_t, ref
        )
        assert torch.equal(_bits(out), _bits(ref))
        assert torch.all(out[valid:] == 7.0)
    finally:
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

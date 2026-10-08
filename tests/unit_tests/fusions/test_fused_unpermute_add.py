# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_unpermute_add import HAVE_TRITON, fused_unpermute_add


def _make_routing(num_tokens, num_experts, topk, pad_multiple, drop_frac, generator):
    """Dense [num_tokens, num_experts] map into expert-major, padded permuted rows (-1 = none)."""
    scores = torch.rand(num_tokens, num_experts, generator=generator)
    experts = scores.topk(topk, dim=1).indices
    routed = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    routed.scatter_(1, experts, True)
    routed &= torch.rand(num_tokens, num_experts, generator=generator) >= drop_frac
    counts = routed.sum(dim=0)
    padded = (counts + pad_multiple - 1) // pad_multiple * pad_multiple
    starts = torch.cumsum(padded, 0) - padded
    rank_in_expert = torch.cumsum(routed.int(), dim=0) - 1
    dense_map = torch.where(routed, starts[None, :] + rank_in_expert, -1).to(torch.int32)
    return dense_map, int(padded.sum())


def _reference(permuted, dense_map, num_dense, add):
    num_tokens = dense_map.shape[0]
    acc = torch.zeros(num_tokens, permuted.shape[1], dtype=torch.float32, device=permuted.device)
    if add is not None:
        acc += add.float()
    valid = torch.arange(num_tokens, device=permuted.device)[:, None] < num_dense
    for e in range(dense_map.shape[1]):
        src = dense_map[:, e].long()
        ok = (src >= 0)[:, None] & valid
        acc += torch.where(ok, permuted[src.clamp(min=0)].float(), 0.0)
    return acc.to(add.dtype if add is not None else permuted.dtype)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_TRITON, reason="Requires CUDA, Triton"
)
@pytest.mark.parametrize("num_experts", [8, 12, 32])
@pytest.mark.parametrize("with_add", [True, False])
@pytest.mark.parametrize("num_dense_offset", [0, 5])
def test_fused_unpermute_add(num_experts, with_add, num_dense_offset):
    generator = torch.Generator().manual_seed(0)
    num_tokens, hidden, topk = 300, 520, 4
    dense_map, num_permuted = _make_routing(num_tokens, num_experts, topk, 16, 0.05, generator)
    dense_map = dense_map.cuda()
    permuted = torch.randn(num_permuted, hidden, generator=generator).to(torch.bfloat16).cuda()
    add = torch.randn(num_tokens, hidden, generator=generator).to(torch.bfloat16).cuda()
    add = add if with_add else None
    num_dense = torch.tensor([num_tokens - num_dense_offset], dtype=torch.int32, device="cuda")

    out = fused_unpermute_add(permuted, dense_map, num_dense, topk, add=add, num_tokens=num_tokens)
    ref = _reference(permuted, dense_map, num_tokens - num_dense_offset, add)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)

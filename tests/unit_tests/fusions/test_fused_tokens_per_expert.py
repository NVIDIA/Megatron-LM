# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions import fused_tokens_per_expert as fused_module
from megatron.core.fusions.fused_tokens_per_expert import fused_tokens_per_expert
from megatron.core.transformer.moe.moe_utils import get_tokens_per_expert_and_token_count
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and fused_module.HAVE_TRITON), reason="needs a GPU and Triton"
)


def _random_map(rows, experts, density=0.05, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(rows, experts, device="cuda", generator=g) < density


@pytest.mark.parametrize(
    "rows, experts",
    [
        (4096, 256),  # power-of-two experts, rows a multiple of the row block
        (4097, 256),  # rows not a multiple of the row block
        (1000, 64),
        (1, 8),
        (333, 72),  # experts not a power of two (masked columns)
        (129, 7),
        (16, 1),
        (64, 1024),  # wide map: fewer rows per program
    ],
)
def test_matches_torch_sum_exactly(rows, experts):
    routing_map = _random_map(rows, experts, seed=rows * 7 + experts)
    got = fused_tokens_per_expert(routing_map)
    ref = routing_map.sum(dim=0)
    assert got.dtype == ref.dtype == torch.int64
    assert torch.equal(got, ref)


def test_all_tokens_to_one_expert_and_dense_map():
    rows, experts = 5000, 32
    one_hot = torch.zeros(rows, experts, dtype=torch.bool, device="cuda")
    one_hot[:, 3] = True
    assert torch.equal(fused_tokens_per_expert(one_hot), one_hot.sum(dim=0))
    dense = torch.ones(rows, experts, dtype=torch.bool, device="cuda")
    assert torch.equal(fused_tokens_per_expert(dense), dense.sum(dim=0))


def test_empty_map():
    routing_map = torch.zeros(0, 16, dtype=torch.bool, device="cuda")
    got = fused_tokens_per_expert(routing_map)
    assert got.dtype == torch.int64
    assert torch.equal(got, routing_map.sum(dim=0))


@pytest.mark.parametrize(
    "make",
    [
        lambda: _random_map(64, 128).t(),  # non-contiguous
        lambda: _random_map(256, 32)[:, :16],  # column slice
        lambda: _random_map(256, 32).int(),  # integer map
        lambda: _random_map(8, 4).view(8, 4, 1).expand(8, 4, 2),  # not 2D
        lambda: _random_map(256, 32).cpu(),  # CPU
    ],
)
def test_unsupported_layouts_fall_back_to_torch_sum(make):
    routing_map = make()
    got = fused_tokens_per_expert(routing_map)
    ref = routing_map.sum(dim=0)
    assert got.dtype == ref.dtype
    assert got.device == ref.device
    assert torch.equal(got, ref)


@pytest.mark.parametrize("with_padding_mask", [False, True])
def test_get_tokens_per_expert_and_token_count(with_padding_mask):
    """Counts reduced over the tensor-parallel group (size 1 here) equal the torch reference."""
    Utils.initialize_model_parallel()
    try:
        topk = 4
        routing_map = torch.zeros(2048, 64, dtype=torch.bool, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(0)
        idx = torch.rand(2048, 64, device="cuda", generator=g).topk(topk, dim=-1).indices
        routing_map.scatter_(1, idx, True)
        if with_padding_mask:
            routing_map[-100:] = False
        group = parallel_state.get_tensor_model_parallel_group()
        global_counts, local_num, total_num = get_tokens_per_expert_and_token_count(
            routing_map, group, topk=topk, with_padding_mask=with_padding_mask
        )
        assert torch.equal(global_counts, routing_map.sum(dim=0))
        expected_local = 2048 - 100 if with_padding_mask else 2048
        assert int(local_num) == expected_local
        assert int(total_num) == expected_local * group.size()
    finally:
        Utils.destroy_model_parallel()

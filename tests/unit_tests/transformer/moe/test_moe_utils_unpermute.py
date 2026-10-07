# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``moe_utils.unpermute`` accumulates permuted rows back into token order with ``scatter_add_``.

Under ``torch.use_deterministic_algorithms(True)`` (``--deterministic-mode``) torch >= 2.9 routes
``scatter_add_`` to the same sort-based ``index_put_`` that ``index_add_`` uses, so the output is
bit-reproducible and bit-identical to the ``index_add_`` fallback kept for older torch; without the
flag the atomic kernel is used. Both compute the same sums.
"""

import pytest
import torch

from megatron.core.transformer.moe import moe_utils
from megatron.core.utils import is_torch_min_version
from tests.unit_tests.stream_contention import SideStreamContention, replay_count

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _case(num_tokens, hidden, num_experts=64, topk=8, dtype=torch.bfloat16):
    torch.manual_seed(0)
    tokens = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)
    top = torch.rand(num_tokens, num_experts, device="cuda").topk(topk, dim=1).indices
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool, device="cuda")
    routing_map.scatter_(1, top, True)
    probs = torch.rand(num_tokens, num_experts, device="cuda") * routing_map
    permuted, _, sorted_indices, _, _ = moe_utils.permute(
        tokens, routing_map, num_out_tokens=num_tokens * topk
    )
    return permuted, sorted_indices, routing_map, probs


def _unpermute(permuted, sorted_indices, routing_map, probs):
    restore_shape = (probs.shape[0], permuted.shape[1])
    return moe_utils.unpermute(
        permuted, sorted_indices, restore_shape, probs=probs, routing_map=routing_map
    )


@pytest.fixture
def deterministic_algorithms():
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


def test_deterministic_branch_replays_bit_exact(deterministic_algorithms):
    """Under deterministic algorithms the output is byte-identical across replays (under side-stream
    contention where the process allows it; see tests/unit_tests/stream_contention.py)."""
    permuted, sorted_indices, routing_map, probs = _case(8192, 4096)
    reference = _unpermute(permuted, sorted_indices, routing_map, probs)
    for _ in range(replay_count(with_contention=4) - 1):
        with SideStreamContention():
            out = _unpermute(permuted, sorted_indices, routing_map, probs)
        assert torch.equal(out, reference)


def test_branches_compute_the_same_sums(deterministic_algorithms):
    """scatter_add_ (default) and index_add_ (deterministic) differ only in accumulation order."""
    permuted, sorted_indices, routing_map, probs = _case(4096, 2048, dtype=torch.float32)
    deterministic = _unpermute(permuted, sorted_indices, routing_map, probs)
    torch.use_deterministic_algorithms(False)
    default = _unpermute(permuted, sorted_indices, routing_map, probs)
    torch.testing.assert_close(deterministic, default, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not is_torch_min_version("2.9.0a0"),
    reason="deterministic scatter_add_ fast path needs torch >= 2.9",
)
def test_deterministic_scatter_add_matches_index_add(deterministic_algorithms):
    """What the single scatter_add_ path relies on: under the flag torch's scatter_add_ takes the
    sort-based index_put_ route and lands on exactly the bits index_add_ would produce."""
    permuted, sorted_indices, _, probs = _case(8192, 4096)
    hidden = permuted.shape[1]
    via_scatter = torch.zeros(probs.shape[0], hidden, device="cuda", dtype=permuted.dtype)
    via_scatter.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted)
    via_index_add = torch.zeros_like(via_scatter)
    via_index_add.index_add_(0, sorted_indices, permuted)
    assert torch.equal(via_scatter, via_index_add)

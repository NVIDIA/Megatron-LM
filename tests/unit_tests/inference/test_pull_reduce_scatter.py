# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The symmetric-memory pull reduce-scatter is bitwise NCCL's reduce-scatter at TP=2.

Batch-invariant mode uses it for the TP=2 sequence-parallel reduce-scatter in inference,
where training reduces the same partials with NCCL. Every element is one BF16 addition, so
the two agree exactly when both round the sum to nearest even.
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
    is_device_nvls_capable,
    pull_reduce_scatter,
)
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.parallel_state import get_tensor_model_parallel_group
from tests.unit_tests.test_utilities import Utils


def _special_bf16_bits():
    """BF16 bit patterns covering zeros, subnormals, the normal boundary, large values,
    infinities and every exponent, with both signs."""
    pats = [0x0000, 0x0001, 0x0002, 0x007F, 0x0080, 0x0081, 0x00FF, 0x3F80, 0x3F81, 0x3FFF]
    pats += [0x7F7F, 0x7F7E, 0x7F80, 0x0100, 0x4000]
    pats += [(e << 7) | m for e in range(0, 255, 7) for m in (0, 1, 0x40, 0x7F)]
    pats = sorted(set(pats))
    pats += [p | 0x8000 for p in pats]
    return torch.tensor(pats, dtype=torch.int32).to(torch.int16)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_device_nvls_capable(torch.device("cuda")),
    reason="needs symmetric memory over NVLink (Hopper+)",
)
class TestPullReduceScatter:

    @classmethod
    def setup_class(cls):
        if Utils.world_size < 2 or Utils.world_size % 2:
            pytest.skip("needs an even number of ranks for TP=2 groups")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

    @classmethod
    def teardown_class(cls):
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def _reduce_scatter_both(self, x):
        group = get_tensor_model_parallel_group()
        expected = torch.empty(x.shape[0] // 2, *x.shape[1:], dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(expected, x, group=group)
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=group)
        symm = buf.maybe_get_tensor(list(x.shape), dtype=x.dtype)
        assert symm["handle"] is not None
        symm["tensor"].copy_(x)
        actual = torch.empty_like(expected)
        pull_reduce_scatter(actual, symm["tensor"], symm["handle"])
        torch.cuda.synchronize()
        dist.barrier(group=group)
        return actual, expected

    @pytest.mark.parametrize("tokens", [2, 64, 128, 1024, 4096])
    def test_matches_nccl_on_activations(self, tokens):
        torch.manual_seed(1234 + dist.get_rank())
        x = (torch.randn(tokens, 2688, device="cuda") * 3).to(torch.bfloat16)
        actual, expected = self._reduce_scatter_both(x)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    def test_matches_nccl_on_all_bit_patterns(self):
        """Every BF16 pattern on one rank against a set of edge-case patterns on the other.

        Covers signed zeros (-0 + -0 = -0), subnormal sums, overflow to infinity, round to
        nearest even at ties, and NaN (compared as NaN on both sides).
        """
        tp_rank = dist.get_rank(get_tensor_model_parallel_group())
        everything = torch.arange(65536, dtype=torch.int32).to(torch.int16).cuda()
        special = _special_bf16_bits().cuda()
        if tp_rank == 0:
            half = everything.repeat(special.numel())
        else:
            half = special.repeat_interleave(everything.numel())
        x = torch.cat([half, half]).view(torch.bfloat16).view(-1, 64)
        actual, expected = self._reduce_scatter_both(x)
        both_nan = torch.isnan(actual) & torch.isnan(expected)
        differ = (actual.view(torch.int16) != expected.view(torch.int16)) & ~both_nan
        assert int(differ.sum()) == 0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact stacked-expert staging and writeback under side-stream contention."""

import pytest
import torch

from megatron.core.resharding.copy_services.nccl_m2n_copy_service import NCCLM2NCopyService
from megatron.core.resharding.utils import TensorReshardSpec
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("adjacent", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_stacked_expert_writeback_replays(adjacent, dtype):
    """Adjacent views and staged noncontiguous destinations preserve every input bit."""
    count, rows, cols = 4, 8, 16
    names = tuple(f"weight{i}" for i in range(count))
    spec = TensorReshardSpec(
        resolved_name="weight[experts]",
        src_ranks=(0,),
        dst_ranks=(1,),
        global_shape=(count, rows, cols),
        src_local_shape=(count, rows, cols),
        dst_local_shape=(count, rows, cols),
        dtype=dtype,
        src_shard_dim=0,
        dst_shard_dim=0,
        src_stack_names=names,
        dst_stack_names=names,
    )
    source = torch.arange(count * rows * cols, device="cuda", dtype=torch.float32).to(dtype)
    source = source.view(count, rows, cols)
    source[0, 0, :3] = torch.tensor([-0.0, float("inf"), float("nan")], device="cuda", dtype=dtype)

    def transfer(value):
        experts = list(value.unbind(0))
        if not adjacent:
            experts = [expert.clone() for expert in experts]
        src = NCCLM2NCopyService._local_stacked_tensor(spec, 0, dict(zip(names, experts)), {})
        if adjacent:
            backing = torch.empty_like(value)
            destinations = list(backing.unbind(0))
        else:
            destinations = [
                torch.empty(rows, cols * 2, device="cuda", dtype=dtype)[:, ::2] for _ in names
            ]
        dst = NCCLM2NCopyService._local_stacked_tensor(spec, 1, {}, dict(zip(names, destinations)))
        assert (dst.dst_scatter is None) == adjacent
        dst.dst.copy_(src.src)
        if dst.dst_scatter is not None:
            NCCLM2NCopyService._scatter_stacked(dst, None)
        return torch.stack(destinations)

    outputs, _ = assert_replays_bit_exact(
        transfer, (source,), backward=False, contention=True, what="stacked expert writeback"
    )
    actual = next(iter(outputs.values()))
    assert torch.equal(actual.contiguous().view(torch.uint8), source.contiguous().view(torch.uint8))

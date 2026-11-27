# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bit-exact replay of the fused fixed-threshold FP8 snap."""

import pytest
import torch

from megatron.core.fp8_param_rounding import _blockwise_scales, snap_master_shard
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows,cols,start", [(300, 520, 0), (4096, 4096, 6207488)])
def test_snap_master_shard_replays_bit_exact(rows, cols, start):
    """The compiled snap is elementwise with no reductions, so replays must match byte for byte."""
    seeded()
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).to(torch.bfloat16).float()
    scale = _blockwise_scales(weight, 128, power_of_two=True)
    shard = weight.reshape(-1)[start:].contiguous()
    assert_replays_bit_exact(
        lambda master: snap_master_shard(master, scale, rows, cols, start, 128, seed=5),
        (shard,),
        backward=False,
        contention=True,
        what="snap_master_shard",
    )

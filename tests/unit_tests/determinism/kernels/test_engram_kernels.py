# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay coverage for Engram row-sharded lookup accumulation."""

import pytest
import torch

from megatron.core.transformer.engram import RowShardedMultiHeadEmbedding
from tests.unit_tests.determinism.kernels.harness import bytes_equal, seeded
from tests.unit_tests.models.engram.test_integration import model_parallel
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or Utils.world_size not in (2, 4, 8),
    reason="requires two, four, or eight GPU ranks",
)


def test_row_sharded_lookup_forward_and_main_grad_replay_bit_exact():
    """Repeated addresses and cross-rank owner accumulation replay byte exactly."""
    with model_parallel(tensor_model_parallel_size=Utils.world_size):
        seeded()
        memory = RowShardedMultiHeadEmbedding(
            [129, 131], D=16, layer_id=3, params_dtype=torch.bfloat16, use_cpu_initialization=True
        ).cuda()
        weight = memory.embedding.weight
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        with torch.no_grad():
            weight.copy_(
                torch.arange(weight.numel(), device="cuda", dtype=torch.float32)
                .reshape_as(weight)
                .remainder(97)
                .div(128)
            )

        rank = torch.distributed.get_rank()
        rows = torch.arange(64, device="cuda", dtype=torch.long).reshape(4, 8, 2)
        rows[..., 0].add_(rank).remainder_(129)
        rows[..., 1].mul_(3).add_(rank).remainder_(131)
        output_grad = (
            torch.arange(4 * 8 * 2 * 16, device="cuda", dtype=torch.float32)
            .reshape(4, 8, 2, 16)
            .remainder(17)
            .div(32)
            .to(torch.bfloat16)
        )

        reference = None
        for replay in range(4):
            weight.main_grad.zero_()
            output = memory(rows)
            output.backward(output_grad)
            torch.cuda.synchronize()
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            current = (output.detach().clone(), weight.main_grad.clone())
            if reference is None:
                reference = current
                continue
            for name, expected, actual in zip(("output", "main_grad"), reference, current):
                assert bytes_equal(expected, actual), f"{name} differs on replay {replay}"

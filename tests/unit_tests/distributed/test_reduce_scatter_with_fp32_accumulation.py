# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

# Import our reduce_scatter implementation and shard_buffer (used for
# checks in the test).
from megatron.core.distributed.param_and_grad_buffer import (
    reduce_scatter_with_fp32_accumulation,
    shard_buffer,
)
from tests.unit_tests.test_utilities import Utils


def get_non_matching_values(tensor1_shard, tensor2_shard):
    mask = torch.isclose(tensor1_shard, tensor2_shard)
    indices = (~mask).nonzero()
    return indices, tensor1_shard[indices], tensor2_shard[indices]


class TestReduceScatterWithFP32Accumulation:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("async_op", [True, False])
    @pytest.mark.parametrize("baseline_reduce_scatter_in_fp32", [True, False])
    def test_reduce_scatter_with_fp32_accumulation(
        self, async_op: bool, baseline_reduce_scatter_in_fp32: bool
    ):
        num_tests = 20
        rank = Utils.rank
        world_size = Utils.world_size
        for _ in range(num_tests):
            # Initialize input tensors.
            tensor1 = torch.rand(100000, device='cuda', dtype=torch.bfloat16)
            tensor2 = tensor1.clone()

            # Make sure the two APIs are *identical*.
            kwargs = {"op": torch.distributed.ReduceOp.SUM, "group": None, "async_op": async_op}

            # Reduce-scatter with all-to-alls.
            args = [
                shard_buffer(tensor1, world_size)[rank],
                tensor1,
            ]  # Output tensor is view into original input.
            handle = reduce_scatter_with_fp32_accumulation(*args, **kwargs)
            if async_op:
                assert handle is not None
                handle.wait()
            tensor1_shard = shard_buffer(tensor1, world_size)[rank]

            if baseline_reduce_scatter_in_fp32:
                tensor2 = tensor2.float()

            # Reduce-scatter with reduce-scatter API.
            args = [
                shard_buffer(tensor2, world_size)[rank],
                tensor2,
            ]  # Output tensor is view into original input.
            handle = torch.distributed.reduce_scatter_tensor(*args, **kwargs)
            if async_op:
                assert handle is not None
                handle.wait()
            tensor2_shard = shard_buffer(tensor2, world_size)[rank]
            if baseline_reduce_scatter_in_fp32:  # Cast result back to bfloat16.
                tensor2_shard = tensor2_shard.bfloat16()

            # Compare results: results should match when doing FP32 reduction and not match when
            # doing direct BF16 reduction. We only look at relevant shard of tensor1 and tensor2.
            assert (
                torch.allclose(tensor1_shard, tensor2_shard) == baseline_reduce_scatter_in_fp32
            ), f"{get_non_matching_values(tensor1_shard, tensor2_shard)}"

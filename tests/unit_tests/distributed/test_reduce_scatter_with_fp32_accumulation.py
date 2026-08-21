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

    @pytest.mark.parametrize("axis_size", [2, 4, 8, 16])
    def test_power_of_two_prescale_equals_scaling_the_fp32_sum(self, axis_size: int):
        """Pre-scaling BF16 by 1/2^k is exact, which is why GTP pre-scales its 1/gtp_remat mean
        onto the wgrad instead of accumulating it here. Local arithmetic only, no collective."""
        scale = 1.0 / axis_size
        contributions = (torch.randn(axis_size, 100000, device='cuda') * 1e-3).bfloat16()

        prescaled = (contributions * scale).sum(dim=0, dtype=torch.float32).bfloat16()
        scaled_sum = (contributions.sum(dim=0, dtype=torch.float32) * scale).bfloat16()

        assert torch.equal(prescaled, scaled_sum), (
            f"1/{axis_size} pre-scale is not exact: "
            f"{(prescaled != scaled_sum).sum().item()} elements differ"
        )

    def test_caller_provided_workspace_tensors(self):
        """Caller-supplied A2A and FP32 scratch tensors must be honored.

        A mis-sized workspace must be rejected before the all-to-all: bailing out mid-flight
        desynchronizes the group.
        """
        rank, world_size = Utils.rank, Utils.world_size
        kwargs = {"op": torch.distributed.ReduceOp.SUM, "group": None, "async_op": False}
        tensor = torch.rand(100000, device='cuda', dtype=torch.bfloat16)

        internal = tensor.clone()
        reduce_scatter_with_fp32_accumulation(
            shard_buffer(internal, world_size)[rank], internal, **kwargs
        )
        provided = tensor.clone()
        provided_output = shard_buffer(provided, world_size)[rank]
        reduce_scatter_with_fp32_accumulation(
            provided_output,
            provided,
            all_to_all_output_tensor=torch.empty_like(tensor),
            fp32_accumulation_output_tensor=torch.empty_like(provided_output, dtype=torch.float32),
            **kwargs,
        )
        torch.testing.assert_close(
            shard_buffer(provided, world_size)[rank],
            shard_buffer(internal, world_size)[rank],
            rtol=0,
            atol=0,
        )

        with pytest.raises(AssertionError):
            reduce_scatter_with_fp32_accumulation(
                shard_buffer(tensor, world_size)[rank],
                tensor,
                all_to_all_output_tensor=torch.empty_like(tensor)[:-world_size],
                **kwargs,
            )

        with pytest.raises(AssertionError):
            reduce_scatter_with_fp32_accumulation(
                shard_buffer(tensor, world_size)[rank],
                tensor,
                fp32_accumulation_output_tensor=torch.empty(
                    tensor.numel() // world_size - 1, device=tensor.device, dtype=torch.float32
                ),
                **kwargs,
            )

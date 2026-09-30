# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mixed-dtype MFSDP reductions must preserve scaling and replay bit-exactly."""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager

from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    gradient_reduce_preprocessing,
)
from tests.unit_tests.test_utilities import Utils


class TestMfsdpV1GradientReduction:
    """Exercise the production scaling helper inside real coalesced NCCL calls."""

    @classmethod
    def setup_class(cls) -> None:
        """Initialize the distributed test environment."""
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls) -> None:
        """Release model-parallel groups for subsequent tests."""
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("scale_multiplier", [1.0, 0.5])
    @pytest.mark.parametrize("bf16_first", [False, True])
    @pytest.mark.parametrize("division_fusion", [False, True])
    def test_mixed_dtype_reduce_scatter(
        self, bf16_first: bool, division_fusion: bool, scale_multiplier: float
    ) -> None:
        """Both dtype orders must apply the requested gradient scale exactly once."""
        world_size = dist.get_world_size()
        if world_size < 2:
            pytest.skip("Mixed coalesced reductions require at least two ranks.")
        rank = dist.get_rank()
        dtypes = [torch.float32, torch.bfloat16]
        if bf16_first:
            dtypes.reverse()
        config = DistributedDataParallelConfig(gradient_reduce_div_fusion=division_fusion)
        expected = scale_multiplier * (world_size + 1) / 2
        reference_outputs = None

        for _ in range(4):
            inputs = [
                torch.full((64 * world_size,), rank + 1, dtype=dtype, device="cuda")
                for dtype in dtypes
            ]
            outputs = [value.chunk(world_size)[rank] for value in inputs]
            with _coalescing_manager(dist.group.WORLD):
                for value, output in zip(inputs, outputs):
                    op = gradient_reduce_preprocessing(
                        value, scale_multiplier / world_size, config, group_size=world_size
                    )
                    dist.reduce_scatter_tensor(output, value, op=op)

            for output in outputs:
                torch.testing.assert_close(
                    output, torch.full_like(output, expected), rtol=0, atol=0
                )
            if reference_outputs is None:
                reference_outputs = [output.clone() for output in outputs]
            else:
                for output, reference in zip(outputs, reference_outputs):
                    assert torch.equal(output, reference)

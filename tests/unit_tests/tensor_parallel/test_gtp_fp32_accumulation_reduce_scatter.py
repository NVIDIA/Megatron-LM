# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The fp32-accumulating GTP wgrad reduce-scatter must round a BF16 main_grad once.

Run with a torchrun launch of four or eight ranks; the fp32-accumulation path is
bypassed on groups of two ranks or fewer, and the per-rank test vectors below are
exact in BF16 only for those two world sizes.
"""

import os

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

requires_group = pytest.mark.skipif(
    WORLD_SIZE not in (4, 8) or not torch.cuda.is_available(),
    reason="needs a 4- or 8-rank torchrun launch",
)

# The shards sum to 1001, which BF16 cannot represent (spacing 4 near 1000), while every
# per-rank value is exact in BF16. Starting from main_grad = 2: one rounding of 2 + 1001 = 1003
# gives 1004; rounding the sum to BF16 first (1000) and then adding gives 1000.
INITIAL_MAIN_GRAD = 2.0
TARGET_SUM = 1001.0
EXPECTED_SINGLE_ROUNDING = 1004.0
EXPECTED_DOUBLE_ROUNDING = 1000.0


def _rank_value(rank: int, world: int) -> float:
    base = TARGET_SUM // world
    return base + (TARGET_SUM - base * world if rank == world - 1 else 0.0)


@requires_group
@pytest.mark.parametrize("async_op", [False, True])
def test_gtp_fp32_accumulation_reduce_scatter_rounds_bf16_main_grad_once(async_op):
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        GTP_CONFIG,
        get_global_GTP_cache,
        get_rs_stream,
        reset_gtp_state,
        update_gtp_config,
        wrap_module_params_gtp,
    )

    Utils.initialize_distributed()
    group = dist.group.WORLD
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    device = torch.device("cuda", torch.cuda.current_device())
    previous = (
        GTP_CONFIG.reduce_scatter_with_fp32_accumulation,
        GTP_CONFIG.calculate_per_token_loss,
        GTP_CONFIG.async_reduction,
    )
    # Per-token loss keeps the reduce-scatter a plain SUM; sync reduction runs main_grad.add_
    # inline so the test can read the result immediately.
    update_gtp_config(
        reduce_scatter_with_fp32_accumulation=True,
        calculate_per_token_loss=True,
        async_reduction=False,
    )
    try:
        # dim 0 divisible by 16 * world, so no GTP padding rows are involved.
        linear = torch.nn.Linear(64, 16 * world, bias=False, dtype=torch.bfloat16, device=device)
        wrap_module_params_gtp(linear, ["weight"], group)
        weight = linear.weight
        assert weight.is_gtp_weight_remat and weight.pad_length == 0
        weight.main_grad = torch.full(
            weight.shape, INITIAL_MAIN_GRAD, dtype=torch.bfloat16, device=device
        )
        weight.grad_added_to_main_grad = False
        value = _rank_value(rank, world)
        assert torch.tensor(value, dtype=torch.bfloat16).item() == value
        wgrad = torch.full(weight._unsharded_shape, value, dtype=torch.bfloat16, device=device)

        if async_op:
            # The async path reserves a persistent output ticket; it must be FP32 and hold
            # the exact shard sum before main_grad.add_ rounds it.
            outputs, handle, release_bufs = weight._reduce_scatter([wgrad], async_op=True)
            with torch.cuda.stream(get_rs_stream(weight.chain_id, group)):
                handle.wait()
            torch.cuda.synchronize(device)
            (out,) = outputs
            assert out.dtype == torch.float32
            assert out.shape == weight.shape
            assert torch.equal(out, torch.full_like(out, TARGET_SUM))
            assert get_global_GTP_cache().get(weight._rs_ticket).dtype == torch.float32
            weight._wgrad_input_bufs = release_bufs
            weight._release_wgrad_scratch()
            weight.main_grad.add_(out)
        else:
            dummy = weight.wgrad_reduce_scatter(wgrad)
            assert dummy.dtype == weight.dtype and dummy.shape == weight.shape
            assert weight.grad_added_to_main_grad

        torch.cuda.synchronize(device)
        assert weight.main_grad.dtype == torch.bfloat16
        assert torch.equal(
            weight.main_grad, torch.full_like(weight.main_grad, EXPECTED_SINGLE_ROUNDING)
        )
        # What a BF16 reduce-scatter output followed by the add would have produced.
        assert EXPECTED_SINGLE_ROUNDING != EXPECTED_DOUBLE_ROUNDING
        assert (
            torch.tensor(INITIAL_MAIN_GRAD, dtype=torch.bfloat16)
            + torch.tensor(TARGET_SUM, dtype=torch.bfloat16)
        ).item() == EXPECTED_DOUBLE_ROUNDING
    finally:
        update_gtp_config(
            reduce_scatter_with_fp32_accumulation=previous[0],
            calculate_per_token_loss=previous[1],
            async_reduction=previous[2],
        )
        reset_gtp_state()
        dist.barrier(group)

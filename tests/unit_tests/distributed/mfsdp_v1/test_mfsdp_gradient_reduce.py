# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise group-local scaling through real MFSDP buffers and NCCL collectives."""

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as buffer_module
from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    BucketingPolicy,
    GradReducePipeline,
    ParamAndGradBuffer,
    gradient_reduce_preprocessing,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import FSDPDistributedIndex


class GradientUnit(torch.nn.Module):
    """Different row sizes keep two buckets even when their dtypes are identical."""

    def __init__(self, dtypes, frozen=False) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.ones(16, 8, device="cuda", dtype=dtypes[0]))
        self.second = torch.nn.Parameter(torch.ones(16, 12, device="cuda", dtype=dtypes[1]))
        if frozen:
            self.frozen = torch.nn.Parameter(
                torch.ones(16, 8, device="cuda", dtype=torch.bfloat16), requires_grad=False
            )


@pytest.fixture(scope="module")
def reduction_meshes():
    """Build both multi-rank and singleton DP groups under torchrun."""
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        pytest.skip("Requires torchrun with at least two CUDA devices")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip("Requires at least two ranks to detect coalesced reduction errors")
    return {
        False: init_device_mesh("cuda", (world_size, 1), mesh_dim_names=("dp", "tp")),
        True: init_device_mesh("cuda", (world_size, 1, 1), mesh_dim_names=("replica", "dp", "tp"))[
            "dp", "tp"
        ],
    }


def make_pipeline(
    mesh, dtypes, strategy, average=False, fusion=True, override=None, frozen=False, hybrid=False
):
    """Use the production constructor, including grouping and communication-buffer allocation."""
    config = DistributedDataParallelConfig(
        data_parallel_sharding_strategy=strategy,
        gradient_reduce_div_fusion=fusion,
        average_in_collective=average,
    )
    # A separate homogeneous unit must retain fusion when the first unit falls back.
    model = torch.nn.ModuleList(
        [GradientUnit(dtypes, frozen=frozen), GradientUnit((torch.float32, torch.float32))]
    )
    buffer = ParamAndGradBuffer(
        config,
        model,
        BucketingPolicy(fsdp_unit_modules=[GradientUnit]),
        FSDPDistributedIndex(
            mesh,
            dp_shard_dim="dp",
            tp_dim="tp",
            dp_outer_dim="outer" if hybrid else None,
            hybrid_fsdp_group=mesh["hybrid"].get_group() if hybrid else None,
        ),
        mixed_precision_policy=MixedPrecisionPolicy(grad_comm_dtype=override),
        gradient_scaling_factor=0.25,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    return GradReducePipeline(buffer)


def unit_buckets(pipeline, unit_id, reverse=False):
    """Return the actual ready group, excluding frozen parameters, in either bucket order."""
    groups = pipeline.buffer.parameter_groups
    ids = [i for i, group in enumerate(groups) if group.fsdp_unit_id == unit_id]
    for bucket_id in ids:
        if groups[bucket_id].main_grad_buffer is not None:
            pipeline.bucket_grad_ready_params[bucket_id] = set(groups[bucket_id].params)
    ready = pipeline.get_ready_bucket_group_for_reduction(ids[0])
    assert ready is not None and len(ready) == 2
    assert set(ready).issubset(pipeline.buffer.bucket_to_bucket_group[ids[0]])
    return sorted(ready, reverse=reverse)


def reduce_and_compare(pipeline, bucket_ids, factors, expected_op, force_sum, repeats=2):
    """Compare real coalesced AR/RS against independent explicit-scale + SUM reductions."""
    config = pipeline.buffer.ddp_config
    pg = pipeline.get_fsdp_buffer(bucket_ids[0]).data_parallel_group
    for bucket_id, factor in zip(bucket_ids, factors):
        pipeline.get_fsdp_buffer(bucket_id).gradient_scaling_factor = factor
    assert pipeline._group_requires_sum(bucket_ids) == force_sum

    for iteration in range(repeats):
        expected = {}
        for index, bucket_id in enumerate(bucket_ids):
            gbuf = pipeline.get_fsdp_buffer(bucket_id)
            dtype = pipeline.buffer.mp_policy.grad_comm_dtype or gbuf.dtype
            # Dyadic inputs/factors make a missing or double scale unambiguous in BF16.
            values = (
                torch.arange(gbuf.bucket_index.size, device="cuda") % 16
                + 4 * (pg.rank() + 1)
                + index
                + iteration
            ).to(dtype) / 8
            old_grad = gbuf.get_shard_from_local_buffer().clone()
            bucket = gbuf.fetch_bucket(dtype=dtype if gbuf.is_data_distributed else None)
            bucket.data.copy_(values)
            reference = values.clone()
            factor = factors[index]
            if factor is not None:
                reference.mul_(1.0 / pg.size() if config.average_in_collective else factor)
            # Separate SUM calls ensure the reference cannot reuse the wrong coalesced op.
            dist.all_reduce(reference, op=dist.ReduceOp.SUM, group=pg)
            strategy = pipeline.buffer.parameter_groups[bucket_id].sharding_strategy
            if strategy != "no_shard":
                shard = gbuf.shard_bucket_index
                reference = reference[
                    shard.bucket_data_index : shard.bucket_data_index + shard.size
                ]
            reference = reference.to(gbuf.dtype)
            if gbuf.is_data_distributed and strategy not in ("no_shard", "optim"):
                reference = old_grad + reference
            expected[bucket_id] = reference

        with (
            patch.object(
                buffer_module, "gradient_reduce_preprocessing", wraps=gradient_reduce_preprocessing
            ) as preprocess,
            patch.object(dist, "_make_nccl_premul_sum", wraps=dist._make_nccl_premul_sum) as premul,
        ):
            pipeline._bucket_group_gradient_reduce(bucket_ids, async_op=iteration == 0)
            pipeline.wait_for_previous_grad_reduce(0)
            torch.cuda.synchronize()
            assert preprocess.call_count == len(bucket_ids)
            assert all(call.kwargs["force_sum"] == force_sum for call in preprocess.call_args_list)
            assert premul.call_count == (len(bucket_ids) if expected_op == "premul" else 0)
        for bucket_id, reference in expected.items():
            gbuf = pipeline.get_fsdp_buffer(bucket_id)
            actual = (
                gbuf.data
                if pipeline.buffer.parameter_groups[bucket_id].sharding_strategy == "no_shard"
                else gbuf.get_shard_from_local_buffer()
            )
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


BF16_FP32 = (torch.bfloat16, torch.float32)
FP32_FP32 = (torch.float32, torch.float32)


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "dtypes,factors,average,fusion,override,expected_op,force_sum",
    [
        pytest.param(BF16_FP32, (0.25, 0.25), False, True, None, "sum", True, id="mixed"),
        pytest.param(BF16_FP32, (0.25, 0.125), False, True, None, "sum", True, id="factors"),
        pytest.param(FP32_FP32, (0.25, 0.125), False, True, None, "sum", True, id="fp32-factors"),
        pytest.param(FP32_FP32, (0.25, 0.25), False, True, None, "premul", False, id="fp32-fused"),
        pytest.param(
            (torch.bfloat16,) * 2, (0.25, 0.125), False, True, None, "sum", False, id="bf16"
        ),
        pytest.param(BF16_FP32, (0.25, 0.125), True, True, None, "avg", False, id="average"),
        pytest.param(BF16_FP32, (None, None), False, True, None, "sum", False, id="per-token"),
        pytest.param(
            BF16_FP32, (None, None), True, True, None, "sum", False, id="per-token-average"
        ),
        pytest.param(FP32_FP32, (None, 0.25), False, True, None, "sum", True, id="sum-premul"),
        pytest.param(BF16_FP32, (None, 0.25), True, True, None, "sum", True, id="sum-avg"),
        pytest.param(BF16_FP32, (0.25, 0.25), False, False, None, "sum", False, id="fusion-off"),
        pytest.param(
            BF16_FP32, (0.25, 0.25), False, True, torch.float32, "premul", False, id="override-fp32"
        ),
        pytest.param(
            BF16_FP32, (0.25, 0.125), False, True, torch.bfloat16, "sum", False, id="override-bf16"
        ),
        pytest.param(
            (torch.float16, torch.float32),
            (0.25, 0.25),
            False,
            True,
            None,
            "sum",
            True,
            id="premul-dtypes",
        ),
    ],
)
def test_coalesced_gradient_reduction(
    reduction_meshes,
    strategy,
    reverse,
    dtypes,
    factors,
    average,
    fusion,
    override,
    expected_op,
    force_sum,
):
    """Changing bucket order must preserve scaling and keep other units eligible for fusion."""
    pipeline = make_pipeline(reduction_meshes[False], dtypes, strategy, average, fusion, override)
    bucket_ids = unit_buckets(pipeline, 0, reverse)
    # Reverse factors together with buckets to keep each bucket's own factor unchanged.
    reduce_and_compare(
        pipeline, bucket_ids, factors[::-1] if reverse else factors, expected_op, force_sum
    )
    other_op = "avg" if average else "premul" if fusion and override != torch.bfloat16 else "sum"
    reduce_and_compare(pipeline, unit_buckets(pipeline, 1), (0.25, 0.25), other_op, False)
    assert pipeline.buffer.ddp_config.gradient_reduce_div_fusion is fusion


@pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
@pytest.mark.parametrize("reverse", [False, True])
def test_single_rank_group_scales_once(reduction_meshes, strategy, reverse):
    """A singleton DP group still applies a non-unit scale without PREMUL_SUM."""
    pipeline = make_pipeline(reduction_meshes[True], BF16_FP32, strategy)
    reduce_and_compare(pipeline, unit_buckets(pipeline, 0, reverse), (0.25, 0.125), "sum", False)


@pytest.mark.parametrize("strategy", ["no_shard", "optim_grads"])
def test_runtime_dtype_override_and_frozen_parameters(reduction_meshes, strategy):
    """Re-evaluate group compatibility after a dtype change and ignore absent grad buffers."""
    pipeline = make_pipeline(reduction_meshes[False], BF16_FP32, strategy, frozen=True)
    for override, expected_op, force_sum in [
        (None, "sum", True),
        (torch.float32, "premul", False),
        (None, "sum", True),
    ]:
        pipeline.buffer.mp_policy = MixedPrecisionPolicy(grad_comm_dtype=override)
        reduce_and_compare(
            pipeline, unit_buckets(pipeline, 0), (0.25, 0.25), expected_op, force_sum
        )
    # Frozen BF16 parameters must not force an otherwise homogeneous FP32 unit to SUM.
    pipeline = make_pipeline(reduction_meshes[False], FP32_FP32, strategy, frozen=True)
    reduce_and_compare(pipeline, unit_buckets(pipeline, 0), (0.25, 0.25), "premul", False)


@pytest.mark.parametrize("strategy", ["no_shard", "optim_grads"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("average", [False, True])
def test_hybrid_outer_all_reduce(reduction_meshes, strategy, reverse, average):
    """Mixed-dtype DP-Outer all-reduce must split calls without scaling a second time."""
    world_size = dist.get_world_size()
    if world_size % 2:
        pytest.skip("Requires an even world size for the hybrid mesh")
    mesh = init_device_mesh("cuda", (2, world_size // 2, 1), mesh_dim_names=("outer", "dp", "tp"))
    mesh["outer", "dp"]._flatten("hybrid")
    pipeline = make_pipeline(mesh, BF16_FP32, strategy, average=average, hybrid=True)
    ids = unit_buckets(pipeline, 0, reverse)
    expected = {}
    for bucket_id in ids:
        gbuf = pipeline.get_fsdp_buffer(bucket_id)
        data = gbuf.fetch_bucket().data
        data.copy_(
            (torch.arange(data.numel(), device="cuda") % 16 + dist.get_rank() + 1).to(data.dtype)
        )
        reference = data.clone()
        # Normalize across both dimensions exactly once. AVG normalizes each dimension.
        gbuf.gradient_scaling_factor = 0.25
        reference.mul_(1.0 / world_size if average else 0.25)
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        if strategy != "no_shard":
            shard = gbuf.shard_bucket_index
            reference = reference[shard.bucket_data_index : shard.bucket_data_index + shard.size]
        expected[bucket_id] = reference
    pipeline._bucket_group_gradient_reduce(ids, outer_fsdp_group_grad_reduce=True)
    torch.cuda.synchronize()
    for bucket_id in ids:
        torch.testing.assert_close(
            pipeline.get_fsdp_buffer(bucket_id).data, expected[bucket_id], rtol=0, atol=0
        )

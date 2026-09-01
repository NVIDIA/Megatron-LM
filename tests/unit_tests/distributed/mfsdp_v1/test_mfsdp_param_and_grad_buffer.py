# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    AllGatherPipeline,
    BucketingPolicy,
    FixedPoolAllocator,
    MaxPoolAllocator,
    ParameterGroup,
    _get_parameter_groups,
)


class _ExpertTestModule(torch.nn.Module):
    """
    Mock module whose params are routed under `.experts.` to trigger
    is_expert_param=True. The outer `layer` attribute puts a dot before
    `experts` in the parameter path (e.g. `layer.experts.linear_fc1`).
    """

    def __init__(self, shapes):
        super().__init__()
        self.layer = torch.nn.Module()
        self.layer.experts = torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.empty(shape)) for name, shape in shapes.items()}
        )


def _get_bucket_signatures(module):
    bucket_groups, _, _ = _get_parameter_groups(
        module, BucketingPolicy(suggested_bucket_size=None), meta_device_init_fp8_params={}
    )
    param_to_name = {param: name for name, param in module.named_parameters()}
    return [
        {
            "chunk_size_factor": group.chunk_size_factor,
            "params": [(param_to_name[param], tuple(param.shape)) for param in group.params],
        }
        for group in bucket_groups
    ]


def _make_uniform_parameter_groups(count=4):
    return [
        ParameterGroup(
            [torch.nn.Parameter(torch.empty(8, dtype=torch.bfloat16))],
            dtype=torch.bfloat16,
            fsdp_unit_id=unit_id,
        )
        for unit_id in range(count)
    ]


class _CpuMemoryBuffer:
    """CPU stand-in for the CUDA global memory buffer used by pool allocators."""

    def __init__(self):
        self.buffers = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        required_len = math.prod(tensor_shape)
        key = (name, dtype)
        if key not in self.buffers or self.buffers[key].numel() < required_len:
            self.buffers[key] = torch.empty(required_len, dtype=dtype)
        return self.buffers[key][:required_len].view(*tensor_shape)


def _allocate(allocator, bucket_id):
    return allocator.allocate(
        bucket_id=bucket_id,
        size=8,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        strict_assignments=False,
    )


def _make_all_gather_pipeline(parameter_groups, allocator_by_bucket):
    for bucket_id, parameter_group in enumerate(parameter_groups):
        parameter_group.model_weight_buffer = SimpleNamespace(
            temporary_bucket_allocator=allocator_by_bucket[bucket_id]
        )
    buffer = SimpleNamespace(
        num_buckets=len(parameter_groups),
        parameter_groups=parameter_groups,
        bucket_to_bucket_group={
            bucket_id: [bucket_id] for bucket_id in range(len(parameter_groups))
        },
        dist_index=SimpleNamespace(use_hybrid_fsdp=False),
        ddp_config=SimpleNamespace(outer_dp_sharding_strategy="no_shard"),
    )
    return AllGatherPipeline(buffer)


@pytest.mark.parametrize("allocator_cls", [FixedPoolAllocator, MaxPoolAllocator])
def test_triple_buffer_pool_capacity_and_reuse(allocator_cls, monkeypatch):
    """Three live buckets fit, a fourth waits, and a freed slot is reused."""
    cpu_memory_buffer = _CpuMemoryBuffer()
    monkeypatch.setattr(
        param_and_grad_buffer, "get_global_memory_buffer", lambda: cpu_memory_buffer
    )
    allocator = allocator_cls("triple", _make_uniform_parameter_groups(), size=3)

    live_buckets = [_allocate(allocator, bucket_id) for bucket_id in range(3)]
    assert len({bucket.data.data_ptr() for bucket in live_buckets}) == 3
    assert not allocator.can_allocate([3])
    assert allocator.can_allocate([3], releasable_bucket_ids={0})

    released_data_ptr = live_buckets[0].data.data_ptr()
    allocator.free(0)
    replacement_bucket = _allocate(allocator, 3)
    assert replacement_bucket.data.data_ptr() == released_data_ptr


@pytest.mark.parametrize("allocator_cls", [FixedPoolAllocator, MaxPoolAllocator])
def test_all_gather_capacity_check_groups_allocators_and_lazy_releases(allocator_cls, monkeypatch):
    """Capacity prediction accounts for every pool and its pending lazy releases."""
    cpu_memory_buffer = _CpuMemoryBuffer()
    monkeypatch.setattr(
        param_and_grad_buffer, "get_global_memory_buffer", lambda: cpu_memory_buffer
    )
    parameter_groups = _make_uniform_parameter_groups(count=8)
    first_allocator = allocator_cls("first", parameter_groups, size=3)
    second_allocator = allocator_cls("second", parameter_groups, size=3)
    allocator_by_bucket = {
        bucket_id: first_allocator if bucket_id < 4 else second_allocator for bucket_id in range(8)
    }
    pipeline = _make_all_gather_pipeline(parameter_groups, allocator_by_bucket)

    for bucket_id in (0, 1, 2):
        _allocate(first_allocator, bucket_id)
    for bucket_id in (4, 5, 6):
        _allocate(second_allocator, bucket_id)

    assert not pipeline._persistent_allocators_can_fit([3, 7], bwd=False)

    pipeline.bucket_can_be_released[pipeline.get_bucket_key(0, False)] = True
    assert not pipeline._persistent_allocators_can_fit([3, 7], bwd=False)

    pipeline.bucket_can_be_released[pipeline.get_bucket_key(4, False)] = True
    assert pipeline._persistent_allocators_can_fit([3, 7], bwd=False)


def test_grouped_expert_weights_split_when_chunk_size_factors_differ():
    """Grouped expert weights with mismatched chunk size factors get routed to separate buckets."""
    num_local_experts = 4
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (num_local_experts, 2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (num_local_experts, hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc1"])[1:].numel(),
            "params": [("layer.experts.linear_fc1", shapes["linear_fc1"])],
        },
        {
            "chunk_size_factor": torch.Size(shapes["linear_fc2"])[1:].numel(),
            "params": [("layer.experts.linear_fc2", shapes["linear_fc2"])],
        },
    ]


def test_per_expert_2d_weights_merge_via_lcm():
    """Per-expert 2D weights merge into a single bucket via LCM chunk size factor."""
    hidden_size = 12
    moe_ffn_hidden_size = 8
    shapes = {
        "linear_fc1": (2 * moe_ffn_hidden_size, hidden_size),
        "linear_fc2": (hidden_size, moe_ffn_hidden_size),
    }
    module = _ExpertTestModule(shapes)

    assert _get_bucket_signatures(module) == [
        {
            "chunk_size_factor": math.lcm(
                torch.Size(shapes["linear_fc1"])[1:].numel(),
                torch.Size(shapes["linear_fc2"])[1:].numel(),
            ),
            "params": [
                ("layer.experts.linear_fc1", shapes["linear_fc1"]),
                ("layer.experts.linear_fc2", shapes["linear_fc2"]),
            ],
        }
    ]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import param_and_grad_buffer as pgb_module
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    Bucket,
    BucketingPolicy,
    FixedPoolAllocator,
    GradReducePipeline,
    ParamAndGradBuffer,
    PlannedBucketAllocator,
    TemporaryBucketAllocator,
    _get_allocator_namespace,
    _get_parameter_groups,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.utils import safe_get_rank


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


class _CpuGlobalMemoryBuffer:
    """CPU test double with the same grow-on-demand contract as GlobalMemoryBuffer."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        required_len = math.prod(tensor_shape)
        key = (name, dtype)
        if key not in self.buffer or self.buffer[key].numel() < required_len:
            allocation_context = mem_alloc_context or nullcontext
            with allocation_context():
                self.buffer[key] = torch.empty(required_len, dtype=dtype)
        return self.buffer[key][:required_len].view(*tensor_shape)


class _TrackingBucketAllocator(TemporaryBucketAllocator):
    """CPU allocator that records ownership without resizing untyped storage."""

    def allocate(self, bucket_id, size, dtype, device, mem_alloc_context=None):
        self.buckets[bucket_id] = Bucket(data=torch.empty(size, dtype=dtype, device=device))
        return self.buckets[bucket_id]

    def free(self, bucket_id):
        self.buckets.pop(bucket_id, None)


def _allocator_param_groups(*unit_ids):
    return [
        SimpleNamespace(
            fsdp_unit_id=unit_id, dtype=torch.float32, params=[torch.nn.Parameter(torch.empty(2))]
        )
        for unit_id in unit_ids
    ]


@pytest.fixture
def cpu_global_memory_buffer(monkeypatch):
    global_memory_buffer = _CpuGlobalMemoryBuffer()
    monkeypatch.setattr(pgb_module, "get_global_memory_buffer", lambda: global_memory_buffer)
    monkeypatch.setattr(pgb_module.torch.distributed, "get_rank", lambda: 0)
    return global_memory_buffer


def test_allocator_namespaces_are_disjoint_across_model_chunks(cpu_global_memory_buffer):
    first_module = torch.nn.Linear(2, 2)
    first_module._megatron_fsdp_buffer_namespace = "model_chunk_0"
    second_module = torch.nn.Linear(2, 2)
    second_module._megatron_fsdp_buffer_namespace = "model_chunk_1"

    first_namespace = _get_allocator_namespace(first_module)
    second_namespace = _get_allocator_namespace(second_module)
    assert first_namespace != second_namespace

    param_groups = _allocator_param_groups(0)
    first_pool = FixedPoolAllocator(f"{first_namespace}_fsdp_params", param_groups, size=1)
    second_pool = FixedPoolAllocator(f"{second_namespace}_fsdp_params", param_groups, size=1)
    first_bucket = first_pool.allocate(0, 8, torch.float32, torch.device("cpu"))
    second_bucket = second_pool.allocate(0, 8, torch.float32, torch.device("cpu"))

    assert first_pool._get_gbuf_name(0, 0) != second_pool._get_gbuf_name(0, 0)
    assert first_bucket.data.data_ptr() != second_bucket.data.data_ptr()

    first_planned = PlannedBucketAllocator(
        f"{first_namespace}_fsdp_grads", _TrackingBucketAllocator(), param_groups
    )
    second_planned = PlannedBucketAllocator(
        f"{second_namespace}_fsdp_grads", _TrackingBucketAllocator(), param_groups
    )
    assert first_planned._get_planned_buf_name(0, 0) != second_planned._get_planned_buf_name(0, 0)


def test_allocator_namespace_fallback_is_unique_per_standalone_buffer():
    first_namespace = _get_allocator_namespace(torch.nn.Linear(2, 2))
    second_namespace = _get_allocator_namespace(torch.nn.Linear(2, 2))

    assert first_namespace != second_namespace
    assert first_namespace.startswith("param_and_grad_buffer_")
    assert second_namespace.startswith("param_and_grad_buffer_")


def test_allocator_namespace_appends_unique_id_to_same_chunk_label():
    """Human-readable labels never become process-global storage identities."""
    first_module = torch.nn.Linear(2, 2)
    first_module._megatron_fsdp_buffer_namespace = "model_chunk_0"
    second_module = torch.nn.Linear(2, 2)
    second_module._megatron_fsdp_buffer_namespace = "model_chunk_0"

    first_namespace = _get_allocator_namespace(first_module)
    second_namespace = _get_allocator_namespace(second_module)

    assert first_namespace != second_namespace
    assert first_namespace.startswith("model_chunk_0_param_and_grad_buffer_")
    assert second_namespace.startswith("model_chunk_0_param_and_grad_buffer_")


def test_planned_allocator_rejects_nccl_ub_for_programmatic_config():
    """Standalone API callers cannot bypass the unsupported NCCL registration boundary."""
    ddp_config = SimpleNamespace(
        data_parallel_sharding_strategy="optim_grads_params",
        nccl_ub=True,
        fsdp_double_buffer=False,
        megatron_fsdp_use_planned_allocator=True,
    )

    with pytest.raises(ValueError, match="does not yet support NCCL user buffers"):
        ParamAndGradBuffer(ddp_config, torch.nn.Linear(2, 2), SimpleNamespace(), SimpleNamespace())


def test_planned_allocator_rejects_double_buffer_for_programmatic_config():
    """Planned slots are the only temporary-buffer reuse policy supported by TE graphs."""
    ddp_config = SimpleNamespace(
        data_parallel_sharding_strategy="optim_grads_params",
        nccl_ub=False,
        fsdp_double_buffer=True,
        megatron_fsdp_use_planned_allocator=True,
    )

    with pytest.raises(ValueError, match="does not support.*fsdp_double_buffer"):
        ParamAndGradBuffer(ddp_config, torch.nn.Linear(2, 2), SimpleNamespace(), SimpleNamespace())


def test_planned_allocator_rejects_optim_grads_for_programmatic_config():
    """Standalone callers cannot bypass the missing per-layer fused-main-grad claim hook."""
    ddp_config = SimpleNamespace(
        data_parallel_sharding_strategy="optim_grads",
        nccl_ub=False,
        fsdp_double_buffer=False,
        megatron_fsdp_use_planned_allocator=True,
    )

    with pytest.raises(ValueError, match="does not yet support optim_grads"):
        ParamAndGradBuffer(ddp_config, torch.nn.Linear(2, 2), SimpleNamespace(), SimpleNamespace())


@pytest.mark.parametrize("uses_planned_allocator", [False, True])
def test_parameter_group_buffer_initialization_uses_instance_planned_policy(
    uses_planned_allocator,
):
    """Both eager and graph PGB initialization read the policy saved by the constructor."""
    pgb = ParamAndGradBuffer.__new__(ParamAndGradBuffer)
    pgb.ddp_config = SimpleNamespace(
        data_parallel_sharding_strategy="optim_grads_params",
        outer_dp_sharding_strategy="no_shard",
        nccl_ub=False,
        fsdp_double_buffer=False,
        fsdp_db_use_persist_buf_on_alloc_fail=False,
        fp8_param_gather=False,
    )
    pgb._uses_planned_allocator = uses_planned_allocator
    pgb.dist_index = SimpleNamespace(use_hybrid_fsdp=False)
    pgb.bucketing_policy = SimpleNamespace(fsdp_unit_modules=[])
    pgb.parameter_groups = []
    pgb._allocator_namespace = "test_param_and_grad_buffer"
    pgb.mem_alloc_context = nullcontext
    pgb.device = torch.device("cpu")
    pgb.reset_parameters_for_meta_device_init_module = False

    pgb._init_each_parameter_group_buffers({})

    assert isinstance(pgb.weight_alloc, PlannedBucketAllocator) is uses_planned_allocator
    assert isinstance(pgb.transpose_weight_alloc, PlannedBucketAllocator) is uses_planned_allocator
    assert isinstance(pgb.main_grad_alloc, PlannedBucketAllocator) is uses_planned_allocator


def test_planned_allocator_colors_nonoverlapping_lifetimes_and_detects_conflicts(
    cpu_global_memory_buffer,
):
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_params", _TrackingBucketAllocator(), _allocator_param_groups(0, 1)
    )
    for bucket_id, size in ((0, 8), (1, 6)):
        allocator.allocate(bucket_id, size, torch.float32, torch.device("cpu"))
        allocator.free(bucket_id)

    allocator.freeze_plan([0, 1])

    assert allocator._plan[0] == allocator._plan[1]
    first = allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    first_data_ptr = first.data.data_ptr()
    allocator.free(0)
    second = allocator.allocate(1, 6, torch.float32, torch.device("cpu"))
    assert second.data.data_ptr() == first_data_ptr
    allocator.free(1)

    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="planned slot conflict"):
        allocator.allocate(1, 6, torch.float32, torch.device("cpu"))


def test_planned_allocator_separates_overlapping_lifetimes(cpu_global_memory_buffer):
    overlapping = PlannedBucketAllocator(
        "overlapping", _TrackingBucketAllocator(), _allocator_param_groups(0, 1)
    )
    overlapping.allocate(0, 8, torch.float32, torch.device("cpu"))
    overlapping.allocate(1, 8, torch.float32, torch.device("cpu"))
    overlapping.free(1)
    overlapping.free(0)
    overlapping.freeze_plan([0, 1])

    assert overlapping._plan[0] != overlapping._plan[1]


def test_planned_allocator_keeps_max_warmup_size_and_rejects_dtype_changes(
    cpu_global_memory_buffer,
):
    allocator = PlannedBucketAllocator(
        "warmup_metadata", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    for size in (8, 6):
        allocator.allocate(0, size, torch.float32, torch.device("cpu"))
        allocator.free(0)
    allocator.freeze_plan([0])

    assert allocator._slot_materialization[(allocator._plan[0], 0)][1] == 8

    dtype_allocator = PlannedBucketAllocator(
        "warmup_dtype", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    dtype_allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    dtype_allocator.free(0)
    with pytest.raises(RuntimeError, match="dtype"):
        dtype_allocator.allocate(0, 8, torch.float64, torch.device("cpu"))


def test_planned_allocator_rejects_over_capacity_and_dtype_changes(cpu_global_memory_buffer):
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_params", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)
    allocator.freeze_plan([0])

    with pytest.raises(RuntimeError, match="materialized at 8 elements"):
        allocator.allocate(0, 9, torch.float32, torch.device("cpu"))
    assert allocator._slot_using == {}

    with pytest.raises(RuntimeError, match="materialized with dtype"):
        allocator.allocate(0, 8, torch.float64, torch.device("cpu"))
    assert allocator._slot_using == {}

    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))


def test_planned_allocator_freeze_materializes_inside_configured_context(cpu_global_memory_buffer):
    context_entries = []

    @contextmanager
    def allocation_context():
        context_entries.append(True)
        yield

    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_params",
        _TrackingBucketAllocator(),
        _allocator_param_groups(0),
        mem_alloc_context=allocation_context,
    )
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)

    allocator.freeze_plan([0])

    assert len(context_entries) == 1
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    assert len(context_entries) == 1


def test_planned_allocator_rejects_pointer_drift(cpu_global_memory_buffer):
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_grads", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)
    allocator.freeze_plan([0])

    slot = (allocator._plan[0], 0)
    key = (allocator._get_planned_buf_name(*slot), torch.float32)
    original_tensor = cpu_global_memory_buffer.buffer[key]
    cpu_global_memory_buffer.buffer[key] = torch.empty(original_tensor.numel() + 1)

    with pytest.raises(RuntimeError, match="moved from address"):
        allocator.allocate(0, 8, torch.float32, torch.device("cpu"))

    # Keep the original allocation alive until after the pointer comparison so
    # the CPU caching allocator cannot reuse its address for the replacement.
    assert original_tensor.data_ptr() != cpu_global_memory_buffer.buffer[key].data_ptr()


def test_planned_allocator_lazy_materialization_uses_configured_context(cpu_global_memory_buffer):
    context_entries = []

    @contextmanager
    def allocation_context():
        context_entries.append(True)
        yield

    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_grads",
        _TrackingBucketAllocator(),
        _allocator_param_groups(0),
        mem_alloc_context=allocation_context,
    )
    allocator.freeze_plan([0])
    assert context_entries == []

    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))

    assert len(context_entries) == 1
    assert allocator._slot_materialization[(0, 0)][1] == 8


def test_planned_allocator_stops_and_clears_observation_metadata(cpu_global_memory_buffer):
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_params", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)
    assert allocator._lifetime_events
    assert allocator._observed_alloc_meta

    allocator.freeze_plan([])

    assert not allocator._recording_lifetimes
    assert allocator._lifetime_events == []
    assert allocator._observed_alloc_meta == {}
    with pytest.raises(RuntimeError, match="new graph buckets appeared"):
        allocator.freeze_plan([0])


    second_allocator = PlannedBucketAllocator(
        "model_chunk_1_fsdp_params", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    second_allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    second_allocator.free(0)
    second_allocator.freeze_plan([0])

    assert not second_allocator._recording_lifetimes
    assert second_allocator._lifetime_events == []
    assert second_allocator._observed_alloc_meta == {}


def test_planned_graph_bucket_claim_is_idempotent_and_released_for_reuse(cpu_global_memory_buffer):
    """Graph claims enforce colormate occupancy and reuse the normal free lifecycle."""
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_grads", _TrackingBucketAllocator(), _allocator_param_groups(0, 1)
    )
    for bucket_id in (0, 1):
        allocator.allocate(bucket_id, 8, torch.float32, torch.device("cpu"))
        allocator.free(bucket_id)
    allocator.freeze_plan([0, 1])
    assert allocator._plan[0] == allocator._plan[1]

    allocator.claim_graph_bucket(0, 8, torch.float32)
    allocator.claim_graph_bucket(0, 8, torch.float32)
    slot = (allocator._plan[0], 0)
    assert allocator._slot_using[slot] == 0

    with pytest.raises(RuntimeError, match="planned slot conflict"):
        allocator.claim_graph_bucket(1, 8, torch.float32)

    allocator.free(0)
    allocator.claim_graph_bucket(1, 8, torch.float32)
    assert allocator._slot_using[slot] == 1
    allocator.free(1)
    assert slot not in allocator._slot_using


def test_planned_prewrite_claim_extends_recorded_warmup_lifetime(cpu_global_memory_buffer):
    """Coloring includes the earlier graph pre-write boundary, not only eager allocation."""
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_grads", _TrackingBucketAllocator(), _allocator_param_groups(0, 1)
    )
    allocator.record_graph_bucket_claim(0, 8, torch.float32)
    allocator.allocate(1, 8, torch.float32, torch.device("cpu"))
    allocator.free(1)
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)

    allocator.freeze_plan([0, 1])

    assert allocator._plan[0] != allocator._plan[1]
    allocator.claim_graph_bucket(0, 8, torch.float32)
    allocator.claim_graph_bucket(1, 8, torch.float32)
    allocator.free(0)
    allocator.free(1)


def test_planned_graph_bucket_claim_rejects_pointer_drift_before_write(cpu_global_memory_buffer):
    """Pre-replay claim detects main_grad relocation before the graph can write."""
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_grads", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)
    allocator.freeze_plan([0])

    slot = (allocator._plan[0], 0)
    key = (allocator._get_planned_buf_name(*slot), torch.float32)
    original_tensor = cpu_global_memory_buffer.buffer[key]
    cpu_global_memory_buffer.buffer[key] = torch.empty(original_tensor.numel() + 1)

    with pytest.raises(RuntimeError, match="moved from address"):
        allocator.claim_graph_bucket(0, 8, torch.float32)

    assert allocator._slot_using == {}
    assert original_tensor.data_ptr() != cpu_global_memory_buffer.buffer[key].data_ptr()


def test_planned_graph_bucket_claim_rejects_live_dtype_and_capacity_drift(cpu_global_memory_buffer):
    """Current main_grad requirements are checked before graph replay writes."""
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_grads", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)
    allocator.freeze_plan([0])

    with pytest.raises(RuntimeError, match="materialized with dtype"):
        allocator.claim_graph_bucket(0, 8, torch.float64)
    with pytest.raises(RuntimeError, match="materialized at 8 elements"):
        allocator.claim_graph_bucket(0, 9, torch.float32)

    assert allocator._slot_using == {}


def test_planned_graph_bucket_claim_requires_frozen_materialized_slot(cpu_global_memory_buffer):
    """Claims never fall back for an unknown or lazily unmaterialized graph bucket."""
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_grads", _TrackingBucketAllocator(), _allocator_param_groups(0)
    )
    allocator.freeze_plan([0])

    with pytest.raises(RuntimeError, match="was not materialized"):
        allocator.claim_graph_bucket(0, 8, torch.float32)
    with pytest.raises(RuntimeError, match="not covered by the frozen"):
        allocator.claim_graph_bucket(1, 8, torch.float32)


def test_planned_allocator_rejects_new_buckets_after_plan_is_frozen(cpu_global_memory_buffer):
    """Repeated freeze is idempotent only when it does not extend the plan."""
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_params", _TrackingBucketAllocator(), _allocator_param_groups(0, 1)
    )
    allocator.allocate(0, 8, torch.float32, torch.device("cpu"))
    allocator.free(0)
    allocator.freeze_plan([0])
    original_plan = dict(allocator._plan)

    allocator.freeze_plan([0])
    with pytest.raises(RuntimeError, match=r"new graph buckets appeared: \[1\]"):
        allocator.freeze_plan([0, 1])

    assert allocator._plan == original_plan


def test_planned_allocator_diagnostics_report_materialized_storage_once(
    cpu_global_memory_buffer, caplog, monkeypatch
):
    """Diagnostics count shared slot storage once and do not touch allocator state."""
    allocator = PlannedBucketAllocator(
        "model_chunk_0_fsdp_params",
        _TrackingBucketAllocator(),
        _allocator_param_groups(0, 0, 1, 1),
    )
    allocations = (
        (0, 8, torch.float32),
        (1, 4, torch.float64),
        (2, 6, torch.float32),
        (3, 10, torch.float64),
    )
    for bucket_id, size, dtype in allocations:
        allocator.allocate(bucket_id, size, dtype, torch.device("cpu"))
        allocator.free(bucket_id)

    with caplog.at_level(logging.DEBUG, logger=pgb_module.__name__):
        allocator.freeze_plan(range(4))
    allocator.claim_graph_bucket(0, 8, torch.float32)

    plan_before = dict(allocator._plan)
    materialization_before = dict(allocator._slot_materialization)
    slot_using_before = dict(allocator._slot_using)
    storage_before = {
        key: (tensor.data_ptr(), tensor.numel())
        for key, tensor in cpu_global_memory_buffer.buffer.items()
    }

    def reject_tensor_allocation(*args, **kwargs):
        raise AssertionError("plan diagnostics must not allocate tensors")

    monkeypatch.setattr(pgb_module.torch, "empty", reject_tensor_allocation)
    monkeypatch.setattr(cpu_global_memory_buffer, "get_tensor", reject_tensor_allocation)
    diagnostics = allocator.get_plan_diagnostics()
    assert allocator.get_plan_diagnostics() == diagnostics

    assert allocator._plan == plan_before
    assert allocator._slot_materialization == materialization_before
    assert allocator._slot_using == slot_using_before
    assert {
        key: (tensor.data_ptr(), tensor.numel())
        for key, tensor in cpu_global_memory_buffer.buffer.items()
    } == storage_before

    assert diagnostics["storage_kind"] == "weight"
    assert diagnostics["planned_bucket_count"] == 4
    assert diagnostics["color_counts_by_offset"] == [
        {"bucket_offset": 0, "color_count": 1},
        {"bucket_offset": 1, "color_count": 1},
    ]
    assert diagnostics["max_color_count_per_offset"] == 1
    assert diagnostics["total_color_count_across_offsets"] == 2
    assert diagnostics["logical_slot_count"] == 2
    assert diagnostics["materialized_slot_count"] == 2
    assert diagnostics["occupied_slot_count"] == 1
    assert diagnostics["unique_storage_count"] == 2
    assert diagnostics["materialized_bytes"] == 8 * 4 + 10 * 8
    assert diagnostics["bucket_to_slot"] == [
        {"bucket_id": 0, "color": 0, "bucket_offset": 0},
        {"bucket_id": 1, "color": 0, "bucket_offset": 1},
        {"bucket_id": 2, "color": 0, "bucket_offset": 0},
        {"bucket_id": 3, "color": 0, "bucket_offset": 1},
    ]
    assert [slot["occupant_bucket_id"] for slot in diagnostics["slots"]] == [0, None]
    assert diagnostics["dtype_summaries"] == [
        {
            "storage_kind": "weight",
            "dtype": "torch.float32",
            "planned_bucket_count": 2,
            "distinct_color_id_count": 1,
            "slot_count": 1,
            "unique_storage_count": 1,
            "capacity_elements": 8,
            "materialized_bytes": 32,
        },
        {
            "storage_kind": "weight",
            "dtype": "torch.float64",
            "planned_bucket_count": 2,
            "distinct_color_id_count": 1,
            "slot_count": 1,
            "unique_storage_count": 1,
            "capacity_elements": 10,
            "materialized_bytes": 80,
        },
    ]
    assert diagnostics["offset_dtype_summaries"] == [
        {
            "storage_kind": "weight",
            "bucket_offset": 0,
            "dtype": "torch.float32",
            "planned_bucket_count": 2,
            "distinct_color_id_count": 1,
            "slot_count": 1,
            "unique_storage_count": 1,
            "capacity_elements": 8,
            "materialized_bytes": 32,
        },
        {
            "storage_kind": "weight",
            "bucket_offset": 1,
            "dtype": "torch.float64",
            "planned_bucket_count": 2,
            "distinct_color_id_count": 1,
            "slot_count": 1,
            "unique_storage_count": 1,
            "capacity_elements": 10,
            "materialized_bytes": 80,
        },
    ]
    assert len(diagnostics["plan_checksum"]) == 64
    assert diagnostics["plan_checksum_scope"] == (
        "freeze-time-plan-and-materialization"
    )
    if safe_get_rank() == 0:
        assert "materialized_bytes=112" in caplog.text
    assert "bucket_to_slot=" in caplog.text
    assert "slot=" in caplog.text


def test_planned_allocator_plan_checksum_is_namespace_and_pointer_independent(
    cpu_global_memory_buffer, caplog
):
    """Equivalent logical plans hash identically across process-local storage names."""

    def freeze(name, first_size=8, overlap=False):
        allocator = PlannedBucketAllocator(
            name, _TrackingBucketAllocator(), _allocator_param_groups(0, 1)
        )
        allocator.allocate(0, first_size, torch.float32, torch.device("cpu"))
        if overlap:
            allocator.allocate(1, 6, torch.float32, torch.device("cpu"))
            allocator.free(1)
            allocator.free(0)
        else:
            allocator.free(0)
            allocator.allocate(1, 6, torch.float32, torch.device("cpu"))
            allocator.free(1)
        allocator.freeze_plan([0, 1])
        return allocator

    first = freeze("model_chunk_0_param_and_grad_buffer_2_fsdp_params")
    second = freeze("model_chunk_0_param_and_grad_buffer_9_fsdp_params")

    assert first.get_plan_diagnostics()["plan_checksum"] == second.get_plan_diagnostics()[
        "plan_checksum"
    ]
    assert {
        materialization[2] for materialization in first._slot_materialization.values()
    }.isdisjoint(
        materialization[2] for materialization in second._slot_materialization.values()
    )

    different_capacity = freeze(
        "model_chunk_0_param_and_grad_buffer_10_fsdp_params", first_size=9
    )
    different_coloring = freeze(
        "model_chunk_0_param_and_grad_buffer_11_fsdp_params", overlap=True
    )
    assert different_capacity._plan == first._plan
    assert different_capacity._plan_checksum != first._plan_checksum
    assert different_coloring._plan != first._plan
    assert different_coloring._plan_checksum != first._plan_checksum

    diagnostics_before = first.get_plan_diagnostics()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=pgb_module.__name__):
        first.freeze_plan([0, 1])
    assert first.get_plan_diagnostics() == diagnostics_before
    assert caplog.text == ""


def test_fused_wgrad_claim_records_warmup_lifetimes_and_skips_persistent_grad_buffers():
    """Warmup records all temporary buckets while replay claims only fused buckets."""
    distributed_param = torch.nn.Parameter(torch.empty(1))
    persistent_param = torch.nn.Parameter(torch.empty(1))
    non_fused_param = torch.nn.Parameter(torch.empty(1))
    record_events, claim_events, wait_events = [], [], []
    distributed_grad_buffer = SimpleNamespace(
        is_data_distributed=True, bucket_index=SimpleNamespace(size=8), dtype=torch.float32
    )
    persistent_grad_buffer = SimpleNamespace(
        is_data_distributed=False, bucket_index=SimpleNamespace(size=8), dtype=torch.float32
    )
    parameter_groups = [
        SimpleNamespace(hfsdp_helper_gbuf=None, main_grad_buffer=distributed_grad_buffer),
        SimpleNamespace(hfsdp_helper_gbuf=None, main_grad_buffer=persistent_grad_buffer),
        SimpleNamespace(hfsdp_helper_gbuf=None, main_grad_buffer=distributed_grad_buffer),
    ]
    pgb = SimpleNamespace(
        main_grad_alloc=SimpleNamespace(
            record_graph_bucket_claim=lambda bucket_id, size, dtype: record_events.append(
                (bucket_id, size, dtype)
            ),
            claim_graph_bucket=lambda bucket_id, size, dtype: claim_events.append(
                (bucket_id, size, dtype)
            )
        ),
        mp_policy=SimpleNamespace(grad_comm_dtype=torch.float16),
        param_to_param_group={distributed_param: 0, persistent_param: 1, non_fused_param: 2},
        parameter_groups=parameter_groups,
    )
    fsdp = MegatronFSDP.__new__(MegatronFSDP)
    torch.nn.Module.__init__(fsdp)
    fsdp.param_and_grad_buffer = pgb
    fsdp._cuda_graph_fused_wgrad_params = {id(distributed_param), id(persistent_param)}
    fsdp.grad_reduce_pipeline = SimpleNamespace(
        wait_for_pending_buckets=lambda bucket_ids: wait_events.append(tuple(bucket_ids))
    )

    fsdp._claim_cuda_graph_fused_wgrad_buckets(
        [distributed_param, distributed_param, persistent_param, non_fused_param]
    )

    assert record_events == [(0, 8, torch.float16), (2, 8, torch.float16)]
    assert claim_events == [(0, 8, torch.float16)]
    assert wait_events == [(0,)]


@pytest.mark.parametrize(
    (
        "queued_bucket_ids",
        "pending_bucket_ids",
        "expected_remaining",
        "expected_released",
    ),
    [
        pytest.param([0, 1], [0], [1], [0], id="keep-safe-suffix"),
        pytest.param([1, 0], [0], [], [1, 0], id="drain-through-newest-match"),
        pytest.param([0, 1], [2], [0, 1], [], id="unrelated-bucket"),
    ],
)
def test_grad_reduce_waits_for_exact_pending_buckets_before_graph_write(
    monkeypatch,
    queued_bucket_ids,
    pending_bucket_ids,
    expected_remaining,
    expected_released,
):
    """Only an older RS reading the replayed bucket forces a pre-write wait."""
    consumer_stream = object()
    events = []
    pipeline = GradReducePipeline.__new__(GradReducePipeline)
    pipeline.buffer = SimpleNamespace(num_buckets=3)
    pipeline.grad_reduce_queue = []
    for bucket_id in queued_bucket_ids:
        event = SimpleNamespace(
            wait=lambda stream=None, bucket_id=bucket_id: events.append(
                ("wait", bucket_id, stream)
            )
        )
        free_bucket = lambda bucket_id=bucket_id: events.append(("free", bucket_id))
        pipeline.grad_reduce_queue.append((event, free_bucket, bucket_id))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: consumer_stream)

    pipeline.wait_for_pending_buckets(pending_bucket_ids)
    events.append(("write",))

    assert [event[0] for event in events] == [
        step for _ in expected_released for step in ("wait", "free")
    ] + ["write"]
    assert [event[1] for event in events if event[0] == "free"] == expected_released
    assert all(event[2] is consumer_stream for event in events if event[0] == "wait")
    assert [bucket_id for _, _, bucket_id in pipeline.grad_reduce_queue] == expected_remaining


def test_planned_force_param_sync_rejects_before_state_change():
    """Force-sync cannot discard planned unit weights while promising all params are ready."""
    events = []
    fake_fsdp = SimpleNamespace(
        data_parallel_sharding_strategy="optim_grads_params",
        param_and_grad_buffer=SimpleNamespace(_uses_planned_allocator=True),
        _replace_param_with_raw_if_needed=lambda: events.append("replace"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"start_param_sync\(force_sync=True\) is incompatible with planned allocation",
    ):
        MegatronFSDP.start_param_sync(fake_fsdp, force_sync=True)

    assert events == []

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import importlib
import logging
import math
import os
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from megatron.core._rank_utils import safe_get_rank
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.fine_grained_activation_offload import ChunkOffloadHandler
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedOffloadingBackwardRecordFunction,
    OffloadTensorGroup,
    OffloadTensorPool,
    PipelineOffloadManager,
)
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    is_checkpoint_without_output_tensor,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

# Tolerance for memory expectation check (GPU allocator jitter etc).
EPSILON = 0.30
# The EP-overlap schedule saves more than the recorded bytes of the requested groups (about
# 1.3x for the all-module cases): offloading most activations moves the peak and shrinks
# allocator fragments, and combined requests are recorded under the fused group.
EPSILON_A2A = 0.35
DELTA = 20  # MiB
CUDA_GRAPH_DELTA = 200  # MiB — Considering some CG overhead


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.7, 0.8, 1.0])
def test_partial_offload_backward_does_not_accumulate_reloaded_groups(monkeypatch, fraction):
    """Actual H2D reloads must stay one group ahead of a sequential backward pass.

    A chain of sin operations saves one equally sized activation per group. Warmup
    uses an online fraction quota; the final policy reserves the last group and
    selects an independent prefix. Reloading at every resident boundary would pull
    all earlier offloaded groups back to GPU before any of them are consumed.

    Observe real reload events and saved-tensor consumption without changing stream
    ordering or calling the scheduler ourselves. Use interfaces shared with the
    unfixed implementation so the same test can demonstrate the original regression.
    """
    num_groups = 9
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    off_interface.reset_instance()
    input_tensor = torch.full((512, 512), 0.1, device="cuda", requires_grad=True)
    group_bytes = input_tensor.numel() * input_tensor.element_size()
    backward_order = []
    reload_events = []
    samples = []
    current_backward_group = None
    offloaded_groups = {}

    def sample(event):
        # Count only initially offloaded groups. A CUDA tensor here was allocated
        # by reload but has not yet been retrieved through saved_tensors_hooks.
        # No synchronize is inserted: allocation/retention already occurs when the
        # asynchronous H2D copy is enqueued, even if the copy has not completed.
        pending = {
            index: sum(
                tensor.numel() * tensor.element_size()
                for tensor in group._tensors.values()
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda
            )
            for group, index in offloaded_groups.items()
        }
        pending = {index: size for index, size in pending.items() if size}
        samples.append((event, current_backward_group, tuple(pending), sum(pending.values())))

    def record_backward(grad, index):
        nonlocal current_backward_group
        current_backward_group = index
        backward_order.append(index)
        sample("backward")
        return grad

    def forward(record=False):
        off_interface.init_chunk_handler(
            pp_rank=0,
            vp_size=None,
            vp_stage=None,
            min_offloaded_tensor_size=1,
            delta_offload_bytes_across_pp_ranks=0,
            activation_offload_fraction=fraction,
        )
        output = input_tensor
        for index in range(num_groups):
            scope = off_interface(True, output, "core_attn")
            with scope as activation:
                output = activation.sin()
            output = scope.group_offload(output)
            if record:
                output.register_hook(lambda grad, index=index: record_backward(grad, index))
        return output

    original_reload_event = OffloadTensorGroup.record_reload_event
    original_pop = OffloadTensorGroup.pop_tensor

    def record_reload_event(group, stream):
        original_reload_event(group, stream)
        if group in offloaded_groups:
            reload_events.append((current_backward_group, offloaded_groups[group]))
            sample("reload")

    def pop_tensor(group, tag):
        tensor = original_pop(group, tag)
        if group in offloaded_groups:
            sample("consume")
        return tensor

    try:
        # Discover the groups and let the production warmup policy apply the
        # fraction; do not manually construct or edit either scheduling queue.
        reference = input_tensor
        for _ in range(num_groups):
            reference = reference.sin()
        expected_grad = torch.autograd.grad(reference.sum(), input_tensor)[0]
        expected_output = reference.detach()
        del reference

        output = forward()
        manager = PipelineOffloadManager.get_instance()
        warmup_chunk = manager.cur_forward_chunk()
        warmup_indices = [
            index
            for index, group in enumerate(warmup_chunk.offload_groups)
            if any(isinstance(state, tuple) for state in group._tensors.values())
        ]
        assert len(warmup_indices) == math.ceil(num_groups * fraction)
        # Check the quota at every prefix, including the three-group case where
        # fraction 0.7 deliberately offloads all three to favor GPU headroom.
        for count in range(1, num_groups + 1):
            assert sum(index < count for index in warmup_indices) == math.ceil(count * fraction)
        assert all(group.offload for group in warmup_chunk.offload_groups)
        assert all(
            group.total_offload_bytes == group_bytes and group.total_tensor_count == 1
            for group in warmup_chunk.offload_groups
        )
        pool = manager._cpu_tensor_pool
        warmup_allocations = pool._stats["total_allocated"]
        assert warmup_allocations == len(warmup_indices)
        torch.testing.assert_close(output, expected_output)
        torch.cuda.synchronize()
        output.sum().backward()
        torch.testing.assert_close(input_tensor.grad, expected_grad)
        torch.cuda.synchronize()
        del output
        input_tensor.grad = None
        off_interface.reset()

        output = forward(record=True)
        chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
        assert len(chunk.offload_groups) == num_groups
        offloaded_groups = {
            group: index
            for index, group in enumerate(chunk.offload_groups)
            if any(isinstance(state, tuple) for state in group._tensors.values())
        }
        eligible_count = num_groups - 1
        num_offloaded = eligible_count - int(eligible_count * (1 - fraction))
        assert list(offloaded_groups.values()) == list(range(num_offloaded))
        assert all(len(group._tensors) == 1 for group in chunk.offload_groups)
        # Equal-shaped buffers are reused across the different warmup/steady masks.
        assert pool._stats["total_allocated"] == warmup_allocations
        torch.testing.assert_close(output, expected_output)

        monkeypatch.setattr(OffloadTensorGroup, "record_reload_event", record_reload_event)
        monkeypatch.setattr(OffloadTensorGroup, "pop_tensor", pop_tensor)
        torch.cuda.synchronize()
        output.sum().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(input_tensor.grad, expected_grad)
        assert backward_order == list(reversed(range(num_groups)))
        assert [index for _, index in reload_events] == list(reversed(range(num_offloaded)))
        assert all(not group._tensors for group in chunk.offload_groups)
        peak_pending = max(len(pending) for _, _, pending, _ in samples)
        peak_pending_bytes = max(size for _, _, _, size in samples)
        print(
            f"Partial offload cadence: fraction={fraction}, groups={num_groups}, "
            f"warmup_offloaded={len(warmup_indices)}, pool_bytes={warmup_allocations * group_bytes}, "
            f"offloaded={num_offloaded}, "
            f"peak_pending_groups={peak_pending}, peak_pending_bytes={peak_pending_bytes}, "
            f"reloads=(backward_group, reload_group) "
            f"{reload_events}"
        )
        assert peak_pending <= 1 and peak_pending_bytes <= group_bytes, (
            "Backward reload accumulation: expected at most one prefetched activation "
            f"({group_bytes} bytes), got {peak_pending} groups / {peak_pending_bytes} bytes; "
            f"samples={samples}"
        )
        assert all(
            target == current - 1 for current, target in reload_events
        ), f"Reload exceeded the next-group prefetch window: {reload_events}"
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize("fraction", [0.5, 1.0])
@pytest.mark.parametrize("core_layers", [(True, True, True), (True, False, False, False)])
@pytest.mark.parametrize("pre_reload", [False, True])
def test_margin_groups_allow_prefetch_after_delayed_norm_commit(
    monkeypatch, fraction, core_layers, pre_reload
):
    """Norm commits after attention; resident margins must not consume prefetch slots."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    off_interface.reset_instance()
    input_tensor = torch.full((128, 128), 0.1, device="cuda", requires_grad=True)

    def forward():
        off_interface.init_chunk_handler(
            pp_rank=0,
            vp_size=None,
            vp_stage=None,
            min_offloaded_tensor_size=1,
            delta_offload_bytes_across_pp_ranks=0,
            activation_offload_fraction=fraction,
        )
        output = input_tensor
        for has_core in core_layers:
            norm_scope = off_interface(True, output, "attn_norm")
            with norm_scope as residual:
                output = residual.sin()
            if has_core:
                core_scope = off_interface(True, output, "core_attn")
                with core_scope as activation:
                    output = activation.sin()
                output = core_scope.group_offload(output)
            # Match the production residual path: norm's commit is delayed until
            # after attention and residual addition, but its start marker is first.
            output = norm_scope.group_offload(output + residual)
        return output

    try:
        reference = input_tensor
        for has_core in core_layers:
            residual = reference
            reference = reference.sin()
            if has_core:
                reference = reference.sin()
            reference = reference + residual
        expected_grad = torch.autograd.grad(reference.sum(), input_tensor)[0]
        expected_output = reference.detach()
        del reference

        output = forward()
        torch.cuda.synchronize()
        output.sum().backward()
        torch.cuda.synchronize()
        del output
        input_tensor.grad = None
        off_interface.reset()

        output = forward()
        chunk = PipelineOffloadManager.get_instance().cur_forward_chunk()
        last_by_name = {group._name: group for group in chunk.offload_groups}
        for group in chunk.offload_groups:
            if group is last_by_name[group._name]:
                assert group.is_offload_margin and not group.offload
                assert group not in chunk._groups_to_reload
            else:
                assert not group.is_offload_margin
                # This includes groups kept resident by fraction, whose slots
                # must remain present to avoid reverting partial-offload cadence.
                assert group in chunk._groups_to_reload

        reload_streams = []
        original_reload = chunk.reload

        def record_reload(state, *args, **kwargs):
            reload_streams.append(torch.cuda.current_stream())
            return original_reload(state, *args, **kwargs)

        monkeypatch.setattr(chunk, "reload", record_reload)
        torch.cuda.synchronize()
        if pre_reload:
            chunk.pre_reload_last_layer()
        output.sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(input_tensor.grad, expected_grad)
        assert reload_streams
        assert all(
            stream == chunk.h2d_stream for stream in reload_streams
        ), "A delayed norm commit caused a compute-stream fallback reload"
        assert not chunk._groups_to_reload
        assert all(not group._tensors for group in chunk.offload_groups)
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


def _reset_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _make_chunk_handler_for_offload_checker(min_offloaded_tensor_size: int = 1):
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.min_offloaded_tensor_size = min_offloaded_tensor_size
    return handler


@pytest.fixture(autouse=True)
def reset_hybrid_ep():
    yield
    reset_hybrid_ep_buffer()


def _offload_summary_groups(offload_modules):
    """Map requested offload modules to the group names the offload summary records.

    GroupedMLP offloads "expert_fc1" + "moe_act" together as one "fused_group_mlp" group, so a
    request naming both is reported under that single name.
    """
    groups = list(offload_modules)
    if "expert_fc1" in groups and "moe_act" in groups:
        groups = [m for m in groups if m not in ("expert_fc1", "moe_act")]
        groups.append("fused_group_mlp")
    return groups


def _enable_te_cpu_offload_v1(monkeypatch) -> None:
    """Match the V1 CPU-offload launch Megatron requires for fine-grained offloading.

    ``megatron/training/arguments.py`` insists on ``NVTE_CPU_OFFLOAD_V1=1`` with TE >= 2.10 so
    that TE's ``is_cpu_offload_enabled()`` follows the V1 ``CPUOffloadEnabled`` flag toggled by
    ``PipelineOffloadManager.__enter__``; without it TE builds its saved tensors for a run
    without offloading (e.g. fused attention keeps the packed ``sbh3d`` q/k/v layout that the
    reloaded tensors no longer satisfy). Tests that build ``TransformerConfig`` directly bypass
    that check, and TE caches the flag at import time, so patch the cached module attributes
    as well. No-op when the installed TE does not know the flag.
    """
    monkeypatch.setenv("NVTE_CPU_OFFLOAD_V1", "1")
    for module_name in (
        "transformer_engine.pytorch.cpu_offload",
        "transformer_engine.pytorch.attention.dot_product_attention.backends",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "NVTE_CPU_OFFLOAD_V1"):
            monkeypatch.setattr(module, "NVTE_CPU_OFFLOAD_V1", True)


def test_offload_summary_uses_explicit_process_group(monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off_module

    process_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_rank",
        lambda *, group=None: 0 if group is process_group else pytest.fail("wrong process group"),
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda *, group=None: 1 if group is process_group else pytest.fail("wrong process group"),
    )

    def all_gather_object(output, value, *, group=None):
        assert group is process_group
        output[0] = value

    def barrier(*, group=None):
        assert group is process_group

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    warning_messages = []
    monkeypatch.setattr(off_module.logger, "warning", warning_messages.append)

    off_module.print_offload_summary_table(
        {"decoder": 1024}, {"decoder": (1, 1024)}, process_group=process_group
    )

    assert len(warning_messages) == 1
    assert "same tensor storage" in warning_messages[0]


def test_chunk_offload_handler_skips_non_offloadable_tensor_types():
    handler = _make_chunk_handler_for_offload_checker()

    cpu_tensor = torch.empty(1024)
    assert not handler.tensor_need_offloading_checker(cpu_tensor)
    assert handler.tensor_push(cpu_tensor) is cpu_tensor
    assert handler.tensor_pop(cpu_tensor) is cpu_tensor

    parameter = torch.nn.Parameter(torch.empty(1024))
    assert not handler.tensor_need_offloading_checker(parameter)
    assert handler.tensor_push(parameter) is parameter
    assert handler.tensor_pop(parameter) is parameter

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:
        pytest.skip("FakeTensorMode is not available in this PyTorch version.")

    with FakeTensorMode():
        fake_tensor = torch.empty(1024, device="cuda")
    assert not handler.tensor_need_offloading_checker(fake_tensor)
    assert handler.tensor_push(fake_tensor) is fake_tensor
    assert handler.tensor_pop(fake_tensor) is fake_tensor


def test_cuda_graph_backward_completion_callback_is_per_graph_task(monkeypatch):
    """Each TE-style GraphTask queues its own sync-back event after AccumulateGrad."""
    calls = []
    active_task = None

    class RecordingStream:
        def record_event(self, event):
            calls.append((active_task, "record_event", event))

        def wait_stream(self, stream):
            calls.append((active_task, "wait_stream", stream))

    recording_stream = RecordingStream()

    def current_stream():
        calls.append((active_task, "callback_boundary", recording_stream))
        return recording_stream

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)

    for task_name in ("first", "second"):
        active_task = task_name
        graph_event = object()
        h2d_stream = object()
        monkeypatch.setattr(
            PipelineOffloadManager,
            "OFFLOAD_MGR",
            type("Manager", (), {"cuda_graph_event": graph_event, "h2d_stream": h2d_stream})(),
        )

        hidden_states = torch.tensor(2.0, requires_grad=True)
        weight = torch.nn.Parameter(torch.tensor(3.0))
        hidden_states.register_post_accumulate_grad_hook(
            lambda _tensor, name=task_name, tensor=hidden_states: calls.append(
                (name, "input_grad", tensor)
            )
        )
        weight.register_post_accumulate_grad_hook(
            lambda _param, name=task_name, param=weight: calls.append((name, "param_grad", param))
        )

        task_start = len(calls)
        recorded_input = FineGrainedOffloadingBackwardRecordFunction.apply(hidden_states)
        (recorded_input * weight).backward()
        task_calls = calls[task_start:]

        assert [call[0] for call in task_calls] == [task_name] * 5
        assert {call[1] for call in task_calls[:2]} == {"input_grad", "param_grad"}
        assert task_calls[2:] == [
            (task_name, "callback_boundary", recording_stream),
            (task_name, "record_event", graph_event),
            (task_name, "wait_stream", h2d_stream),
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_chunk_offload_handler_respects_tensor_opt_out_flags():
    handler = _make_chunk_handler_for_offload_checker()

    tensor = torch.empty(1024, device="cuda")
    assert handler.tensor_need_offloading_checker(tensor)

    tensor._TE_do_not_offload = True
    assert not handler.tensor_need_offloading_checker(tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_chunk_offload_handler_handles_storage_less_tensor_wrapper():
    """TE-style wrapper tensors must reach their opt-out without requiring storage."""

    data_tensor = torch.empty(1, device="cuda")
    data_tensor._TE_do_not_offload = True

    class StorageLessTensor(torch.Tensor):
        """Minimal TE-style tensor wrapper without backing storage."""

        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_wrapper_subclass(
                cls,
                (1024,),
                strides=(1,),
                storage_offset=0,
                dtype=torch.float32,
                layout=torch.strided,
                device="cuda",
                requires_grad=False,
            )

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise NotImplementedError

        def get_data_tensors(self):
            """Return the backing tensor inspected by TE's offload opt-out."""
            return [data_tensor]

        def untyped_storage(self):
            raise RuntimeError("StorageLessTensor has no backing storage")

    handler = _make_chunk_handler_for_offload_checker()
    tensor = StorageLessTensor()

    assert not handler.tensor_need_offloading_checker(tensor)
    assert not is_checkpoint_without_output_tensor(tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_chunk_offload_handler_skips_checkpoint_without_output_storage():
    """Checkpoint outputs and their views must not be copied or released by offload."""
    handler = _make_chunk_handler_for_offload_checker()
    Utils.initialize_model_parallel()
    try:
        input_tensor = torch.randn(8, device="cuda", requires_grad=True)
        checkpoint = CheckpointWithoutOutput()
        output = checkpoint.checkpoint(lambda tensor: tensor * 2, input_tensor)

        assert not handler.tensor_need_offloading_checker(output)
        assert not handler.tensor_need_offloading_checker(output.view(2, 4))
        assert handler.tensor_need_offloading_checker(output.clone())
    finally:
        Utils.destroy_model_parallel()


def _make_chunk_handler_for_offload_reload(capacity_bytes=0):
    handler = ChunkOffloadHandler.__new__(ChunkOffloadHandler)
    handler.cpu_tensor_pool = OffloadTensorPool(
        device="cpu", pin_memory=True, capacity_bytes=capacity_bytes
    )
    return handler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
@pytest.mark.parametrize("capacity_bytes", [0, 2**20])
def test_chunk_offload_handler_offloads_base_storage_for_covering_views(capacity_bytes):
    handler = _make_chunk_handler_for_offload_reload(capacity_bytes)

    base = torch.randn(64, 96, device="cuda")
    view = base[:, :80]  # non-contiguous last-dim slice covering >50% of the storage
    assert not view.is_contiguous()

    state = handler.offload(view)
    _, cpu_backup, _, view_meta = state
    # The full flat storage is offloaded, not a gathered copy of the view.
    assert cpu_backup.shape == (base.numel(),)
    assert view_meta == (view.size(), view.stride(), view.storage_offset())

    reloaded = handler.reload(state)
    torch.cuda.synchronize()
    assert reloaded.shape == view.shape
    assert reloaded.stride() == view.stride()
    assert torch.equal(reloaded, view)
    # The pool backup was returned after the reload.
    assert handler.cpu_tensor_pool._stats["current_in_use"] == 0

    # Autograd hands the saved-tensor hooks a detach()-ed alias (e.g. the
    # checkpoint's saved input), which has no ._base; the storage-based
    # resolution must still take the base path.
    detached = base[:, :80].detach()
    state_d = handler.offload(detached)
    assert state_d[3] is not None
    reloaded_d = handler.reload(state_d)
    torch.cuda.synchronize()
    assert torch.equal(reloaded_d, detached)
    assert handler.cpu_tensor_pool._stats["current_in_use"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
@pytest.mark.parametrize("capacity_bytes", [0, 2**20])
@pytest.mark.parametrize("use_cpu_pool", [False, True])
def test_chunk_offload_handler_gathers_low_coverage_views(capacity_bytes, use_cpu_pool):
    handler = _make_chunk_handler_for_offload_reload(capacity_bytes)

    base = torch.randn(64, 96, device="cuda")
    view = base[:, :8]  # covers well under BASE_OFFLOAD_MIN_COVERAGE of base

    state = handler.offload(view, use_cpu_pool=use_cpu_pool)
    _, cpu_backup, uses_pool, view_meta = state
    # Fixed storage also handles variable-shape groups that bypass the legacy cache.
    assert uses_pool is bool(capacity_bytes or use_cpu_pool)
    # Low-coverage views fall back to the contiguous gather.
    assert view_meta is None
    assert cpu_backup.shape == view.shape

    reloaded = handler.reload(state)
    torch.cuda.synchronize()
    assert reloaded.is_contiguous()
    assert torch.equal(reloaded, view)
    status = handler.cpu_tensor_pool.get_pool_status()
    assert status["global_stats"]["current_in_use"] == 0
    if capacity_bytes:
        assert status["pools"] == {}
        assert status["fixed_buffer"]["capacity_bytes"] == capacity_bytes


def _make_warmup_chunk(groups: List["OffloadTensorGroup"]) -> ChunkOffloadHandler:
    """Build a warmup chunk handler that already owns ``groups``."""
    handler = ChunkOffloadHandler(
        min_offloaded_tensor_size=1,
        cpu_tensor_pool=OffloadTensorPool(device="cpu", pin_memory=True),
    )
    handler.offload_groups = list(groups)
    handler._max_group_size = len(groups)
    return handler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_warmup_fraction_counts_every_group_per_module_and_keeps_view_metadata():
    base = torch.ones(64, 96, device="cuda")
    opt_out = torch.ones(16, device="cuda")
    opt_out._do_not_offload = True
    tensors_by_group = [
        ("qkv_linear", [torch.ones(16, device="cuda")]),
        ("core_attn", [torch.ones(16, device="cuda")]),
        ("attn_proj", [torch.ones(16, device="cuda")]),
        ("qkv_linear", []),
        ("core_attn", [torch.ones(16), torch.ones(1, device="cuda"), opt_out]),
        ("attn_proj", [torch.ones(16, device="cuda")]),
        ("qkv_linear", [torch.ones(16, device="cuda")]),
        ("core_attn", [torch.ones(16, device="cuda")]),
        ("attn_proj", [torch.ones(16, device="cuda")]),
        ("expert_fc1", []),
        ("expert_fc1", [base[:, :80].detach(), base[:, 16:], base[::8, ::8]]),
    ]
    groups = []
    for group_index, (name, tensors) in enumerate(tensors_by_group):
        group = OffloadTensorGroup(name)
        for tensor_index, tensor in enumerate(tensors):
            group.push_tensor((group_index, tensor_index), tensor)
        groups.append(group)
    chunk = _make_warmup_chunk(groups)
    chunk.min_offloaded_tensor_size = 16
    chunk._activation_offload_fraction = 0.5
    chunk._groups_to_offload = list(groups)
    try:
        for group in groups:
            chunk.bulk_offload(group._name, [])
        torch.cuda.synchronize()
        # Each module selects occurrences 1 and 3. The empty QKV and ineligible
        # attention groups in occurrence 2 must not shift the next selection.
        assert [
            i
            for i, group in enumerate(groups)
            if any(isinstance(state, tuple) for state in group._tensors.values())
        ] == [0, 1, 2, 6, 7, 8]
        assert groups[3].total_offload_bytes == groups[4].total_offload_bytes == 0
        assert chunk.cpu_tensor_pool._stats["allocation_requests"] == 6
        # The skipped MoE group has no CPU pool, but its storage-sized copies,
        # logical gather, and redundant storage still have complete metadata.
        skipped = groups[10]
        storage_bytes = base.untyped_storage().nbytes()
        gather_bytes = base[::8, ::8].numel() * base.element_size()
        assert not skipped.use_cpu_pool
        assert all(isinstance(tensor, torch.Tensor) for tensor in skipped._tensors.values())
        assert skipped.total_tensor_count == 3
        assert skipped.total_offload_bytes == 2 * storage_bytes + gather_bytes
        assert skipped.duplicate_storage_tensor_count == 2
        assert skipped.duplicate_storage_bytes == storage_bytes + gather_bytes
        assert all(group.offload for group in groups)
    finally:
        torch.cuda.synchronize()


def _run_post_warmup_callback(chunk: ChunkOffloadHandler) -> PipelineOffloadManager:
    """Drive post_warmup_callback over a single hand-built chunk and return the manager."""
    manager = PipelineOffloadManager.__new__(PipelineOffloadManager)
    manager._is_warmup = True
    manager._cached_chunks_forward = [chunk]
    manager._cached_chunks_backward = [chunk]
    manager._offload_margin = 0
    manager._pp_rank = 0
    manager._delta_offload_bytes_across_pp_ranks = 0
    manager._activation_offload_fraction = 1.0
    manager.post_warmup_callback()
    return manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
@pytest.mark.skipif(
    Utils.world_size < 2, reason="Rank-0 duplicate aggregation requires at least two ranks."
)
def test_post_warmup_callback_warns_on_duplicated_storage_offload(caplog, monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off_module

    # post_warmup_callback gathers the per-rank duplicate summaries, so this test needs a
    # live process group. The skipif above reads the launcher's world size rather than
    # torch.distributed.get_world_size(), which raises when no group exists yet.
    Utils.initialize_model_parallel()

    try:
        rank = safe_get_rank()
        duplicate_rank = 1
        base = torch.randn(1024, 512, device="cuda")
        storage_bytes = base.numel() * base.element_size()
        # Two strided views that each cover >50% of the shared storage, so both take
        # the whole-storage offload path and copy the same bytes to CPU twice. Only
        # rank 1 creates this duplication, exercising the rank-0 aggregation path.
        views = (base[:, :400], base[:, 112:]) if rank == duplicate_rank else (base[:, :400],)
        assert all(not view.is_contiguous() for view in views)

        # post_warmup_callback disables the last group of each name (offload margin),
        # so build two same-named groups and assert on the surviving one.
        groups = []
        for group_idx in range(2):
            group = OffloadTensorGroup("core_attn")
            for tensor_idx, view in enumerate(views):
                group.push_tensor((group_idx, tensor_idx), view)
            groups.append(group)

        chunk = _make_warmup_chunk(groups)
        for group in groups:
            chunk.bulk_offload_group(group)
        torch.cuda.synchronize()

        # The second view of each group on rank 1 is redundant; other ranks have no
        # local duplication to report.
        for group in groups:
            assert group.duplicate_storage_tensor_count == int(rank == duplicate_rank)
            assert group.duplicate_storage_bytes == (storage_bytes if rank == duplicate_rank else 0)

        real_all_gather_object = torch.distributed.all_gather_object
        all_gather_object_calls = 0

        def counting_all_gather_object(*args, **kwargs):
            nonlocal all_gather_object_calls
            all_gather_object_calls += 1
            return real_all_gather_object(*args, **kwargs)

        monkeypatch.setattr(torch.distributed, "all_gather_object", counting_all_gather_object)
        with caplog.at_level(logging.WARNING, logger=off_module.__name__):
            manager = _run_post_warmup_callback(chunk)

        assert all_gather_object_calls == 1

        # The margin rule keeps only the first group offloaded. Rank 0 gathers and
        # reports the duplication even though it occurred only on rank 1.
        if rank == 0:
            assert "copying the same tensor storage to CPU more than once" in caplog.text
            assert (
                f"rank {duplicate_rank}, core_attn: 1 redundant copies, "
                f"{storage_bytes / (1024 * 1024):.2f} MB" in caplog.text
            )
            assert "Offloading proceeds with the duplicated copies." in caplog.text
        else:
            assert "copying the same tensor storage to CPU more than once" not in caplog.text
        # The duplicated offloading is diagnosed, not suppressed.
        assert groups[0].offload
        assert groups[0].total_tensor_count == len(views)
        # The redundant bytes of the surviving group are cached per name next to
        # the offload bytes so consumers can count each storage once.
        assert manager.offload_summary_bytes == {"core_attn": groups[0].total_offload_bytes}
        assert manager.offload_summary_duplicate_bytes == (
            {"core_attn": storage_bytes} if rank == duplicate_rank else {}
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_bulk_offload_group_flags_disjoint_views_taking_the_whole_storage_path():
    base = torch.randn(1024, 512, device="cuda")
    storage_bytes = base.numel() * base.element_size()
    # Disjoint halves, but each covers exactly BASE_OFFLOAD_MIN_COVERAGE of the
    # storage, so both are offloaded as the full buffer: the views not sharing a
    # single element does not make the second copy any less redundant.
    group = OffloadTensorGroup("core_attn")
    group.push_tensor((1, 0), base[:, :256])
    group.push_tensor((1, 1), base[:, 256:])

    chunk = _make_warmup_chunk([group])
    chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    assert all(state[3] is not None for state in group._tensors.values())
    assert group.total_offload_bytes == 2 * storage_bytes
    assert group.duplicate_storage_tensor_count == 1
    assert group.duplicate_storage_bytes == storage_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_bulk_offload_group_flags_contiguous_whole_storage_aliases(monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off_module

    base = torch.randn(1024, 512, device="cuda")
    storage_bytes = base.numel() * base.element_size()
    # Two autograd aliases of one contiguous storage (MLA's q/kv down projections
    # both save the same hidden_states under qkv_linear) each copy the whole
    # storage through the plain contiguous path. Neither is a strided view, yet
    # the second copy is as redundant as a second whole-storage view copy.
    aliases = (base.detach(), base.detach())
    assert all(alias.is_contiguous() for alias in aliases)
    # post_warmup_callback disables the last group of each name (offload margin),
    # so build two same-named groups and assert on the surviving one.
    groups = []
    for group_idx in range(2):
        group = OffloadTensorGroup("qkv_linear")
        for tensor_idx, alias in enumerate(aliases):
            group.push_tensor((group_idx, tensor_idx), alias)
        groups.append(group)

    chunk = _make_warmup_chunk(groups)
    for group in groups:
        chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    for group in groups:
        assert all(state[3] is None for state in group._tensors.values())
        assert group.total_offload_bytes == 2 * storage_bytes
        assert group.duplicate_storage_tensor_count == 1
        assert group.duplicate_storage_bytes == storage_bytes

    # Diagnosed, not suppressed: both copies stay in the summary, and the
    # redundant bytes are cached per group so the GPU saving (one storage) can
    # be derived from it.
    monkeypatch.setattr(off_module, "print_offload_summary_table", lambda *args, **kwargs: None)
    manager = _run_post_warmup_callback(chunk)
    assert groups[0].offload and not groups[1].offload
    assert manager.offload_summary_bytes == {"qkv_linear": 2 * storage_bytes}
    assert manager.offload_summary_duplicate_bytes == {"qkv_linear": storage_bytes}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
@pytest.mark.parametrize("full_storage_first", [False, True])
def test_bulk_offload_group_counts_full_storage_and_logical_view_duplicates(full_storage_first):
    base = torch.randn(1024, 512, device="cuda")
    storage_bytes = base.numel() * base.element_size()
    logical_view = base[:128]
    logical_view_bytes = logical_view.numel() * logical_view.element_size()
    full_storage_view = base[:, :400]
    tensors = (
        (full_storage_view, logical_view)
        if full_storage_first
        else (logical_view, full_storage_view)
    )

    group = OffloadTensorGroup("core_attn")
    for tensor_idx, tensor in enumerate(tensors):
        group.push_tensor((1, tensor_idx), tensor)

    chunk = _make_warmup_chunk([group])
    chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    assert group.total_offload_bytes == storage_bytes + logical_view_bytes
    assert group.duplicate_storage_tensor_count == 1
    assert group.duplicate_storage_bytes == logical_view_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_bulk_offload_group_does_not_flag_views_copying_their_own_bytes(caplog, monkeypatch):
    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off_module

    base = torch.randn(1024, 512, device="cuda")
    storage_bytes = base.numel() * base.element_size()
    # Disjoint contiguous slices copy their own bytes, together exactly the
    # storage once.
    group = OffloadTensorGroup("core_attn")
    group.push_tensor((1, 0), base[:512])
    group.push_tensor((1, 1), base[512:])
    other_group = OffloadTensorGroup("core_attn")
    other_group.push_tensor((2, 0), base[:, :8])
    other_group.push_tensor((2, 1), base[:, 8:16])

    chunk = _make_warmup_chunk([group, other_group])
    chunk.bulk_offload_group(group)
    chunk.bulk_offload_group(other_group)
    torch.cuda.synchronize()

    assert group.total_offload_bytes == storage_bytes
    for offload_group in (group, other_group):
        assert all(state[3] is None for state in offload_group._tensors.values())
        assert offload_group.duplicate_storage_tensor_count == 0
        assert offload_group.duplicate_storage_bytes == 0

    monkeypatch.setattr(off_module, "print_offload_summary_table", lambda *args, **kwargs: None)
    with caplog.at_level(logging.WARNING, logger=off_module.__name__):
        _run_post_warmup_callback(chunk)
    assert "more than once" not in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_bulk_offload_group_does_not_analyze_logical_view_overlap():
    base = torch.randn(1024, 512, device="cuda")
    half_bytes = base[:512].numel() * base.element_size()
    # General overlap analysis for logical views is intentionally out of scope;
    # duplicate-byte accounting is exact only when the storage is copied in full.
    group = OffloadTensorGroup("core_attn")
    group.push_tensor((1, 0), base[:512])
    group.push_tensor((1, 1), base[:512])

    chunk = _make_warmup_chunk([group])
    chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    assert group.total_offload_bytes == 2 * half_bytes
    assert group.duplicate_storage_tensor_count == 0
    assert group.duplicate_storage_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offload check.")
def test_duplicate_storage_accounting_is_scoped_to_one_group():
    base = torch.randn(1024, 512, device="cuda")
    groups = []
    for group_idx, view in enumerate((base[:, :400], base[:, 112:])):
        group = OffloadTensorGroup("core_attn")
        group.push_tensor((group_idx, 0), view)
        groups.append(group)

    chunk = _make_warmup_chunk(groups)
    for group in groups:
        chunk.bulk_offload_group(group)
    torch.cuda.synchronize()

    for group in groups:
        assert group.duplicate_storage_tensor_count == 0
        assert group.duplicate_storage_bytes == 0


def _build_gpt_model(
    *,
    seed: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    seq_length: int,
    num_experts: Optional[int],
    fine_grained_activation_offloading: bool,
    offload_modules: Optional[List[str]],
    min_offloaded_tensor_size: int,
    is_mla: bool,
    activation_offload_fraction: float = 1.0,
    attention_backend: AttnBackend = AttnBackend.unfused,
    qk_layernorm: bool = True,
    fine_grained_offloading_buffer_size_gib: float = 0.0,
    enable_hyper_connections: bool = False,
    num_residual_streams: int = 4,
    mhc_recompute_layer_num: Optional[int] = None,
    enable_attention_residuals: bool = False,
    attn_res_block_layers: int = 2,
    attn_res_impl: str = "eager",
    mtp_num_layers: Optional[int] = None,
    use_transformer_engine_op_fuser: bool = False,
    recompute_modules: Optional[List[str]] = None,
) -> GPTModel:
    """Build a GPTModel that uses TE-based transformer layer spec."""
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    ConfigClass = MLATransformerConfig if is_mla else TransformerConfig
    if recompute_modules is None:
        recompute_modules = ["layernorm"]
        if num_experts is not None and not use_transformer_engine_op_fuser:
            recompute_modules.append("moe_act")
    else:
        recompute_modules = list(recompute_modules)
    if enable_hyper_connections and mhc_recompute_layer_num is not None:
        recompute_modules.append("mhc")
    transformer_config = ConfigClass(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        attention_backend=attention_backend,
        qk_layernorm=qk_layernorm and not is_mla,
        bf16=True,
        params_dtype=torch.bfloat16,
        # Recompute
        recompute_modules=recompute_modules,
        recompute_granularity="selective" if recompute_modules else None,
        # MoE
        num_moe_experts=num_experts,
        moe_grouped_gemm=(num_experts is not None),
        gated_linear_unit=use_transformer_engine_op_fuser,
        # The TE operation-fuser GroupedLinear reduces its bias gradients with fp32 atomic
        # adds across row splits (TE refuses that kernel under NVTE_ALLOW_NONDETERMINISTIC_ALGO=0),
        # so two otherwise identical runs can round a BF16 bias gradient to neighbouring values
        # (one BF16 ulp is 0.25 at |grad| ~ 40, far above this test's 1e-3 tolerance). Keep the
        # op-fuser model bias-free so the baseline-vs-offload comparison below only sees the
        # deterministic GEMMs; biases are parameters, not activations, so the offload boundary
        # under test is unchanged.
        add_bias_linear=not use_transformer_engine_op_fuser,
        activation_func=(
            torch.nn.functional.silu
            if use_transformer_engine_op_fuser
            else torch.nn.functional.gelu
        ),
        # Fine-grained activation offloading
        fine_grained_activation_offloading=fine_grained_activation_offloading,
        offload_modules=offload_modules,
        min_offloaded_tensor_size=min_offloaded_tensor_size,
        activation_offload_fraction=activation_offload_fraction,
        fine_grained_offloading_buffer_size_gib=fine_grained_offloading_buffer_size_gib,
        use_transformer_engine_op_fuser=use_transformer_engine_op_fuser,
        # Hyper Connection settings
        enable_hyper_connections=enable_hyper_connections,
        num_residual_streams=num_residual_streams,
        mhc_recompute_layer_num=mhc_recompute_layer_num,
        # Attention Residual settings
        enable_attention_residuals=enable_attention_residuals,
        attn_res_block_layers=attn_res_block_layers if enable_attention_residuals else None,
        attn_res_impl=attn_res_impl,
        mtp_num_layers=mtp_num_layers,
    )
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_experts,
        moe_grouped_gemm=num_experts is not None,
        multi_latent_attention=is_mla,
        qk_layernorm=qk_layernorm and not is_mla,
        use_te_op_fuser=use_transformer_engine_op_fuser,
        enable_hyper_connection=enable_hyper_connections,
        enable_attention_residual=enable_attention_residuals,
    )
    mtp_block_spec = (
        get_gpt_mtp_block_spec(
            config=transformer_config, spec=transformer_layer_spec, use_transformer_engine=True
        )
        if mtp_num_layers is not None
        else None
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
    ).bfloat16()
    return gpt_model


def _make_gpt_inputs(
    *, seq_length: int, micro_batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = list(range(seq_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
    attention_mask = torch.ones((micro_batch_size, 1, seq_length, seq_length), dtype=bool).to(
        device
    )
    return input_ids, position_ids, attention_mask


def _capture_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    params: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        params[name] = p.detach().cpu().clone()
    return params


def _restore_params(model: torch.nn.Module, params: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(params[name].to(device=p.device, dtype=p.dtype))


def _expected_unique_offload_bytes(manager, offload_modules):
    # For these PP=1 tests, the manager's summary window ends at the first
    # backward chunk, including when the schedule runs multiple microbatches.
    # Multiple copies of one storage increase PCIe traffic, but that GPU
    # allocation can only be freed once. Compare memory savings to unique bytes.
    chunk = manager._cached_chunks_backward[0]
    return sum(
        group.total_offload_bytes - group.duplicate_storage_bytes
        for group in chunk.offload_groups
        if group.offload and group._name in offload_modules
    )


@pytest.mark.parametrize("num_chunks", [1, 2])
def test_expected_unique_offload_bytes_uses_first_backward_chunk(num_chunks):
    def group(name, offload, total_bytes, duplicate_bytes=0):
        return SimpleNamespace(
            _name=name,
            offload=offload,
            total_offload_bytes=total_bytes,
            duplicate_storage_bytes=duplicate_bytes,
        )

    first_chunk = SimpleNamespace(
        offload_groups=[
            group("core_attn", True, 100, 20),
            group("core_attn", False, 900),
            group("attn_proj", True, 700),
        ]
    )
    second_chunk = SimpleNamespace(offload_groups=[group("core_attn", True, 500)])
    chunks = [first_chunk, second_chunk][:num_chunks]
    manager = SimpleNamespace(_cached_chunks_forward=chunks, _cached_chunks_backward=chunks)

    assert _expected_unique_offload_bytes(manager, ["core_attn"]) == 80


def _run_one_iter_and_capture(
    model: GPTModel,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    enable_offload_reset: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
    """
    Run a single forward+backward iteration.

    Returns:
      - logits (CPU float32)
      - selected grads (CPU float32)
      - peak_memory_allocated (bytes) during the iteration
    """

    if enable_offload_reset:
        off_interface.reset()

    # Keep warmup-created grad buffers resident so the peak-memory check still
    # compares the steady-state allocator footprint, but remove accumulated
    # warmup values before capturing correctness grads.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    torch.cuda.reset_peak_memory_stats()
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
    loss = logits.float().sum()
    loss.backward()
    torch.cuda.synchronize()
    peak_bytes = int(torch.cuda.max_memory_allocated())

    # capture all gradients for correctness
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        grads[name] = p.grad.detach().float().cpu() if p.grad is not None else None

    return logits.detach().float().cpu(), grads, peak_bytes


def _build_one_layer_vpp_models(*, seed: int) -> List[GPTModel]:
    """Build PP=2/VPP=2 GPT chunks with exactly one TransformerLayer per chunk."""
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType

    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        fine_grained_activation_offloading=True,
        offload_modules=["core_attn", "attn_proj"],
        min_offloaded_tensor_size=1,
    )

    models = []
    for vp_stage in range(2):
        model = (
            GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=128,
                max_sequence_length=32,
                pre_process=parallel_state.is_pipeline_first_stage(
                    ignore_virtual=False, vp_stage=vp_stage
                ),
                post_process=parallel_state.is_pipeline_last_stage(
                    ignore_virtual=False, vp_stage=vp_stage
                ),
                position_embedding_type="rope",
                vp_stage=vp_stage,
                share_embeddings_and_output_weights=False,
            )
            .bfloat16()
            .cuda()
        )
        model.model_type = ModelType.encoder_or_decoder
        model.train()
        assert len(model.decoder.layers) == 1
        models.append(model)
    return models


def _run_one_vpp_iteration(models: List[GPTModel]) -> List[Dict[str, torch.Tensor]]:
    """Run one complete interleaved pipeline forward/backward iteration."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    seq_length = 32
    micro_batch_size = 1
    num_microbatches = 4

    def forward_step_func(data_iterator, model):
        next(data_iterator)
        tokens = torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0)
        labels = tokens.clone()
        output = model(tokens, position_ids, None, labels=labels)

        def loss_func(output_tensor):
            loss = output_tensor.float().mean()
            return loss, {"lm loss": loss.detach()}

        return output, loss_func

    losses = get_forward_backward_func()(
        forward_step_func=forward_step_func,
        data_iterator=[iter(range(num_microbatches)) for _ in models],
        model=models,
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
    )
    torch.cuda.synchronize()
    return losses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_one_layer_vpp_chunk_runs_full_iteration_with_activation_offload():
    """Do not advance a one-layer VPP chunk before its later offload group.

    The warmup iteration records the real group layout for four TransformerLayers.
    The second iteration executes the PP=2/VPP=2 interleaved 1F1B schedule, including
    the immediate backward whose preceding virtual chunk contains only one layer.
    """
    from megatron.core import parallel_state

    if Utils.world_size < 2 or Utils.world_size % 2 != 0:
        pytest.skip("PP=2 requires an even WORLD_SIZE of at least 2.")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    off_interface.reset_instance()

    try:
        models = _build_one_layer_vpp_models(seed=123)

        # Build the real cached ChunkOffloadHandlers, then run one steady-state iteration.
        _run_one_vpp_iteration(models)
        for model in models:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        losses = _run_one_vpp_iteration(models)

        grads = [
            param.grad for model in models for param in model.parameters() if param.grad is not None
        ]
        assert grads, "Expected the interleaved schedule to produce parameter gradients."
        assert all(torch.isfinite(grad).all() for grad in grads)
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            assert len(losses) == 4
            assert all(torch.isfinite(item["lm loss"]) for item in losses)
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("offload_core_attn", [False, True])
def test_packed_qkv_offload_requires_unpacked_inputs(fraction, offload_core_attn):
    """Reject packed inputs only when core attention can actually be offloaded."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    off_interface.reset_instance()
    try:
        model = _build_gpt_model(
            seed=123,
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            vocab_size=1024,
            seq_length=128,
            num_experts=None,
            fine_grained_activation_offloading=True,
            offload_modules=["core_attn", "attn_proj"] if offload_core_attn else ["qkv_linear"],
            min_offloaded_tensor_size=1024,
            is_mla=False,
            activation_offload_fraction=fraction,
            attention_backend=AttnBackend.fused,
            qk_layernorm=False,
        ).cuda()
        input_ids, position_ids, attention_mask = _make_gpt_inputs(
            seq_length=128, micro_batch_size=1, device=torch.device("cuda")
        )
        off_interface.reset()
        if offload_core_attn and fraction > 0:
            with pytest.raises(ValueError, match="core_attn offloading does not support packed"):
                model(input_ids, position_ids, attention_mask)
        else:
            output = model(input_ids, position_ids, attention_mask)
            output.float().sum().backward()
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize("attention_backend", [AttnBackend.fused, AttnBackend.flash])
def test_partial_offload_warmup_preserves_attention_output(monkeypatch, attention_backend):
    """Projection must not free the output retained by fused/FlashAttention backward."""
    from transformer_engine.pytorch.attention.dot_product_attention import dot_product_attention

    # TE caches backend selection across models; changing only the environment
    # can otherwise make both parameters exercise the same cached backend.
    monkeypatch.setattr(
        dot_product_attention,
        "_attention_backends",
        dict(dot_product_attention._attention_backends, attention_params=None),
    )
    for name in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    off_interface.reset_instance()
    try:
        inputs = dict(
            zip(
                ("input_ids", "position_ids", "attention_mask"),
                _make_gpt_inputs(seq_length=128, micro_batch_size=1, device=torch.device("cuda")),
            )
        )
        model_args = dict(
            seed=123,
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            vocab_size=1024,
            seq_length=128,
            num_experts=None,
            min_offloaded_tensor_size=1024,
            is_mla=False,
            attention_backend=attention_backend,
        )
        baseline = _build_gpt_model(
            **model_args, fine_grained_activation_offloading=False, offload_modules=None
        ).cuda()
        params = _capture_params(baseline)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        expected_output, expected_grads, _ = _run_one_iter_and_capture(
            baseline, **inputs, enable_offload_reset=False
        )
        backend_key = (
            "use_fused_attention"
            if attention_backend == AttnBackend.fused
            else "use_flash_attention"
        )
        assert dot_product_attention._attention_backends[backend_key]
        del baseline
        model = _build_gpt_model(
            **model_args,
            fine_grained_activation_offloading=True,
            offload_modules=["qkv_linear", "core_attn", "attn_proj"],
            activation_offload_fraction=0.5,
        ).cuda()
        _restore_params(model, params)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        output, grads, _ = _run_one_iter_and_capture(model, **inputs, enable_offload_reset=True)
        torch.testing.assert_close(output, expected_output, rtol=1e-3, atol=1e-3)
        assert grads.keys() == expected_grads.keys()
        for name, expected_grad in expected_grads.items():
            if expected_grad is None:
                assert grads[name] is None
            else:
                torch.testing.assert_close(
                    grads[name],
                    expected_grad,
                    rtol=1e-3,
                    atol=1e-3,
                    msg=lambda msg, name=name: f"{name}: {msg}",
                )
    finally:
        torch.cuda.synchronize()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize(
    "is_moe, is_mla, offload_modules, activation_offload_fraction, buffer_size_gib, "
    "enable_attention_residuals, attn_res_impl, mtp_num_layers, recompute_modules",
    [
        # Dense GPT modules
        (False, True, ["attn_norm"], 1.0, 0.0, False, "eager", None, None),
        (True, False, ["qkv_linear"], 1.0, 0.0, False, "eager", None, None),
        (True, False, ["core_attn"], 1.0, 0.0, False, "eager", None, None),
        # # attn_proj depends on core_attn (validated in TransformerConfig.__post_init__)
        (True, True, ["core_attn", "attn_proj"], 1.0, 0.0, False, "eager", None, None),
        (True, False, ["mlp_norm"], 1.0, 0.0, False, "eager", None, None),
        (True, False, ["expert_fc1"], 1.0, 0.0, False, "eager", None, None),
        (True, False, ["moe_act"], 1.0, 0.0, False, "eager", None, None),
        # One eager partial-offload case across attention and MoE groups.
        (
            True,
            False,
            ["core_attn", "attn_proj", "expert_fc1"],
            0.5,
            0.0,
            False,
            "eager",
            None,
            None,
        ),
        # Non-power-of-two pinned storage with fixed and variable activation shapes.
        (False, True, ["attn_norm"], 1.0, 1.5, False, "eager", None, None),
        (True, False, ["moe_act"], 1.0, 1.5, False, "eager", None, None),
        (
            True,
            False,
            ["core_attn", "attn_proj", "expert_fc1"],
            1.0,
            1.5,
            False,
            "eager",
            None,
            None,
        ),
        # Attention Residuals with every non-fused offload boundary enabled together.
        pytest.param(
            True,
            True,
            [
                "attn_norm",
                "qkv_linear",
                "core_attn",
                "attn_proj",
                "mlp_norm",
                "expert_fc1",
                "moe_act",
            ],
            1.0,
            0.0,
            True,
            "eager",
            None,
            [],
            id="attnres-all-non-fused-eager",
        ),
        pytest.param(
            True,
            True,
            [
                "attn_norm",
                "qkv_linear",
                "core_attn",
                "attn_proj",
                "mlp_norm",
                "expert_fc1",
                "moe_act",
            ],
            1.0,
            0.0,
            True,
            "compile",
            None,
            [],
            id="attnres-all-non-fused-compile",
        ),
        pytest.param(
            True,
            True,
            [
                "attn_norm",
                "qkv_linear",
                "core_attn",
                "attn_proj",
                "mlp_norm",
                "expert_fc1",
                "moe_act",
            ],
            1.0,
            0.0,
            True,
            "fla",
            None,
            [],
            id="attnres-all-non-fused-fla",
        ),
        # The whole fused grouped MLP boundary is mutually exclusive with expert_fc1/moe_act.
        pytest.param(
            True,
            False,
            ["fused_group_mlp"],
            1.0,
            0.0,
            True,
            "fla",
            None,
            [],
            id="attnres-fused-group-mlp",
        ),
        # Two MTP depths exercise the longer-lived trunk source tuple with all non-fused groups.
        pytest.param(
            True,
            True,
            [
                "attn_norm",
                "qkv_linear",
                "core_attn",
                "attn_proj",
                "mlp_norm",
                "expert_fc1",
                "moe_act",
            ],
            1.0,
            0.0,
            True,
            "eager",
            2,
            ["layernorm", "moe_act"],
            id="attnres-mtp2-all-non-fused",
        ),
        pytest.param(
            True,
            True,
            [
                "attn_norm",
                "qkv_linear",
                "core_attn",
                "attn_proj",
                "mlp_norm",
                "expert_fc1",
                "moe_act",
            ],
            1.0,
            0.0,
            True,
            "fla",
            None,
            ["layernorm", "moe_act"],
            id="attnres-all-non-fused-selective-recompute",
        ),
    ],
)
def test_gpt_fine_grained_activation_offloading_correctness_and_memory(
    is_moe: bool,
    is_mla: bool,
    offload_modules: List[str],
    activation_offload_fraction: float,
    buffer_size_gib: float,
    enable_attention_residuals: bool,
    attn_res_impl: str,
    mtp_num_layers: Optional[int],
    recompute_modules: Optional[List[str]],
    monkeypatch,
):
    """
    Initialize a GPTModel and verify:
    - forward output correctness under each offload_modules setting
    - backward gradient correctness (subset)
    - peak GPU memory is reduced roughly as expected (based on recorded offload bytes)
    """
    if attn_res_impl == "fla":
        from megatron.core.transformer.attention_residual import _get_fla_fused_attnres

        if _get_fla_fused_attnres() is None:
            pytest.skip("FLA fused AttnRes is not installed")

    # setup distributed/model-parallel (same pattern as other UTs)
    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)
    # os.environ["NVTE_FLASH_ATTN"] = "1"
    _enable_te_cpu_offload_v1(monkeypatch)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    seed = 123
    # Choose shapes large enough to make memory deltas stable but still fast.
    num_experts = 4 if is_moe else None
    num_layers = 8
    hidden_size = 2048 if num_experts is None else 1024
    num_attention_heads = 16 if hidden_size >= 2048 else 8
    vocab_size = 1024
    seq_length = 1024
    micro_batch_size = 2
    device = torch.device("cuda")
    use_transformer_engine_op_fuser = "fused_group_mlp" in offload_modules
    if use_transformer_engine_op_fuser:
        if not is_te_min_version("2.14.0"):
            pytest.skip("TE operation fuser requires transformer-engine>=2.14.0")
        try:
            from transformer_engine.pytorch.ops import GroupedLinear  # noqa: F401
        except ImportError:
            pytest.skip("Installed Transformer Engine does not provide the operation-fuser API")
        monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")

    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )

    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    off_interface.reset_instance()

    # Shape-level record of what the qkv_linear group offloads during warm-up,
    # taken before bulk_offload_group replaces the group's tensors by CPU states.
    qkv_linear_offloaded_shapes = set()
    real_bulk_offload_group = off.ChunkOffloadHandler.bulk_offload_group

    def spying_bulk_offload_group(handler, group_to_offload):
        if handler.is_warmup and group_to_offload._name == "qkv_linear":
            for tensor in group_to_offload._tensors.values():
                if isinstance(tensor, torch.Tensor) and handler.tensor_need_offloading_checker(
                    tensor
                ):
                    qkv_linear_offloaded_shapes.add(tuple(tensor.shape))
        return real_bulk_offload_group(handler, group_to_offload)

    monkeypatch.setattr(off.ChunkOffloadHandler, "bulk_offload_group", spying_bulk_offload_group)

    try:
        # 1) Baseline run (no offloading)
        _reset_cuda_memory()
        base_model = _build_gpt_model(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=False,
            offload_modules=None,
            min_offloaded_tensor_size=1024 * 1024,
            is_mla=is_mla,
            enable_attention_residuals=enable_attention_residuals,
            attn_res_impl=attn_res_impl,
            mtp_num_layers=mtp_num_layers,
            use_transformer_engine_op_fuser=use_transformer_engine_op_fuser,
            recompute_modules=recompute_modules,
        ).cuda()
        base_model.train()
        base_params = _capture_params(base_model)

        # Warmup baseline once for allocator stability
        _run_one_iter_and_capture(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=False,
        )
        _reset_cuda_memory()
        base_logits, base_grads, base_peak = _run_one_iter_and_capture(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=False,
        )
        # Free baseline model GPU memory before offload path
        del base_model
        _reset_cuda_memory()

        # 2) Offload run (warmup to record bytes + steady-state measurement)
        off_model = _build_gpt_model(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
            min_offloaded_tensor_size=1024,  # force offloading for UT determinism
            is_mla=is_mla,
            activation_offload_fraction=activation_offload_fraction,
            fine_grained_offloading_buffer_size_gib=buffer_size_gib,
            enable_attention_residuals=enable_attention_residuals,
            attn_res_impl=attn_res_impl,
            mtp_num_layers=mtp_num_layers,
            use_transformer_engine_op_fuser=use_transformer_engine_op_fuser,
            recompute_modules=recompute_modules,
        ).cuda()
        _restore_params(off_model, base_params)
        off_model.train()

        # Warmup 1 iter to populate cached chunks, then reset to finish warmup bookkeeping.
        _run_one_iter_and_capture(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=True,
        )
        # Reset once more to trigger post_warmup_callback and apply steady-state offload decisions.
        off_interface.reset()

        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            PipelineOffloadManager,
        )

        mgr = PipelineOffloadManager.get_instance()
        # The summary records GroupedMLP's expert_fc1 + moe_act pair under the single
        # "fused_group_mlp" group, so check the recorded group names, not the request.
        for module in _offload_summary_groups(offload_modules):
            if (
                enable_attention_residuals
                and module == "qkv_linear"
                and "layernorm" in off_model.config.recompute_modules
            ):
                # AttnRes uses an explicit norm: checkpointing its output makes
                # the QKV input [s, b, h] recompute-owned rather than
                # offload-owned, so that shape must not be among the group's
                # offloads. MLA still legitimately offloads its own saved
                # activations in this group (the down-projected latents
                # [s, b, lora_rank] and the RoPE tables [s, 1, 1, rotary_dim]),
                # so its byte count is > 0 while a plain QKV linear saves only
                # the recompute-owned input and records 0.
                assert (
                    seq_length,
                    micro_batch_size,
                    hidden_size,
                ) not in qkv_linear_offloaded_shapes, (
                    f"qkv_linear offloaded the recompute-owned QKV input: "
                    f"{sorted(qkv_linear_offloaded_shapes)}"
                )
                if is_mla:
                    assert mgr.offload_summary_bytes.get(module, 0) > 0, (
                        f"Expected MLA {module} to offload its latents/RoPE tables, got "
                        f"{mgr.offload_summary_bytes.get(module, 0)} bytes"
                    )
                else:
                    assert mgr.offload_summary_bytes.get(module, 0) == 0
                continue
            assert mgr.offload_summary_bytes.get(module, 0) > 0, (
                f"Expected {module} to offload at least one activation, got "
                f"{mgr.offload_summary_bytes.get(module, 0)} bytes"
            )
        # A storage copied twice within one group (MLA's q/kv down projections both
        # save hidden_states) frees GPU memory only once, so expect the per-storage
        # saving.
        expected_offload_bytes = _expected_unique_offload_bytes(
            mgr, _offload_summary_groups(offload_modules)
        )
        expected_offload_mib = expected_offload_bytes / (1024**2)
        pool = mgr.cpu_tensor_pool
        assert pool.uses_fixed_buffer == bool(buffer_size_gib)
        if buffer_size_gib:
            fixed = pool.get_pool_status()["fixed_buffer"]
            assert fixed["capacity_bytes"] == int(buffer_size_gib * 2**30)
            assert fixed["peak_live_bytes"] > 0

        _reset_cuda_memory()
        off_logits, off_grads, off_peak = _run_one_iter_and_capture(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=True,
        )
        if buffer_size_gib:
            assert pool.get_pool_status()["fixed_buffer"]["live_bytes"] == 0
        del off_model
        _reset_cuda_memory()

        # 3) Correctness checks (forward + selected grads)
        assert torch.allclose(off_logits, base_logits, rtol=1e-3, atol=1e-3)
        assert set(off_grads.keys()) == set(base_grads.keys())
        for name, gb in base_grads.items():
            go = off_grads[name]
            if gb is None or go is None:
                assert gb is None and go is None, f"Grad None mismatch for {name}"
                continue
            assert torch.allclose(go, gb, rtol=1e-3, atol=1e-3), f"Grad mismatch for {name}"

        # 4) Memory checks (peak allocated over forward+backward)
        saved_mib = (base_peak - off_peak) / (1024**2)
        assert saved_mib > 0.0, (
            f"Expected GPU peak memory reduction for offload_modules={offload_modules}, "
            f"but got saved={saved_mib:.2f}MiB (base={base_peak/(1024**2):.2f}MiB, "
            f"off={off_peak/(1024**2):.2f}MiB)"
        )

        # If expectation is large enough, enforce approximate match.
        # For tiny expectations, allocator noise may dominate; we only require a positive reduction.
        if expected_offload_mib >= 2.0:
            rel_err = abs(saved_mib - expected_offload_mib) / max(expected_offload_mib, 1e-6)
            abs_err = abs(saved_mib - expected_offload_mib)
            assert rel_err <= EPSILON and abs_err <= DELTA, (
                f"Memory saving mismatch for offload_modules={offload_modules}: "
                f"saved={saved_mib:.2f}MiB expected~={expected_offload_mib:.2f}MiB "
                f"(rel_err={rel_err:.2f}, abs_err={abs_err:.2f})"
            )
            print(
                f"Rank {torch.distributed.get_rank()}: Saved {saved_mib:.2f}MiB, expected {expected_offload_mib:.2f}MiB"
            )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.flaky_in_dev
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("1.9.0.dev0"),
    reason="EP A2A overlap requires TE 1.9.0.dev0+ in this repo's tests.",
)
@pytest.mark.parametrize(
    "dispatcher_backend, is_mla, offload_modules",
    [
        ("alltoall", True, ["attn_norm"]),
        ("alltoall", True, ["core_attn"]),
        ("alltoall", True, ["attn_norm", "core_attn", "attn_proj"]),
        ("alltoall", True, ["mlp_norm"]),
        ("alltoall", False, ["expert_fc1"]),
        ("alltoall", False, ["moe_act"]),
        (
            "alltoall",
            True,
            ["attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act"],
        ),
        (
            "alltoall",
            False,
            ["attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act"],
        ),
    ],
)
def test_fine_grained_activation_offload_with_ep_a2a_overlap_compatibility(
    dispatcher_backend: str, is_mla: bool, offload_modules: List[str]
):
    """
    Compatibility test for:
      - fine-grained activation offloading
      - EP all-to-all overlap (overlap_moe_expert_parallel_comm)
      - memory saving roughly matches expected offload bytes (when expectation is large enough)

    The EP A2A overlap initialization pattern is aligned with
    `tests/unit_tests/a2a_overlap/test_schedule_chunk_1f1b.py`.
    """
    from megatron.core.models.common.model_chunk_schedule_plan import (
        TransformerModelChunkSchedulePlan,
    )
    from megatron.core.pipeline_parallel.utils import set_streams
    from tests.unit_tests.a2a_overlap.utils import deterministic_mode

    # EP overlap requires distributed initialization with EP groups.
    ep_size = 4
    if Utils.world_size % ep_size != 0:
        pytest.skip(
            f"Skipping: WORLD_SIZE={Utils.world_size} must be divisible by ep_size={ep_size}."
        )

    seed = 123
    num_experts = 8  # must be divisible by ep_size
    if num_experts % ep_size != 0:
        pytest.skip(
            f"Skipping: num_moe_experts={num_experts} must be divisible by ep_size={ep_size}."
        )

    # Small shapes to keep this compatibility test fast.
    num_layers = 8
    hidden_size = 1024
    num_attention_heads = 16
    vocab_size = 1024
    seq_length = 1024
    micro_batch_size = 2
    device = torch.device("cuda")

    from megatron.core.pipeline_parallel import fine_grained_activation_offload as off

    def _make_schedule_inputs() -> Dict[str, torch.Tensor]:
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
        position_ids = (
            torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).to(device)
        )
        attention_mask = torch.ones((micro_batch_size, 1, seq_length, seq_length), dtype=bool).to(
            device
        )
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def _capture_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        params: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            params[name] = p.detach().clone()
        return params

    def _restore_params(model: torch.nn.Module, params: Dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            p.data.copy_(params[name])

    def _build_overlap_moe_gpt(
        *, enable_offload: bool, is_mla: bool, dispatcher_backend: str
    ) -> GPTModel:
        model_parallel_cuda_manual_seed(seed)
        torch.manual_seed(seed)
        ConfigClass = MLATransformerConfig if is_mla else TransformerConfig
        transformer_config = ConfigClass(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            attention_backend=AttnBackend.unfused,
            qk_layernorm=not is_mla,
            # Recompute
            recompute_modules=["layernorm", "moe_act"],
            recompute_granularity="selective",
            bf16=True,
            params_dtype=torch.bfloat16,
            # MoE + EP overlap
            num_moe_experts=num_experts,
            moe_grouped_gemm=True,
            expert_model_parallel_size=ep_size,
            moe_token_dispatcher_type="alltoall" if dispatcher_backend == "alltoall" else "flex",
            moe_flex_dispatcher_backend=dispatcher_backend,
            moe_router_dtype="fp32" if dispatcher_backend == "hybridep" else "fp64",
            overlap_moe_expert_parallel_comm=True,
            delay_wgrad_compute=True,
            # Fine-grained activation offloading
            fine_grained_activation_offloading=enable_offload,
            offload_modules=offload_modules if enable_offload else None,
            min_offloaded_tensor_size=1024,  # force offloading to exercise the code path
        )
        return (
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                    num_experts=num_experts,
                    moe_grouped_gemm=True,
                    multi_latent_attention=is_mla,
                    qk_layernorm=not is_mla,
                ),
                vocab_size=vocab_size,
                max_sequence_length=seq_length,
            )
            .bfloat16()
            .cuda()
        )

    def _run_schedule_1f1b_two_microbatches(
        model: GPTModel, *, enable_offload_reset: bool
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], int]:
        """
        Run a minimal 1F1B schedule (2 microbatches) using ModelChunkSchedulePlan.run().
        This is the execution path that exercises EP A2A overlap scheduling.
        """
        if enable_offload_reset:
            off_interface.reset()

        data0 = _make_schedule_inputs()
        data1 = _make_schedule_inputs()
        plan0 = model.build_schedule_plan(**data0)

        torch.cuda.reset_peak_memory_stats()
        out0 = TransformerModelChunkSchedulePlan.run(plan0, None)
        plan1 = model.build_schedule_plan(**data1)
        out1 = TransformerModelChunkSchedulePlan.run(plan1, plan0, b_grad=torch.ones_like(out0))
        TransformerModelChunkSchedulePlan.run(None, plan1, b_grad=torch.ones_like(out1))
        torch.cuda.synchronize()
        peak_bytes = int(torch.cuda.max_memory_allocated())

        # capture outputs and grads
        outputs = [out0.detach().float().cpu(), out1.detach().float().cpu()]
        grads: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            grads[name] = p.grad.detach().float().cpu() if p.grad is not None else None
        return outputs, grads, peak_bytes

    # setup distributed/model-parallel
    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )
    set_streams()

    off_interface.reset_instance()

    try:
        with deterministic_mode():
            # Baseline: EP overlap on, offload off.
            _reset_cuda_memory()
            base_model = _build_overlap_moe_gpt(
                enable_offload=False, is_mla=is_mla, dispatcher_backend=dispatcher_backend
            )
            base_model.train()
            base_params = _capture_params(base_model)
            # Warmup once for allocator stability / graph caching
            _run_schedule_1f1b_two_microbatches(base_model, enable_offload_reset=False)
            _reset_cuda_memory()
            base_outs, base_grads, base_peak = _run_schedule_1f1b_two_microbatches(
                base_model, enable_offload_reset=False
            )
            del base_model
            _reset_cuda_memory()

            # Offload: EP overlap on, fine-grained offload on.
            off_model = _build_overlap_moe_gpt(
                enable_offload=True, is_mla=is_mla, dispatcher_backend=dispatcher_backend
            )
            _restore_params(off_model, base_params)
            off_model.train()
            # Warmup once to populate cached chunks, then reset to apply steady-state offload decisions.
            off_interface.reset()
            _run_schedule_1f1b_two_microbatches(off_model, enable_offload_reset=False)
            off_interface.reset()
            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                PipelineOffloadManager,
            )

            mgr = PipelineOffloadManager.get_instance()
            expected_offload_bytes = _expected_unique_offload_bytes(
                mgr, _offload_summary_groups(offload_modules)
            )
            expected_offload_mib = expected_offload_bytes / (1024**2)

            _reset_cuda_memory()
            off_outs, off_grads, off_peak = _run_schedule_1f1b_two_microbatches(
                off_model, enable_offload_reset=True
            )
            del off_model
            _reset_cuda_memory()

            # Correctness (forward outputs + all grads)
            assert len(off_outs) == len(base_outs) == 2
            for i in range(2):
                assert torch.allclose(off_outs[i], base_outs[i], rtol=1e-3, atol=1e-3)
            assert set(off_grads.keys()) == set(base_grads.keys())
            for name, gb in base_grads.items():
                go = off_grads[name]
                if gb is None or go is None:
                    assert gb is None and go is None, f"Grad None mismatch for {name}"
                    continue
                assert torch.allclose(
                    go, gb, rtol=1e-3, atol=1e-3
                ), f"Rank {torch.distributed.get_rank()}: Grad mismatch for {name}"

            # Memory checks (peak allocated during the scheduled 1F1B run)
            saved_mib = (base_peak - off_peak) / (1024**2)
            assert saved_mib > 0.0, (
                f"Expected GPU peak memory reduction for offload_modules={offload_modules}, "
                f"but got saved={saved_mib:.2f}MiB (base={base_peak/(1024**2):.2f}MiB, "
                f"off={off_peak/(1024**2):.2f}MiB)"
            )
            # If expectation is large enough, enforce approximate match.
            if expected_offload_mib >= 2.0:
                rel_err = abs(saved_mib - expected_offload_mib) / max(expected_offload_mib, 1e-6)
                abs_err = abs(saved_mib - expected_offload_mib)
                print(
                    f"Rank {torch.distributed.get_rank()}: Saved {saved_mib:.2f}MiB, expected {expected_offload_mib:.2f}MiB"
                )
                if abs_err > DELTA:
                    assert rel_err <= EPSILON_A2A, (
                        f"Memory saving mismatch for offload_modules={offload_modules}: "
                        f"saved={saved_mib:.2f}MiB expected~={expected_offload_mib:.2f}MiB "
                        f"(rel_err={rel_err:.2f}, abs_err={abs_err:.2f})"
                    )
    finally:
        Utils.destroy_model_parallel()


# =============================================================================
# CUDA Graph + Fine-grained Activation Offloading Tests
# =============================================================================


def _build_gpt_model_with_cuda_graph(
    *,
    seed: int,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    vocab_size: int,
    seq_length: int,
    num_experts: Optional[int],
    fine_grained_activation_offloading: bool,
    offload_modules: Optional[List[str]],
    min_offloaded_tensor_size: int,
    is_mla: bool,
    cuda_graph_impl: str,
    cuda_graph_modules: Optional[List[str]],
    cuda_graph_warmup_steps: int,
    delay_offload_until_cuda_graph: bool = False,
    activation_offload_fraction: float = 1.0,
    enable_hyper_connections: bool = False,
    num_residual_streams: int = 4,
    position_embedding_type: str = "learned_absolute",
    config_attrs: Optional[Dict[str, object]] = None,
    **config_overrides,
) -> GPTModel:
    """Build a GPTModel with CUDA Graph support and fine-grained activation offloading.

    ``config_overrides`` are applied last on top of the defaults below; ``config_attrs`` are set
    on the built config (yarn's parameters are attributes, not TransformerConfig fields)."""
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    ConfigClass = MLATransformerConfig if is_mla else TransformerConfig
    recompute_modules = ["layernorm", "moe_act"] if num_experts is not None else ["layernorm"]
    config_kwargs = dict(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.unfused,
        qk_layernorm=not is_mla,
        bf16=True,
        params_dtype=torch.bfloat16,
        # Recompute (note: "mhc" recompute is incompatible with offloading)
        recompute_modules=recompute_modules,
        recompute_granularity="selective",
        # MoE
        num_moe_experts=num_experts,
        moe_grouped_gemm=(num_experts is not None),
        # Fine-grained activation offloading
        fine_grained_activation_offloading=fine_grained_activation_offloading,
        offload_modules=offload_modules,
        min_offloaded_tensor_size=min_offloaded_tensor_size,
        delay_offload_until_cuda_graph=delay_offload_until_cuda_graph,
        activation_offload_fraction=activation_offload_fraction,
        # CUDA Graph settings
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=cuda_graph_modules,
        cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        use_te_rng_tracker=True,
        # Hyper Connection settings
        enable_hyper_connections=enable_hyper_connections,
        num_residual_streams=num_residual_streams,
    )
    config_kwargs.update(config_overrides)
    transformer_config = ConfigClass(**config_kwargs)
    for name, value in (config_attrs or {}).items():
        setattr(transformer_config, name, value)
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=num_experts is not None,
            multi_latent_attention=is_mla,
            qk_layernorm=not is_mla,
            enable_hyper_connection=enable_hyper_connections,
        ),
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
        position_embedding_type=position_embedding_type,
    ).bfloat16()
    return gpt_model


def _run_iters_with_cuda_graph(
    model: GPTModel,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_warmup_iters: int,
    num_measure_iters: int,
    enable_offload_reset: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
    """
    Run multiple forward+backward iterations with CUDA graph capture.

    Returns:
      - logits from last iteration (CPU float32)
      - selected grads from last iteration (CPU float32)
      - peak_memory_allocated (bytes) during measurement iterations
    """
    from megatron.core.num_microbatches_calculator import (
        destroy_num_microbatches_calculator,
        init_num_microbatches_calculator,
    )
    from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

    # Switch to a non-default stream so AccumulateGrad nodes (created during warmup backward
    # passes) are associated with this stream rather than the default stream.  If they are on
    # the default stream, CUDA graph capture will fail with cudaErrorStreamCaptureImplicit.
    te_side_stream = torch.cuda.Stream()
    te_side_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(te_side_stream)

    micro_batch_size = input_ids.shape[0]
    init_num_microbatches_calculator(
        rank=0,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
        decrease_batch_size_if_needed=False,
    )

    if enable_offload_reset:
        off_interface.reset()

    # Warmup iterations (before CUDA graph capture)
    for _ in range(num_warmup_iters):
        if enable_offload_reset:
            off_interface.reset()
        logits = model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        loss = logits.float().sum()
        loss.backward()
        # Zero grads for next iteration
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # Trigger post-warmup offload decisions
    if enable_offload_reset:
        off_interface.reset()

    # TECudaGraphHelper expects model chunks to have zero_grad_buffer() (from DDP wrappers).
    # For a plain GPTModel, this is a no-op (same as the DataParallelBase base implementation).
    if not hasattr(model, 'zero_grad_buffer'):
        model.zero_grad_buffer = lambda: None

    # Create CUDA graphs after warmup
    cuda_graph_helper = TECudaGraphHelper(
        model=[model],
        config=model.config,
        seq_length=input_ids.shape[1],
        micro_batch_size=input_ids.shape[0],
        optimizers=[],
    )
    cuda_graph_helper.create_cudagraphs()

    # Measurement iterations (with CUDA graph replay)
    torch.cuda.reset_peak_memory_stats()
    for i in range(num_measure_iters):
        if enable_offload_reset:
            off_interface.reset()
        logits = model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        loss = logits.float().sum()
        loss.backward()
        if i < num_measure_iters - 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    torch.cuda.synchronize()
    peak_bytes = int(torch.cuda.max_memory_allocated())

    # Capture grads from last iteration
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        grads[name] = p.grad.detach().float().cpu() if p.grad is not None else None

    # Cleanup CUDA graphs
    if cuda_graph_helper.graphs_created():
        cuda_graph_helper.delete_cuda_graphs()

    destroy_num_microbatches_calculator()

    return logits.detach().float().cpu(), grads, peak_bytes


def _build_partial_cg_pipeline_model(
    *, seed: int, num_layers: int, moe_layer_freq, offload_modules: List[str]
) -> GPTModel:
    """Build one PP=2 model chunk with graph-scoped attention and eager experts."""
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=256,
        num_attention_heads=8,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.unfused,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        pipeline_model_parallel_size=2,
        recompute_modules=["layernorm", "moe_act"],
        recompute_granularity="selective",
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_layer_freq=moe_layer_freq,
        fine_grained_activation_offloading=True,
        offload_modules=offload_modules,
        min_offloaded_tensor_size=1,
        delay_offload_until_cuda_graph=False,
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=["attn", "moe_router"],
        cuda_graph_warmup_steps=3,
        use_te_rng_tracker=True,
    )
    model = (
        GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            ),
            vocab_size=128,
            max_sequence_length=128,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            position_embedding_type="rope",
            share_embeddings_and_output_weights=False,
        )
        .bfloat16()
        .cuda()
    )
    model.model_type = ModelType.encoder_or_decoder
    model.train()
    model.zero_grad_buffer = lambda: None
    return model


def _run_partial_cg_pipeline_iteration(model: GPTModel) -> List[Dict[str, torch.Tensor]]:
    """Run one PP=2 pipeline iteration with four live CUDA-graph microbatch slots."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    seq_length = 128
    micro_batch_size = 1
    num_microbatches = 4

    def forward_step_func(data_iterator, model):
        microbatch_id = next(data_iterator)
        tokens = (
            (torch.arange(seq_length, dtype=torch.long, device="cuda") + microbatch_id)
            .remainder(128)
            .unsqueeze(0)
        )
        position_ids = torch.arange(seq_length, dtype=torch.long, device="cuda").unsqueeze(0)
        output = model(tokens, position_ids, None, labels=tokens.clone())

        def loss_func(output_tensor):
            loss = output_tensor.float().mean()
            return loss, {"lm loss": loss.detach()}

        return output, loss_func

    return get_forward_backward_func()(
        forward_step_func=forward_step_func,
        data_iterator=[iter(range(num_microbatches))],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=False,
    )


def _run_partial_cg_offload_case(
    *, num_layers: int, moe_layer_freq, offload_modules: List[str]
) -> None:
    """Validate one real PP=2 partial-CG/offload configuration."""
    from megatron.core import parallel_state
    from megatron.core.num_microbatches_calculator import (
        destroy_num_microbatches_calculator,
        init_num_microbatches_calculator,
    )
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        PipelineOffloadManager,
    )
    from megatron.core.tensor_parallel.random import initialize_rng_tracker
    from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
    from megatron.core.transformer.enums import CudaGraphModule
    from megatron.core.transformer.moe.moe_layer import MoELayer

    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
    off_interface.reset_instance()
    destroy_num_microbatches_calculator()
    data_parallel_size = parallel_state.get_data_parallel_world_size()
    init_num_microbatches_calculator(
        rank=torch.distributed.get_rank(),
        rampup_batch_size=None,
        global_batch_size=4 * data_parallel_size,
        micro_batch_size=1,
        data_parallel_size=data_parallel_size,
        decrease_batch_size_if_needed=False,
    )
    original_cuda_stream = torch.cuda.current_stream()
    te_side_stream = torch.cuda.Stream()
    te_side_stream.wait_stream(original_cuda_stream)
    torch.cuda.set_stream(te_side_stream)

    try:
        model = _build_partial_cg_pipeline_model(
            seed=123,
            num_layers=num_layers,
            moe_layer_freq=moe_layer_freq,
            offload_modules=offload_modules,
        )
        local_num_layers = num_layers // 2
        assert len(model.decoder.layers) == local_num_layers

        if isinstance(moe_layer_freq, int):
            global_moe_pattern = [
                1 if layer_idx % moe_layer_freq == 0 else 0 for layer_idx in range(num_layers)
            ]
        else:
            global_moe_pattern = moe_layer_freq
        local_layer_offset = parallel_state.get_pipeline_model_parallel_rank() * local_num_layers
        expected_local_moe_pattern = global_moe_pattern[
            local_layer_offset : local_layer_offset + local_num_layers
        ]
        actual_local_moe_pattern = [
            int(isinstance(layer.mlp, MoELayer)) for layer in model.decoder.layers
        ]
        assert actual_local_moe_pattern == expected_local_moe_pattern

        assert model.config.cuda_graph_modules == [CudaGraphModule.attn, CudaGraphModule.moe_router]
        attention_offload_modules = {"qkv_linear", "core_attn", "attn_proj"}
        expected_offload_in_graph = bool(attention_offload_modules & set(offload_modules))
        assert all(
            layer.offload_module_in_cuda_graph == expected_offload_in_graph
            for layer in model.decoder.layers
        )
        cuda_graph_helper = TECudaGraphHelper(
            model=[model], config=model.config, seq_length=128, micro_batch_size=1, optimizers=[]
        )

        # Record real handlers and stabilize grad buffers before graph capture.
        for _ in range(3):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            eager_losses = _run_partial_cg_pipeline_iteration(model)

        eager_grads = {
            name: param.grad.detach().float().cpu().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        eager_loss_values = [item["lm loss"].float().cpu() for item in eager_losses]
        offload_summary = PipelineOffloadManager.get_instance().offload_summary_bytes
        for offload_module in _offload_summary_groups(offload_modules):
            global_offload_bytes = torch.tensor(
                offload_summary.get(offload_module, 0), dtype=torch.int64, device="cuda"
            )
            torch.distributed.all_reduce(global_offload_bytes)
            assert global_offload_bytes.item() > 0, offload_summary

        cuda_graph_helper.create_cudagraphs()
        assert cuda_graph_helper.num_microbatches == 4
        assert cuda_graph_helper.graphs_created()

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        losses = _run_partial_cg_pipeline_iteration(model)
        torch.cuda.synchronize()

        graph_grads = {
            name: param.grad.detach().float().cpu()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        assert graph_grads.keys() == eager_grads.keys()
        for name, eager_grad in eager_grads.items():
            assert torch.allclose(graph_grads[name], eager_grad, rtol=1e-2, atol=1e-2), name
        if parallel_state.is_pipeline_last_stage():
            assert len(losses) == 4
            graph_loss_values = [item["lm loss"].float().cpu() for item in losses]
            assert all(
                torch.allclose(graph_loss, eager_loss, rtol=1e-2, atol=1e-2)
                for graph_loss, eager_loss in zip(graph_loss_values, eager_loss_values)
            )
    finally:
        if "cuda_graph_helper" in locals() and cuda_graph_helper.graphs_created():
            cuda_graph_helper.delete_cuda_graphs()
        original_cuda_stream.wait_stream(te_side_stream)
        torch.cuda.set_stream(original_cuda_stream)
        destroy_num_microbatches_calculator()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"), reason="Partial CUDA graph offloading requires TE >= 2.14.0"
)
def test_partial_cuda_graph_mixed_scope_offload_runs_full_iteration():
    """Run graph-scoped core_attn and eager moe_act offload in one real iteration."""
    _run_partial_cg_offload_case(
        num_layers=4, moe_layer_freq=1, offload_modules=["core_attn", "moe_act"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"), reason="Partial CUDA graph offloading requires TE >= 2.14.0"
)
@pytest.mark.parametrize(
    "offload_modules",
    [
        pytest.param(["moe_act"], id="moe-act-only"),
        pytest.param(["expert_fc1", "moe_act"], id="moe-stack"),
        pytest.param(["core_attn"], id="attention-core-only"),
        pytest.param(["qkv_linear", "core_attn", "attn_proj"], id="attention-stack"),
        pytest.param(["core_attn", "moe_act"], id="attention-and-moe-act"),
        pytest.param(
            ["qkv_linear", "core_attn", "attn_proj", "expert_fc1", "moe_act"],
            id="attention-and-moe-stack",
        ),
    ],
)
def test_hybrid_dense_moe_partial_cuda_graph_offload_variants(offload_modules: List[str]):
    """Exercise DeepSeek-V3-like dense-to-MoE layers across offload group subsets."""
    _run_partial_cg_offload_case(
        num_layers=6, moe_layer_freq=[0, 0, 1, 1, 1, 1], offload_modules=offload_modules
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"), reason="CUDA Graph with TE RNG tracker requires TE >= 2.13.0"
)
@pytest.mark.flaky_in_dev
@pytest.mark.parametrize(
    "is_mla, offload_modules, cuda_graph_modules, activation_offload_fraction, delay_offload",
    [
        # MoE model with attention CUDA graph + attn offloading
        (False, ["core_attn", "attn_proj"], ["attn", "moe_router"], 1.0, True),
        (False, ["expert_fc1", "moe_act"], ["attn", "moe_router", "moe_preprocess"], 1.0, True),
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 1.0, True),
        (
            False,
            ["core_attn", "attn_proj", "expert_fc1", "moe_act"],
            ["attn", "moe_router"],
            1.0,
            True,
        ),
        (
            False,
            ["core_attn", "expert_fc1", "moe_act"],
            ["attn", "moe_router", "moe_preprocess"],
            1.0,
            True,
        ),
        (
            True,
            ["core_attn", "attn_proj", "expert_fc1", "moe_act"],
            ["attn", "moe_router", "moe_preprocess"],
            1.0,
            True,
        ),
        # Test activation_offload_fraction parameter
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 0.0, True),
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 0.5, True),
        # Test delay_offload_until_cuda_graph parameter
        (False, ["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], 1.0, False),
    ],
)
def test_fine_grained_activation_offloading_with_cuda_graph(
    is_mla: bool,
    offload_modules: List[str],
    cuda_graph_modules: List[str],
    activation_offload_fraction: float,
    delay_offload: bool,
):
    """
    Test fine-grained activation offloading combined with CUDA graph capture.

    Verifies:
    - Forward output correctness with CUDA graph + offloading
    - Backward gradient correctness
    - Memory savings from offloading are preserved with CUDA graphs
    - Different activation_offload_fraction values work correctly
    - Both delay_offload_until_cuda_graph=True/False produce correct results
    """
    from megatron.core.tensor_parallel.random import initialize_rng_tracker

    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    seed = 123
    num_experts = 4  # Always MoE model
    num_layers = 4  # Smaller for faster test with CUDA graphs
    hidden_size = 1024
    num_attention_heads = 8
    vocab_size = 512
    seq_length = 512
    micro_batch_size = 2
    device = torch.device("cuda")
    cuda_graph_warmup_steps = 3

    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )

    off_interface.reset_instance()

    try:
        # 1) Baseline: CUDA graph enabled, offloading disabled
        _reset_cuda_memory()
        base_model = _build_gpt_model_with_cuda_graph(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=False,
            offload_modules=None,
            min_offloaded_tensor_size=1024 * 1024,
            is_mla=is_mla,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        ).cuda()
        base_model.train()

        base_logits, base_grads, base_peak = _run_iters_with_cuda_graph(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_warmup_iters=cuda_graph_warmup_steps,
            num_measure_iters=2,
            enable_offload_reset=False,
        )
        del base_model
        _reset_cuda_memory()

        # 2) Test: CUDA graph enabled + offloading enabled
        off_interface.reset_instance()

        off_model = _build_gpt_model_with_cuda_graph(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
            min_offloaded_tensor_size=1024,  # Force offloading for determinism
            is_mla=is_mla,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            delay_offload_until_cuda_graph=delay_offload,
            activation_offload_fraction=activation_offload_fraction,
        ).cuda()
        off_model.train()

        off_logits, off_grads, off_peak = _run_iters_with_cuda_graph(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_warmup_iters=cuda_graph_warmup_steps,
            num_measure_iters=2,
            enable_offload_reset=True,
        )
        del off_model
        _reset_cuda_memory()

        # 3) Correctness checks
        assert torch.allclose(
            off_logits, base_logits, rtol=1e-2, atol=1e-2
        ), f"Logits mismatch: max_diff={torch.max(torch.abs(off_logits - base_logits))}"
        assert set(off_grads.keys()) == set(base_grads.keys())
        for name, gb in base_grads.items():
            go = off_grads[name]
            if gb is None or go is None:
                assert gb is None and go is None, f"Grad None mismatch for {name}"
                continue
            assert torch.allclose(
                go, gb, rtol=1e-2, atol=1e-2
            ), f"Grad mismatch for {name}: max_diff={torch.max(torch.abs(go - gb))}"

        # 4) Memory checks - offloading should still reduce memory with CUDA graphs
        saved_mib = (base_peak - off_peak) / (1024**2)
        print(
            f"CUDA Graph + Offload test (fraction={activation_offload_fraction}, delay={delay_offload}): "
            f"base_peak={base_peak/(1024**2):.2f}MiB, "
            f"off_peak={off_peak/(1024**2):.2f}MiB, "
            f"saved={saved_mib:.2f}MiB"
        )

        # Basic sanity checks
        assert not torch.isnan(off_logits).any(), "NaN detected in logits"
        assert not torch.isinf(off_logits).any(), "Inf detected in logits"

        # Check gradients are valid
        for name, g in off_grads.items():
            if g is not None:
                assert not torch.isnan(g).any(), f"NaN detected in grad for {name}"
                assert not torch.isinf(g).any(), f"Inf detected in grad for {name}"

        # Note: With CUDA graphs, memory behavior may differ from eager mode.
        # We check that offloading doesn't significantly increase memory.
        # In some cases, graph capture overhead may offset offload savings.
        assert saved_mib >= -CUDA_GRAPH_DELTA, (
            f"Offloading with CUDA graph significantly increased memory: "
            f"saved={saved_mib:.2f}MiB (negative means increase)"
        )

    finally:
        Utils.destroy_model_parallel()


@pytest.mark.flaky_in_dev
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"), reason="CUDA Graph with TE RNG tracker requires TE >= 2.13.0"
)
@pytest.mark.parametrize(
    "offload_modules, cuda_graph_modules, delay_offload",
    [
        # mHC + MoE with attention CUDA graph + attn offloading
        (["core_attn", "attn_proj"], ["attn", "moe_router"], True),
        # mHC + MoE with expert offloading
        (["expert_fc1", "moe_act"], ["attn", "moe_router", "moe_preprocess"], True),
        # mHC + MoE with combined offloading
        (["core_attn", "attn_proj", "expert_fc1", "moe_act"], ["attn", "moe_router"], True),
        # mHC + delay_offload_until_cuda_graph=False
        (["core_attn", "attn_proj", "expert_fc1"], ["attn", "moe_router"], False),
    ],
)
def test_mhc_fine_grained_activation_offloading_with_cuda_graph(
    offload_modules: List[str], cuda_graph_modules: List[str], delay_offload: bool
):
    """
    Test mHC (Hyper Connection) + fine-grained activation offloading + CUDA graph.

    This validates that the fix to HyperConnectionTransformerLayer._te_cuda_graph_replay_impl
    correctly preserves the delay_offload_until_cuda_graph lifecycle (enter_replay /
    flush_delayed_groups / exit_replay) from the parent class.

    Note: "mhc" recompute is incompatible with fine_grained_activation_offloading,
    so mhc recompute is NOT enabled here. HC still functions, just without the mHC-specific
    selective recompute optimization.
    """
    from megatron.core.tensor_parallel.random import initialize_rng_tracker

    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    seed = 123
    num_experts = 4
    num_layers = 4
    hidden_size = 1024
    num_attention_heads = 8
    vocab_size = 512
    seq_length = 512
    micro_batch_size = 2
    device = torch.device("cuda")
    cuda_graph_warmup_steps = 3

    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )

    off_interface.reset_instance()

    try:
        # 1) Baseline: mHC + CUDA graph enabled, offloading disabled
        _reset_cuda_memory()
        base_model = _build_gpt_model_with_cuda_graph(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=False,
            offload_modules=None,
            min_offloaded_tensor_size=1024 * 1024,
            is_mla=False,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            enable_hyper_connections=True,
            num_residual_streams=4,
        ).cuda()
        base_model.train()

        base_logits, base_grads, base_peak = _run_iters_with_cuda_graph(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_warmup_iters=cuda_graph_warmup_steps,
            num_measure_iters=2,
            enable_offload_reset=False,
        )
        del base_model
        _reset_cuda_memory()

        # 2) Test: mHC + CUDA graph enabled + offloading enabled
        off_interface.reset_instance()

        off_model = _build_gpt_model_with_cuda_graph(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
            min_offloaded_tensor_size=1024,
            is_mla=False,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
            delay_offload_until_cuda_graph=delay_offload,
            enable_hyper_connections=True,
            num_residual_streams=4,
        ).cuda()
        off_model.train()

        off_logits, off_grads, off_peak = _run_iters_with_cuda_graph(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_warmup_iters=cuda_graph_warmup_steps,
            num_measure_iters=2,
            enable_offload_reset=True,
        )
        del off_model
        _reset_cuda_memory()

        # 3) Correctness checks
        assert torch.allclose(
            off_logits, base_logits, rtol=1e-2, atol=1e-2
        ), f"mHC logits mismatch: max_diff={torch.max(torch.abs(off_logits - base_logits))}"
        assert set(off_grads.keys()) == set(base_grads.keys())
        for name, gb in base_grads.items():
            go = off_grads[name]
            if gb is None or go is None:
                assert gb is None and go is None, f"Grad None mismatch for {name}"
                continue
            assert torch.allclose(
                go, gb, rtol=1e-2, atol=1e-2
            ), f"mHC grad mismatch for {name}: max_diff={torch.max(torch.abs(go - gb))}"

        # 4) Memory and sanity checks
        saved_mib = (base_peak - off_peak) / (1024**2)
        print(
            f"mHC + CUDA Graph + Offload test (delay={delay_offload}): "
            f"base_peak={base_peak/(1024**2):.2f}MiB, "
            f"off_peak={off_peak/(1024**2):.2f}MiB, "
            f"saved={saved_mib:.2f}MiB"
        )

        assert not torch.isnan(off_logits).any(), "NaN detected in mHC logits"
        assert not torch.isinf(off_logits).any(), "Inf detected in mHC logits"

        for name, g in off_grads.items():
            if g is not None:
                assert not torch.isnan(g).any(), f"NaN detected in mHC grad for {name}"
                assert not torch.isinf(g).any(), f"Inf detected in mHC grad for {name}"

        assert saved_mib >= -CUDA_GRAPH_DELTA, (
            f"mHC offloading with CUDA graph significantly increased memory: "
            f"saved={saved_mib:.2f}MiB (negative means increase)"
        )

    finally:
        Utils.destroy_model_parallel()


@pytest.mark.flaky_in_dev
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_mhc_recompute_with_non_conflicting_offload_modules():
    """
    Test that mHC recompute ('mhc' in recompute_modules) works correctly with
    offload modules that do NOT overlap with mHC checkpoints.

    mHC checkpoints wrap input_layernorm (inside attn_norm) and pre_mlp_layernorm
    (inside mlp_norm). Other offload modules (qkv_linear, core_attn, attn_proj,
    expert_fc1, moe_act) live inside self_attention/MLP which are not wrapped by
    mHC checkpoints, so there is no backward hook ordering conflict.
    """
    offload_modules = ["qkv_linear", "core_attn", "attn_proj", "expert_fc1", "moe_act"]

    os.environ.pop("NVTE_FUSED_ATTN", None)
    os.environ.pop("NVTE_FLASH_ATTN", None)
    os.environ.pop("NVTE_UNFUSED_ATTN", None)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    seed = 123
    num_experts = 4
    num_layers = 8
    hidden_size = 1024
    num_attention_heads = 8
    vocab_size = 1024
    seq_length = 1024
    micro_batch_size = 2
    device = torch.device("cuda")

    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )

    off_interface.reset_instance()

    try:
        # 1) Baseline: mHC + mhc recompute, no offloading
        _reset_cuda_memory()
        base_model = _build_gpt_model(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=False,
            offload_modules=None,
            min_offloaded_tensor_size=1024 * 1024,
            is_mla=False,
            enable_hyper_connections=True,
            mhc_recompute_layer_num=2,
        ).cuda()
        base_model.train()

        _run_one_iter_and_capture(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=False,
        )
        _reset_cuda_memory()
        base_logits, base_grads, base_peak = _run_one_iter_and_capture(
            base_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=False,
        )
        del base_model
        _reset_cuda_memory()

        # 2) mHC + mhc recompute + offloading (non-conflicting modules only)
        off_model = _build_gpt_model(
            seed=seed,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_experts=num_experts,
            fine_grained_activation_offloading=True,
            offload_modules=offload_modules,
            min_offloaded_tensor_size=1024,
            is_mla=False,
            enable_hyper_connections=True,
            mhc_recompute_layer_num=2,
        ).cuda()
        off_model.train()

        _run_one_iter_and_capture(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=True,
        )
        off_interface.reset()

        _reset_cuda_memory()
        off_logits, off_grads, off_peak = _run_one_iter_and_capture(
            off_model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            enable_offload_reset=True,
        )
        del off_model
        _reset_cuda_memory()

        # 3) Correctness checks
        assert torch.allclose(off_logits, base_logits, rtol=1e-3, atol=1e-3), (
            f"mHC recompute + offload logits mismatch: "
            f"max_diff={torch.max(torch.abs(off_logits - base_logits))}"
        )
        assert set(off_grads.keys()) == set(base_grads.keys())
        for name, gb in base_grads.items():
            go = off_grads[name]
            if gb is None or go is None:
                assert gb is None and go is None, f"Grad None mismatch for {name}"
                continue
            assert torch.allclose(go, gb, rtol=1e-3, atol=1e-3), (
                f"mHC recompute + offload grad mismatch for {name}: "
                f"max_diff={torch.max(torch.abs(go - gb))}"
            )

        # 4) Memory checks
        saved_mib = (base_peak - off_peak) / (1024**2)
        assert saved_mib > 0.0, (
            f"Expected GPU peak memory reduction for offload_modules={offload_modules}, "
            f"but got saved={saved_mib:.2f}MiB"
        )
        print(f"mHC recompute + offload ({offload_modules}): " f"saved={saved_mib:.2f}MiB")

    finally:
        Utils.destroy_model_parallel()


def _block_capture_group_fn():
    """``group(x, name, weight, forced_released_tensors=None)``: one offload group (start,
    saved-tensor hooks, commit)."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        fine_grained_offloading_group_offload,
        fine_grained_offloading_group_start,
    )

    def group(x, name, weight, forced_released_tensors=None):
        x = fine_grained_offloading_group_start(x, name)
        with PipelineOffloadManager.get_instance():
            y = torch.nn.functional.gelu(torch.nn.functional.linear(x, weight))
        return fine_grained_offloading_group_offload(
            y, name, forced_released_tensors=forced_released_tensors
        )

    return group


def _block_capture_reference(x, weights):
    """Offload-free gradient of ``sum(gelu(linear(...)))`` chained over ``weights``."""
    x = x.detach().clone().requires_grad_()
    y = x
    for weight in weights:
        y = torch.nn.functional.gelu(torch.nn.functional.linear(y, weight.detach()))
    y.sum().backward()
    return x.grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_block_capture_mode_serves_one_handler_per_capture_forward():
    """Chunk-wise CUDA graphs drive the offload manager in block-capture mode.

    Warm-up records two virtual pipeline stages (two ``mlp`` groups on stage 0; two ``mlp``
    groups and a ``post`` group on stage 1). In block-capture mode every capture-time decoder
    forward gets a fresh handler cloned from its stage's warm-up template (same names and
    policy flags, never the pooled host buffers); every post-process call (Transformer Engine
    warms every callable up several times) gets a handler of its own that resumes at the
    decoder's last group, so neither callable's graph prefetches or waits on the other's
    groups; reloads stay inside each handler, and the gradients equal the offload-free
    reference. Once the capture is over, an eager caller (the post-process of fixed-length chunk
    graphs) gets no handler and runs offload-free.
    """
    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = [torch.randn(64, 64, device=device, requires_grad=True) for _ in range(3)]
    group = _block_capture_group_fn()

    def decoder(x):
        return group(group(x, "mlp", weights[0]), "mlp", weights[1])

    def post_process(x):
        # Two groups of one name: the warm-up policy keeps only the last one resident, so the
        # post-process handler has a group of its own to offload and reload.
        return group(group(x, "post", weights[2]), "post", weights[2])

    try:
        manager = PipelineOffloadManager.get_instance()
        # Warm-up iteration: the eager model forward opens one handler per virtual stage; the
        # manager queues the backward handlers when the last virtual stage opens, and the
        # interleaved schedule runs the backwards in reverse stage order.
        off_interface.reset()
        inputs = [torch.randn(8, 64, device=device, requires_grad=True) for _ in range(2)]
        outputs = []
        for vp_stage in range(2):
            off_interface.init_chunk_handler(
                pp_rank=0,
                vp_size=2,
                vp_stage=vp_stage,
                min_offloaded_tensor_size=1,
                delta_offload_bytes_across_pp_ranks=0,
                activation_offload_fraction=1.0,
            )
            out = decoder(inputs[vp_stage])
            if vp_stage == 1:
                out = post_process(out)
            outputs.append(out)
        for out in reversed(outputs):
            out.sum().backward()
        off_interface.reset()  # closes the warm-up and fixes the policy
        assert not manager._is_warmup
        templates = list(manager._cached_chunks_forward)
        template = next(chunk for chunk in templates if chunk.vpp_rank == 1)
        assert [g._name for g in template.offload_groups] == ["mlp", "mlp", "post", "post"]

        off_interface.enter_block_capture(keep_last_group=True)
        # Capture-time decoder forward of stage 1: a fresh handler cloned from the template.
        off_interface.begin_block_capture_chunk(1)
        handler = manager.cur_forward_chunk()
        assert handler not in templates and handler.vpp_rank == 1
        assert manager.pop_forward_chunk("mlp") is handler
        assert [g._name for g in handler.offload_groups] == ["mlp", "mlp", "post", "post"]
        # Warm-up policy flags, minus the last group of every name (keep_last_group=True: kept
        # resident so that no backward graph starts with a synchronous reload).
        expected_flags = [g.offload for g in template.offload_groups]
        for last in (1, 3):
            expected_flags[last] = False
        assert [g.offload for g in handler.offload_groups] == expected_flags
        assert not any(g.use_cpu_pool for g in handler.offload_groups)
        x = torch.randn(8, 64, device=device, requires_grad=True)
        hidden = decoder(x)
        off_interface.end_block_capture_chunk()
        assert manager._block_capture_decoder_end[1] == 2
        offloaded = [g for g in handler.offload_groups if g.offload]
        assert offloaded, "the warm-up policy should offload at least one group"
        assert len(handler._block_capture_host_buffers) > 0

        # Every post-process call gets a handler of its own that resumes at the decoder's end:
        # the two callables are separate graphs, and one graph may neither prefetch the other
        # graph's groups nor wait on events recorded in the other capture.
        off_interface.begin_block_capture_chunk(1, continue_current=True)
        post = manager.cur_forward_chunk()
        assert post is not handler and post.vpp_rank == 1
        assert post._offloaded_group_index == 2
        assert [g._name for g in post.offload_groups] == ["mlp", "mlp", "post", "post"]
        out = post_process(hidden)
        off_interface.end_block_capture_chunk(record_decoder_end=False)
        # The first post-process group was offloaded by the post-process handler itself ...
        assert post.offload_groups[2].offload and not post.offload_groups[3].offload
        assert post._offloaded_group_index == 4
        assert len(post._block_capture_host_buffers) > 0
        # ... and the decoder's groups still belong to the decoder's handler alone.
        decoder_reloads = list(handler._groups_to_reload)
        assert decoder_reloads
        off_interface.begin_block_capture_chunk(1, continue_current=True)
        fresh = manager.cur_forward_chunk()
        assert fresh is not handler and fresh is not post and fresh.vpp_rank == 1
        assert fresh._offloaded_group_index == 2
        hidden_again = hidden.detach().clone().requires_grad_()
        out_again = post_process(hidden_again)
        off_interface.end_block_capture_chunk(record_decoder_end=False)

        # The post-process backward (up to the decoder's output) reloads only its own group:
        # the decoder's pending reloads are untouched until the decoder's backward runs.
        (grad_hidden_again,) = torch.autograd.grad(out_again.sum(), hidden_again)
        assert not fresh._groups_to_reload and not fresh._reloading_group
        assert list(handler._groups_to_reload) == decoder_reloads
        out.sum().backward()
        for chunk_handler in (handler, post, fresh):
            assert not chunk_handler._groups_to_reload and not chunk_handler._reloading_group
        torch.cuda.synchronize()
        assert torch.equal(
            x.grad, _block_capture_reference(x, [weights[0], weights[1], weights[2], weights[2]])
        )
        assert torch.equal(
            grad_hidden_again, _block_capture_reference(hidden, [weights[2], weights[2]])
        )
        off_interface.exit_block_capture()
        assert manager.cur_forward_chunk() is None
        # After the capture the replayed graphs own every offload; an eager caller (the
        # post-process of fixed-length chunk graphs runs after the decoder replay) gets no
        # handler, so its groups stay on the GPU and its backward reloads nothing.
        off_interface.reset()
        assert manager.pop_forward_chunk("post") is None
        eager_in = torch.randn(8, 64, device=device, requires_grad=True)
        eager_out = group(eager_in, "post", weights[2])
        assert manager.cur_forward_chunk() is None
        eager_out.sum().backward()
        assert torch.equal(eager_in.grad, _block_capture_reference(eager_in, [weights[2]]))
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


def _block_capture_warmup(forward, inputs, after_forward=None):
    """One eager warm-up iteration (single stage) recording the template; returns its handler.

    ``after_forward(output)`` runs between the forward and the backward, like the backward-time
    recompute that appends its groups to the template after the model-chunk forward returned."""
    off_interface.reset()
    off_interface.init_chunk_handler(
        pp_rank=0,
        vp_size=1,
        vp_stage=None,
        min_offloaded_tensor_size=1,
        delta_offload_bytes_across_pp_ranks=0,
        activation_offload_fraction=1.0,
    )
    output = forward(inputs)
    if after_forward is not None:
        after_forward(output)
    output.sum().backward()
    off_interface.reset()  # closes the warm-up and fixes the policy
    manager = PipelineOffloadManager.get_instance()
    assert not manager._is_warmup
    return manager._cached_chunks_forward[0]


def _block_capture_offload_everything(template):
    """Replace the post-warm-up policy (the schedule's last group of every name stays resident
    as a margin group) by "offload every group", so that the capture's own keep-last policy is
    the only thing that keeps a group of these templates resident."""
    for group in template.offload_groups:
        group.offload = True
        group.is_offload_margin = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_block_capture_hooks_are_noops_outside_a_capture_time_forward():
    """Full activation recompute inside a chunk graph re-runs the layer forward from the backward
    capture (Transformer Engine's checkpoint), after ``end_block_capture_chunk`` closed the
    capture-time forward. The group hooks then serve no handler: the start / commit markers return
    their input, the saved-tensor hooks are not installed, and the capture handler's groups are
    neither advanced nor reloaded by the recompute's backward."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        fine_grained_offloading_group_offload,
        fine_grained_offloading_group_start,
    )

    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = [torch.randn(64, 64, device=device, requires_grad=True) for _ in range(2)]
    group = _block_capture_group_fn()

    def decoder(x):
        return group(group(x, "mlp", weights[0]), "mlp", weights[1])

    try:
        manager = PipelineOffloadManager.get_instance()
        _block_capture_warmup(decoder, torch.randn(8, 64, device=device, requires_grad=True))

        off_interface.enter_block_capture()
        # Capture-time forward: the handler serves the hooks.
        off_interface.begin_block_capture_chunk(None)
        handler = manager.cur_forward_chunk()
        assert handler is not None and manager.pop_forward_chunk("mlp") is handler
        x = torch.randn(8, 64, device=device, requires_grad=True)
        out = decoder(x)
        assert handler._offloaded_group_index == 2
        off_interface.end_block_capture_chunk()
        pending_reloads = list(handler._groups_to_reload)
        assert pending_reloads, "the warm-up policy should offload at least one group"

        # Backward-time recompute: no handler, every hook is an identity.
        assert manager.cur_forward_chunk() is None
        assert manager.pop_forward_chunk("mlp") is None
        probe = torch.randn(8, 64, device=device)
        assert fine_grained_offloading_group_start(probe, "mlp") is probe
        with manager:
            assert not getattr(manager, "inside_context", False)
        assert fine_grained_offloading_group_offload(probe, "mlp") is probe
        recompute_in = x.detach().clone().requires_grad_()
        recompute_out = decoder(recompute_in)
        assert handler._offloaded_group_index == 2
        recompute_out.sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(recompute_in.grad, _block_capture_reference(x, weights))
        # The recompute's backward touched none of the capture handler's groups.
        assert list(handler._groups_to_reload) == pending_reloads
        assert not handler._reloading_group

        # The next capture-time forward (next slot) is served again, by a handler of its own.
        off_interface.begin_block_capture_chunk(None)
        other = manager.cur_forward_chunk()
        assert other is not None and other is not handler
        assert manager.pop_forward_chunk("mlp") is other
        off_interface.end_block_capture_chunk()

        # The captured forward's own backward reloads its groups as before.
        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(x.grad, _block_capture_reference(x, weights))
        assert not handler._groups_to_reload and not handler._reloading_group
        off_interface.exit_block_capture()
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_block_capture_keeps_static_input_storage_out_of_the_release():
    """The captured callable's static input (Transformer Engine's hidden_states sample arg) is the
    first layer's input and residual, which the attn_norm commit force-releases. Registered as a
    block static input, any view sharing its storage is neither offloaded nor resized to zero
    (TE keeps the storage alive for the slot and refreshes it at replay); the registration lives
    for the capture only."""
    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weight = torch.randn(64, 64, device=device, requires_grad=True)
    group = _block_capture_group_fn()

    try:
        manager = PipelineOffloadManager.get_instance()
        _block_capture_warmup(
            lambda x: group(x, "attn_norm", weight),
            torch.randn(8, 64, device=device, requires_grad=True),
        )
        # Outside a capture the registration is ignored.
        off_interface.register_block_static_input(torch.randn(8, 64, device=device))
        assert not manager.block_static_storages

        off_interface.enter_block_capture()
        off_interface.begin_block_capture_chunk(None)
        handler = manager.cur_forward_chunk()
        static_input = torch.randn(8, 64, device=device)  # TE's sample arg
        layer_input = static_input[:4]  # a view: the first layer's input / residual
        other = torch.randn(8, 64, device=device)
        assert handler.tensor_need_offloading_checker(layer_input)
        off_interface.register_block_static_input(static_input)
        assert manager.is_block_static_storage(layer_input)
        assert not handler.tensor_need_offloading_checker(layer_input)
        assert handler.tensor_need_offloading_checker(other)
        off_interface.register_block_static_input(None)  # tolerated

        # The forced release of a committed group frees the other tensor only.
        group_obj = OffloadTensorGroup("attn_norm")
        group_obj.record_offload_event(torch.cuda.current_stream())
        handler._block_capture_pending_release.append((group_obj, [layer_input, other]))
        handler._block_capture_release(keep=0)
        # As plain numbers: a failed assert must not print a released tensor.
        storage_bytes = [t.untyped_storage().size() for t in (static_input, other)]
        assert storage_bytes == [8 * 64 * 4, 0]
        off_interface.end_block_capture_chunk()
        off_interface.exit_block_capture()
        assert not manager.block_static_storages
        assert not manager.is_block_static_storage(layer_input)
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_block_capture_no_grad_commit_keeps_its_forced_release_tensors():
    """A group commit force-releases the tensors its caller names (attn_norm: the layer input)
    once the group's saved tensors are copied out. The no-grad forward of an activation
    checkpoint (full recompute inside a chunk graph runs every layer that way) saves nothing,
    so nothing is copied out, and the tensor it names is the input the checkpoint holds for its
    backward-time recompute: that commit releases nothing. A commit that runs with autograd
    enabled still releases its tensor, and its backward reloads the copy."""
    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = [torch.randn(64, 64, device=device, requires_grad=True) for _ in range(2)]
    group = _block_capture_group_fn()
    num_bytes = 8 * 64 * 4

    def layers(x):
        # Like the attn_norm group of two layers: each commit names its layer input.
        hidden = group(x, "attn_norm", weights[0], forced_released_tensors=[x])
        return hidden, group(hidden, "attn_norm", weights[1], forced_released_tensors=[hidden])

    try:
        manager = PipelineOffloadManager.get_instance()
        template = _block_capture_warmup(
            lambda x: group(group(x, "attn_norm", weights[0]), "attn_norm", weights[1]),
            torch.randn(8, 64, device=device, requires_grad=True),
        )
        # Offload both groups, so that the policy never keeps a commit from releasing.
        _block_capture_offload_everything(template)

        off_interface.enter_block_capture()
        # Capture-time forward under a checkpoint: autograd is off, as in its first pass.
        off_interface.begin_block_capture_chunk(None)
        handler = manager.cur_forward_chunk()
        checkpoint_input = torch.randn(8, 64, device=device)
        expected_input = checkpoint_input.clone()
        with torch.no_grad():
            hidden, _ = layers(checkpoint_input)
        expected_hidden = hidden.clone()
        assert handler._offloaded_group_index == 2
        off_interface.end_block_capture_chunk()
        assert not handler._block_capture_pending_release
        # Sizes first, as plain numbers: a failed assert must not print a released tensor.
        kept_bytes = [t.untyped_storage().size() for t in (checkpoint_input, hidden)]
        assert kept_bytes == [num_bytes, num_bytes]
        assert torch.equal(checkpoint_input, expected_input)
        assert torch.equal(hidden, expected_hidden)
        assert not handler._block_capture_host_buffers, "a no-grad group has nothing to copy"

        # Capture-time forward with autograd on: both layer inputs are copied out and released.
        off_interface.begin_block_capture_chunk(None)
        handler = manager.cur_forward_chunk()
        x = torch.randn(8, 64, device=device, requires_grad=True)
        layer_input = x.clone()
        hidden, out = layers(layer_input)
        off_interface.end_block_capture_chunk()
        assert handler._block_capture_host_buffers
        released_bytes = [t.untyped_storage().size() for t in (layer_input, hidden)]
        assert released_bytes == [0, 0]
        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(x.grad, _block_capture_reference(x, weights))
        off_interface.exit_block_capture()
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_eager_no_grad_commit_keeps_its_forced_release_tensors():
    """Outside a capture the group commit follows the same rule: without autograd (the first
    pass of an activation checkpoint, an evaluation forward) it releases nothing; with autograd
    on it releases the tensor it names as soon as its group is copied out, and the backward
    reloads the copy. The schedule's last group of a name stays resident, so only the first of
    the two commits offloads."""
    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = [torch.randn(64, 64, device=device, requires_grad=True) for _ in range(2)]
    group = _block_capture_group_fn()
    num_bytes = 8 * 64 * 4

    def layers(x):
        hidden = group(x, "attn_norm", weights[0], forced_released_tensors=[x])
        return group(hidden, "attn_norm", weights[1], forced_released_tensors=[hidden])

    try:
        template = _block_capture_warmup(
            lambda x: group(group(x, "attn_norm", weights[0]), "attn_norm", weights[1]),
            torch.randn(8, 64, device=device, requires_grad=True),
        )
        assert [g.offload for g in template.offload_groups] == [True, False]

        off_interface.reset()
        checkpoint_input = torch.randn(8, 64, device=device)
        expected_input = checkpoint_input.clone()
        with torch.no_grad():
            layers(checkpoint_input)
        # As a plain number: a failed assert must not print a released tensor.
        kept_bytes = checkpoint_input.untyped_storage().size()
        assert kept_bytes == num_bytes
        assert torch.equal(checkpoint_input, expected_input)

        off_interface.reset()
        x = torch.randn(8, 64, device=device, requires_grad=True)
        layer_input = x.clone()
        out = layers(layer_input)
        released_bytes = layer_input.untyped_storage().size()
        assert released_bytes == 0
        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(x.grad, _block_capture_reference(x, weights))
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize(
    "ending", ["capture_raised", "capture_raised_in_te_warmup", "graphs_deleted"]
)
def test_block_capture_hands_the_offloads_back_to_the_eager_policy(ending):
    """Captured block graphs own every offload of the step, so while they live an eager caller
    gets no handler and the template marks are inert. That ownership ends with the graphs: a
    capture that raised (``exit_block_capture(captured=False)``) never had them, and
    ``release_block_capture`` follows their deletion. Either way the handlers the graphs kept
    alive are dropped, and the next eager step is served by the warm-up handler again, offloads
    and reloads as before the capture. Transformer Engine switches the offload off around its
    warm-up pass and skips the hook that switches it back on when that pass raises."""
    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = [torch.randn(64, 64, device=device, requires_grad=True) for _ in range(2)]
    group = _block_capture_group_fn()

    def decoder(x):
        return group(group(x, "mlp", weights[0]), "mlp", weights[1])

    def marks_are_live():
        # The marks act on the warm-up handler of the running eager forward.
        probe = ChunkOffloadHandler(1, manager.cpu_tensor_pool)
        probe._offloaded_group_index = 2
        manager._cur_forward_chunk = probe
        off_interface.mark_decoder_groups_end()
        off_interface.mark_forward_groups_end()
        manager._cur_forward_chunk = None
        return (probe._decoder_group_end, probe._forward_group_end) == (2, 2)

    try:
        manager = PipelineOffloadManager.get_instance()
        template = _block_capture_warmup(
            decoder, torch.randn(8, 64, device=device, requires_grad=True)
        )

        off_interface.enter_block_capture()
        off_interface.begin_block_capture_chunk(None)
        handler = manager.cur_forward_chunk()
        decoder(torch.randn(8, 64, device=device, requires_grad=True))
        off_interface.end_block_capture_chunk()
        # Kept alive for the graphs when the stream is capturing (not in this test).
        manager._block_capture_live_chunks.append(handler)

        if ending.startswith("capture_raised"):
            if ending == "capture_raised_in_te_warmup":
                off_interface.disable_offload()
                assert not manager.do_offload and not template.do_offload
            off_interface.exit_block_capture(captured=False)
            assert manager.do_offload and template.do_offload
        else:
            off_interface.exit_block_capture()
            assert manager._block_graphs_captured
            assert manager._block_capture_live_chunks == [handler]
            off_interface.reset()
            assert manager.pop_forward_chunk("mlp") is None
            assert not marks_are_live()
            off_interface.release_block_capture()
        assert not manager._block_graphs_captured
        assert not manager._block_capture_live_chunks
        assert marks_are_live()

        # The next eager step: the warm-up handler serves it and offloads its first group (the
        # last one of a name stays resident), and the backward reloads it.
        off_interface.reset()
        x = torch.randn(8, 64, device=device, requires_grad=True)
        out = decoder(x)
        assert manager.cur_forward_chunk() is template
        assert template._group_has_offloaded_tensor(template.offload_groups[0])
        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(x.grad, _block_capture_reference(x, weights))
        assert not template._groups_to_reload and not template._reloading_group
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_capture_scope_ends_a_capture_the_offload_manager_refuses():
    """The offload manager refuses a whole-block capture that no eager warm-up preceded (it has
    no policy to clone). The helper's capture scope treats that like any failed capture: the
    capture flag is cleared, so a later eager forward does not take the capture path, and the
    manager, which never entered the capture, is not told that one ended."""
    from megatron.core.transformer import cuda_graphs

    off_interface.reset_instance()
    helper = SimpleNamespace(
        config=SimpleNamespace(
            fine_grained_activation_offloading=True,
            cuda_graph_impl="chunk_wise",
            fine_grained_offloading_graph_keep_last_group=False,
        )
    )
    try:
        manager = PipelineOffloadManager.get_instance()
        cuda_graphs._set_capture_start()
        with pytest.raises(AssertionError, match="at least one eager warm-up iteration"):
            with cuda_graphs.TECudaGraphHelper._te_capture_scope(helper):
                pytest.fail("the scope must not be entered")
        assert not cuda_graphs.is_graph_capturing()
        assert not manager._block_capture and not manager._block_graphs_captured
    finally:
        cuda_graphs._set_capture_end()
        off_interface.reset_instance()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_block_capture_clones_margin_flags_and_marks_kept_groups_as_margin():
    """A slot's handler clones the warm-up policy including ``is_offload_margin`` (a margin group
    takes no reload slot), and ``keep_last_group`` marks the groups it keeps resident as margin
    groups too, so the backward's prefetch steps are spent on offloaded groups only."""
    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = [torch.randn(64, 64, device=device, requires_grad=True) for _ in range(2)]
    group = _block_capture_group_fn()

    def decoder(x):
        return group(group(x, "mlp", weights[0]), "mlp", weights[1])

    try:
        manager = PipelineOffloadManager.get_instance()
        template = _block_capture_warmup(
            decoder, torch.randn(8, 64, device=device, requires_grad=True)
        )
        assert [g._name for g in template.offload_groups] == ["mlp", "mlp"]
        # The post-warm-up policy keeps the schedule's last mlp group resident as a margin
        # group; make the first group the margin one instead, so the clone is told apart from
        # the keep-last policy below (which marks the last group).
        assert [g.is_offload_margin for g in template.offload_groups] == [False, True]
        _block_capture_offload_everything(template)
        template.offload_groups[0].is_offload_margin = True

        off_interface.enter_block_capture(keep_last_group=False)
        off_interface.begin_block_capture_chunk(None)
        plain = manager.cur_forward_chunk()
        assert [g.is_offload_margin for g in plain.offload_groups] == [True, False]
        assert [g.offload for g in plain.offload_groups] == [True, True]
        x = torch.randn(8, 64, device=device, requires_grad=True)
        out = decoder(x)
        off_interface.end_block_capture_chunk()  # records the decoder's end (2)
        assert manager._block_capture_decoder_end == {0: 2}

        manager._block_capture_keep_last_group = True
        off_interface.begin_block_capture_chunk(None)
        kept = manager.cur_forward_chunk()
        assert [g.is_offload_margin for g in kept.offload_groups] == [True, True]
        assert [g.offload for g in kept.offload_groups] == [True, False]
        off_interface.end_block_capture_chunk()
        # The template itself is never rewritten by a slot's policy.
        assert [g.is_offload_margin for g in template.offload_groups] == [True, False]
        assert [g.offload for g in template.offload_groups] == [True, True]
        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(x.grad, _block_capture_reference(x, weights))
        off_interface.exit_block_capture()
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
def test_block_capture_keep_last_covers_the_decoder_groups_marked_by_the_eager_warm_up():
    """The eager warm-up appends the post-process groups (an MTP layer reuses the decoder's group
    names) and then the backward-time recompute groups to the same template, after the
    decoder's. ``mark_decoder_groups_end`` / ``mark_forward_groups_end`` (called by the model
    forward when the decoder chunk returns / when the forward returns) record where those ranges
    end, so the first capture slot's ``keep_last_group`` keeps the decoder's own last group of
    every name resident, and the post-process handler the last groups of its own range: never a
    later group that the callable does not visit (which would leave its real last group offloaded
    and reloaded synchronously at the start of its backward)."""
    Utils.initialize_model_parallel(1, 1)
    off_interface.reset_instance()
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = [torch.randn(64, 64, device=device, requires_grad=True) for _ in range(4)]
    group = _block_capture_group_fn()

    def decoder(x):
        return group(group(x, "mlp", weights[0]), "mlp", weights[1])

    def post_process(x):
        return group(group(x, "mlp", weights[2]), "post", weights[3])

    def model(x):
        hidden = decoder(x)
        off_interface.mark_decoder_groups_end()  # GPTModel.forward, after the decoder
        off_interface.mark_decoder_groups_end()  # idempotent
        out = post_process(hidden)
        off_interface.mark_forward_groups_end()  # GPTModel.forward, after the post-process
        off_interface.mark_forward_groups_end()  # idempotent
        return out

    def backward_time_recompute(out):
        # A checkpointed layer re-running its forward from the warm-up backward: the group
        # hooks still find the warm-up handler and append a group of the decoder's name.
        with torch.no_grad():
            group(out.detach(), "mlp", weights[1])

    try:
        manager = PipelineOffloadManager.get_instance()
        template = _block_capture_warmup(
            model,
            torch.randn(8, 64, device=device, requires_grad=True),
            after_forward=backward_time_recompute,
        )
        names = ["mlp", "mlp", "mlp", "post", "mlp"]
        assert [g._name for g in template.offload_groups] == names
        assert template._decoder_group_end == 2
        assert template._forward_group_end == 4
        # The post-warm-up policy keeps the schedule's last group of every name resident (the
        # recompute's mlp and the post-process's post); offload everything so that only the
        # capture's keep-last policy decides below.
        assert [g.offload for g in template.offload_groups] == [True, True, True, False, False]
        _block_capture_offload_everything(template)

        off_interface.enter_block_capture(keep_last_group=True)
        assert manager._block_capture_decoder_end == {0: 2}
        assert manager._block_capture_forward_end == {0: 4}
        off_interface.mark_decoder_groups_end()  # eager only: no-ops inside the capture
        off_interface.mark_forward_groups_end()
        assert (template._decoder_group_end, template._forward_group_end) == (2, 4)

        # First capture slot, decoder callable: keep-last over the decoder's range [0, 2).
        # Without the decoder-end mark the range would be the whole template, keeping the
        # recompute's mlp (index 4) and the post (index 3): [True, True, True, False, False].
        off_interface.begin_block_capture_chunk(None)
        handler = manager.cur_forward_chunk()
        assert handler._decoder_group_end is None and handler._forward_group_end is None
        assert [g.offload for g in handler.offload_groups] == [True, False, True, True, True]
        x = torch.randn(8, 64, device=device, requires_grad=True)
        hidden = decoder(x)
        assert handler._offloaded_group_index == 2
        off_interface.end_block_capture_chunk()
        assert manager._block_capture_decoder_end == {0: 2}
        # Post-process callable of the slot: a clone of the warm-up template (not of the decoder
        # handler) that resumes at the decoder's end, keep-last over its own range [2, 4). The
        # decoder-range entries keep the template's flags, which is inert: this handler never
        # starts or commits those groups. Without the forward-end mark the range would run to
        # the template's end and keep the recompute's mlp instead of its own:
        # [True, True, True, False, False].
        off_interface.begin_block_capture_chunk(None, continue_current=True)
        post = manager.cur_forward_chunk()
        assert post is not handler and post._offloaded_group_index == 2
        assert [g.offload for g in post.offload_groups] == [True, True, False, False, True]
        out = post_process(hidden)
        assert post._offloaded_group_index == 4
        off_interface.end_block_capture_chunk(record_decoder_end=False)
        assert manager._block_capture_decoder_end == {0: 2}
        assert manager._block_capture_forward_end == {0: 4}

        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(x.grad, _block_capture_reference(x, weights))
        for chunk_handler in (handler, post):
            assert not chunk_handler._groups_to_reload and not chunk_handler._reloading_group
        off_interface.exit_block_capture()
    finally:
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


def _run_eager_iters(
    model: GPTModel,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_iters: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Eager forward+backward iterations; logits and grads of the last one (CPU float32)."""
    for i in range(num_iters):
        logits = model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        logits.float().sum().backward()
        if i < num_iters - 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    torch.cuda.synchronize()
    grads = {
        name: (p.grad.detach().float().cpu() if p.grad is not None else None)
        for name, p in model.named_parameters()
    }
    return logits.detach().float().cpu(), grads


def _assert_chunk_graph_matches_eager(graph_logits, graph_grads, base_logits, base_grads):
    assert torch.isfinite(graph_logits).all()
    assert torch.allclose(
        graph_logits, base_logits, rtol=1e-2, atol=1e-2
    ), f"Logits mismatch: max_diff={torch.max(torch.abs(graph_logits - base_logits))}"
    assert set(graph_grads.keys()) == set(base_grads.keys())
    for name, gb in base_grads.items():
        go = graph_grads[name]
        if gb is None or go is None:
            assert gb is None and go is None, f"Grad None mismatch for {name}"
            continue
        assert torch.isfinite(go).all(), f"non-finite grad for {name}"
        assert torch.allclose(
            go, gb, rtol=1e-2, atol=1e-2
        ), f"Grad mismatch for {name}: max_diff={torch.max(torch.abs(go - gb))}"


@pytest.fixture
def chunk_graph_process_state(monkeypatch):
    """Model-parallel state for a chunk-graph GPU test, plus the teardown of everything such a
    test changes process-wide, also when it fails.

    ``_run_iters_with_cuda_graph`` moves the thread to a side stream, initializes the
    num-microbatches calculator and captures graphs into the shared graph pool; it destroys the
    calculator and the graphs only when it returns normally. The graphs go first: their
    deletion still reads the process groups.
    """
    from megatron.core.full_cuda_graph import release_shared_graph_pool
    from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
    from megatron.core.tensor_parallel.random import initialize_rng_tracker
    from megatron.core.transformer import cuda_graphs

    helpers = []
    create_cudagraphs = cuda_graphs.TECudaGraphHelper.create_cudagraphs

    def tracked_create_cudagraphs(helper):
        helpers.append(helper)
        return create_cudagraphs(helper)

    monkeypatch.setattr(
        cuda_graphs.TECudaGraphHelper, "create_cudagraphs", tracked_create_cudagraphs
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    # Read after the rank selected its device: setting a stream makes its device the current
    # one, so a stream saved before that would move the rank back to device 0.
    original_stream = torch.cuda.current_stream()
    off_interface.reset_instance()
    yield
    try:
        torch.cuda.synchronize()
        torch.cuda.set_stream(original_stream)
        for helper in helpers:
            if helper.graphs_created():
                helper.delete_cuda_graphs()
    finally:
        cuda_graphs._set_capture_end()
        release_shared_graph_pool()
        destroy_num_microbatches_calculator()
        off_interface.reset_instance()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"), reason="CUDA Graph with TE RNG tracker requires TE >= 2.14.0"
)
@pytest.mark.parametrize(
    "case",
    [
        "attn_norm_standalone_input_norm",
        "attn_norm_keep_last_group",
        "attn_norm_capture_raises",
        "full_recompute",
        "attn_norm_full_recompute",
    ],
)
def test_chunk_wise_cuda_graph_with_fine_grained_activation_offloading(
    case: str, chunk_graph_process_state, monkeypatch
):
    """Chunk-wise TE CUDA graphs with fine-grained activation offloading, against eager:

    - ``attn_norm_standalone_input_norm``: MLA layers (a standalone TENorm input layernorm, not
      fused into the first linear) offload ``attn_norm``, whose commit force-releases the layer
      input. For the block's first layer that input is the captured graph's static hidden_states,
      which the capture keeps intact (``register_block_static_input``); the last layer's group
      stays resident; the middle layer's input, a full hidden state, is copied out, released and
      reloaded inside the graphs.
    - ``attn_norm_keep_last_group``: the same with
      ``fine_grained_offloading_graph_keep_last_group``, which the capture hands to the manager.
    - ``attn_norm_capture_raises``: the graph creation fails. The helper ends the capture on its
      way out: the capture flag is clear (a decoder block that still saw it would take the
      capture path in its next forward) and no graphs own the offloads, so the eager step that
      follows is served by the warm-up handler, offloads, and matches the reference.
    - ``full_recompute``: full activation recompute captured into the backward graph together
      with ``core_attn`` offloading; the recompute re-runs the layer forward with the group hooks
      as no-ops (no handler outside a capture-time forward).
    - ``attn_norm_full_recompute``: full activation recompute with ``attn_norm`` offloading. The
      capture-time forward runs every layer under the checkpoint without autograd, where the
      attn_norm commit must leave the layer input alone: it is the tensor the checkpoint
      recomputes the layer from inside the backward graph.
    """
    seq_length, micro_batch_size, hidden_size = 128, 2, 256
    device = torch.device("cuda")
    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )
    common = dict(
        seed=123,
        num_layers=3,
        hidden_size=hidden_size,
        num_attention_heads=4,
        vocab_size=128,
        seq_length=seq_length,
        num_experts=None,
        cuda_graph_modules=[],
        cuda_graph_warmup_steps=1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    full_recompute = dict(
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=[],
    )
    keep_last_group = case == "attn_norm_keep_last_group"
    capture_raises = case == "attn_norm_capture_raises"
    if case in (
        "attn_norm_standalone_input_norm",
        "attn_norm_keep_last_group",
        "attn_norm_capture_raises",
    ):
        common.update(is_mla=True, recompute_granularity=None)
        offload_modules = ["attn_norm"]
    elif case == "attn_norm_full_recompute":
        common.update(is_mla=True, **full_recompute)
        offload_modules = ["attn_norm"]
    else:
        common.update(is_mla=False, num_layers=2, **full_recompute)
        offload_modules = ["core_attn"]

    # What the capture leaves behind when it ends: the keep-last policy it ran with and the sizes
    # of the host buffers its graphs address (the handlers the manager keeps alive for them).
    captures = []
    exit_block_capture = PipelineOffloadManager.exit_block_capture

    def recording_exit_block_capture(manager, *args, **kwargs):
        host_buffer_sizes = [
            host_buffer.numel()
            for handler in manager._block_capture_live_chunks
            for host_buffer in handler._block_capture_host_buffers
        ]
        captures.append((manager._block_capture_keep_last_group, host_buffer_sizes))
        return exit_block_capture(manager, *args, **kwargs)

    monkeypatch.setattr(PipelineOffloadManager, "exit_block_capture", recording_exit_block_capture)

    # The host tensors of the eager warm-up's shared pool, when the capture starts and when it
    # ends: the graphs address buffers of their own, so the pool is handed back, not kept.
    def pooled_host_tensors(manager):
        return sum(len(pool["all"]) for pool in manager.cpu_tensor_pool._pools.values())

    pooled_around_capture = []
    enter_block_capture = PipelineOffloadManager.enter_block_capture

    def recording_enter_block_capture(manager, *args, **kwargs):
        pooled_around_capture.append(pooled_host_tensors(manager))
        enter_block_capture(manager, *args, **kwargs)
        pooled_around_capture.append(pooled_host_tensors(manager))

    monkeypatch.setattr(
        PipelineOffloadManager, "enter_block_capture", recording_enter_block_capture
    )

    # Eager reference: no graphs, no offloading.
    _reset_cuda_memory()
    base_model = _build_gpt_model_with_cuda_graph(
        fine_grained_activation_offloading=False,
        offload_modules=None,
        min_offloaded_tensor_size=1,
        cuda_graph_impl="none",
        **common,
    ).cuda()
    base_model.train()
    base_logits, base_grads = _run_eager_iters(
        base_model,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        num_iters=3,
    )
    del base_model
    _reset_cuda_memory()

    off_interface.reset_instance()
    graph_model = _build_gpt_model_with_cuda_graph(
        fine_grained_activation_offloading=True,
        offload_modules=offload_modules,
        min_offloaded_tensor_size=1,
        cuda_graph_impl="chunk_wise",
        fine_grained_offloading_graph_keep_last_group=keep_last_group,
        **common,
    ).cuda()
    graph_model.train()
    if offload_modules == ["attn_norm"]:
        assert all(layer.offload_attn_norm for layer in graph_model.decoder.layers)
    graph_run = dict(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        num_warmup_iters=1,
        num_measure_iters=2,
        enable_offload_reset=True,
    )
    if capture_raises:
        from megatron.core.transformer import cuda_graphs

        def failing_make_graphed_callables(*args, **kwargs):
            raise RuntimeError("injected graph creation failure")

        monkeypatch.setattr(cuda_graphs, "make_graphed_callables", failing_make_graphed_callables)
        with pytest.raises(RuntimeError, match="injected graph creation failure"):
            _run_iters_with_cuda_graph(graph_model, **graph_run)
        assert not cuda_graphs.is_graph_capturing()
        # No capture-time forward ran: the manager holds no host buffer for a graph.
        assert captures == [(False, [])]
        # The warm-up pool was handed back all the same; the eager step below refills it.
        assert pooled_around_capture[0] > 0 and pooled_around_capture[1] == 0
    else:
        graph_logits, graph_grads, _ = _run_iters_with_cuda_graph(graph_model, **graph_run)
        _assert_chunk_graph_matches_eager(graph_logits, graph_grads, base_logits, base_grads)
        assert len(captures) == 1
        captured_keep_last_group, host_buffer_sizes = captures[0]
        assert captured_keep_last_group == keep_last_group
        if common["recompute_granularity"] is None:
            # A full hidden state went through a captured offload and reload.
            hidden_state_size = seq_length * micro_batch_size * hidden_size
            assert max(host_buffer_sizes, default=0) >= hidden_state_size, host_buffer_sizes
            # The eager warm-up offloaded through the shared pool, which the capture emptied
            # before it allocated the buffers above.
            assert pooled_around_capture[0] > 0 and pooled_around_capture[1] == 0
        else:
            # A fully recomputed block saves no group tensor in its capture-time forward.
            assert not host_buffer_sizes
    if common["recompute_granularity"] is None:
        # No graphs own the offloads any more (the helper deleted them when it returned) or ever
        # did (the capture raised), which leaves them to the eager policy: an eager step is
        # served by the warm-up handler, offloads, and matches the reference.
        manager = PipelineOffloadManager.get_instance()
        assert not manager._block_graphs_captured
        assert not manager._block_capture_live_chunks
        off_interface.reset()
        graph_model.zero_grad()
        eager_logits = graph_model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        handler = manager.cur_forward_chunk()
        assert handler is not None and handler in manager._cached_chunks_forward
        assert any(handler._group_has_offloaded_tensor(group) for group in handler.offload_groups)
        assert pooled_host_tensors(manager) > 0  # refilled on demand
        eager_logits.float().sum().backward()
        torch.cuda.synchronize()
        eager_grads = {
            name: (p.grad.detach().float().cpu() if p.grad is not None else None)
            for name, p in graph_model.named_parameters()
        }
        _assert_chunk_graph_matches_eager(
            eager_logits.detach().float().cpu(), eager_grads, base_logits, base_grads
        )
    del graph_model
    _reset_cuda_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not is_te_min_version("2.14.0"), reason="CUDA Graph with TE RNG tracker requires TE >= 2.14.0"
)
@pytest.mark.parametrize("position_embedding_type", ["yarn", "mrope"])
def test_chunk_wise_cuda_graph_replays_yarn_and_mrope_rotary_tables(
    position_embedding_type, chunk_graph_process_state
):
    """Chunk-wise TE CUDA graphs with yarn / fixed-length mRoPE, against eager.

    The block's sample inputs carry a rotary table of the runtime shape (``get_rotary_pos_emb``
    materializes it for block callables), which the runtime table overwrites at every replay;
    a capture without the table would replay attention without RoPE and diverge from eager."""
    seq_length, micro_batch_size = 128, 2
    device = torch.device("cuda")
    input_ids, position_ids, attention_mask = _make_gpt_inputs(
        seq_length=seq_length, micro_batch_size=micro_batch_size, device=device
    )
    common = dict(
        seed=123,
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        vocab_size=128,
        seq_length=seq_length,
        num_experts=None,
        is_mla=False,
        fine_grained_activation_offloading=False,
        offload_modules=None,
        min_offloaded_tensor_size=1,
        cuda_graph_modules=[],
        cuda_graph_warmup_steps=1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        recompute_granularity=None,
        position_embedding_type=position_embedding_type,
    )
    if position_embedding_type == "yarn":
        common.update(
            config_attrs=dict(
                yarn_rotary_scaling_factor=2.0,
                yarn_original_max_position_embeddings=seq_length // 2,
                yarn_beta_fast=32.0,
                yarn_beta_slow=1.0,
                yarn_mscale=1.0,
                yarn_mscale_all_dim=0.0,
                yarn_correction_range_round_to_int=True,
            )
        )
    else:
        # kv_channels = 64 -> rotary dim 64 -> the T/H/W sections span dim/2 = 32 frequencies.
        common.update(mrope_section=[16, 8, 8])
        # Runtime mRoPE position ids are [3, batch, seq]; give the three axes distinct values.
        position_ids = torch.stack(
            [position_ids, position_ids // 4, position_ids // 8]
        ).contiguous()

    _reset_cuda_memory()
    base_model = _build_gpt_model_with_cuda_graph(cuda_graph_impl="none", **common).cuda()
    base_model.train()
    base_logits, base_grads = _run_eager_iters(
        base_model,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        num_iters=3,
    )
    del base_model
    _reset_cuda_memory()

    graph_model = _build_gpt_model_with_cuda_graph(cuda_graph_impl="chunk_wise", **common).cuda()
    graph_model.train()
    graph_logits, graph_grads, _ = _run_iters_with_cuda_graph(
        graph_model,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        num_warmup_iters=1,
        num_measure_iters=2,
        enable_offload_reset=False,
    )
    del graph_model
    _reset_cuda_memory()

    _assert_chunk_graph_matches_eager(graph_logits, graph_grads, base_logits, base_grads)

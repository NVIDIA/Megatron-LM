# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    OffloadTensorGroup,
    PipelineOffloadManager,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for offloading tests.")
@pytest.mark.parametrize("num_groups", [9, 17])
def test_partial_offload_backward_does_not_accumulate_reloaded_groups(monkeypatch, num_groups):
    """Actual H2D reloads must stay one group ahead of a sequential backward pass.

    A chain of sin operations saves one equally sized activation per group. Warmup
    reserves the final group, and fraction=0.5 keeps the latter half of the remaining
    groups on GPU. Reloading at every resident group's backward boundary would pull
    all earlier offloaded groups back to GPU before any of them are consumed.

    Observe real reload events and saved-tensor consumption without changing stream
    ordering or calling the scheduler ourselves. Use interfaces shared with the
    unfixed implementation so the same test can demonstrate the original regression.
    """
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
            activation_offload_fraction=0.5,
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
        output = forward()
        torch.cuda.synchronize()
        output.sum().backward()
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
        num_offloaded = (num_groups - 1) // 2
        assert list(offloaded_groups.values()) == list(range(num_offloaded))
        assert all(len(group._tensors) == 1 for group in chunk.offload_groups)

        monkeypatch.setattr(OffloadTensorGroup, "record_reload_event", record_reload_event)
        monkeypatch.setattr(OffloadTensorGroup, "pop_tensor", pop_tensor)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        output.sum().backward()
        torch.cuda.synchronize()
        backward_peak = torch.cuda.max_memory_allocated()

        assert backward_order == list(reversed(range(num_groups)))
        assert [index for _, index in reload_events] == list(reversed(range(num_offloaded)))
        assert all(not group._tensors for group in chunk.offload_groups)
        peak_pending = max(len(pending) for _, _, pending, _ in samples)
        peak_pending_bytes = max(size for _, _, _, size in samples)
        print(
            f"Partial offload cadence: groups={num_groups}, offloaded={num_offloaded}, "
            f"peak_pending_groups={peak_pending}, peak_pending_bytes={peak_pending_bytes}, "
            f"backward_peak_bytes={backward_peak}, reloads=(backward_group, reload_group) "
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

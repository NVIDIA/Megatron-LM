# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Complete recorded execution with mocked physical communication buffers."""

from contextlib import ExitStack, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import FsdpContext
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionAction,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.runtime_policy import (
    PrefetchOrder,
    RuntimePolicy,
)


def test_complete_recorded_execution_and_invalid_schedules():
    with ExitStack() as stack:
        for name in ("range_push", "range_pop"):
            stack.enter_context(patch.object(torch.cuda.nvtx, name))
        stack.enter_context(patch.object(torch.cuda, "Stream", return_value=Mock(device="cpu")))
        stack.enter_context(patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()))
        stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        sample = stack.enter_context(
            patch.object(
                torch.cuda,
                "memory_allocated",
                side_effect=lambda _: sum(
                    group.unsharded_storage_bytes
                    for owner in owners
                    for group in owner.parameter_groups
                    if group._unshard_events
                ),
            )
        )
        context = FsdpContext(
            device=torch.device("cuda"),
            runtime_policy=RuntimePolicy(
                prefetch_order=PrefetchOrder.RECORDED, autograd_completes_execution=False
            ),
        )
        context.current_stream = Mock(return_value=Mock())
        owners = [torch.nn.Identity(), torch.nn.Identity()]
        for owner in owners:
            group = Mock(unsharded_storage_bytes=40, _unshard_events={}, is_rowwise_unsharded=False)
            group.unshard_parameters.side_effect = (
                lambda phase, group=group: group._unshard_events.update(
                    {Phase.FORWARD: object(), Phase.BACKWARD: object()}
                )
            )
            group.release_unsharded_storage.side_effect = (
                lambda *, rowwise_only, group=group: group._unshard_events.clear()
            )
            owner.parameter_groups = owner._parameter_groups = (group,)
            owner.context = context
            owner.phase = Phase.RESTING
            owner.name = "block"
            owner.is_root = Mock(return_value=False)
            owner._has_delayed_weight_gradients = Mock(return_value=False)
            owner._schedule_policy = SimpleNamespace(
                register_hooks=True,
                unshard_all_planes_in_backward=False,
                forward_prefetch_size=0,
                backward_prefetch_size=0,
            )
            owner._nvtx_range = Mock(side_effect=lambda _: nullcontext())
            owner._nvtx_label = Mock(return_value="record-host-test")
            for method in (
                "unshard",
                "_unshard_parameter_groups",
                "_prefetch_parameter_groups",
                "reshard",
                "bind_sharded_parameters",
                "release_unsharded_storage",
                "_release_unsharded_storage",
                "_execution_event",
            ):
                setattr(owner, method, MethodType(getattr(FsdpModule, method), owner))
            context.register_module(owner)
        context.finalize()

        def cycle(owner):
            FsdpModule.pre_forward(owner)
            FsdpModule.post_forward(owner)
            FsdpModule.pre_backward(owner)
            FsdpModule.post_backward(owner)

        def schedule():
            for _ in range(2):
                for owner in owners:
                    cycle(owner)
                context._post_autograd()

        for owner in owners:
            owner.eval()
        for owner in owners:
            FsdpModule.pre_forward(owner)
            FsdpModule.post_forward(owner)
        for owner in owners:
            owner.train()
        assert context._recording_execution_order and len(context.execution_order) == 0
        gather = owners[0]._parameter_groups[0].unshard_parameters
        gather.reset_mock()
        context.prefetch_next()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with pytest.raises(RuntimeError, match="before CUDA graph capture"):
                owners[0].unshard(Phase.FORWARD)
        gather.assert_not_called()
        assert len(context.execution_order) == 0
        schedule()
        assert context.is_recording_execution_order and len(context.execution_order) == 16
        context.post_backward()
        recorded = tuple(context.execution_order)
        view = context.execution_order_view
        assert view.position == 0 and len(view) == 16
        assert [event.allocated_bytes for event in recorded] == [0, 40] * 8
        assert all(event.unsharded_bytes == 40 for event in recorded)
        assert [event.action for event in recorded[:4]] == [
            ExecutionAction.UNSHARD,
            ExecutionAction.RESHARD,
            ExecutionAction.UNSHARD,
            ExecutionAction.RESHARD,
        ]
        sample.reset_mock()
        for owner in owners:
            owner.eval()
        gather.reset_mock()
        context.prefetch_next()
        gather.assert_not_called()
        for owner in owners:
            FsdpModule.pre_forward(owner)
            FsdpModule.post_forward(owner)
        assert view.position == 0
        for owner in owners:
            owner.train()
        schedule()
        assert context.execution_order_view is view and view.position == len(view)
        gather = owners[0]._parameter_groups[0].unshard_parameters
        gather.reset_mock()
        context.prefetch_next()  # Do not wrap to the first event of the next iteration.
        with pytest.raises(RuntimeError, match="diverged"):
            owners[0].unshard(Phase.FORWARD)
        gather.assert_not_called()
        context.post_backward()
        assert view.position == 0
        owner = owners[0]
        owner.phase = Phase.FORWARD
        context.prefetch_next()
        gather.assert_called_once_with(Phase.FORWARD)
        assert view.position == context.prefetch_order_view.position == 0
        owner.unshard(Phase.FORWARD)
        gather.reset_mock()
        context.prefetch_next()  # Do not look past the next reshard.
        gather.assert_not_called()
        # Both filtered orders could accept this, but the recorded interleaving cannot.
        other = owners[1]
        other.phase = Phase.FORWARD
        other._parameter_groups[0].unshard_parameters.reset_mock()
        with pytest.raises(RuntimeError, match="diverged"):
            other.unshard(Phase.FORWARD)
        other._parameter_groups[0].unshard_parameters.assert_not_called()
        with pytest.raises(RuntimeError, match="complete"):
            context.post_backward()
        assert view.position == 1
        owner.phase = Phase.BACKWARD
        with pytest.raises(RuntimeError, match="diverged"):
            owner.reshard()
        owner.phase = Phase.FORWARD
        with pytest.raises(RuntimeError, match="diverged"):
            owner._release_unsharded_storage(rowwise_only=True)
        position = view.position
        owner.reshard(force=True)
        assert view.position == position
        source = context.execution_order
        context.execution_order = ExecutionOrder(recorded)
        with pytest.raises(RuntimeError, match="replaced"):
            owner.unshard(Phase.FORWARD)
        context.execution_order = source
        assert tuple(source) == recorded
        sample.assert_not_called()

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Recorded weight retention through MCore's iteration boundary, without CUDA."""

from contextlib import ExitStack, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.distributed.fsdp.mcore_adapter_v2 import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.context import (
    FsdpContext,
    GradientFinalizationMode,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.runtime_policy import (
    PrefetchOrder,
    RuntimePolicy,
)


def _owner(context, size, groups, residency, prefetched):
    owner = torch.nn.Identity()
    group = Mock(unsharded_storage_bytes=size, _unshard_events={}, is_rowwise_unsharded=False)
    group.buffer, group.allocations = None, 0

    def gather(phase):
        if not group._unshard_events:
            group.buffer = object()
            group.allocations += 1
            if owner.phase is Phase.RESTING:
                prefetched.append(owner)
        residency.append(sum(item.buffer is not None for item in groups))
        group._unshard_events.update({Phase.FORWARD: object(), Phase.BACKWARD: object()})

    def release(*, rowwise_only):
        group._unshard_events.clear()
        group.buffer = None
        residency.append(sum(item.buffer is not None for item in groups))

    groups.append(group)
    group.unshard_parameters.side_effect = gather
    group.release_unsharded_storage.side_effect = release
    owner.parameter_groups = owner._parameter_groups = (group,)
    owner.context, owner.phase, owner.name = context, Phase.RESTING, "layer"
    owner.is_root = Mock(return_value=False)
    owner._has_delayed_weight_gradients = Mock(return_value=False)
    owner._schedule_policy = SimpleNamespace(
        register_hooks=True,
        unshard_all_planes_in_backward=False,
        forward_prefetch_size=None,
        backward_prefetch_size=None,
    )
    owner._nvtx_range = Mock(side_effect=lambda _: nullcontext())
    owner._nvtx_label = Mock(return_value="retention-host-test")
    for name in (
        "unshard",
        "_unshard_parameter_groups",
        "_prefetch_parameter_groups",
        "reshard",
        "bind_sharded_parameters",
        "release_unsharded_storage",
        "_release_unsharded_storage",
        "_execution_event",
    ):
        setattr(owner, name, MethodType(getattr(FsdpModule, name), owner))
    context.register_module(owner)
    return owner


@pytest.mark.parametrize(
    "mode", (GradientFinalizationMode.REDUCE_OUTER_AND_WAIT, GradientFinalizationMode.REDUCE_OUTER)
)
def test_record_compile_replay_and_iteration_cleanup(mode):
    with ExitStack() as stack:
        for name in ("range_push", "range_pop"):
            stack.enter_context(patch.object(torch.cuda.nvtx, name))
        stack.enter_context(patch.object(torch.cuda, "Stream", return_value=Mock(device="cpu")))
        stack.enter_context(patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()))
        stack.enter_context(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )
        stack.enter_context(patch.object(torch.cuda, "memory_allocated", return_value=100))
        agree = stack.enter_context(patch.object(torch.distributed, "all_reduce"))
        context = FsdpContext(
            device=torch.device("cuda"),
            gradient_finalization_mode=mode,
            runtime_policy=RuntimePolicy(
                prefetch_order=PrefetchOrder.RECORDED, autograd_completes_execution=False
            ),
        )
        context.current_stream = Mock(return_value=Mock())
        physical_groups, residency, prefetched = [], [], []
        owners = [
            _owner(context, size, physical_groups, residency, prefetched) for size in (40, 60000)
        ]
        context.finalize()
        chunks = [Mock(spec=FullyShardedDataParallelV2) for _ in owners]
        for chunk, owner in zip(chunks, owners):
            chunk.module = torch.nn.Sequential(owner)
            chunk.module.context = context
            chunk.ddp_config = SimpleNamespace(megatron_fsdp_weight_retention=True)
            chunk.finish_grad_sync = Mock(side_effect=lambda **_: context.finish_grad_sync())
        config = SimpleNamespace(
            timers=None,
            gtp_weight_remat_size=1,
            expert_gtp_weight_remat_size=1,
            moe_router_enable_expert_bias=False,
            moe_router_load_balancing_type="none",
            calculate_per_token_loss=False,
        )
        replacements = {
            name: Mock()
            for name in finalize_model_grads.__globals__
            if name.startswith("_allreduce_") or name == "reset_model_temporary_tensors"
        }
        replacements["get_model_config"] = lambda _: config
        stack.enter_context(patch.dict(finalize_model_grads.__globals__, replacements))
        compiler_globals = FullyShardedDataParallelV2.compile_weight_retention.__globals__
        planner = Mock(wraps=compiler_globals["build_weight_plan"])
        stack.enter_context(patch.dict(compiler_globals, build_weight_plan=planner))
        group = object()
        groups = SimpleNamespace(tp=None, pp=None, embd=None, pos_embd=None, dp_cp=group)
        allocations = []
        for iteration in range(2):
            before = sum(owner._parameter_groups[0].allocations for owner in owners)
            for microbatch in range(2):
                for owner in owners:
                    if iteration and microbatch == 0 and owner is owners[1]:
                        assert owner._parameter_groups[0].buffer is not None
                    FsdpModule.pre_forward(owner)
                    buffer = owner._parameter_groups[0].buffer
                    FsdpModule.post_forward(owner)
                    if iteration and microbatch == 0:
                        assert owner._parameter_groups[0].buffer is buffer
                    FsdpModule.pre_backward(owner)
                    FsdpModule.post_backward(owner)
                waits = context.current_stream().wait_stream.call_count
                context._post_autograd()
                assert context.gradient_finalization_mode is mode
                assert context.current_stream().wait_stream.call_count - waits == (
                    mode is GradientFinalizationMode.REDUCE_OUTER_AND_WAIT
                )
                if microbatch == 0:
                    if iteration:
                        assert all(
                            owner._parameter_groups[0].buffer is not None for owner in owners
                        )
                    else:
                        assert context.is_recording_execution_order
            # Speculative gathering is outside the logical trace and must still be flushed.
            owners[0]._unshard_parameter_groups(Phase.FORWARD)
            assert owners[0]._parameter_groups[0].buffer is not None
            finalize_model_grads(chunks, pg_collection=groups)
            assert all(owner._parameter_groups[0].buffer is None for owner in owners)
            assert context.execution_order_view.position == 0
            assert len(context.execution_order) == 16
            allocations.append(
                sum(owner._parameter_groups[0].allocations for owner in owners) - before
            )
        assert context.retain_mask
        assert allocations[1] < allocations[0]
        planner.assert_called_once()
        assert planner.call_args.args == (context.execution_order, group)
        assert prefetched and max(residency) == 2
        assert agree.call_count == 1 and agree.call_args.kwargs["group"] is group

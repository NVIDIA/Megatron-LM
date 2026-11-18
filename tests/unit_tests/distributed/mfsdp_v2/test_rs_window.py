# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static RS packets respect real sharding, byte capacity and gradient numerics."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from torch.distributed.tensor import Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionAction,
    ExecutionEvent,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.gradient_window import (
    build_rs_plan,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase


def test_plan_cuts_at_outer_and_four_and_leaves_unprofitable_groups_on_baseline():
    agreement_group = object()

    def group(numel, *, size=2, placement=Shard(0)):
        mesh = Mock(ndim=1)
        mesh.size.return_value = size
        mesh.get_group.return_value = agreement_group
        result = Mock(requires_grad=True, requires_inner_reduction=True)
        result.mesh = mesh
        result.main_grad.dtype = torch.float32
        result.main_grad.placements = (placement,)
        result.grad_comm_dtype = torch.bfloat16
        layout = SimpleNamespace(
            tensor_shapes=((numel,),),
            tensor_to_offset=(0,),
            size=numel,
            rank_to_offset=(0,),
            get_local_range=lambda *_: SimpleNamespace(numel=numel),
        )
        result.main_grad.layout = result.main_weight.layout = layout
        result.grad_scale = 1.0
        return result

    owner = nn.Module()
    # The first group costs16B:8B accumulator +4B cast +4B conservative scratch.
    owner.parameter_groups = (
        group(2),
        group(100),
        group(2, size=1),
        group(2, placement=Partial("avg")),
        group(2, placement=Replicate()),
    )
    order = ExecutionOrder()
    ids = []
    for index in range(8):
        if index == 3:
            order.append(ExecutionEvent(owner, Phase.BACKWARD, ExecutionAction.UNSHARD))
            order.append(ExecutionEvent(owner, Phase.BACKWARD, ExecutionAction.RESHARD))
        ids.append(
            order.append(
                ExecutionEvent(
                    owner, Phase.BACKWARD, ExecutionAction.GRAD_READY, finalizes_outer=index == 2
                )
            )
        )

    with (
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch("torch.distributed.all_reduce"),
    ):
        plan = build_rs_plan(order, agreement_group, device=torch.device("cpu"), budget_bytes=16)
        assert dict(plan) == {
            **{(event_id, 0): ids[2] for event_id in ids[:3]},
            **{(event_id, 0): ids[6] for event_id in ids[3:7]},
        }
        assert not build_rs_plan(
            order, agreement_group, device=torch.device("cpu"), budget_bytes=15
        )
    with pytest.raises(TypeError):
        plan[(ids[0], 0)] = ids[0]


def test_real_packing_accumulates_changed_bf16_gradients_once_before_rs():
    class Buffer:
        def __init__(self, dtype):
            self.dtype = dtype
            self.tensor = torch.empty(4, dtype=dtype)

        def get_tensor_view(self, index):
            assert index == 0
            return self.tensor

        def cast(self, dtype):
            result = Buffer(dtype)
            result.tensor = self.tensor.to(dtype)
            return result

    parameter = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    group = Mock(grad_comm_dtype=torch.bfloat16)
    group.main_grad.dtype = torch.float32
    group.fsdp_parameters = (SimpleNamespace(unsharded=parameter, fqns=("weight",)),)
    group.allocate_partial_grad_buffer.side_effect = lambda *, dtype: Buffer(dtype)
    group.copy_gradients_to_partial_buffer = MethodType(
        FsdpParameterGroup.copy_gradients_to_partial_buffer, group
    )
    reduced = []
    group.reduce_inner_gradients.side_effect = lambda buffer: reduced.append(buffer.tensor.clone())
    compute, inner = Mock(), Mock()
    context = SimpleNamespace(
        rs_inputs={}, current_stream=lambda: compute, inner_reduction_stream=inner
    )
    owner = SimpleNamespace(context=context)
    with (
        patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()),
        patch.object(torch.cuda, "current_stream", return_value=compute),
        patch.object(torch.Tensor, "record_stream"),
    ):
        for index, value in enumerate((1, 3, 5)):
            (parameter * value).sum().backward()
            FsdpModule._accumulate_rs_window(owner, group, flush=index == 2)
            assert parameter.grad is None
            assert len(reduced) == (1 if index == 2 else 0)
    assert len(reduced) == 1
    assert reduced[0].dtype == torch.bfloat16
    assert torch.equal(reduced[0], torch.full((4,), 9, dtype=torch.bfloat16))
    assert not context.rs_inputs

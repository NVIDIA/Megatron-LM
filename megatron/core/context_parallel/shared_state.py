# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit CP redistribution for shared tensor fields, independent of their meaning."""

from dataclasses import replace
from typing import Callable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core.context_parallel.layout import (
    CPLayout,
    THDCPLayoutPlan,
    _get_layout_parallel_context,
    _local_segment_ids,
    convert_cp_layout,
)
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.state_boundary import TensorField, TensorSchema


def _map_sharded_fields(
    state: Mapping[str, Tensor | None],
    schema: TensorSchema,
    convert: Callable[[TensorField, Tensor], tuple[Tensor, str]],
) -> tuple[dict[str, Tensor | None], TensorSchema]:
    """Convert present sequence-sharded fields; replicated/local/strided fields stay in place."""
    values = schema.unpack(schema.pack(state))
    fields = []
    for field in schema.fields:
        if field.present and field.layout in ("contiguous", "zigzag"):
            values[field.key], layout = convert(field, values[field.key])
            field = replace(field, shape=tuple(values[field.key].shape), layout=layout)
        fields.append(field)
    return {**state, **values}, TensorSchema(tuple(fields))


def redistribute_state(
    state: Mapping[str, Tensor | None],
    schema: TensorSchema,
    target_layout: CPLayout,
    cp_group: dist.ProcessGroup,
    *,
    sequence_parallel: bool = False,
    tp_group: dist.ProcessGroup | None = None,
    tp_cp_group: dist.ProcessGroup | None = None,
    thd_plans: Mapping[str, THDCPLayoutPlan] | None = None,
) -> tuple[dict[str, Tensor | None], TensorSchema]:
    """Reuse main's contiguous/zigzag exchange for every sharded field.

    Sharded fields have sequence on dimension zero. Owners choose which fields
    are sharded and provide packed layout metadata; this function does not infer
    compression groups, halo ownership or logical token positions.
    """
    if target_layout not in ("contiguous", "zigzag"):
        raise ValueError("Shared state redistribution requires contiguous or zigzag layout")
    _get_layout_parallel_context(cp_group, sequence_parallel, tp_group, tp_cp_group)

    def convert(field, tensor):
        plan = None if thd_plans is None else thd_plans.get(field.key)
        return (
            convert_cp_layout(
                tensor,
                field.layout,
                target_layout,
                cp_group,
                sequence_parallel=sequence_parallel,
                tp_group=tp_group,
                tp_cp_group=tp_cp_group,
                thd_plan=plan,
            ),
            target_layout,
        )

    return _map_sharded_fields(state, schema, convert)


def gather_state(
    state: Mapping[str, Tensor | None],
    schema: TensorSchema,
    cp_group: dist.ProcessGroup,
    *,
    sequence_parallel: bool = False,
    tp_group: dist.ProcessGroup | None = None,
    tp_cp_group: dist.ProcessGroup | None = None,
    tp_replicated: bool = False,
) -> tuple[dict[str, Tensor | None], TensorSchema]:
    """Gather equal CP/SP shards in sequence order; sum consumer gradients to each owner.

    One all-gather writes all shards, followed by a local reorder without host
    synchronization, so the call is safe inside CUDA graph capture. Backward
    reduce-scatters, which is correct when every rank contributes a distinct
    gradient (e.g. TP-partitioned heads). With sequence parallelism, set
    ``tp_replicated`` when TP ranks consume the gathered state identically, so
    their duplicate contributions are averaged instead of summed.
    """
    context = _get_layout_parallel_context(cp_group, sequence_parallel, tp_group, tp_cp_group)
    group = context.communication_group

    def convert(field, tensor):
        tensor = tensor.contiguous()
        if group.size() == 1:
            return tensor, "replicated"
        if field.differentiable and torch.is_grad_enabled():
            tensor = gather_from_sequence_parallel_region(tensor, group=group)
        else:
            gathered = tensor.new_empty((tensor.shape[0] * group.size(), *tensor.shape[1:]))
            dist.all_gather_into_tensor(gathered, tensor, group=group)
            tensor = gathered
        if field.layout == "contiguous":
            order = context.group_rank_by_logical_rank
        else:
            owners = {}
            for cp_rank in range(context.cp_size):
                for tp_rank in range(context.tp_size):
                    segments = _local_segment_ids(
                        "zigzag", context.cp_size, cp_rank, context.tp_size, tp_rank
                    )
                    rank = context.group_rank_by_logical_rank[cp_rank * context.tp_size + tp_rank]
                    owners.update(
                        (segment, rank * len(segments) + i) for i, segment in enumerate(segments)
                    )
            order = tuple(owners[segment] for segment in sorted(owners))
        if tensor.shape[0] % len(order):
            raise ValueError("Shared state length is not divisible by its layout segments")
        if order != tuple(range(len(order))):
            chunks = tensor.chunk(len(order))
            tensor = torch.cat([chunks[slot] for slot in order])
        if tp_replicated and context.tp_size > 1 and tensor.requires_grad:
            tensor.register_hook(lambda grad: grad / context.tp_size)
        return tensor, "replicated"

    return _map_sharded_fields(state, schema, convert)

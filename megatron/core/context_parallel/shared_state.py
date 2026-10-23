# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit CP redistribution for shared tensor fields, independent of their meaning."""

from dataclasses import replace
from typing import Mapping

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
from megatron.core.transformer.state_boundary import TensorSchema


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
    """Reuse main's contiguous/zigzag exchange; replicated/local fields stay in place.

    Sharded fields have sequence on dimension zero. Owners choose which fields
    are sharded and provide packed layout metadata; this function does not infer
    compression groups, halo ownership or logical token positions.
    """
    if target_layout not in ("contiguous", "zigzag"):
        raise ValueError("Shared state redistribution requires contiguous or zigzag layout")
    _get_layout_parallel_context(cp_group, sequence_parallel, tp_group, tp_cp_group)
    values = schema.unpack(schema.pack(state))
    fields = []
    for field in schema.fields:
        if field.layout not in ("contiguous", "zigzag", "replicated", "local", "strided"):
            raise ValueError(f"State field {field.key} requires a model-owned CP layout adapter")
        if field.present and field.layout in ("contiguous", "zigzag"):
            values[field.key] = convert_cp_layout(
                values[field.key],
                field.layout,
                target_layout,
                cp_group,
                sequence_parallel=sequence_parallel,
                tp_group=tp_group,
                tp_cp_group=tp_cp_group,
                thd_plan=None if thd_plans is None else thd_plans.get(field.key),
            )
            field = replace(field, shape=tuple(values[field.key].shape), layout=target_layout)
        fields.append(field)
    updated = dict(state)
    updated.update(values)
    return updated, TensorSchema(tuple(fields))


def gather_state(
    state: Mapping[str, Tensor | None],
    schema: TensorSchema,
    cp_group: dist.ProcessGroup,
    *,
    sequence_parallel: bool = False,
    tp_group: dist.ProcessGroup | None = None,
    tp_cp_group: dist.ProcessGroup | None = None,
) -> tuple[dict[str, Tensor | None], TensorSchema]:
    """Gather equal CP/SP shards in sequence order; sum consumer gradients to each owner.

    Zigzag requires one all-gather followed by a local reorder. The existing
    sequence-parallel collective uses a single output buffer and reduce-scatter
    backward. Replicated/local fields stay in place; owners select the schema.
    """
    context = _get_layout_parallel_context(cp_group, sequence_parallel, tp_group, tp_cp_group)
    group = context.communication_group
    values = schema.unpack(schema.pack(state))
    fields = []
    for field in schema.fields:
        if field.layout not in ("contiguous", "zigzag", "replicated", "local", "strided"):
            raise ValueError(f"State field {field.key} requires a model-owned CP layout adapter")
        if field.present and field.layout in ("contiguous", "zigzag"):
            tensor = values[field.key].contiguous()
            if group.size() > 1:
                if field.differentiable and torch.is_grad_enabled():
                    tensor = gather_from_sequence_parallel_region(tensor, group=group)
                else:
                    gathered = tensor.new_empty((tensor.shape[0] * group.size(), *tensor.shape[1:]))
                    dist.all_gather_into_tensor(gathered, tensor, group=group)
                    tensor = gathered
                if field.layout == "contiguous":
                    permutation = context.group_rank_by_logical_rank
                else:
                    owners = []
                    for cp_rank in range(context.cp_size):
                        for tp_rank in range(context.tp_size):
                            segments = _local_segment_ids(
                                "zigzag", context.cp_size, cp_rank, context.tp_size, tp_rank
                            )
                            rank = context.group_rank_by_logical_rank[
                                cp_rank * context.tp_size + tp_rank
                            ]
                            owners.extend(
                                (segment, rank * len(segments) + i)
                                for i, segment in enumerate(segments)
                            )
                    permutation = tuple(slot for _, slot in sorted(owners))
                if tensor.shape[0] % len(permutation):
                    raise ValueError("Shared state length is not divisible by its layout segments")
                if permutation != tuple(range(len(permutation))):
                    shape = tensor.shape
                    chunks = tensor.reshape(len(permutation), -1, *shape[1:])
                    indices = torch.tensor(permutation, device=tensor.device)
                    tensor = chunks.index_select(0, indices).reshape(shape)
            values[field.key] = tensor
            field = replace(field, shape=tuple(tensor.shape), layout="replicated")
        fields.append(field)
    updated = dict(state)
    updated.update(values)
    return updated, TensorSchema(tuple(fields))

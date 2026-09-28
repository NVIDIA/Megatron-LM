# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit CP redistribution for shared tensor fields, independent of their meaning."""

from dataclasses import replace
from typing import Mapping

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather

from megatron.core.context_parallel.layout import CPLayout, THDCPLayoutPlan, convert_cp_layout
from megatron.core.transformer.state_boundary import TensorSchema


def redistribute_state(
    state: Mapping[str, Tensor | None],
    schema: TensorSchema,
    target_layout: CPLayout,
    cp_group: dist.ProcessGroup,
    *,
    thd_plans: Mapping[str, THDCPLayoutPlan] | None = None,
) -> tuple[dict[str, Tensor | None], TensorSchema]:
    """Reuse main's contiguous/zigzag exchange; replicated/local fields stay in place.

    Sharded fields have sequence on dimension zero. Owners choose which fields
    are sharded and provide packed layout metadata; this function does not infer
    compression groups, halo ownership or logical token positions.
    """
    if target_layout not in ("contiguous", "zigzag"):
        raise ValueError("Shared state redistribution requires contiguous or zigzag layout")
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
                thd_plan=None if thd_plans is None else thd_plans.get(field.key),
            )
            field = replace(field, shape=tuple(values[field.key].shape), layout=target_layout)
        fields.append(field)
    updated = dict(state)
    updated.update(values)
    return updated, TensorSchema(tuple(fields))


def gather_state(
    state: Mapping[str, Tensor | None], schema: TensorSchema, cp_group: dist.ProcessGroup
) -> tuple[dict[str, Tensor | None], TensorSchema]:
    """Gather equal contiguous shards; backward sums consumer contributions to each owner.

    Replicated/local fields are unchanged. Integer and nondifferentiable fields
    use ordinary collectives. A model with compressed or padded layouts must
    canonicalize its own ownership before declaring a contiguous field.
    """
    values = schema.unpack(schema.pack(state))
    fields = []
    for field in schema.fields:
        if field.layout not in ("contiguous", "zigzag", "replicated", "local", "strided"):
            raise ValueError(f"State field {field.key} requires a model-owned CP layout adapter")
        if field.present and field.layout == "contiguous":
            tensor = values[field.key].contiguous()
            if cp_group.size() > 1:
                if field.differentiable and torch.is_grad_enabled():
                    pieces = all_gather(tensor, group=cp_group)
                else:
                    pieces = [torch.empty_like(tensor) for _ in range(cp_group.size())]
                    dist.all_gather(pieces, tensor, group=cp_group)
                tensor = torch.cat(pieces, dim=0)
            values[field.key] = tensor
            field = replace(field, shape=tuple(tensor.shape), layout="replicated")
        elif field.present and field.layout == "zigzag":
            raise ValueError("Convert zigzag state to contiguous before gathering")
        fields.append(field)
    updated = dict(state)
    updated.update(values)
    return updated, TensorSchema(tuple(fields))

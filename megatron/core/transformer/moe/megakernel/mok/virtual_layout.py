# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoK descriptors for the native and MoonEP replica slots in physical expert order."""

import torch

from megatron.core.transformer.moe.megakernel.mok.weights import (
    _parameter_storage_attr,
    _storage_view,
)
from megatron.core.transformer.moe.megakernel.parameter_bridge import main_grad_buffer


def _grouped_members(grouped):
    """Split a grouped payload and its already-swizzled scales into zero-copy expert views."""
    if isinstance(grouped, torch.Tensor):
        return grouped.unbind(0)
    row, row_scale, col, col_scale, native_columnwise = grouped
    if not native_columnwise:
        raise ValueError("MoK virtual experts require native-columnwise MXFP8 weights")
    count = row.shape[0]
    return tuple(
        (r, rs, c, cs, True)
        for r, rs, c, cs in zip(
            row.unbind(), row_scale.chunk(count), col.unbind(), col_scale.chunk(count), strict=True
        )
    )


def _owner_members(params, native, single_grouped):
    """Recover each native matrix address while reusing MoK's existing scale conversion."""
    if single_grouped:
        return _grouped_members(native)

    members = []
    for i, param in enumerate(params):
        row = _parameter_storage_attr(param, "rowwise_data")
        if native.scale is None:
            storage = param if row is None else row
            members.append(_storage_view(storage, param.shape, dtype=torch.bfloat16, name="owner"))
            continue
        col = _parameter_storage_attr(param, "columnwise_data")
        members.append(
            (
                _storage_view(row, param.shape, dtype=torch.float8_e4m3fn, name="owner rowwise"),
                native.scale_tensors[i],
                _storage_view(col, param.shape, dtype=torch.float8_e4m3fn, name="owner columnwise"),
                native.transposed_scale_tensors[i],
                True,
            )
        )
    return tuple(members)


def _weight_descriptor(members):
    """Point a single MoK descriptor table at native DDP payloads and the shared replica pool."""
    from mok import functional, ops

    first = members[0]
    if isinstance(first, torch.Tensor):
        return functional.SplitRoutedWeight(
            data=first, storage_table=ops.make_routed_weight_storage_table_bf16(list(members))
        )
    row, row_scales, col, col_scales, _ = zip(*members, strict=True)
    return functional.SplitRoutedWeight(
        data=row[0],
        storage_table=ops.make_routed_weight_storage_table_mxfp8(list(row)),
        scale=row_scales[0],
        scale_storage_table=ops.make_routed_scale_storage_table(list(row_scales)),
        scale_tensors=row_scales,
        transposed_data=col[0],
        transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(list(col)),
        transposed_scale=col_scales[0],
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(list(col_scales)),
        transposed_scale_tensors=col_scales,
        native_columnwise=True,
    )


class MoKVirtualLayout:
    """Exchange replicas through MoonEP while leaving MoK and the optimizer on their native APIs."""

    def __init__(self, config, group, num_local_experts):
        from megatron.core.transformer.moe.mok_virtual_experts import MoKVirtualExperts

        self.transport = MoKVirtualExperts(config, group, num_local_experts)
        self.clear()

    def clear(self):
        """Drop views and descriptors tied to the previous owner storage."""
        self.transport.clear()
        self.native_views = None
        self.owner_members = None
        self.weight_descriptors = None
        self.grad_fingerprint = None
        self.grad_tables = None

    def prepare_weights(self, module, plan, native_views, *, backward):
        """Publish replica weights and return MoK descriptors in physical slot order."""
        # MoK refreshes its converted scales in place after an optimizer update. A new native
        # view may also change a backing address, e.g. on model reload; rebuild descriptors then.
        if self.native_views is None or any(
            view is not cached for view, cached in zip(native_views, self.native_views, strict=True)
        ):
            self.native_views = native_views
            self.owner_members = tuple(
                _owner_members(params, native, module.native_single_grouped_weights)
                for params, native in zip(
                    (module.routed_fc1_parameters, module.routed_fc2_parameters),
                    native_views,
                    strict=True,
                )
            )
            self.weight_descriptors = None
        slots = self.transport.prepare_weights(plan, self.owner_members, backward=backward)
        if self.weight_descriptors is None:
            self.weight_descriptors = tuple(
                _weight_descriptor(owner + _grouped_members(slot))
                for owner, slot in zip(self.owner_members, slots, strict=True)
            )
        return self.weight_descriptors

    def main_grad_arguments(self, module):
        """Use optimizer gradients for native slots and clearable arena gradients for replicas."""
        from mok import ops

        owners = tuple(
            tuple(main_grad_buffer(p) for p in params)
            for params in (module.routed_fc1_parameters, module.routed_fc2_parameters)
        )
        if module.native_single_grouped_weights:
            owners = tuple(group[0].unbind() for group in owners)
        physical = tuple(
            owner + slot.unbind()
            for owner, slot in zip(owners, self.transport.gradient_slots(), strict=True)
        )
        fingerprint = tuple((g.data_ptr(), g.dtype, g.shape) for group in physical for g in group)
        if fingerprint != self.grad_fingerprint:
            self.grad_tables = tuple(
                ops.make_routed_d_weight_storage_table(list(group)) for group in physical
            )
            self.grad_fingerprint = fingerprint
        shared_fc1 = module.shared_fc1_weight
        shared_fc2 = module.shared_fc2_weight
        grads = (
            main_grad_buffer(shared_fc1) if shared_fc1 is not None else None,
            owners[0][0],
            main_grad_buffer(shared_fc2) if shared_fc2 is not None else None,
            owners[1][0],
        )
        return grads, self.grad_tables, owners

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Megatron-FSDP fully_shard entrypoint."""

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .module import FsdpContext, FsdpModule
from .placement import Placement

MeshAxis = int | str


@dataclasses.dataclass(frozen=True)
class Placements:
    """Per-data-parallel-axis placements for FSDP buffers.

    ``dp_axes`` identifies the axes of the parent mesh that form FSDP's
    data-parallel mesh. Placement lists are ordered to match those axes.
    """

    dp_axes: list[MeshAxis]
    parameter: list[Placement]
    gradient: list[Placement]
    optimizer: list[Placement]

    def __post_init__(self) -> None:
        """Validate placement list lengths."""
        axis_count = len(self.dp_axes)
        for name, placements in (
            ("parameter", self.parameter),
            ("gradient", self.gradient),
            ("optimizer", self.optimizer),
        ):
            if len(placements) != axis_count:
                raise ValueError(f"Expected {axis_count} {name} placements, got {len(placements)}.")


def fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
    use_symm_mem: bool = False,
) -> None:
    """Apply FSDP to a module in place.

    This attaches the FSDP mixin to the original module instance, so parent
    modules do not need to replace existing child-module references.

    Args:
        module: Module whose currently unowned parameters are managed by FSDP.
        mesh: Parent device mesh containing the data-parallel axes.
        placements: Parameter, gradient, and optimizer placements.
        mixed_precision_policy: Optional precision policy. Defaults to FP32 main weights
            and parameter-dtype main gradients.
        use_symm_mem: Allocate all-gather and reduce-scatter staging buffers from
            PyTorch's NCCL symmetric-memory pool.
    """
    if isinstance(module, FsdpModule):
        raise ValueError("This module is already managed by FSDP.")

    dp_axes = _normalize_dp_axes(mesh, placements.dp_axes)
    model_weight_placements = tuple(placements.parameter)
    main_grad_placements = tuple(placements.gradient)
    main_weight_placements = tuple(placements.optimizer)
    _validate_placement_tuples(
        len(dp_axes),
        model_weight_placements=model_weight_placements,
        main_grad_placements=main_grad_placements,
        main_weight_placements=main_weight_placements,
    )

    mixed_precision_policy = mixed_precision_policy or MixedPrecisionPolicy()
    original_cls = module.__class__
    _attach_mixin(module)
    try:
        assert isinstance(module, FsdpModule)
        FsdpModule.__init__(
            module,
            mesh=mesh,
            dp_axes=dp_axes,
            model_weight_placements=model_weight_placements,
            main_grad_placements=main_grad_placements,
            main_weight_placements=main_weight_placements,
            mixed_precision_policy=mixed_precision_policy,
            use_symm_mem=use_symm_mem,
        )
    except Exception:
        module.__class__ = original_cls
        raise


@contextmanager
def microbatch(module: nn.Module, is_last: bool) -> Iterator[None]:
    """Scope FSDP state to one microbatch.

    Args:
        module: Module tree whose FSDP roots should use this microbatch state.
        is_last: Whether forwards in this scope are for the last microbatch.
    """
    contexts: list[FsdpContext] = []
    _collect_fsdp_contexts(module, contexts)
    previous_states = [(context, context.is_last_microbatch) for context in contexts]
    for context in contexts:
        context.is_last_microbatch = is_last

    try:
        yield
    finally:
        for context, is_last_microbatch in previous_states:
            context.is_last_microbatch = is_last_microbatch


def _attach_mixin(module: nn.Module) -> None:
    if isinstance(module, FsdpModule):
        return
    module_cls = module.__class__
    fsdp_cls = type(f"ExperimentalFsdp{module_cls.__name__}", (FsdpModule, module_cls), {})
    module.__class__ = fsdp_cls


def _collect_fsdp_contexts(module: nn.Module, contexts: list[FsdpContext]) -> None:
    if isinstance(module, FsdpModule):
        module._lazy_init_context()
        contexts.append(module.context)
        return

    for child in module.children():
        _collect_fsdp_contexts(child, contexts)


def _normalize_dp_axes(mesh: DeviceMesh, dp_axes: list[MeshAxis]) -> tuple[int, ...]:
    """Normalize and validate the data-parallel axes of a parent mesh.

    Subset selection is not supported yet. Keeping normalized axes separate
    from the parent mesh preserves the information needed to derive a DP
    submesh and reconstruct composed DTensors later.
    """
    axis_indices = tuple(_axis_index(mesh, axis) for axis in dp_axes)
    if len(set(axis_indices)) != len(axis_indices):
        raise ValueError(f"Placements.dp_axes must reference distinct mesh axes, got {dp_axes}.")
    if axis_indices != tuple(sorted(axis_indices)):
        raise ValueError(f"Placements.dp_axes must be in mesh-axis order, got {dp_axes}.")
    if axis_indices != tuple(range(mesh.ndim)):
        raise NotImplementedError(
            "FSDP requires Placements.dp_axes to select every parent-mesh axis for now."
        )
    return axis_indices


def _axis_index(mesh: DeviceMesh, axis: MeshAxis) -> int:
    """Normalize an integer or named parent-mesh axis to a non-negative index."""
    if isinstance(axis, int):
        axis_index = axis
        if axis_index < 0:
            axis_index += mesh.ndim
        if axis_index < 0 or axis_index >= mesh.ndim:
            raise ValueError(f"Mesh axis {axis} is out of bounds for mesh ndim {mesh.ndim}.")
        return axis_index

    dim_names = mesh.mesh_dim_names
    if dim_names is None or axis not in dim_names:
        raise ValueError(f"Mesh axis {axis!r} is not present in mesh dim names {dim_names}.")
    return dim_names.index(axis)


def _validate_placement_tuples(
    dp_ndim: int,
    *,
    model_weight_placements: tuple[Placement, ...],
    main_grad_placements: tuple[Placement, ...],
    main_weight_placements: tuple[Placement, ...],
) -> None:
    """Require one placement per dimension of the selected DP mesh."""
    for name, placements in (
        ("model_weight", model_weight_placements),
        ("main_grad", main_grad_placements),
        ("main_weight", main_weight_placements),
    ):
        if len(placements) != dp_ndim:
            raise ValueError(
                f"Expected {dp_ndim} {name} placements for the DP mesh, got {len(placements)}."
            )

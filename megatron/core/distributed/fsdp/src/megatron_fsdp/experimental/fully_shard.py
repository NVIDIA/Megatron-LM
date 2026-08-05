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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import MixedPrecisionPolicy
from .module import FsdpContext, FsdpModule

_FSDP_CONTEXT = ContextVar[FsdpContext | None]("mfsdp_context", default=None)

MeshAxis = int | str


@dataclasses.dataclass(frozen=True)
class Placements:
    """Per-data-parallel-axis placements for MFSDP buffers.

    ``dp_axes`` identifies the parent-mesh axes that form MFSDP's data-parallel
    mesh. Placement sequences are ordered to match those axes. Use Torch's
    ``Shard(0)`` for public parameter, gradient, and optimizer sharding.
    """

    dp_axes: Sequence[MeshAxis]
    parameter: Sequence[Placement]
    gradient: Sequence[Placement]
    optimizer: Sequence[Placement]

    def __post_init__(self) -> None:
        """Validate placement sequence lengths."""
        axis_count = len(self.dp_axes)
        for name, placements in (
            ("parameter", self.parameter),
            ("gradient", self.gradient),
            ("optimizer", self.optimizer),
        ):
            if len(placements) != axis_count:
                raise ValueError(f"Expected {axis_count} {name} placements, got {len(placements)}.")


@contextmanager
def fully_shard_context(
    device: torch.device | None = None,
    *,
    use_symmetric_memory: bool = False,
    unify_communication_stream: bool = False,
) -> Iterator[FsdpContext]:
    """Construct FSDP modules that share runtime streams and prefetch orders.

    Independent roots are ordered by their root-level ``fully_shard`` calls.
    Construction must finish before any of the registered modules run forward.

    Args:
        device: CUDA device on which to create communication streams. Defaults to
            the current CUDA device.
        use_symmetric_memory: Allocate communication staging buffers from PyTorch's
            NCCL symmetric-memory pool.
        unify_communication_stream: Whether all-gathers and reduce-scatters share one
            communication stream to reduce peak transient memory. See
            https://github.com/NVIDIA/Megatron-LM/issues/6471.
    """
    if _FSDP_CONTEXT.get() is not None:
        raise RuntimeError("fully_shard_context does not support nesting.")

    device = device or torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise ValueError(f"fully_shard_context requires a CUDA device, got {device}.")

    context = FsdpContext(
        device=device,
        use_symmetric_memory=use_symmetric_memory,
        unify_communication_stream=unify_communication_stream,
    )
    token = _FSDP_CONTEXT.set(context)
    try:
        yield context
    except Exception:
        raise
    else:
        context.finalize()
    finally:
        _FSDP_CONTEXT.reset(token)


def fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
    fine_grained: bool = False,
    skip_backward_callback: bool = False,
    grad_divisor: int = 1,
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
        fine_grained: Register pre-forward and pre-backward hooks on every sub-module
            so the 1F1B EP overlap schedule can call sub-modules directly.
        skip_backward_callback: Skip per-param post_accumulate_grad_hook. Required
            when ``delay_wgrad_compute=True`` so gradient reduction waits for
            ``backward_dw()`` to complete.
        grad_divisor: Additional divisor applied to the reduced gradient, on top of the
            averaging the mesh already performs. Defaults to 1, which is correct whenever
            each mesh rank contributes exactly one term to the gradient.

            Expert parallelism is the motivating case. A rank's experts process tokens
            routed to them from every rank in the expert-parallel group, and the backward
            pass routes those tokens' gradients back, so a rank's expert gradient already
            sums over ``ep_size`` ranks' data before any reduction happens. Averaging over
            the expert-data-parallel mesh alone therefore divides by too little, and
            ``grad_divisor=ep_size`` makes up the difference. Dense parameters see only
            their own rank's tokens and need no divisor.

        Parameters that are TE MXFP8 primary weights (detected via
        ``is_float8tensor`` + ``fp8_need_transpose_data``) are grouped into
        ``Fp8ParameterGroup`` automatically; no flag is needed.
    """
    if isinstance(module, FsdpModule):
        raise ValueError("This module is already managed by FSDP.")
    context = _FSDP_CONTEXT.get()
    if context is None:
        raise RuntimeError("fully_shard must run inside fully_shard_context.")
    for submodule in module.modules():
        if isinstance(submodule, FsdpModule) and submodule.context is not context:
            raise ValueError(
                "Cannot fully_shard a module containing an FSDP child from another "
                "fully_shard_context."
            )

    _validate_dp_axes(mesh, placements.dp_axes)
    mixed_precision_policy = mixed_precision_policy or MixedPrecisionPolicy()
    original_cls = module.__class__
    _attach_mixin(module)
    try:
        assert isinstance(module, FsdpModule)
        FsdpModule.__init__(
            module,
            context=context,
            mesh=mesh,
            model_weight_placements=tuple(placements.parameter),
            main_grad_placements=tuple(placements.gradient),
            main_weight_placements=tuple(placements.optimizer),
            mixed_precision_policy=mixed_precision_policy,
            fine_grained=fine_grained,
            skip_backward_callback=skip_backward_callback,
            grad_divisor=grad_divisor,
            use_symmetric_memory=context.use_symmetric_memory,
        )
    except Exception:
        module.__class__ = original_cls
        raise


def _validate_dp_axes(mesh: DeviceMesh, dp_axes: Sequence[MeshAxis]) -> None:
    """Validate the parent mesh's data-parallel axes."""
    normalized_dp_axes = tuple(_axis_index(mesh, axis) for axis in dp_axes)
    if len(set(normalized_dp_axes)) != len(normalized_dp_axes):
        raise ValueError(f"Data-parallel axes must be distinct, got {dp_axes!r}.")
    if normalized_dp_axes != tuple(sorted(normalized_dp_axes)):
        raise ValueError(f"Data-parallel axes must be in mesh-axis order, got {dp_axes!r}.")
    if normalized_dp_axes != tuple(range(mesh.ndim)):
        raise NotImplementedError(
            "MFSDP currently requires dp_axes to match every mesh axis in mesh order."
        )


def _axis_index(mesh: DeviceMesh, axis: MeshAxis) -> int:
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


@contextmanager
def microbatch(context: FsdpContext, is_last: bool) -> Iterator[None]:
    """Mark an FSDP microbatch as the last accumulation microbatch.

    At present, this is only needed for HSDP/HFSDP gradient accumulation, so
    FSDP finalizes gradients only on the last backward. Plain all-``Shard(0)`` data
    parallelism finalizes gradients on every backward and does not need it.

    Args:
        context: FSDP context whose roots should use this microbatch state.
        is_last: Whether forwards in this scope are for the last microbatch.
    """
    context.ensure_finalized()
    previous_state = context.is_last_microbatch
    context.is_last_microbatch = is_last

    try:
        yield
    finally:
        context.is_last_microbatch = previous_state


def _attach_mixin(module: nn.Module) -> None:
    if isinstance(module, FsdpModule):
        return
    module_cls = module.__class__
    fsdp_cls = type(f"ExperimentalFsdp{module_cls.__name__}", (FsdpModule, module_cls), {})
    module.__class__ = fsdp_cls

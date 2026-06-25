# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sharded gradient norm and clipping primitive for FSDP2 optimizers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

try:  # pragma: no cover - import availability is PyTorch-version dependent.
    from torch.distributed.tensor import DTensor, Partial, Replicate
except ImportError:  # pragma: no cover
    DTensor = Partial = Replicate = None  # type: ignore[assignment]


def sharded_grad_sq_sum(
    params: Iterable[nn.Parameter],
    *,
    accum_dtype: str | torch.dtype = torch.float32,
    default_device: torch.device | None = None,
    chunk_size_numel: int = 0,
    scalar_all_reduce: (
        Callable[[torch.Tensor, dist.ProcessGroup, dist.ReduceOp], None] | None
    ) = None,
) -> torch.Tensor:
    """Return global L2 grad squared-sum for Tensor/DTensor parameters.

    The primitive reduces one scalar per DTensor sharding group over mesh
    dimensions whose placement is not replicated. Pipeline/expert reductions
    are intentionally left to the runtime adapter because they are model-layout
    policy, not a DTensor property.
    """

    dtype = resolve_torch_dtype(accum_dtype)
    groups = _group_grads(params)
    total: torch.Tensor | None = None
    for group in groups.values():
        local_sq = _group_local_sq_sum(group, dtype=dtype, chunk_size_numel=chunk_size_numel)
        meta = group[0][2]
        if meta is not None and not _has_partial_placement(meta) and dist.is_initialized():
            _reduce_dtensor_scalar_(
                local_sq, meta, op=dist.ReduceOp.SUM, scalar_all_reduce=scalar_all_reduce
            )
        total = local_sq if total is None else total.to(local_sq.device) + local_sq

    if total is None:
        return torch.zeros((), device=default_device or torch.device("cpu"), dtype=dtype)
    return total


def sharded_grad_norm(
    params: Iterable[nn.Parameter],
    *,
    norm_type: float = 2.0,
    pp_group: dist.ProcessGroup | None = None,
    accum_dtype: str | torch.dtype = torch.float32,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    """Return global grad norm for Tensor/DTensor parameters.

    Only L2 and infinity norms are implemented because they cover Megatron Lite's
    optimizer path today and avoid ambiguous cross-placement semantics for
    arbitrary p-norms.
    """

    if math.isinf(float(norm_type)):
        total = sharded_grad_abs_max(
            params, pp_group=pp_group, accum_dtype=accum_dtype, default_device=default_device
        )
        return total
    if float(norm_type) != 2.0:
        raise ValueError(f"sharded_grad_norm supports norm_type=2.0 or inf, got {norm_type!r}.")
    sq_sum = sharded_grad_sq_sum(params, accum_dtype=accum_dtype, default_device=default_device)
    if pp_group is not None and dist.is_initialized() and dist.get_world_size(pp_group) > 1:
        all_reduce_scalar_(sq_sum, op=dist.ReduceOp.SUM, group=pp_group)
    return sq_sum.sqrt()


def sharded_grad_abs_max(
    params: Iterable[nn.Parameter],
    *,
    pp_group: dist.ProcessGroup | None = None,
    accum_dtype: str | torch.dtype = torch.float32,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    """Return global infinity grad norm for Tensor/DTensor parameters."""

    dtype = resolve_torch_dtype(accum_dtype)
    groups = _group_grads(params)
    total: torch.Tensor | None = None
    for group in groups.values():
        local_max = _group_local_abs_max(group, dtype=dtype)
        meta = group[0][2]
        if meta is not None and not _has_partial_placement(meta) and dist.is_initialized():
            _reduce_dtensor_scalar_(local_max, meta, op=dist.ReduceOp.MAX)
        total = local_max if total is None else torch.maximum(total.to(local_max.device), local_max)

    if total is None:
        total = torch.zeros((), device=default_device or torch.device("cpu"), dtype=dtype)
    if pp_group is not None and dist.is_initialized() and dist.get_world_size(pp_group) > 1:
        all_reduce_scalar_(total, op=dist.ReduceOp.MAX, group=pp_group)
    return total


def all_reduce_scalar_(
    value: torch.Tensor,
    *,
    op: dist.ReduceOp,
    group: dist.ProcessGroup,
) -> None:
    """All-reduce a scalar on a device compatible with the process group backend."""

    reduced = _scalar_for_process_group(value, group)
    dist.all_reduce(reduced, op=op, group=group)
    if reduced is not value:
        value.copy_(reduced.to(device=value.device, dtype=value.dtype))


@torch.no_grad()
def clip_grads_with_sharded_norm_(
    params: Iterable[nn.Parameter], max_norm: float, total_norm: torch.Tensor | float
) -> None:
    """Scale gradients in-place using a precomputed global norm."""

    max_norm = float(max_norm)
    if max_norm <= 0:
        return
    if isinstance(total_norm, torch.Tensor):
        if not bool(torch.isfinite(total_norm).item()):
            return
        clip_coef = (max_norm / (total_norm + 1.0e-6)).clamp(max=1.0)
        if float(clip_coef.item()) >= 1.0:
            return
    else:
        norm_value = float(total_norm)
        if not math.isfinite(norm_value):
            return
        clip_coef = max_norm / (norm_value + 1.0e-6)
        if clip_coef >= 1.0:
            return
    for param in params:
        if param.grad is not None:
            _scale_grad_(param.grad, clip_coef)


def resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        name = dtype.removeprefix("torch.")
        resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported torch dtype for grad norm accumulation: {dtype!r}")
    if not torch.empty((), dtype=resolved).is_floating_point():
        raise ValueError(f"Grad norm accumulation dtype must be floating point: {dtype!r}")
    return resolved


def _group_grads(
    params: Iterable[nn.Parameter],
) -> dict[tuple[Any, ...], list[tuple[nn.Parameter, torch.Tensor, Any | None]]]:
    groups: dict[tuple[Any, ...], list[tuple[nn.Parameter, torch.Tensor, Any | None]]] = (
        defaultdict(list)
    )
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        meta = _dtensor_meta(param, grad)
        if meta is None:
            key = ("tensor", grad.device)
        else:
            key = (
                "dtensor",
                id(meta.device_mesh),
                tuple((type(placement).__name__, repr(placement)) for placement in meta.placements),
            )
        groups[key].append((param, grad, meta))
    return groups


def _group_local_sq_sum(
    group: list[tuple[nn.Parameter, torch.Tensor, Any | None]],
    *,
    dtype: torch.dtype,
    chunk_size_numel: int = 0,
) -> torch.Tensor:
    device = _local_grad(group[0][1], group[0][2]).device
    total = torch.zeros((), device=device, dtype=dtype)
    for _param, grad, meta in group:
        local_grad = _local_grad(grad, meta)
        total += _tensor_sq_sum(local_grad.detach(), dtype=dtype, chunk_size_numel=chunk_size_numel)
    return total


def _tensor_sq_sum(
    tensor: torch.Tensor, *, dtype: torch.dtype, chunk_size_numel: int = 0
) -> torch.Tensor:
    if chunk_size_numel <= 0 or tensor.numel() <= chunk_size_numel:
        return tensor.to(dtype).pow(2).sum()
    try:
        flat = tensor.view(-1)
    except RuntimeError:
        flat = tensor.reshape(-1)
    total = torch.zeros((), device=tensor.device, dtype=dtype)
    for start in range(0, flat.numel(), chunk_size_numel):
        chunk = flat.narrow(0, start, min(chunk_size_numel, flat.numel() - start))
        total += chunk.to(dtype).pow(2).sum()
    return total


def _group_local_abs_max(
    group: list[tuple[nn.Parameter, torch.Tensor, Any | None]], *, dtype: torch.dtype
) -> torch.Tensor:
    device = _local_grad(group[0][1], group[0][2]).device
    total = torch.zeros((), device=device, dtype=dtype)
    for _param, grad, meta in group:
        local_grad = _local_grad(grad, meta)
        if local_grad.numel() > 0:
            total = torch.maximum(total, local_grad.detach().to(dtype).abs().max())
    return total


def _local_grad(grad: torch.Tensor, meta: Any | None) -> torch.Tensor:
    if meta is not None and _has_partial_placement(meta):
        full_tensor = getattr(grad, "full_tensor", None)
        if callable(full_tensor):
            return full_tensor()
    to_local = getattr(grad, "to_local", None)
    if callable(to_local):
        return to_local()
    return grad


def _scale_grad_(grad: torch.Tensor, scale: float | torch.Tensor) -> None:
    # clip_coef is a scalar; scale every shard by it in place. A plain scalar mul_
    # is correct for ANY DTensor placement (Shard/Replicate/Partial) and avoids a
    # to_local()/from_local() round-trip, which mis-reconstructs the global shape of
    # an unevenly-sharded param -- e.g. a (3,) mHC scale FSDP-sharded over 8 ranks:
    # from_local assumes even sharding and infers dim0=8, so copy_ raises
    # "tensor a (3) must match tensor b (8) at dim 0".
    grad.mul_(float(scale) if isinstance(scale, torch.Tensor) else scale)


def _dtensor_meta(param: nn.Parameter, grad: torch.Tensor) -> Any | None:
    if _is_dtensor_like(grad):
        return grad
    if _is_dtensor_like(param):
        return param
    return None


def _is_dtensor_like(tensor: Any) -> bool:
    if DTensor is not None and isinstance(tensor, DTensor):
        return True
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def _has_partial_placement(dtensor: Any) -> bool:
    return any(_placement_name(placement) == "Partial" for placement in dtensor.placements)


def _is_replicate_placement(placement: Any) -> bool:
    return _placement_name(placement) == "Replicate"


def _placement_name(placement: Any) -> str:
    return type(placement).__name__


def _reduce_dtensor_scalar_(
    value: torch.Tensor,
    dtensor: Any,
    *,
    op: dist.ReduceOp,
    scalar_all_reduce: (
        Callable[[torch.Tensor, dist.ProcessGroup, dist.ReduceOp], None] | None
    ) = None,
) -> None:
    for mesh_dim, placement in enumerate(dtensor.placements):
        if _is_replicate_placement(placement):
            continue
        group = dtensor.device_mesh.get_group(mesh_dim)
        if dist.get_world_size(group) > 1:
            if scalar_all_reduce is None:
                all_reduce_scalar_(value, op=op, group=group)
            else:
                scalar_all_reduce(value, group, op)


def _scalar_for_process_group(
    value: torch.Tensor, group: dist.ProcessGroup
) -> torch.Tensor:
    backend = _process_group_backend(group)
    if "nccl" in backend and value.device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL scalar all_reduce requires a CUDA tensor.")
        return value.to(device=torch.device("cuda", torch.cuda.current_device()))
    if "gloo" in backend and value.device.type != "cpu":
        return value.to(device=torch.device("cpu"))
    return value


def _process_group_backend(group: dist.ProcessGroup) -> str:
    try:
        return str(dist.get_backend(group)).lower()
    except (RuntimeError, ValueError, TypeError):
        return ""


__all__ = [
    "all_reduce_scalar_",
    "clip_grads_with_sharded_norm_",
    "resolve_torch_dtype",
    "sharded_grad_abs_max",
    "sharded_grad_norm",
    "sharded_grad_sq_sum",
]

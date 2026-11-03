# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Model parameter initialization constraints."""

import math
from collections.abc import Iterable

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size


def validate_hyperball_config(eps: float, radius: float | None) -> None:
    """Validate Hyperball configuration; None selects the sampled initial radius."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"hyperball_eps must be finite and positive, got {eps}")
    if radius is None:
        return
    if not math.isfinite(radius) or radius < eps:
        raise ValueError(
            f"hyperball_radius must be finite and at least hyperball_eps={eps}, got {radius}"
        )


def is_muon_parameter(param: torch.Tensor) -> bool:
    """Return whether a parameter is managed by Muon rather than its scalar fallback."""
    return (
        param.requires_grad
        and getattr(param, "use_muon", True)
        and not getattr(param, "is_embedding_or_output_parameter", False)
        and param.ndim == 2
    )


def _is_tensor_parallel_shard(param: torch.Tensor) -> bool:
    is_parallel = getattr(param, "tensor_model_parallel", None)
    if is_parallel is not None:
        return bool(is_parallel)
    return getattr(param, "partition_dim", None) not in (None, -1)


def _is_expert_parameter(param: torch.Tensor) -> bool:
    return bool(getattr(param, "expert_tp", False) or not getattr(param, "allreduce", True))


def _parameter_shard_groups(
    param: torch.Tensor, pg_collection: ProcessGroupCollection
) -> tuple[torch.distributed.ProcessGroup, ...]:
    """Return process groups spanning every unique shard of a parameter."""
    is_expert = _is_expert_parameter(param)
    groups = []
    if _is_tensor_parallel_shard(param):
        tp_group = pg_collection.expt_tp if is_expert else pg_collection.tp
        if tp_group is None:
            raise RuntimeError("Muon Hyperball is missing the parameter's TP group")
        if get_pg_size(tp_group) > 1:
            groups.append(tp_group)

    if getattr(param, "is_gtp_weight_remat", False):
        gtp_group = pg_collection.expt_gtp_remat if is_expert else pg_collection.gtp_remat
        if gtp_group is None:
            raise RuntimeError("Muon Hyperball is missing the parameter's GTP-remat group")
        if get_pg_size(gtp_group) > 1 and all(gtp_group is not group for group in groups):
            groups.append(gtp_group)

    return tuple(groups)


def logical_matrix_rows(param: torch.Tensor, pg_collection: ProcessGroupCollection) -> int:
    """Return the number of non-padding rows in this rank's matrix shard."""
    if not getattr(param, "is_gtp_weight_remat", False):
        return param.shape[0]
    group = pg_collection.expt_gtp_remat if _is_expert_parameter(param) else pg_collection.gtp_remat
    if group is None:
        raise RuntimeError("Muon Hyperball is missing the parameter's GTP-remat group")
    rows = param.shape[0]
    logical_rows = rows * get_pg_size(group) - getattr(param, "pad_length", 0)
    return max(0, min(rows, logical_rows - get_pg_rank(group) * rows))


def _initialize_fan_in_normal(param: torch.Tensor, pg_collection: ProcessGroupCollection) -> None:
    """Sample a logical matrix with variance 1/fan_in using Megatron's RNG streams."""
    from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu
    from megatron.core.tensor_parallel.random import (
        get_cuda_rng_tracker,
        get_data_parallel_rng_tracker_name,
        get_expert_parallel_rng_tracker_name,
        get_gtp_remat_rng_tracker_name,
    )

    is_expert = _is_expert_parameter(param)
    tp_group = pg_collection.expt_tp if is_expert else pg_collection.tp
    tp_size = get_pg_size(tp_group) if _is_tensor_parallel_shard(param) else 1
    partition_dim = getattr(param, "partition_dim", -1)
    fan_in = param.shape[1] * (tp_size if partition_dim == 1 else 1)
    std = fan_in**-0.5
    gtp_group = None
    if getattr(param, "is_gtp_weight_remat", False):
        gtp_group = pg_collection.expt_gtp_remat if is_expert else pg_collection.gtp_remat
    gtp_size = get_pg_size(gtp_group)

    if param.device.type == "cpu":
        # Reuse master-matrix sampling and strided TP slicing, so CPU initialization is
        # invariant to TP/GTP partitioning. Slice GTP using the existing parameter's
        # padding metadata, not the current global GTP alignment configuration.
        tp_rows = param.shape[0] * gtp_size - getattr(param, "pad_length", 0)
        output_size = tp_rows * (tp_size if partition_dim == 0 else 1)
        dim = partition_dim if partition_dim in (0, 1) else 0
        tp_weight = torch.empty(tp_rows, param.shape[1], dtype=param.dtype, device="cpu")
        _initialize_affine_weight_cpu(
            tp_weight,
            output_size,
            fan_in,
            tp_rows if dim == 0 else param.shape[1],
            dim,
            lambda weight: torch.nn.init.normal_(weight, mean=0.0, std=std),
            stride=getattr(param, "partition_stride", 1),
            params_dtype=param.dtype,
            rank=get_pg_rank(tp_group) if tp_size > 1 else 0,
            world_size=tp_size,
            skip_set_tensor_parallel_attributes=True,
        )
        row_start = get_pg_rank(gtp_group) * param.shape[0]
        rows = logical_matrix_rows(param, pg_collection)
        param.zero_()
        param[:rows].copy_(tp_weight[row_start : row_start + rows])
    else:
        if gtp_size > 1:
            rng_name = get_gtp_remat_rng_tracker_name(is_expert=is_expert)
        elif is_expert:
            rng_name = get_expert_parallel_rng_tracker_name()
        elif _is_tensor_parallel_shard(param):
            rng_name = None
        else:
            rng_name = get_data_parallel_rng_tracker_name()
        tracker = get_cuda_rng_tracker()
        rng_context = tracker.fork() if rng_name is None else tracker.fork(rng_name)
        with rng_context:
            torch.nn.init.normal_(param, mean=0.0, std=std)

    param[logical_matrix_rows(param, pg_collection) :].zero_()
    # Quantized primary weights can carry an initialization snapshot used to seed
    # master weights. Keep it consistent with the new initialization.
    high_precision_init_val = getattr(param, "_high_precision_init_val", None)
    if high_precision_init_val is not None:
        high_precision_init_val.copy_(param)


def logical_frobenius_norm(
    tensor: torch.Tensor, param: torch.Tensor, pg_collection: ProcessGroupCollection
) -> torch.Tensor:
    """Compute an FP32 Frobenius norm over all unique shards of a parameter."""
    if pg_collection is None:
        raise ValueError("Muon Hyperball requires an explicit ProcessGroupCollection")
    squared_norm = logical_frobenius_inner_product(tensor, tensor, param, pg_collection)
    return squared_norm.sqrt()


def logical_frobenius_inner_product(
    lhs: torch.Tensor, rhs: torch.Tensor, param: torch.Tensor, pg_collection: ProcessGroupCollection
) -> torch.Tensor:
    """Compute an FP32 Frobenius inner product over all unique shards of a parameter."""
    if pg_collection is None:
        raise ValueError("Muon Hyperball requires an explicit ProcessGroupCollection")
    rows = logical_matrix_rows(param, pg_collection)
    inner_product = (lhs[:rows].float() * rhs[:rows].float()).sum()
    for group in _parameter_shard_groups(param, pg_collection):
        reduction_input = inner_product
        if inner_product.device.type == "cpu" and torch.distributed.get_backend(group) == "nccl":
            reduction_input = inner_product.to(
                device=torch.device("cuda", torch.cuda.current_device())
            )
        torch.distributed.all_reduce(
            reduction_input, op=torch.distributed.ReduceOp.SUM, group=group
        )
        inner_product = reduction_input.to(device=lhs.device)
    return inner_product


@torch.no_grad()  # type: ignore[misc]
def initialize_muon_ht_parameters(
    model_chunks: Iterable[torch.nn.Module],
    radius: float | None,
    eps: float,
    pg_collection: ProcessGroupCollection,
) -> None:
    """Initialize managed matrices with fan-in Gaussian draws or an explicit radius.

    With radius=None, sample N(0, 1/d_in) using the logical (unsharded) fan-in.
    The optimizer discovers each sampled radius lazily from the initialized/restored
    weights, after mixed-precision copies; no radius tensor is attached to parameters.
    An explicit radius instead rescales the existing initialization, as before.
    """
    if pg_collection is None:
        raise ValueError("Muon Hyperball requires an explicit ProcessGroupCollection")
    validate_hyperball_config(eps, radius)

    params_and_norms = []
    seen_params = set()
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            if id(param) in seen_params or not is_muon_parameter(param):
                continue
            seen_params.add(id(param))
            # Validate topology before drawing random numbers or reducing any shard.
            _parameter_shard_groups(param, pg_collection)
            if radius is None:
                _initialize_fan_in_normal(param, pg_collection)
            norm = logical_frobenius_norm(param, param, pg_collection)
            if bool((~torch.isfinite(norm) | (norm < eps)).item()):
                raise ValueError(
                    "Muon Hyperball requires all parameters to have finite, non-zero norm"
                )
            params_and_norms.append((param, norm))

    if radius is not None:
        for param, norm in params_and_norms:
            param.mul_(radius / norm)
            param[logical_matrix_rows(param, pg_collection) :].zero_()

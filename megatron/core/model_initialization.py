# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model parameter initialization constraints."""

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig


def validate_hyperball_config(eps: float, radius: float | None) -> None:
    """Validate fixed-radius Hyperball configuration."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"hyperball_eps must be finite and positive, got {eps}")
    if radius is None:
        raise ValueError("TensorParallelMuonHT requires hyperball_radius to be configured")
    if not math.isfinite(radius) or radius < eps:
        raise ValueError(
            f"hyperball_radius must be finite and at least hyperball_eps={eps}, got {radius}"
        )


def is_muon_parameter(param: torch.Tensor) -> bool:
    """Return whether a parameter is managed by Muon rather than its scalar fallback."""
    return (
        param.requires_grad
        and getattr(param, 'use_muon', True)
        and not getattr(param, 'is_embedding_or_output_parameter', False)
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
            raise RuntimeError("TensorParallelMuonHT is missing the parameter's TP group")
        if get_pg_size(tp_group) > 1:
            groups.append(tp_group)

    if getattr(param, "is_gtp_weight_remat", False):
        gtp_group = pg_collection.expt_gtp_remat if is_expert else pg_collection.gtp_remat
        if gtp_group is None:
            raise RuntimeError("TensorParallelMuonHT is missing the parameter's GTP-remat group")
        if get_pg_size(gtp_group) > 1 and all(gtp_group is not group for group in groups):
            groups.append(gtp_group)

    return tuple(groups)


def logical_frobenius_norm(
    tensor: torch.Tensor, param: torch.Tensor, pg_collection: ProcessGroupCollection
) -> torch.Tensor:
    """Compute an FP32 Frobenius norm over all unique shards of a parameter."""
    if pg_collection is None:
        raise ValueError("TensorParallelMuonHT requires an explicit ProcessGroupCollection")
    squared_norm = tensor.float().square().sum()
    for group in _parameter_shard_groups(param, pg_collection):
        torch.distributed.all_reduce(squared_norm, op=torch.distributed.ReduceOp.SUM, group=group)
    return squared_norm.sqrt()


@torch.no_grad()  # type: ignore[misc]
def initialize_muon_ht_parameters(
    model_chunks: Iterable[torch.nn.Module],
    radius: float,
    eps: float,
    pg_collection: ProcessGroupCollection,
) -> None:
    """Place every Muon-managed model parameter on the configured Hyperball sphere."""
    if pg_collection is None:
        raise ValueError("TensorParallelMuonHT requires an explicit ProcessGroupCollection")
    validate_hyperball_config(eps, radius)

    params_and_norms = []
    seen_params = set()
    for model_chunk in model_chunks:
        for param in model_chunk.parameters():
            if id(param) in seen_params or not is_muon_parameter(param):
                continue
            seen_params.add(id(param))
            norm = logical_frobenius_norm(param, param, pg_collection)
            if bool((~torch.isfinite(norm) | (norm < eps)).item()):
                raise ValueError(
                    "TensorParallelMuonHT requires all parameters to have finite, non-zero norm"
                )
            params_and_norms.append((param, norm))

    for param, norm in params_and_norms:
        param.mul_(radius / norm)


def maybe_initialize_muon_ht_parameters(
    model_chunks: Iterable[torch.nn.Module],
    transformer_config: "TransformerConfig",
    pg_collection: ProcessGroupCollection,
) -> None:
    """Apply configured MuonHT initialization during Megatron model initialization."""
    if pg_collection is None:
        raise ValueError("TensorParallelMuonHT requires an explicit ProcessGroupCollection")
    radius = getattr(transformer_config, 'muon_ht_radius', None)
    if radius is None or not transformer_config.perform_initialization:
        return
    initialize_muon_ht_parameters(
        model_chunks,
        radius=radius,
        eps=getattr(transformer_config, 'muon_ht_eps', 1e-15),
        pg_collection=pg_collection,
    )

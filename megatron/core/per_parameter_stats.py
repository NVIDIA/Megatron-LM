# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for high-cardinality per-parameter statistics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_raw_moments
except ImportError:
    multi_tensor_applier = None
    multi_tensor_raw_moments = None

from megatron.core.parameter_names import CanonicalParameterNameMap
from megatron.core.utils import get_pg_rank, get_pg_size, unwrap_model

RAW_MOMENT_FIELDS = ("count", "sum_1", "sum_2", "sum_3", "sum_4")
_RAW_MOMENTS_DTYPE = torch.float32
_MULTI_TENSOR_RAW_MOMENTS_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MULTI_TENSOR_RAW_MOMENTS_SPLIT_ALIGNMENT = 4
_MAX_MULTI_TENSOR_RAW_MOMENTS_NUMEL = torch.iinfo(torch.int32).max - (
    torch.iinfo(torch.int32).max % _MULTI_TENSOR_RAW_MOMENTS_SPLIT_ALIGNMENT
)
_DISABLE_MULTI_TENSOR_RAW_MOMENTS_ENV = "MEGATRON_DISABLE_MULTI_TENSOR_RAW_MOMENTS"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class NamedTensorBucket:
    """Named tensors that should be reduced over the same process groups."""

    names: Sequence[str]
    tensors: Sequence[torch.Tensor]
    reduce_groups: tuple[torch.distributed.ProcessGroup | None, ...] = ()


class PerParameterStatRegistry:
    """Canonical parameter-name registry for per-parameter statistics."""

    def __init__(
        self,
        model_chunks: Iterable[torch.nn.Module] | torch.nn.Module,
        expert_model_parallel_group: torch.distributed.ProcessGroup | None = None,
    ):
        self.model_chunks = unwrap_model(_normalize_model_chunks(model_chunks))
        self.expert_model_parallel_group = expert_model_parallel_group
        self.expert_model_parallel_rank, self.expert_model_parallel_size = (
            _get_expert_model_parallel_rank_size(expert_model_parallel_group)
        )
        self.cache_key = _registry_cache_key(
            self.model_chunks,
            expert_model_parallel_group,
            self.expert_model_parallel_rank,
            self.expert_model_parallel_size,
        )
        self.parameter_names = CanonicalParameterNameMap(
            self.model_chunks,
            expert_parallel_rank=self.expert_model_parallel_rank,
            expert_parallel_size=self.expert_model_parallel_size,
        )
        self.parameter_name_index = self.parameter_names.all_gather_index()
        self.param_to_name = dict(self.parameter_names.items())
        self.name_to_index = self.parameter_name_index
        self.index_to_name = self.parameter_name_index.names

    def name_for_param(self, param: torch.nn.Parameter) -> str:
        """Return the canonical name for ``param``."""
        return self.param_to_name[param]

    @property
    def num_params(self) -> int:
        """Number of globally known parameters."""
        return len(self.name_to_index)


def get_or_create_per_parameter_stat_registry(
    model_chunks: Iterable[torch.nn.Module] | torch.nn.Module,
    expert_model_parallel_group: torch.distributed.ProcessGroup | None = None,
) -> PerParameterStatRegistry:
    """Return a per-model cached parameter-stat registry."""
    unwrapped_model_chunks = unwrap_model(_normalize_model_chunks(model_chunks))
    if not unwrapped_model_chunks:
        raise ValueError("Cannot build a per-parameter stat registry for an empty model list.")

    expert_model_parallel_rank, expert_model_parallel_size = _get_expert_model_parallel_rank_size(
        expert_model_parallel_group
    )
    cache_key = _registry_cache_key(
        unwrapped_model_chunks,
        expert_model_parallel_group,
        expert_model_parallel_rank,
        expert_model_parallel_size,
    )
    cache_owner = unwrapped_model_chunks[0]
    registry = getattr(cache_owner, "_per_parameter_stat_registry", None)
    if registry is None or registry.cache_key != cache_key:
        registry = PerParameterStatRegistry(
            unwrapped_model_chunks, expert_model_parallel_group=expert_model_parallel_group
        )
        cache_owner._per_parameter_stat_registry = registry
    return registry


def reduce_raw_moments_by_param(
    registry: PerParameterStatRegistry, buckets: Sequence[NamedTensorBucket]
) -> tuple[list[tuple[str, dict[str, float]]], dict[str, float]]:
    """Reduce named tensor raw moments by parameter name.

    Args:
        registry: Canonical parameter-name registry.
        buckets: Named tensor buckets with the process groups needed to assemble each bucket's
            local raw moments into global per-parameter raw moments.

    Returns:
        A ``(values, aggregate_moments)`` tuple. ``values`` is a list of
        ``(name, raw_moment_dict)`` tuples ordered by canonical parameter index.
    """
    device = _select_device(buckets)
    moments = torch.zeros(
        (registry.num_params, len(RAW_MOMENT_FIELDS)), dtype=_RAW_MOMENTS_DTYPE, device=device
    )

    for bucket in buckets:
        if len(bucket.names) != len(bucket.tensors):
            raise ValueError(
                f"NamedTensorBucket has {len(bucket.names)} names but "
                f"{len(bucket.tensors)} tensors."
            )

        bucket_moments = torch.zeros_like(moments)
        if bucket.names:
            indices = torch.tensor(
                [registry.name_to_index[name] for name in bucket.names],
                dtype=torch.long,
                device=device,
            )
            bucket_moments.index_add_(0, indices, _local_raw_moments(bucket.tensors, device))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for group in bucket.reduce_groups:
                torch.distributed.all_reduce(
                    bucket_moments, op=torch.distributed.ReduceOp.SUM, group=group
                )

        moments += bucket_moments

    rows = moments.tolist()
    aggregate_moments = raw_moment_row_to_dict(moments.sum(dim=0).tolist())
    return [
        (name, raw_moment_row_to_dict(rows[idx])) for idx, name in enumerate(registry.index_to_name)
    ], aggregate_moments


def _select_device(buckets: Sequence[NamedTensorBucket]) -> torch.device:
    for bucket in buckets:
        if bucket.tensors:
            return bucket.tensors[0].device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _normalize_model_chunks(
    model_chunks: Iterable[torch.nn.Module] | torch.nn.Module,
) -> list[torch.nn.Module]:
    if isinstance(model_chunks, (list, tuple)):
        return list(model_chunks)
    return [model_chunks]


def _local_raw_moments(tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not tensors:
        return torch.zeros((0, len(RAW_MOMENT_FIELDS)), dtype=_RAW_MOMENTS_DTYPE, device=device)

    if _can_use_multi_tensor_raw_moments(tensors, device):
        return _multi_tensor_raw_moments(tensors, device)

    rows = [_torch_raw_moment_row(tensor, device=device) for tensor in tensors]
    return torch.stack(rows)


def raw_moment_row(tensor: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Return count and raw sums of powers 1-4 for ``tensor`` as an fp32 row."""
    device = tensor.device if device is None else device
    if _can_use_multi_tensor_raw_moments([tensor], device):
        return _multi_tensor_raw_moments([tensor], device)[0]
    return _torch_raw_moment_row(tensor, device=device)


def _torch_raw_moment_row(tensor: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Torch fallback for count and raw sums of powers 1-4."""
    device = tensor.device if device is None else device
    values = tensor.detach().to(device=device, dtype=_RAW_MOMENTS_DTYPE)
    values_2 = values * values
    return torch.stack(
        [
            torch.tensor(float(values.numel()), dtype=_RAW_MOMENTS_DTYPE, device=device),
            values.sum(),
            values_2.sum(),
            (values_2 * values).sum(),
            (values_2 * values_2).sum(),
        ]
    )


def _can_use_multi_tensor_raw_moments(
    tensors: Sequence[torch.Tensor], device: torch.device
) -> bool:
    disabled = os.getenv(_DISABLE_MULTI_TENSOR_RAW_MOMENTS_ENV, "").lower() in _TRUTHY_ENV_VALUES
    return (
        not disabled
        and multi_tensor_applier is not None
        and multi_tensor_raw_moments is not None
        and device.type == "cuda"
        and all(
            tensor.device == device
            and tensor.dtype in _MULTI_TENSOR_RAW_MOMENTS_DTYPES
            and tensor.is_contiguous()
            for tensor in tensors
        )
    )


def _multi_tensor_raw_moments(
    tensors: Sequence[torch.Tensor], device: torch.device
) -> torch.Tensor:
    grouped_indices = _group_tensor_indices_by_device_and_dtype(tensors)
    if len(grouped_indices) == 1:
        return _multi_tensor_raw_moments_for_group(tensors, device)

    rows = torch.empty(
        (len(tensors), len(RAW_MOMENT_FIELDS)), dtype=_RAW_MOMENTS_DTYPE, device=device
    )
    for indices in grouped_indices.values():
        group_tensors = [tensors[index] for index in indices]
        group_rows = _multi_tensor_raw_moments_for_group(group_tensors, group_tensors[0].device)
        rows[torch.tensor(indices, dtype=torch.long, device=device)] = group_rows.to(device=device)
    return rows


def _multi_tensor_raw_moments_for_group(
    tensors: Sequence[torch.Tensor], device: torch.device
) -> torch.Tensor:
    split_tensors, source_indices = _split_tensors_for_multi_tensor_raw_moments(tensors)
    device = split_tensors[0].device
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)
    split_rows = multi_tensor_applier(multi_tensor_raw_moments, dummy_overflow_buf, [split_tensors])
    if len(split_tensors) == len(tensors):
        return split_rows

    rows = torch.zeros(
        (len(tensors), len(RAW_MOMENT_FIELDS)), dtype=_RAW_MOMENTS_DTYPE, device=device
    )
    rows.index_add_(0, torch.tensor(source_indices, dtype=torch.long, device=device), split_rows)
    return rows


def _split_tensors_for_multi_tensor_raw_moments(
    tensors: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[int]]:
    split_tensors = []
    source_indices = []
    for index, tensor in enumerate(tensors):
        flat_tensor = tensor.detach().view(-1)
        if flat_tensor.numel() == 0 or flat_tensor.numel() <= _MAX_MULTI_TENSOR_RAW_MOMENTS_NUMEL:
            split_tensors.append(flat_tensor)
            source_indices.append(index)
            continue

        for start in range(0, flat_tensor.numel(), _MAX_MULTI_TENSOR_RAW_MOMENTS_NUMEL):
            length = min(_MAX_MULTI_TENSOR_RAW_MOMENTS_NUMEL, flat_tensor.numel() - start)
            split_tensors.append(flat_tensor.narrow(0, start, length))
            source_indices.append(index)
    return split_tensors, source_indices


def _group_tensor_indices_by_device_and_dtype(
    tensors: Sequence[torch.Tensor],
) -> dict[tuple[torch.device, torch.dtype], list[int]]:
    groups = {}
    for index, tensor in enumerate(tensors):
        groups.setdefault((tensor.device, tensor.dtype), []).append(index)
    return groups


def raw_moment_row_to_dict(row: Sequence[float]) -> dict[str, float]:
    """Convert a raw-moment row to a JSON-serializable mapping."""
    return {field: float(row[idx]) for idx, field in enumerate(RAW_MOMENT_FIELDS)}


def _get_expert_model_parallel_rank_size(
    expert_model_parallel_group: torch.distributed.ProcessGroup | None,
) -> tuple[int, int]:
    return get_pg_rank(expert_model_parallel_group), get_pg_size(expert_model_parallel_group)


def _registry_cache_key(
    model_chunks: Sequence[torch.nn.Module],
    expert_model_parallel_group: torch.distributed.ProcessGroup | None,
    expert_model_parallel_rank: int,
    expert_model_parallel_size: int,
) -> tuple[tuple[int, ...], int | None, int, int]:
    group_id = id(expert_model_parallel_group) if expert_model_parallel_group is not None else None
    return (
        tuple(id(model_chunk) for model_chunk in model_chunks),
        group_id,
        expert_model_parallel_rank,
        expert_model_parallel_size,
    )

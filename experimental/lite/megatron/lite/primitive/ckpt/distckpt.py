"""Megatron Core distributed checkpoint bridge for MLite distopt."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any

import torch
import torch.nn as nn

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import (
    ExpertClassifierFn,
    PlacementFn,
    default_expert_classifier,
    default_placement_fn,
)


_DISTOPT_METADATA = {
    "distrib_optim_sharding_type": "fully_reshardable",
    "distrib_optim_fully_reshardable_mem_efficient": False,
    "chained_optim_avoid_prefix": True,
}


def attach_model_sharded_state_dict(
    model_chunks: Iterable[nn.Module],
    ps: ParallelState,
    *,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
) -> None:
    """Attach an MLite-local mcore sharded_state_dict method to distopt chunks."""

    for chunk in model_chunks:
        chunk.sharded_state_dict = MethodType(  # type: ignore[method-assign]
            _build_bound_sharded_state_dict(ps, get_placements, is_expert),
            chunk,
        )
        chunk._mlite_distopt_sharded_state_dict = True  # type: ignore[attr-defined]


def supports_distopt_distckpt(model: nn.Module | Iterable[nn.Module], optimizer: Any) -> bool:
    """Return whether this model/optimizer pair can use mcore dist_checkpointing."""

    if optimizer is not None and not callable(getattr(optimizer, "sharded_state_dict", None)):
        return False
    return all(
        bool(getattr(chunk, "_mlite_distopt_sharded_state_dict", False))
        and callable(getattr(chunk, "sharded_state_dict", None))
        for chunk in _model_chunks(model)
    )


def save_distopt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer: Any,
    step: int,
    checkpoint_dir: str,
    *,
    save_model: bool = True,
    save_optimizer: bool = True,
) -> None:
    """Save model and DistributedOptimizer state through mcore dist_checkpointing."""

    os.makedirs(checkpoint_dir, exist_ok=True)
    model_sd = _model_sharded_state_dict(model) if save_model or save_optimizer else {}
    state_dict: dict[str, Any] = {"step": int(step)}
    if save_model:
        state_dict.update(model_sd)
    if save_optimizer and optimizer is not None:
        state_dict["optimizer"] = optimizer.sharded_state_dict(
            _single_or_all_model_state(model_sd),
            metadata=_DISTOPT_METADATA,
        )
    dist_checkpointing.save(
        state_dict,
        checkpoint_dir,
        validate_access_integrity=False,
        content_metadata=_DISTOPT_METADATA,
    )


def load_distopt_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer: Any,
    checkpoint_dir: str,
    *,
    load_model: bool = True,
    load_optimizer: bool = True,
) -> int:
    """Load a mcore dist_checkpointing checkpoint into model and DistributedOptimizer."""

    model_sd = _model_sharded_state_dict(model) if load_model or load_optimizer else {}
    load_sd: dict[str, Any] = {"step": 0}
    if load_model:
        load_sd.update(model_sd)
    if load_optimizer and optimizer is not None:
        load_sd["optimizer"] = optimizer.sharded_state_dict(
            _single_or_all_model_state(model_sd),
            is_loading=True,
            metadata=_DISTOPT_METADATA,
        )
    state_dict = dist_checkpointing.load(load_sd, checkpoint_dir, validate_access_integrity=False)
    if load_model:
        _load_model_state_dict(model, state_dict)
    if load_optimizer and optimizer is not None and "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])
    return int(state_dict.get("step", 0))


def _build_bound_sharded_state_dict(
    ps: ParallelState,
    get_placements: PlacementFn,
    is_expert: ExpertClassifierFn,
) -> Callable:
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int], ...] = (),
        metadata: dict | None = None,
    ) -> dict[str, ShardedTensor]:
        del metadata
        return _module_sharded_state_dict(
            _wrapped_module(self),
            ps,
            get_placements=get_placements,
            is_expert=is_expert,
            prefix=prefix,
            sharded_offsets=sharded_offsets,
        )

    return sharded_state_dict


def _module_sharded_state_dict(
    module: nn.Module,
    ps: ParallelState,
    *,
    get_placements: PlacementFn,
    is_expert: ExpertClassifierFn,
    prefix: str = "",
    sharded_offsets: tuple[tuple[int, int, int], ...] = (),
) -> dict[str, ShardedTensor]:
    state: dict[str, ShardedTensor] = {}
    for name, param in module.named_parameters():
        state[f"{prefix}{name}"] = _make_sharded_tensor(
            f"{prefix}{name}",
            param,
            ps,
            placements=get_placements(name),
            expert=is_expert(name),
            sharded_offsets=sharded_offsets,
        )
    for name, buffer in module.named_buffers():
        state[f"{prefix}{name}"] = _make_sharded_tensor(
            f"{prefix}{name}",
            buffer,
            ps,
            placements=get_placements(name),
            expert=is_expert(name),
            sharded_offsets=sharded_offsets,
        )
    return state


def _make_sharded_tensor(
    key: str,
    tensor: torch.Tensor,
    ps: ParallelState,
    *,
    placements: list,
    expert: bool,
    sharded_offsets: tuple[tuple[int, int, int], ...] = (),
) -> ShardedTensor:
    rank_offsets, replica_id = _rank_offsets_and_replica_id(placements, ps, expert=expert)
    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *sharded_offsets,
        *rank_offsets,
        replica_id=replica_id,
    )


def _rank_offsets_and_replica_id(
    placements: list,
    ps: ParallelState,
    *,
    expert: bool,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, int, int]]:
    ranks, sizes = _mesh_ranks_and_sizes(ps, expert=expert)
    axis_fragments: dict[int, tuple[int, int]] = {}
    for placement, rank, size in zip(placements, ranks, sizes, strict=True):
        if _is_shard_placement(placement):
            dim = _shard_dim(placement)
            if dim is None:
                raise ValueError(f"Unsupported Shard placement without dim: {placement!r}.")
            prev_rank, prev_size = axis_fragments.get(dim, (0, 1))
            axis_fragments[dim] = (prev_rank * size + rank, prev_size * size)
    rank_offsets = tuple((dim, rank, size) for dim, (rank, size) in axis_fragments.items())
    return rank_offsets, _replica_id(placements, ps, expert=expert)


def _replica_id(placements: list, ps: ParallelState, *, expert: bool) -> tuple[int, int, int]:
    if expert:
        return (
            _replica_axis_rank(placements, 0, ps.pp_rank),
            _replica_axis_rank(placements, 2, ps.ep_rank),
            _replica_axis_rank(placements, 1, ps.expert_dp_rank),
        )
    dp_cp_rank = (
        0
        if _placement_is_sharded(placements, 1) or _placement_is_sharded(placements, 2)
        else ps.dp_cp_rank
    )
    return (
        _replica_axis_rank(placements, 0, ps.pp_rank),
        _replica_axis_rank(placements, 3, ps.tp_rank),
        int(dp_cp_rank),
    )


def _replica_axis_rank(placements: list, axis: int, rank: int) -> int:
    return 0 if _placement_is_sharded(placements, axis) else int(rank)


def _placement_is_sharded(placements: list, axis: int) -> bool:
    return axis < len(placements) and _is_shard_placement(placements[axis])


def _mesh_ranks_and_sizes(ps: ParallelState, *, expert: bool) -> tuple[list[int], list[int]]:
    if expert:
        return (
            [ps.pp_rank, ps.expert_dp_rank, ps.ep_rank, ps.etp_rank],
            [ps.pp_size, ps.expert_dp_size, ps.ep_size, ps.etp_size],
        )
    return (
        [ps.pp_rank, ps.dp_rank, ps.cp_rank, ps.tp_rank],
        [ps.pp_size, ps.dp_size, ps.cp_size, ps.tp_size],
    )


def _is_shard_placement(placement: Any) -> bool:
    return type(placement).__name__ == "Shard"


def _shard_dim(placement: Any) -> int | None:
    dim = getattr(placement, "dim", None)
    if dim is None:
        dim = getattr(placement, "_dim", None)
    return None if dim is None else int(dim)


def _model_sharded_state_dict(model: nn.Module | Iterable[nn.Module]) -> dict[str, Any]:
    chunks = _model_chunks(model)
    if len(chunks) == 1:
        return {"model": chunks[0].sharded_state_dict()}  # type: ignore[attr-defined]
    return {
        f"model{idx}": chunk.sharded_state_dict()  # type: ignore[attr-defined]
        for idx, chunk in enumerate(chunks)
    }


def _single_or_all_model_state(model_sd: dict[str, Any]) -> dict[str, Any]:
    if "model" in model_sd:
        return model_sd["model"]
    return model_sd


def _load_model_state_dict(model: nn.Module | Iterable[nn.Module], state_dict: dict[str, Any]) -> None:
    chunks = _model_chunks(model)
    if len(chunks) == 1 and "model" in state_dict:
        _wrapped_module(chunks[0]).load_state_dict(state_dict["model"], strict=False)
        return
    for idx, chunk in enumerate(chunks):
        key = f"model{idx}"
        if key in state_dict:
            _wrapped_module(chunk).load_state_dict(state_dict[key], strict=False)


def _model_chunks(model: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    if isinstance(model, nn.Module):
        return list(model) if isinstance(model, nn.ModuleList) else [model]
    chunks = list(model)
    if not all(isinstance(chunk, nn.Module) for chunk in chunks):
        raise TypeError("distckpt model chunks must be nn.Module instances.")
    return chunks


def _wrapped_module(model: nn.Module) -> nn.Module:
    module = getattr(model, "module", None)
    return module if isinstance(module, nn.Module) else model


__all__ = [
    "attach_model_sharded_state_dict",
    "load_distopt_checkpoint",
    "save_distopt_checkpoint",
    "supports_distopt_distckpt",
]

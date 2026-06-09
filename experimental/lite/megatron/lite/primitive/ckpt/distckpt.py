"""Megatron Core distributed checkpoint bridge for MLite distopt."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, MutableMapping
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
        _synchronize_native_optimizer_steps(optimizer)
        patches = _patch_empty_native_optimizer_state_dicts(optimizer, fallback_step=step)
        try:
            state_dict["optimizer"] = optimizer.sharded_state_dict(
                _single_or_all_model_state(model_sd),
                metadata=_DISTOPT_METADATA,
            )
        finally:
            _restore_state_dict_patches(patches)
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
        patches = _patch_empty_native_optimizer_state_dicts(optimizer, fallback_step=0)
        try:
            load_sd["optimizer"] = optimizer.sharded_state_dict(
                _single_or_all_model_state(model_sd),
                is_loading=True,
                metadata=_DISTOPT_METADATA,
            )
        finally:
            _restore_state_dict_patches(patches)
    state_dict = dist_checkpointing.load(load_sd, checkpoint_dir, validate_access_integrity=False)
    if load_model:
        _load_model_state_dict(model, state_dict)
    if load_optimizer and optimizer is not None and "optimizer" in state_dict:
        load_patches = _patch_native_optimizer_step_load(optimizer)
        try:
            optimizer.load_state_dict(state_dict["optimizer"])
        finally:
            _restore_set_state_patches(load_patches)
        _synchronize_native_optimizer_steps(optimizer)
    elif load_model and optimizer is not None:
        reload_model_params = getattr(optimizer, "reload_model_params", None)
        if callable(reload_model_params):
            reload_model_params()
    return int(state_dict.get("step", 0))


def _synchronize_native_optimizer_steps(optimizer: Any) -> None:
    """Align torch optimizer per-parameter steps before mcore fallback checkpointing."""

    seen: set[int] = set()

    def visit(obj: Any) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        chained = getattr(obj, "chained_optimizers", None)
        if isinstance(chained, Iterable):
            for child in chained:
                visit(child)

        inner = getattr(obj, "optimizer", None)
        if inner is not None and inner is not obj:
            visit(inner)

        state = getattr(obj, "state", None)
        if isinstance(state, MutableMapping):
            _synchronize_step_mapping(state)

    visit(optimizer)


def _patch_empty_native_optimizer_state_dicts(
    optimizer: Any,
    *,
    fallback_step: int,
) -> list[tuple[Any, Any]]:
    patches: list[tuple[Any, Any]] = []
    for distopt in _iter_distributed_optimizers(optimizer):
        inner = getattr(distopt, "optimizer", None)
        state = getattr(inner, "state", None)
        if not isinstance(state, MutableMapping) or state:
            continue
        original_state_dict = distopt.state_dict

        def patched_state_dict(
            original_state_dict=original_state_dict,
            distopt=distopt,
            fallback_step=fallback_step,
        ):
            try:
                return original_state_dict()
            except AssertionError:
                return _empty_native_optimizer_state_dict(distopt, fallback_step)

        distopt.state_dict = patched_state_dict  # type: ignore[method-assign]
        patches.append((distopt, original_state_dict))
    return patches


def _restore_state_dict_patches(patches: list[tuple[Any, Any]]) -> None:
    for distopt, original_state_dict in patches:
        distopt.state_dict = original_state_dict  # type: ignore[method-assign]


def _patch_native_optimizer_step_load(optimizer: Any) -> list[tuple[Any, Any]]:
    patches: list[tuple[Any, Any]] = []
    for distopt in _iter_distributed_optimizers(optimizer):
        original_set_state = distopt._set_main_param_and_optimizer_states

        def patched_set_state(
            model_param,
            tensors,
            distopt=distopt,
            original_set_state=original_set_state,
        ):
            removed_step = _pop_optimizer_step_for_model_param(distopt, model_param, tensors)
            try:
                return original_set_state(model_param, tensors)
            finally:
                if removed_step is not None:
                    state, step = removed_step
                    state["step"] = step

        distopt._set_main_param_and_optimizer_states = patched_set_state  # type: ignore[method-assign]
        patches.append((distopt, original_set_state))
    return patches


def _restore_set_state_patches(patches: list[tuple[Any, Any]]) -> None:
    for distopt, original_set_state in patches:
        distopt._set_main_param_and_optimizer_states = original_set_state  # type: ignore[method-assign]


def _pop_optimizer_step_for_model_param(
    distopt: Any,
    model_param,
    tensors: dict[str, Any],
) -> tuple[MutableMapping, Any] | None:
    if "step" in tensors:
        return None
    try:
        group_index, group_order = distopt.model_param_group_index_map[model_param]
        main_param = distopt.optimizer.param_groups[group_index]["params"][group_order]
        state = distopt.optimizer.state[main_param]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(state, MutableMapping) or "step" not in state:
        return None
    return state, state.pop("step")


def _iter_distributed_optimizers(optimizer: Any) -> Iterable[Any]:
    seen: set[int] = set()

    def visit(obj: Any):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if (
            callable(getattr(obj, "sharded_state_dict", None))
            and hasattr(obj, "gbuf_ranges")
            and hasattr(obj, "buffers")
            and hasattr(obj, "optimizer")
        ):
            yield obj

        chained = getattr(obj, "chained_optimizers", None)
        if isinstance(chained, Iterable):
            for child in chained:
                yield from visit(child)

        sub_optimizers = getattr(obj, "sub_optimizers", None)
        if isinstance(sub_optimizers, Iterable):
            for child in sub_optimizers:
                yield from visit(child)

        inner = getattr(obj, "optimizer", None)
        if inner is not None and inner is not obj:
            yield from visit(inner)

    yield from visit(optimizer)


def _empty_native_optimizer_state_dict(distopt: Any, fallback_step: int) -> dict[str, Any]:
    inner_state_dict = distopt.optimizer.state_dict()
    optimizer_state = {
        key: ([group.copy() for group in value] if key == "param_groups" else value)
        for key, value in inner_state_dict.items()
        if key != "state"
    }
    for param_group in optimizer_state["param_groups"]:
        param_group.pop("params", None)
        param_group["step"] = int(fallback_step)
    state_dict: dict[str, Any] = {"optimizer": optimizer_state}
    grad_scaler = getattr(distopt, "grad_scaler", None)
    if grad_scaler:
        state_dict["grad_scaler"] = grad_scaler.state_dict()
    return state_dict


def _synchronize_step_mapping(state: MutableMapping) -> None:
    steps: list[Any] = []
    for param_state in state.values():
        if isinstance(param_state, MutableMapping) and "step" in param_state:
            steps.append(param_state["step"])
    if not steps:
        return
    target = max(_step_as_int(step) for step in steps)
    for param_state in state.values():
        if isinstance(param_state, MutableMapping) and "step" in param_state:
            param_state["step"] = _step_like(param_state["step"], target)


def _step_as_int(step: Any) -> int:
    if isinstance(step, torch.Tensor):
        return int(step.detach().cpu().item())
    return int(step)


def _step_like(reference: Any, value: int) -> Any:
    if isinstance(reference, torch.Tensor):
        return torch.full_like(reference, value)
    return value


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

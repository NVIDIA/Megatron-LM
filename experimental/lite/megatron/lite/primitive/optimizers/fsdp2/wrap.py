# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""PyTorch FSDP2 wrapping primitive.

This module is intentionally independent from Megatron Lite model and runtime
packages. Model protocols can call it after building modules and before
building the optimizer.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.parallel.state import ParallelState

UnitModule = type[nn.Module] | str

_WARNED_TP_NOT_RECOMMENDED = False


@dataclass(frozen=True)
class FSDP2Config:
    """Configuration for ``wrap_fsdp2``."""

    unit_modules: tuple[UnitModule, ...] = field(default_factory=tuple)
    leaf_module_names: tuple[str, ...] = field(default_factory=tuple)
    reshard_after_forward: bool | int | None = None
    last_unit_reshard_after_forward: bool | int | None = False
    root_reshard_after_forward: bool | int | None = False
    wrap_root: bool = True
    preserve_param_attrs: bool = True
    forward_prefetch_depth: int = 1
    backward_prefetch_depth: int = 0
    mesh_dim_name: str = "dp_cp"
    device_type: str = "cuda"
    param_dtype: str | torch.dtype | None = None
    reduce_dtype: str | torch.dtype | None = None
    output_dtype: str | torch.dtype | None = None
    cast_forward_inputs: bool | None = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_modules", tuple(self.unit_modules))
        object.__setattr__(self, "leaf_module_names", tuple(self.leaf_module_names))
        if not self.wrap_root and not self.unit_modules and not self.leaf_module_names:
            raise ValueError(
                "FSDP2Config requires wrap_root=True, at least one unit module, "
                "or at least one leaf module name."
            )
        for name in self.leaf_module_names:
            if not isinstance(name, str) or not name:
                raise ValueError("leaf_module_names entries must be non-empty strings.")
        if not self.mesh_dim_name:
            raise ValueError("mesh_dim_name must be non-empty.")
        if not self.device_type:
            raise ValueError("device_type must be non-empty.")
        if self.forward_prefetch_depth < 0:
            raise ValueError("forward_prefetch_depth must be >= 0.")
        if self.backward_prefetch_depth < 0:
            raise ValueError("backward_prefetch_depth must be >= 0.")


def fsdp2_available() -> bool:
    """Return whether the installed PyTorch exposes FSDP2 ``fully_shard``."""

    try:
        from torch.distributed import DeviceMesh  # noqa: F401
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError:
        return False
    return True


def build_fsdp2_device_mesh(ps: ParallelState, config: FSDP2Config | None = None) -> Any:
    """Build the default one-dimensional FSDP2 DeviceMesh from ``ParallelState``."""

    cfg = config or FSDP2Config()
    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires torch.distributed to be initialized.")

    group = ps.dp_cp_group or ps.dp_group
    if group is None:
        raise RuntimeError("FSDP2 requires ParallelState.dp_cp_group or dp_group.")

    from torch.distributed import DeviceMesh

    return DeviceMesh.from_group(
        group, device_type=cfg.device_type, mesh_dim_names=(cfg.mesh_dim_name,)
    )


def build_fsdp2_process_group_mesh(
    group: dist.ProcessGroup, *, mesh_dim_name: str, device_type: str = "cuda"
) -> Any:
    """Build a one-dimensional FSDP2 DeviceMesh from an explicit process group."""

    if not dist.is_initialized():
        raise RuntimeError("FSDP2 requires torch.distributed to be initialized.")
    if group is None:
        raise RuntimeError("FSDP2 requires a non-null process group.")
    if not mesh_dim_name:
        raise ValueError("mesh_dim_name must be non-empty.")
    if not device_type:
        raise ValueError("device_type must be non-empty.")

    from torch.distributed import DeviceMesh

    return DeviceMesh.from_group(group, device_type=device_type, mesh_dim_names=(mesh_dim_name,))


def build_fsdp2_shard_placement_fn(fsdp_size: int) -> Callable[[nn.Parameter], Any]:
    """Build AutoModel-style FSDP2 shard placement.

    Choose the first tensor dimension divisible by the FSDP group size to avoid
    padded DTensor shards when possible.
    """

    if fsdp_size <= 0:
        raise ValueError(f"fsdp_size must be positive, got {fsdp_size}.")

    def shard_placement_fn(param: nn.Parameter) -> Any:
        from torch.distributed.tensor import Shard

        for dim, size in enumerate(param.shape):
            if int(size) % fsdp_size == 0:
                return Shard(dim)
        return Shard(0)

    return shard_placement_fn


def wrap_fsdp2(
    model: nn.Module,
    ps: ParallelState,
    config: FSDP2Config | None = None,
    *,
    mesh: Any | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    mp_policy: Any | None = None,
    offload_policy: Any | None = None,
    shard_placement_fn: Callable[[nn.Parameter], Any] | None = None,
) -> nn.Module:
    """Apply PyTorch FSDP2 ``fully_shard`` to selected modules and the root.

    The model is mutated in place and returned for call-site convenience.
    ``wrap_fsdp2`` must run before constructing a regular PyTorch optimizer.
    """

    cfg = config or FSDP2Config()
    _warn_tp_not_recommended(ps)
    fully_shard = _load_fully_shard()
    fsdp_mesh = mesh if mesh is not None else build_fsdp2_device_mesh(ps, cfg)
    unit_types = _resolve_unit_module_types(cfg.unit_modules)

    saved_attrs = _save_param_attrs(model) if cfg.preserve_param_attrs else {}
    common_kwargs = _fully_shard_kwargs(
        mesh=fsdp_mesh,
        reshard_after_forward=None,
        ignored_params=ignored_params,
        mp_policy=mp_policy or _mixed_precision_policy_from_config(cfg),
        offload_policy=offload_policy,
        shard_placement_fn=shard_placement_fn,
    )

    wrapped_units: list[nn.Module] = []
    unit_modules = list(_iter_fsdp2_unit_modules(model, unit_types, cfg.leaf_module_names))
    for idx, sub_module in enumerate(unit_modules):
        kwargs = dict(common_kwargs)
        _set_optional_reshard_after_forward(
            kwargs, _unit_reshard_after_forward(cfg, idx, len(unit_modules))
        )
        fully_shard(sub_module, **kwargs)
        wrapped_units.append(sub_module)

    if cfg.wrap_root:
        kwargs = dict(common_kwargs)
        _set_optional_reshard_after_forward(kwargs, cfg.root_reshard_after_forward)
        fully_shard(model, **kwargs)

    _apply_fsdp2_prefetch(
        model,
        wrapped_units,
        forward_depth=cfg.forward_prefetch_depth,
        backward_depth=cfg.backward_prefetch_depth,
    )
    if saved_attrs:
        _restore_param_attrs(model, saved_attrs)
    return model


def wrap_fsdp2_module(
    module: nn.Module,
    ps: ParallelState,
    config: FSDP2Config | None = None,
    *,
    mesh: Any | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    mp_policy: Any | None = None,
    offload_policy: Any | None = None,
    shard_placement_fn: Callable[[nn.Parameter], Any] | None = None,
    reshard_after_forward: bool | int | None = None,
) -> nn.Module:
    """Apply FSDP2 ``fully_shard`` to exactly one module.

    This is used for nested modules that need a different sharding mesh from
    their parent, e.g. EP-local MoE experts sharded over expert-DP while the
    transformer block is sharded over dense DP/CP.
    """

    cfg = config or FSDP2Config()
    _warn_tp_not_recommended(ps)
    fully_shard = _load_fully_shard()
    fsdp_mesh = mesh if mesh is not None else build_fsdp2_device_mesh(ps, cfg)

    saved_attrs = _save_param_attrs(module) if cfg.preserve_param_attrs else {}
    kwargs = _fully_shard_kwargs(
        mesh=fsdp_mesh,
        reshard_after_forward=(
            cfg.root_reshard_after_forward
            if reshard_after_forward is None
            else reshard_after_forward
        ),
        ignored_params=ignored_params,
        mp_policy=mp_policy or _mixed_precision_policy_from_config(cfg),
        offload_policy=offload_policy,
        shard_placement_fn=shard_placement_fn,
    )
    fully_shard(module, **kwargs)

    if saved_attrs:
        _restore_param_attrs(module, saved_attrs)
    return module


def _warn_tp_not_recommended(ps: ParallelState) -> None:
    global _WARNED_TP_NOT_RECOMMENDED
    if _WARNED_TP_NOT_RECOMMENDED or ps.tp_size <= 1:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    _WARNED_TP_NOT_RECOMMENDED = True
    warnings.warn(
        f"FSDP2 with tp={ps.tp_size} is supported, but TP is not recommended "
        "for FSDP2 V1; prefer tp=1 etp=1 for precision and speed signoff.",
        RuntimeWarning,
        stacklevel=3,
    )


def promote_fsdp2_trainable_params_to_fp32(
    model: nn.Module, *, ignored_params: set[nn.Parameter] | None = None
) -> int:
    """Promote FSDP2-owned trainable floating parameters to FP32 shards.

    Model protocols still use ``FSDP2Config.param_dtype`` to run compute in
    BF16. Keeping the sharded parameters in FP32 makes the torch optimizer path
    closer to MCore dist-opt's main-param semantics and avoids BF16 grad-norm
    and update drift before FSDP2 wrapping.
    """

    ignored_param_ids = {id(param) for param in ignored_params or ()}
    promoted = 0
    with torch.no_grad():
        for param in model.parameters():
            if id(param) in ignored_param_ids:
                continue
            if not param.requires_grad or not param.is_floating_point():
                continue
            if param.dtype == torch.float32:
                continue
            param._fsdp2_model_param_dtype = param.dtype
            param.data = param.data.to(torch.float32)
            if param.grad is not None:
                param.grad = param.grad.to(torch.float32)
            promoted += 1
    return promoted


def set_fsdp2_requires_gradient_sync(
    module: nn.Module, requires_gradient_sync: bool, *, recurse: bool = True
) -> int:
    """Set FSDP2 gradient sync state and return the number of touched roots."""

    setter = getattr(module, "set_requires_gradient_sync", None)
    if callable(setter):
        setter(requires_gradient_sync, recurse=recurse)
        return 1
    if not recurse:
        return 0

    touched = 0
    for child in module.modules():
        if child is module:
            continue
        setter = getattr(child, "set_requires_gradient_sync", None)
        if callable(setter):
            setter(requires_gradient_sync, recurse=False)
            touched += 1
    return touched


def _load_fully_shard():
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch FSDP2 is unavailable; install a PyTorch build with "
            "torch.distributed.fsdp.fully_shard."
        ) from exc
    return fully_shard


def _resolve_unit_module_types(unit_modules: Iterable[UnitModule]) -> tuple[type[nn.Module], ...]:
    resolved: list[type[nn.Module]] = []
    for item in unit_modules:
        if isinstance(item, str):
            item = _import_module_type(item)
        if not isinstance(item, type) or not issubclass(item, nn.Module):
            raise TypeError(
                "FSDP2 unit_modules entries must be nn.Module subclasses or import paths."
            )
        resolved.append(item)
    return tuple(resolved)


def _import_module_type(path: str) -> type[nn.Module]:
    module_name, sep, attr_name = path.rpartition(".")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid FSDP2 unit module path: {path!r}")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if not isinstance(obj, type) or not issubclass(obj, nn.Module):
        raise TypeError(f"FSDP2 unit module path does not resolve to nn.Module: {path!r}")
    return obj


def _iter_fsdp2_unit_modules(
    root: nn.Module, unit_types: tuple[type[nn.Module], ...], leaf_module_names: tuple[str, ...]
) -> Iterable[nn.Module]:
    leaf_names = set(leaf_module_names)
    if not unit_types and not leaf_names:
        return ()
    ordered_units: list[nn.Module] = []
    seen: set[int] = set()

    def visit(module: nn.Module, module_name: str) -> None:
        leaf_name = module_name.rsplit(".", 1)[-1] if module_name else ""
        selected = module is not root and (
            isinstance(module, unit_types) or leaf_name in leaf_names
        )
        if selected and _module_has_trainable_param(module):
            module_id = id(module)
            if module_id not in seen:
                ordered_units.append(module)
                seen.add(module_id)
            return
        for child_name, child in _iter_ordered_named_children(module):
            full_name = child_name if not module_name else f"{module_name}.{child_name}"
            visit(child, full_name)

    visit(root, "")
    return tuple(ordered_units)


def _iter_ordered_named_children(module: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    if isinstance(module, nn.ModuleDict):
        return module.items()
    if isinstance(module, nn.ModuleList):
        return ((str(idx), module[idx]) for idx in range(len(module)))
    return module.named_children()


def _module_has_trainable_param(module: nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters())


def _is_fsdp2_module(module: nn.Module) -> bool:
    return hasattr(module, "set_modules_to_forward_prefetch") and hasattr(
        module, "set_modules_to_backward_prefetch"
    )


def _apply_fsdp2_prefetch(
    root: nn.Module, wrapped_units: list[nn.Module], *, forward_depth: int, backward_depth: int
) -> None:
    fsdp_units = [module for module in wrapped_units if _is_fsdp2_module(module)]
    fsdp_root = root if _is_fsdp2_module(root) else None

    if fsdp_units and forward_depth > 0:
        if fsdp_root is not None:
            fsdp_root.set_modules_to_forward_prefetch(fsdp_units[:forward_depth])
        for idx, current in enumerate(fsdp_units[:-1]):
            targets = fsdp_units[idx + 1 : idx + 1 + forward_depth]
            if targets:
                current.set_modules_to_forward_prefetch(targets)

    if len(fsdp_units) > 1 and backward_depth > 0:
        for idx in range(1, len(fsdp_units)):
            start = max(0, idx - backward_depth)
            targets = list(reversed(fsdp_units[start:idx]))
            if targets:
                fsdp_units[idx].set_modules_to_backward_prefetch(targets)


def _unit_reshard_after_forward(cfg: FSDP2Config, idx: int, total_units: int) -> bool | int | None:
    if total_units > 0 and idx == total_units - 1:
        return cfg.last_unit_reshard_after_forward
    return cfg.reshard_after_forward


def _set_optional_reshard_after_forward(
    kwargs: dict[str, Any], reshard_after_forward: bool | int | None
) -> None:
    if reshard_after_forward is not None:
        kwargs["reshard_after_forward"] = reshard_after_forward


def _fully_shard_kwargs(
    *,
    mesh: Any,
    reshard_after_forward: bool | int | None,
    ignored_params: set[nn.Parameter] | None,
    mp_policy: Any | None,
    offload_policy: Any | None,
    shard_placement_fn: Callable[[nn.Parameter], Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"mesh": mesh}
    if reshard_after_forward is not None:
        kwargs["reshard_after_forward"] = reshard_after_forward
    if ignored_params is not None:
        kwargs["ignored_params"] = ignored_params
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy
    if shard_placement_fn is not None:
        kwargs["shard_placement_fn"] = shard_placement_fn
    return kwargs


def _mixed_precision_policy_from_config(cfg: FSDP2Config) -> Any | None:
    if cfg.param_dtype is None and cfg.reduce_dtype is None and cfg.output_dtype is None:
        return None
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except ImportError as exc:
        raise RuntimeError("FSDP2 mixed precision policy is unavailable.") from exc
    kwargs = dict(
        param_dtype=_resolve_torch_dtype(cfg.param_dtype),
        reduce_dtype=_resolve_torch_dtype(cfg.reduce_dtype),
        output_dtype=_resolve_torch_dtype(cfg.output_dtype),
    )
    if cfg.cast_forward_inputs is not None:
        kwargs["cast_forward_inputs"] = cfg.cast_forward_inputs
    try:
        return MixedPrecisionPolicy(**kwargs)
    except TypeError:
        kwargs.pop("cast_forward_inputs", None)
        return MixedPrecisionPolicy(**kwargs)


def _resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    name = dtype.removeprefix("torch.")
    resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported torch dtype for FSDP2 mixed precision: {dtype!r}")
    return resolved


def _save_param_attrs(module: nn.Module) -> dict[str, dict[str, Any]]:
    return {name: dict(vars(param)) for name, param in module.named_parameters()}


def _restore_param_attrs(module: nn.Module, saved_attrs: dict[str, dict[str, Any]]) -> None:
    for name, param in module.named_parameters():
        for attr_name, attr_value in saved_attrs.get(name, {}).items():
            setattr(param, attr_name, attr_value)


__all__ = [
    "FSDP2Config",
    "build_fsdp2_device_mesh",
    "build_fsdp2_process_group_mesh",
    "build_fsdp2_shard_placement_fn",
    "fsdp2_available",
    "promote_fsdp2_trainable_params_to_fp32",
    "set_fsdp2_requires_gradient_sync",
    "wrap_fsdp2",
    "wrap_fsdp2_module",
]

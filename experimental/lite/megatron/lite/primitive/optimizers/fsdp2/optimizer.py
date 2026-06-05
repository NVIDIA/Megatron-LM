"""FSDP2 optimizer adapter for Megatron Lite runtime contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2.adamw import (
    all_reduce_grad_,
    build_adamw_optimizer,
    fsdp2_model_param_dtype,
    get_bool_opt,
    has_dtensor_grad_or_param,
    local_grad_sq_sum,
)
from megatron.lite.primitive.optimizers.fsdp2.grad_clip import (
    clip_grads_with_sharded_norm_,
    resolve_torch_dtype,
    sharded_grad_sq_sum,
)
from megatron.lite.primitive.optimizers.fsdp2.state import (
    OffloadedStateEntry,
    move_offloaded_optimizer_state_to_device,
    move_optimizer_state_to_cpu,
)
from megatron.lite.primitive.optimizers.fsdp2.wrap import (
    FSDP2Config,
    build_fsdp2_process_group_mesh,
    build_fsdp2_shard_placement_fn,
    promote_fsdp2_trainable_params_to_fp32,
    wrap_fsdp2,
    wrap_fsdp2_module,
)
from megatron.lite.primitive.parallel.state import ParallelState


_DEFAULT_RESHARD_AFTER_FORWARD: bool | int | None = True
_DEFAULT_WRAP_ROOT = True
_DEFAULT_LEAF_MODULE_NAMES = ("embed", "head")
_DEFAULT_FORWARD_PREFETCH_DEPTH = 1
_DEFAULT_BACKWARD_PREFETCH_DEPTH = 0
_DEFAULT_PARAM_DTYPE: str | None = "bfloat16"
_DEFAULT_REDUCE_DTYPE: str | None = "float32"
_DEFAULT_USE_FP32_SHARDS = True
_DEFAULT_USE_FP32_MASTER = True
_DEFAULT_ADAMW_FOREACH: bool | str = "auto"


class FSDP2Optimizer:
    """Adapt an optimizer to Megatron Lite's FSDP2 optimizer contract."""

    name = "fsdp2"

    def __init__(
        self,
        optimizer: Any,
        params: Iterable[nn.Parameter],
        ps: ParallelState | None = None,
        *,
        clip_grad: float = 1.0,
        replicated_grad_params: Iterable[nn.Parameter] | None = None,
        replicated_grad_sync_group: dist.ProcessGroup | None = None,
        replicated_grad_sync_divisor: float | None = None,
        replicated_grad_norm_group: dist.ProcessGroup | None = None,
        expert_sharded_grad_params: Iterable[nn.Parameter] | None = None,
        expert_sharded_grad_scale: float | None = None,
        expert_sharded_grad_norm_group: dist.ProcessGroup | None = None,
        tp_replicated_grad_params: Iterable[nn.Parameter] | None = None,
        tp_replicated_grad_sync_group: dist.ProcessGroup | None = None,
        grad_norm_accum_dtype: str | torch.dtype = torch.float32,
        optimizer_offload_dtensor_state: bool = False,
        param_names: dict[int, str] | None = None,
    ):
        self.optimizer = optimizer
        self.params = list(params)
        self.param_names = dict(param_names or {})
        self.ps = ps
        self.clip_grad = float(clip_grad)
        self.grad_norm_accum_dtype = resolve_torch_dtype(grad_norm_accum_dtype)
        self.replicated_grad_params = list(replicated_grad_params or ())
        self._replicated_grad_param_ids = {id(param) for param in self.replicated_grad_params}
        self.replicated_grad_sync_group = replicated_grad_sync_group
        self.replicated_grad_sync_divisor = replicated_grad_sync_divisor
        self.replicated_grad_norm_group = replicated_grad_norm_group
        self.expert_sharded_grad_params = list(expert_sharded_grad_params or ())
        self._expert_sharded_grad_param_ids = {
            id(param) for param in self.expert_sharded_grad_params
        }
        self.expert_sharded_grad_scale = (
            1.0 if expert_sharded_grad_scale is None else float(expert_sharded_grad_scale)
        )
        self.expert_sharded_grad_norm_group = expert_sharded_grad_norm_group
        self.tp_replicated_grad_params = list(tp_replicated_grad_params or ())
        self._tp_replicated_grad_param_ids = {
            id(param) for param in self.tp_replicated_grad_params
        }
        self.tp_replicated_grad_sync_group = tp_replicated_grad_sync_group
        self.optimizer_offload_dtensor_state = bool(optimizer_offload_dtensor_state)
        self._cpu_offloaded_state: dict[tuple[int, str], OffloadedStateEntry] = {}
        self.grad_sync_enabled = False

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self) -> None:
        self.grad_sync_enabled = False
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> tuple[bool, float, int]:
        self.sync_tp_replicated_grads()
        self.sync_replicated_grads()
        self.scale_expert_sharded_grads()
        grad_norm = self.clip_grad_norm()
        if not math.isfinite(grad_norm):
            self.grad_sync_enabled = False
            return False, float(grad_norm), 0
        self.optimizer.step()
        self.grad_sync_enabled = False
        return True, float(grad_norm), 0

    def sync_replicated_grads(self) -> None:
        group = self.replicated_grad_sync_group
        if not self.replicated_grad_params:
            return
        group_size = 1
        if group is not None and dist.is_initialized():
            group_size = dist.get_world_size(group)
        if group is None and self.replicated_grad_sync_divisor is None:
            return
        divisor = self.replicated_grad_sync_divisor
        if divisor is None:
            divisor = float(group_size)
        for param in self.replicated_grad_params:
            grad = param.grad
            if grad is None:
                continue
            if group is not None and dist.is_initialized() and group_size > 1:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
            if divisor != 1.0:
                grad.div_(divisor)

    def sync_tp_replicated_grads(self) -> None:
        group = self.tp_replicated_grad_sync_group
        if not self.tp_replicated_grad_params:
            return
        if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return
        for param in self.tp_replicated_grad_params:
            grad = param.grad
            if grad is not None:
                all_reduce_grad_(grad, group=group)

    def scale_expert_sharded_grads(self) -> None:
        if not self.expert_sharded_grad_params or self.expert_sharded_grad_scale == 1.0:
            return
        for param in self.expert_sharded_grad_params:
            grad = param.grad
            if grad is not None:
                grad.mul_(self.expert_sharded_grad_scale)

    def clip_grad_norm(self) -> float:
        excluded_sharded_param_ids = (
            self._replicated_grad_param_ids
            | self._tp_replicated_grad_param_ids
            | self._expert_sharded_grad_param_ids
        )
        sharded_params = [
            param for param in self.params if id(param) not in excluded_sharded_param_ids
        ]
        dtensor_sharded_params = [
            param for param in sharded_params if has_dtensor_grad_or_param(param)
        ]
        plain_sharded_params = [
            param for param in sharded_params if not has_dtensor_grad_or_param(param)
        ]
        total_sq = sharded_grad_sq_sum(
            dtensor_sharded_params,
            accum_dtype=self.grad_norm_accum_dtype,
        )
        plain_sharded_sq = local_grad_sq_sum(
            plain_sharded_params,
            dtype=resolve_torch_dtype(self.grad_norm_accum_dtype),
            default_device=total_sq.device,
        )
        plain_sharded_group = None
        if self.ps is not None:
            plain_sharded_group = self.ps.dp_cp_group or self.ps.dp_group
        if (
            plain_sharded_group is not None
            and dist.is_initialized()
            and dist.get_world_size(plain_sharded_group) > 1
        ):
            dist.all_reduce(
                plain_sharded_sq,
                op=dist.ReduceOp.SUM,
                group=plain_sharded_group,
            )
        total_sq = total_sq.to(plain_sharded_sq.device) + plain_sharded_sq
        if (
            self.ps is not None
            and self.ps.tp_group is not None
            and dist.is_initialized()
            and dist.get_world_size(self.ps.tp_group) > 1
        ):
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=self.ps.tp_group)

        tp_replicated_sq = sharded_grad_sq_sum(
            self.tp_replicated_grad_params,
            accum_dtype=self.grad_norm_accum_dtype,
            default_device=total_sq.device,
        )
        replicated_sq = local_grad_sq_sum(
            self.replicated_grad_params,
            dtype=self.grad_norm_accum_dtype,
            default_device=total_sq.device,
        )
        if (
            self.replicated_grad_norm_group is not None
            and dist.is_initialized()
            and dist.get_world_size(self.replicated_grad_norm_group) > 1
        ):
            dist.all_reduce(
                replicated_sq,
                op=dist.ReduceOp.SUM,
                group=self.replicated_grad_norm_group,
            )

        expert_sharded_sq = sharded_grad_sq_sum(
            self.expert_sharded_grad_params,
            accum_dtype=self.grad_norm_accum_dtype,
            default_device=total_sq.device,
        )
        if (
            self.expert_sharded_grad_norm_group is not None
            and dist.is_initialized()
            and dist.get_world_size(self.expert_sharded_grad_norm_group) > 1
        ):
            dist.all_reduce(
                expert_sharded_sq,
                op=dist.ReduceOp.SUM,
                group=self.expert_sharded_grad_norm_group,
            )

        total_sq = (
            total_sq
            + tp_replicated_sq.to(total_sq.device)
            + replicated_sq.to(total_sq.device)
            + expert_sharded_sq.to(total_sq.device)
        )
        if (
            self.ps is not None
            and self.ps.pp_group is not None
            and dist.is_initialized()
            and dist.get_world_size(self.ps.pp_group) > 1
        ):
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=self.ps.pp_group)

        grad_norm = total_sq.sqrt()
        if torch.isfinite(grad_norm):
            clip_grads_with_sharded_norm_(self.params, self.clip_grad, grad_norm)
        return float(grad_norm.float().item())

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def offload_state_to_cpu(self) -> None:
        move_optimizer_state_to_cpu(
            self.optimizer,
            self._cpu_offloaded_state,
            include_dtensor_state=self.optimizer_offload_dtensor_state,
        )

    def load_state_to_device(self) -> None:
        move_offloaded_optimizer_state_to_device(self.optimizer, self._cpu_offloaded_state)


def build_fsdp2_adamw(
    model_chunks: list[nn.Module],
    opt,
    ps: ParallelState,
    *,
    replicated_grad_params: Iterable[nn.Parameter] | None = None,
    replicated_grad_sync_group: dist.ProcessGroup | None = None,
    replicated_grad_sync_divisor: float | None = None,
    replicated_grad_norm_group: dist.ProcessGroup | None = None,
    expert_sharded_grad_params: Iterable[nn.Parameter] | None = None,
    expert_sharded_grad_scale: float | None = None,
    expert_sharded_grad_norm_group: dist.ProcessGroup | None = None,
    tp_replicated_grad_params: Iterable[nn.Parameter] | None = None,
    tp_replicated_grad_sync_group: dist.ProcessGroup | None = None,
    grad_norm_accum_dtype: str | torch.dtype = torch.float32,
    adamw_foreach: bool | str = "auto",
    optimizer_offload_dtensor_state: bool = False,
    use_fp32_master: bool = False,
    model_param_dtypes: dict[tuple[int, str], torch.dtype] | None = None,
) -> FSDP2Optimizer:
    """Build AdamW from Megatron Lite's shared OptimizerConfig-like object."""

    optimizer_name = getattr(opt, "optimizer", "adam")
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError(f"fsdp2 supports adam/adamw, got {optimizer_name!r}.")

    params, param_groups, param_names, param_model_dtypes = _build_adamw_param_groups(
        model_chunks,
        weight_decay=float(getattr(opt, "weight_decay", 0.01)),
        apply_wd_to_qk_layernorm=bool(getattr(opt, "apply_wd_to_qk_layernorm", False)),
        model_param_dtypes=model_param_dtypes,
    )
    beta1 = getattr(opt, "adam_beta1", None)
    beta2 = getattr(opt, "adam_beta2", None)
    eps = getattr(opt, "adam_eps", None)
    offload_fraction = getattr(opt, "offload_fraction", None) or 0.0
    optimizer = build_adamw_optimizer(
        param_groups,
        all_params=params,
        lr=float(getattr(opt, "lr", 1.0e-4)),
        weight_decay=float(getattr(opt, "weight_decay", 0.01)),
        betas=(0.9 if beta1 is None else beta1, 0.999 if beta2 is None else beta2),
        eps=1.0e-8 if eps is None else eps,
        foreach=adamw_foreach,
        use_fp32_master=use_fp32_master,
        cpu_update=use_fp32_master and float(offload_fraction) > 0.0,
        model_param_dtypes=param_model_dtypes,
        opt=opt,
    )
    return FSDP2Optimizer(
        optimizer,
        params,
        ps,
        clip_grad=float(getattr(opt, "clip_grad", 1.0)),
        replicated_grad_params=replicated_grad_params,
        replicated_grad_sync_group=replicated_grad_sync_group,
        replicated_grad_sync_divisor=replicated_grad_sync_divisor,
        replicated_grad_norm_group=replicated_grad_norm_group,
        expert_sharded_grad_params=expert_sharded_grad_params,
        expert_sharded_grad_scale=expert_sharded_grad_scale,
        expert_sharded_grad_norm_group=expert_sharded_grad_norm_group,
        tp_replicated_grad_params=tp_replicated_grad_params,
        tp_replicated_grad_sync_group=tp_replicated_grad_sync_group,
        grad_norm_accum_dtype=grad_norm_accum_dtype,
        optimizer_offload_dtensor_state=(
            optimizer_offload_dtensor_state or float(offload_fraction) > 0.0
        ),
        param_names=param_names,
    )


def build_fsdp2_training_optimizer(
    model_chunks: list[nn.Module],
    opt,
    ps: ParallelState,
    *,
    unit_modules: tuple[type[nn.Module] | str, ...],
    expert_classifier: Callable[[str], bool] | None = None,
    expert_module_leaf_name: str = "experts",
    deterministic: bool | None = None,
    vpp: int | None = 1,
    leaf_module_names: Iterable[str] = _DEFAULT_LEAF_MODULE_NAMES,
    reshard_after_forward: bool | int | None = _DEFAULT_RESHARD_AFTER_FORWARD,
    wrap_root: bool = _DEFAULT_WRAP_ROOT,
    forward_prefetch_depth: int = _DEFAULT_FORWARD_PREFETCH_DEPTH,
    backward_prefetch_depth: int = _DEFAULT_BACKWARD_PREFETCH_DEPTH,
    param_dtype: str | torch.dtype | None = _DEFAULT_PARAM_DTYPE,
    reduce_dtype: str | torch.dtype | None = _DEFAULT_REDUCE_DTYPE,
    use_fp32_shards: bool | None = None,
    use_fp32_master: bool | None = None,
    adamw_foreach: bool | str = _DEFAULT_ADAMW_FOREACH,
) -> FSDP2Optimizer:
    """Wrap model chunks with FSDP2 and build the matching AdamW adapter."""

    if (vpp or 1) > 1 and ps.pp_size <= 1:
        raise ValueError("optimizer='fsdp2' requires pp>1 when vpp>1.")

    expert_params = _collect_expert_params(model_chunks, ps, expert_classifier)
    expert_modules = _collect_expert_modules(
        model_chunks,
        ps,
        expert_classifier,
        expert_module_leaf_name=expert_module_leaf_name,
    )
    if expert_params and not expert_modules:
        raise RuntimeError("FSDP2 expert parameters were found but no expert module was found.")

    if opt is None:
        opt = SimpleNamespace(
            optimizer="adam",
            lr=1e-4,
            weight_decay=0.01,
            clip_grad=1.0,
            adam_beta1=None,
            adam_beta2=None,
            adam_eps=None,
        )
    if deterministic is None:
        from megatron.lite.primitive.deterministic import deterministic_requested

        deterministic = deterministic_requested()
    effective_use_fp32_shards = (
        bool(use_fp32_shards)
        if use_fp32_shards is not None
        else get_bool_opt(
            opt,
            "fsdp2_use_fp32_shards",
            default=_DEFAULT_USE_FP32_SHARDS,
        )
    )
    effective_use_fp32_master = (
        bool(use_fp32_master)
        if use_fp32_master is not None
        else get_bool_opt(
            opt,
            "fsdp2_use_fp32_master",
            default=_DEFAULT_USE_FP32_MASTER,
        )
    )

    unit_reshard_after_forward = _fsdp2_unit_reshard_after_forward(
        ps,
        reshard_after_forward=reshard_after_forward,
    )
    fsdp2_config = FSDP2Config(
        unit_modules=unit_modules,
        leaf_module_names=tuple(leaf_module_names),
        reshard_after_forward=unit_reshard_after_forward,
        last_unit_reshard_after_forward=unit_reshard_after_forward,
        root_reshard_after_forward=False,
        wrap_root=wrap_root,
        forward_prefetch_depth=_fsdp2_prefetch_depth(
            ps,
            default_depth=forward_prefetch_depth,
        ),
        backward_prefetch_depth=_fsdp2_prefetch_depth(
            ps,
            default_depth=backward_prefetch_depth,
        ),
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )
    if effective_use_fp32_shards:
        model_param_dtypes = _collect_model_param_dtypes(model_chunks)
        for chunk in model_chunks:
            promote_fsdp2_trainable_params_to_fp32(chunk)
    else:
        model_param_dtypes = {}

    tp_replicated_grad_param_names = _collect_tp_replicated_grad_param_names(
        model_chunks
    )

    dense_shard_placement_fn = build_fsdp2_shard_placement_fn(ps.dp_cp_size)
    expert_shard_placement_fn = build_fsdp2_shard_placement_fn(ps.expert_dp_size)

    if expert_modules:
        if ps.ep_dp_group is None:
            raise RuntimeError("FSDP2 expert sharding requires ParallelState.ep_dp_group.")
        expert_mesh = build_fsdp2_process_group_mesh(
            ps.ep_dp_group,
            mesh_dim_name="expert_dp",
            device_type=fsdp2_config.device_type,
        )
        for module in expert_modules:
            wrap_fsdp2_module(
                module,
                ps,
                fsdp2_config,
                mesh=expert_mesh,
                shard_placement_fn=expert_shard_placement_fn,
                reshard_after_forward=unit_reshard_after_forward,
            )

    ignored_expert_params = _collect_module_params(expert_modules)
    for chunk in model_chunks:
        wrap_fsdp2(
            chunk,
            ps,
            fsdp2_config,
            ignored_params=ignored_expert_params or None,
            shard_placement_fn=dense_shard_placement_fn,
        )
    if model_param_dtypes:
        _restore_model_param_dtypes(model_chunks, model_param_dtypes)

    tp_replicated_grad_params = _collect_tp_replicated_grad_params(
        model_chunks,
        param_names=tp_replicated_grad_param_names,
    )
    return build_fsdp2_adamw(
        model_chunks,
        opt,
        ps,
        expert_sharded_grad_params=list(ignored_expert_params),
        expert_sharded_grad_scale=(
            float(ps.expert_dp_size) / float(ps.dp_cp_size)
            if ignored_expert_params else None
        ),
        expert_sharded_grad_norm_group=ps.ep_group if ignored_expert_params else None,
        tp_replicated_grad_params=tp_replicated_grad_params,
        tp_replicated_grad_sync_group=ps.tp_group if tp_replicated_grad_params else None,
        grad_norm_accum_dtype="float32",
        adamw_foreach=False if deterministic else adamw_foreach,
        use_fp32_master=effective_use_fp32_master,
        model_param_dtypes=model_param_dtypes,
    )


def _collect_expert_params(
    chunks: Iterable[nn.Module],
    ps: ParallelState,
    expert_classifier: Callable[[str], bool] | None,
) -> set[nn.Parameter]:
    if ps.ep_size <= 1 or expert_classifier is None:
        return set()
    return {
        param
        for chunk in chunks
        for name, param in chunk.named_parameters()
        if expert_classifier(name)
    }


def _collect_expert_modules(
    chunks: Iterable[nn.Module],
    ps: ParallelState,
    expert_classifier: Callable[[str], bool] | None,
    *,
    expert_module_leaf_name: str,
) -> list[nn.Module]:
    if ps.ep_size <= 1 or expert_classifier is None:
        return []
    modules: list[nn.Module] = []
    seen: set[int] = set()
    for chunk in chunks:
        for module_name, module in chunk.named_modules():
            if module_name.rsplit(".", 1)[-1] != expert_module_leaf_name:
                continue
            prefix = f"{module_name}."
            if not any(
                expert_classifier(prefix + name)
                for name, _param in module.named_parameters(recurse=True)
            ):
                continue
            module_id = id(module)
            if module_id not in seen:
                modules.append(module)
                seen.add(module_id)
    return modules


def _collect_module_params(modules: Iterable[nn.Module]) -> set[nn.Parameter]:
    return {param for module in modules for param in module.parameters()}


def _collect_model_param_dtypes(
    chunks: Iterable[nn.Module],
) -> dict[tuple[int, str], torch.dtype]:
    return {
        (chunk_idx, name): param.dtype
        for chunk_idx, chunk in enumerate(chunks)
        for name, param in chunk.named_parameters()
        if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32
    }


def _restore_model_param_dtypes(
    chunks: Iterable[nn.Module],
    model_param_dtypes: dict[tuple[int, str], torch.dtype],
) -> None:
    for chunk_idx, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            model_dtype = model_param_dtypes.get((chunk_idx, name))
            if model_dtype is not None:
                param._fsdp2_model_param_dtype = model_dtype


def _collect_tp_replicated_grad_param_names(
    chunks: Iterable[nn.Module],
) -> list[tuple[int, str]]:
    names: list[tuple[int, str]] = []
    for chunk_idx, chunk in enumerate(chunks):
        sp_param_ids = {id(param) for param in getattr(chunk, "sp_params", ())}
        if not sp_param_ids:
            continue
        for name, param in chunk.named_parameters():
            if id(param) in sp_param_ids:
                names.append((chunk_idx, name))
    return names


def _collect_tp_replicated_grad_params(
    chunks: Iterable[nn.Module],
    *,
    param_names: Iterable[tuple[int, str]] | None = None,
) -> list[nn.Parameter]:
    chunk_list = list(chunks)
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    if param_names is not None:
        named_by_chunk = [dict(chunk.named_parameters()) for chunk in chunk_list]
        for chunk_idx, name in param_names:
            if chunk_idx >= len(named_by_chunk):
                continue
            param = named_by_chunk[chunk_idx].get(name)
            if param is None or not param.requires_grad or id(param) in seen:
                continue
            params.append(param)
            seen.add(id(param))
        return params
    for chunk in chunk_list:
        for param in getattr(chunk, "sp_params", ()):
            if not param.requires_grad or id(param) in seen:
                continue
            params.append(param)
            seen.add(id(param))
    return params


def _fsdp2_unit_reshard_after_forward(
    ps: ParallelState,
    *,
    reshard_after_forward: bool | int | None,
) -> bool | int | None:
    if ps.pp_size > 1:
        return False
    return reshard_after_forward


def _fsdp2_prefetch_depth(ps: ParallelState, *, default_depth: int) -> int:
    if ps.pp_size > 1:
        return 0
    return default_depth


def _build_adamw_param_groups(
    model_chunks: Iterable[nn.Module],
    *,
    weight_decay: float,
    apply_wd_to_qk_layernorm: bool,
    model_param_dtypes: dict[tuple[int, str], torch.dtype] | None = None,
) -> tuple[list[nn.Parameter], list[dict[str, Any]], dict[int, str], dict[int, torch.dtype]]:
    params: list[nn.Parameter] = []
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    param_names: dict[int, str] = {}
    param_model_dtypes: dict[int, torch.dtype] = {}
    seen_param_ids: set[int] = set()

    for chunk_idx, chunk in enumerate(model_chunks):
        for name, param in chunk.named_parameters():
            if not param.requires_grad or id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            params.append(param)
            param_names[id(param)] = f"chunk{chunk_idx}.{name}"
            model_dtype = None
            if model_param_dtypes is not None:
                model_dtype = model_param_dtypes.get((chunk_idx, name))
            if model_dtype is None:
                model_dtype = fsdp2_model_param_dtype(param)
            if model_dtype is not None:
                param_model_dtypes[id(param)] = model_dtype
            if _matches_megatron_no_weight_decay(name, param, apply_wd_to_qk_layernorm):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # Match Megatron's default get_megatron_optimizer(config_overrides=None):
    # 1D params and bias skip AdamW decay unless the Q/K layernorm override is set.
    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay, "wd_mult": 1.0})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "wd_mult": 0.0})
    return params, param_groups, param_names, param_model_dtypes


def _matches_megatron_no_weight_decay(
    name: str,
    param: nn.Parameter,
    apply_wd_to_qk_layernorm: bool,
) -> bool:
    if len(param.shape) != 1 and not name.endswith(".bias"):
        return False
    if not apply_wd_to_qk_layernorm:
        return True
    return not (
        "q_layernorm." in name
        or "k_layernorm." in name
        or "q_norm." in name
        or "k_norm." in name
    )


@dataclass(frozen=True, slots=True)
class FSDP2OptimizerBackend:
    name: str = "fsdp2"
    runtime_backend: str = "fsdp2"

    def zero_grad(self, optimizer: FSDP2Optimizer) -> None:
        optimizer.zero_grad()

    def finish_grad_sync(self, optimizer: FSDP2Optimizer) -> None:
        return None

    def clip_grad_norm(self, optimizer: FSDP2Optimizer):
        return optimizer.clip_grad_norm()

    def step(self, optimizer: FSDP2Optimizer):
        return optimizer.step()

    def state_dict(self, optimizer: FSDP2Optimizer) -> dict:
        return optimizer.state_dict()

    def load_state_dict(
        self,
        optimizer: FSDP2Optimizer,
        state_dict: dict,
    ) -> None:
        optimizer.load_state_dict(state_dict)


BACKEND = FSDP2OptimizerBackend()

__all__ = [
    "BACKEND",
    "FSDP2OptimizerBackend",
    "FSDP2Optimizer",
    "build_fsdp2_adamw",
    "build_fsdp2_training_optimizer",
]

"""AdamW helpers for the FSDP2 optimizer primitive."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


def local_grad_sq_sum(
    params: Iterable[nn.Parameter],
    *,
    dtype: torch.dtype,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    total: torch.Tensor | None = None
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        grad = to_local_tensor(grad)
        if total is None:
            total = torch.zeros((), device=grad.device, dtype=dtype)
        total += grad.detach().to(dtype).pow(2).sum()
    if total is None:
        return torch.zeros((), device=default_device or torch.device("cpu"), dtype=dtype)
    return total


def to_local_tensor(tensor):
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def fsdp2_model_param_dtype(param: nn.Parameter) -> torch.dtype | None:
    dtype = getattr(param, "_fsdp2_model_param_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def has_dtensor_grad_or_param(param: nn.Parameter) -> bool:
    grad = param.grad
    return is_dtensor_like(param) or (grad is not None and is_dtensor_like(grad))


def is_dtensor_like(tensor: Any) -> bool:
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def copy_local_tensor_to_param_(param: nn.Parameter, local_tensor: torch.Tensor) -> None:
    if not is_dtensor_like(param):
        param.detach().copy_(local_tensor.to(device=param.device, dtype=param.dtype))
        return

    local_param = to_local_tensor(param)
    local_value = local_tensor.to(device=local_param.device, dtype=local_param.dtype)
    from torch.distributed.tensor import DTensor

    param.detach().copy_(DTensor.from_local(local_value, param.device_mesh, param.placements))


def all_reduce_grad_(grad: torch.Tensor, *, group: dist.ProcessGroup) -> None:
    local_grad = to_local_tensor(grad)
    dist.all_reduce(local_grad, op=dist.ReduceOp.SUM, group=group)
    if local_grad is grad:
        return
    from torch.distributed.tensor import DTensor

    grad.copy_(DTensor.from_local(local_grad, grad.device_mesh, grad.placements))


class ChainedOptimizer:
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "chained_torch_optimizer",
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers")
        if not isinstance(optimizer_states, list) or len(optimizer_states) != len(self.optimizers):
            raise ValueError("Invalid chained torch optimizer state_dict.")
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(optimizer_state)


class FP32AdamW:
    """AdamW with FP32 master params for BF16/DTensor model weights."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        cpu_update: bool = False,
        model_param_dtypes: dict[int, torch.dtype] | None = None,
    ):
        self.param_groups = normalize_param_groups(params, default_weight_decay=weight_decay)
        self.params: list[nn.Parameter] = []
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.cpu_update = bool(cpu_update)
        self.step_count = 0
        self.state: dict[nn.Parameter, dict[str, torch.Tensor]] = {}
        self._master_for_param: dict[nn.Parameter, torch.Tensor] = {}
        self._model_param_dtypes_by_id = dict(model_param_dtypes or {})
        self._model_dtype_for_param: dict[nn.Parameter, torch.dtype] = {}

        for group in self.param_groups:
            group.setdefault("lr", lr)
            group.setdefault("wd_mult", 1.0)
            group_weight_decay = float(group.get("weight_decay", weight_decay))
            group["weight_decay"] = group_weight_decay
            for param in group["params"]:
                self.params.append(param)
                model_dtype = self._model_param_dtypes_by_id.get(id(param))
                if model_dtype is not None:
                    self._model_dtype_for_param[param] = model_dtype
                master = self._init_master_param(param)
                self.state[param] = {
                    "master_param": master,
                    "exp_avg": torch.zeros_like(master, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(master, dtype=torch.float32),
                    "step": 0,
                }
                self._master_for_param[param] = master

    def _init_master_param(self, param: nn.Parameter) -> torch.Tensor:
        if self.cpu_update:
            local_param = to_local_tensor(param.detach())
            return local_param.detach().to(device="cpu", dtype=torch.float32).clone()
        if self._model_param_dtype(param) is not None:
            return param.detach().to(dtype=torch.float32).clone()
        return (
            param.detach()
            if param.dtype is torch.float32
            else param.detach().to(dtype=torch.float32).clone()
        )

    def _model_param_dtype(self, param: nn.Parameter) -> torch.dtype | None:
        return self._model_dtype_for_param.get(param) or fsdp2_model_param_dtype(param)

    def zero_grad(self, *args, **kwargs) -> None:
        set_to_none = kwargs.get("set_to_none", False)
        if args:
            set_to_none = bool(args[0])
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self) -> None:
        self._step_param_groups()

    def _step_param_groups(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas

        for group in self.param_groups:
            group_lr = float(group.get("lr", self.lr))
            group_weight_decay = float(group.get("weight_decay", self.weight_decay))
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                state["step"] = int(state["step"]) + 1
                param_step = int(state["step"])
                bias_correction1 = 1.0 - beta1**param_step
                bias_correction2_sqrt = (1.0 - beta2**param_step) ** 0.5
                group_step_size = group_lr / bias_correction1
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group_weight_decay != 0.0:
                    master.mul_(1.0 - group_lr * group_weight_decay)
                grad = self._prepare_grad(grad, master)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
                master.addcdiv_(exp_avg.to(dtype=torch.float32), denom, value=-group_step_size)
                self._copy_master_to_param(param, master)

    def _prepare_grad(self, grad: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        if self.cpu_update:
            grad = to_local_tensor(grad)
            return grad.detach().to(device=master.device, dtype=torch.float32)
        return grad.detach().to(dtype=torch.float32)

    def _copy_master_to_param(self, param: nn.Parameter, master: torch.Tensor) -> None:
        model_dtype = self._model_param_dtype(param)
        if model_dtype is not None:
            master = master.to(dtype=model_dtype).to(dtype=param.dtype)
        if not self.cpu_update:
            param.detach().copy_(master.to(dtype=param.dtype))
            return
        copy_local_tensor_to_param_(param, master)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "fp32_adamw",
            "step_count": self.step_count,
            "master_params": [self.state[param]["master_param"] for param in self.params],
            "exp_avgs": [self.state[param]["exp_avg"] for param in self.params],
            "exp_avg_sqs": [self.state[param]["exp_avg_sq"] for param in self.params],
            "steps": [int(self.state[param]["step"]) for param in self.params],
            "weight_decays": [
                float(group.get("weight_decay", self.weight_decay))
                for group in self.param_groups
                for _param in group["params"]
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("type") != "fp32_adamw":
            raise ValueError("Invalid FP32 AdamW state_dict.")
        self.step_count = int(state_dict.get("step_count", 0))
        for target_name, key in (
            ("master_params", "master_param"),
            ("exp_avgs", "exp_avg"),
            ("exp_avg_sqs", "exp_avg_sq"),
        ):
            loaded = state_dict.get(target_name)
            if not isinstance(loaded, list) or len(loaded) != len(self.params):
                raise ValueError(f"Invalid FP32 AdamW {target_name} state.")
            for param, src in zip(self.params, loaded, strict=True):
                self.state[param][key].copy_(src)
        loaded_steps = state_dict.get("steps")
        if loaded_steps is not None:
            if not isinstance(loaded_steps, list) or len(loaded_steps) != len(self.params):
                raise ValueError("Invalid FP32 AdamW steps state.")
            for param, step in zip(self.params, loaded_steps, strict=True):
                self.state[param]["step"] = int(step)
        else:
            for param in self.params:
                self.state[param]["step"] = self.step_count
        loaded_weight_decays = state_dict.get("weight_decays")
        if loaded_weight_decays is not None:
            if not isinstance(loaded_weight_decays, list) or len(loaded_weight_decays) != len(
                self.params
            ):
                raise ValueError("Invalid FP32 AdamW weight_decay state.")
            idx = 0
            for group in self.param_groups:
                if not group["params"]:
                    continue
                group["weight_decay"] = float(loaded_weight_decays[idx])
                idx += len(group["params"])


def build_adamw_optimizer(
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
    *,
    all_params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    foreach: bool | str,
    use_fp32_master: bool,
    cpu_update: bool,
    model_param_dtypes: dict[int, torch.dtype] | None,
    opt,
) -> Any:
    param_groups = normalize_param_groups(params, default_weight_decay=weight_decay)
    fused_adam = maybe_build_te_fused_adam_optimizer(
        param_groups,
        all_params=all_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        opt=opt,
        use_fp32_master=use_fp32_master,
    )
    if fused_adam is not None:
        return fused_adam
    if use_fp32_master:
        return FP32AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            cpu_update=cpu_update,
            model_param_dtypes=model_param_dtypes,
        )
    if foreach not in {True, False, "auto"}:
        raise ValueError(f"adamw_foreach must be True, False, or 'auto', got {foreach!r}.")
    if foreach is False:
        return torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=False
        )

    dtensor_param_groups, tensor_param_groups = split_dtensor_and_tensor_param_groups(
        param_groups, default_weight_decay=weight_decay
    )
    split_param_groups = [group for group in (dtensor_param_groups, tensor_param_groups) if group]
    if foreach == "auto" and not dtensor_param_groups:
        return torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=False
        )
    if len(split_param_groups) <= 1:
        return torch.optim.AdamW(
            split_param_groups[0] if split_param_groups else param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            foreach=True,
        )
    return ChainedOptimizer(
        torch.optim.AdamW(
            group, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=True
        )
        for group in split_param_groups
    )


def maybe_build_te_fused_adam_optimizer(
    param_groups: list[dict[str, Any]],
    *,
    all_params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    opt,
    use_fp32_master: bool,
) -> Any | None:
    if not get_bool_opt(opt, "fsdp2_use_te_fused_adam", default=False):
        return None
    try:
        from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
    except ImportError:
        return None

    all_param_list = list(all_params)
    master_weights = get_bool_opt(
        opt, "master_weights", default=use_fp32_master and should_use_master_weights(all_param_list)
    )
    kwargs = dict(
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        adam_w_mode=True,
        master_weights=master_weights,
        master_weight_dtype=get_dtype_opt(opt, "master_weight_dtype", default=torch.float32),
        store_param_remainders=get_bool_opt(opt, "store_param_remainders", default=master_weights),
        exp_avg_dtype=get_dtype_opt(opt, "exp_avg_dtype", default=torch.float32),
        exp_avg_sq_dtype=get_dtype_opt(opt, "exp_avg_sq_dtype", default=torch.float32),
    )
    return FusedAdam(param_groups, **filter_supported_kwargs(FusedAdam.__init__, kwargs))


def filter_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def should_use_master_weights(params: Iterable[nn.Parameter]) -> bool:
    return any(param.is_floating_point() and param.dtype is not torch.float32 for param in params)


def get_bool_opt(opt, attr: str, *, default: bool) -> bool:
    value = get_opt_value(opt, attr)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_dtype_opt(opt, attr: str, *, default: torch.dtype) -> torch.dtype:
    value = get_opt_value(opt, attr)
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    name = str(value).removeprefix("torch.")
    resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported dtype for FSDP2 TE FusedAdam: {value!r}.")
    return resolved


def get_opt_value(opt, attr: str):
    if opt is None:
        return None
    if isinstance(opt, dict):
        value = opt.get(attr)
        override = opt.get("override_optimizer_config")
    else:
        value = getattr(opt, attr, None)
        override = getattr(opt, "override_optimizer_config", None)
    if value is not None:
        return value
    if isinstance(override, dict):
        return override.get(attr)
    return None


def normalize_param_groups(
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]], *, default_weight_decay: float
) -> list[dict[str, Any]]:
    items = list(params)
    if not items:
        return []
    if all(isinstance(item, dict) for item in items):
        groups: list[dict[str, Any]] = []
        for item in items:
            group = dict(item)
            group_params = list(group.get("params", ()))
            if not group_params:
                continue
            group["params"] = group_params
            group.setdefault("weight_decay", default_weight_decay)
            groups.append(group)
        return groups
    return [{"params": items, "weight_decay": default_weight_decay}]


def split_dtensor_and_tensor_param_groups(
    param_groups: Iterable[dict[str, Any]], *, default_weight_decay: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dtensor_groups: list[dict[str, Any]] = []
    tensor_groups: list[dict[str, Any]] = []
    for group in param_groups:
        dtensor_params, tensor_params = split_dtensor_and_tensor_params(group["params"])
        metadata = {key: value for key, value in group.items() if key != "params"}
        metadata.setdefault("weight_decay", default_weight_decay)
        if dtensor_params:
            dtensor_groups.append({**metadata, "params": dtensor_params})
        if tensor_params:
            tensor_groups.append({**metadata, "params": tensor_params})
    return dtensor_groups, tensor_groups


def split_dtensor_and_tensor_params(
    params: Iterable[nn.Parameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    dtensor_params: list[nn.Parameter] = []
    tensor_params: list[nn.Parameter] = []
    for param in params:
        if is_dtensor_like(param):
            dtensor_params.append(param)
        else:
            tensor_params.append(param)
    return dtensor_params, tensor_params


def iter_torch_optimizers(optimizer: Any) -> Iterable[torch.optim.Optimizer]:
    if isinstance(optimizer, ChainedOptimizer):
        yield from optimizer.optimizers
    else:
        yield optimizer


def dtensor_from_local(
    local_tensor: torch.Tensor, device_mesh: Any, placements: Any
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return DTensor.from_local(local_tensor, device_mesh, placements)


__all__ = [
    "all_reduce_grad_",
    "build_adamw_optimizer",
    "copy_local_tensor_to_param_",
    "dtensor_from_local",
    "filter_supported_kwargs",
    "fsdp2_model_param_dtype",
    "get_bool_opt",
    "get_dtype_opt",
    "get_opt_value",
    "has_dtensor_grad_or_param",
    "is_dtensor_like",
    "iter_torch_optimizers",
    "local_grad_sq_sum",
    "normalize_param_groups",
    "to_local_tensor",
]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-process PyTorch runtime used to validate the Lite contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields as dataclass_fields
from typing import Any

import torch

from megatron.lite.model.registry import get_train_runtime_module, resolve_runtime_model_name
from megatron.lite.runtime.backends import Runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.data import ForwardResult, TrainBatch
from megatron.lite.runtime.contracts.handle import ModelHandle


def _as_backend_config(hf_path: str, cfg: MegatronLiteConfig | dict[str, Any] | None) -> MegatronLiteConfig:
    if cfg is None:
        return MegatronLiteConfig(hf_path=hf_path)
    if isinstance(cfg, MegatronLiteConfig):
        return cfg
    return MegatronLiteConfig.from_dict(hf_path, cfg)


def _dataclass_kwargs(cls, values: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in dataclass_fields(cls)}
    return {name: value for name, value in values.items() if name in names}


def _batch_to_mapping(data: Any) -> dict[str, Any]:
    if isinstance(data, TrainBatch):
        return {"inputs": data.inputs, "targets": data.targets, **data.metadata}
    if isinstance(data, dict):
        return data
    return {"inputs": data, "targets": None}


class MegatronLiteRuntime(Runtime):
    """A minimal local runtime for contract validation."""

    def __init__(self, hf_path: str = "", cfg: MegatronLiteConfig | dict[str, Any] | None = None):
        self._hf_path = hf_path
        self._cfg = _as_backend_config(hf_path, cfg)

    def build_model(
        self,
        hf_path: str | None = None,
        cfg: MegatronLiteConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> ModelHandle:
        rt_cfg = _as_backend_config(hf_path or self._hf_path, cfg) if cfg is not None else self._cfg
        if kwargs:
            rt_cfg.impl_cfg.update(kwargs)

        runtime_key = resolve_runtime_model_name(rt_cfg.model_name, rt_cfg.impl)
        protocol = get_train_runtime_module(runtime_key)
        model_cfg = protocol.build_model_config(rt_cfg.hf_path)

        impl_cfg = None
        if hasattr(protocol, "ImplConfig"):
            impl_cfg = protocol.ImplConfig(**_dataclass_kwargs(protocol.ImplConfig, rt_cfg.impl_cfg))

        bundle = protocol.build_model(model_cfg, impl_cfg=impl_cfg)
        if not bundle.chunks:
            raise ValueError("Model protocol returned an empty ModelBundle.")

        model = bundle.chunks[0].to(rt_cfg.device)
        optimizer = bundle.optimizer
        if optimizer is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=rt_cfg.optimizer.lr)

        return ModelHandle(
            model=model,
            optimizer=optimizer,
            lr_scheduler=None,
            config=rt_cfg,
            extras={
                "forward_step": bundle.forward_step,
                "model_cfg": model_cfg,
                "protocol": protocol,
                **bundle.extras,
            },
        )

    def train_mode(self, handle: ModelHandle) -> Any:
        return handle.model.train()

    def eval_mode(self, handle: ModelHandle) -> Any:
        return handle.model.eval()

    def forward_backward(
        self,
        handle: ModelHandle,
        data: Any,
        loss_fn: Callable[[Any, Any], tuple[torch.Tensor, dict[str, Any]]] | None = None,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
    ) -> ForwardResult:
        if num_microbatches != 1:
            raise ValueError("The PR1 local runtime supports exactly one microbatch.")

        batch = _batch_to_mapping(data)
        model = handle.model
        forward_step = handle.extras.get("forward_step")
        outputs = forward_step(model, batch) if forward_step is not None else model(batch["inputs"])

        if loss_fn is None:
            targets = batch.get("targets")
            if targets is None:
                loss = outputs.mean()
                metrics: dict[str, Any] = {"loss": float(loss.detach())}
            else:
                loss = torch.nn.functional.mse_loss(outputs, targets.to(outputs.device))
                metrics = {"loss": float(loss.detach())}
        else:
            loss, metrics = loss_fn(outputs, batch)

        if not forward_only:
            loss.backward()

        return ForwardResult(loss=loss, metrics=metrics, outputs=outputs)

    def zero_grad(self, handle: ModelHandle) -> None:
        handle.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self, handle: ModelHandle) -> tuple[bool, float, int | None]:
        grad_norm_sq = 0.0
        zero_grad_count = 0
        for parameter in handle.model.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            grad_norm_sq += float(grad.pow(2).sum())
            zero_grad_count += int(torch.count_nonzero(grad == 0).item())
        handle.optimizer.step()
        return True, grad_norm_sq**0.5, zero_grad_count

    def lr_scheduler_step(self, handle: ModelHandle) -> float | list[float]:
        scheduler = handle.lr_scheduler
        if scheduler is not None:
            scheduler.step()
        lrs = [float(group["lr"]) for group in handle.optimizer.param_groups]
        return lrs[0] if len(lrs) == 1 else lrs


def create(hf_path: str = "", cfg: MegatronLiteConfig | dict[str, Any] | None = None) -> MegatronLiteRuntime:
    """Factory used by ``megatron.lite.runtime.create_runtime``."""

    return MegatronLiteRuntime(hf_path, cfg)


__all__ = ["MegatronLiteRuntime", "create"]

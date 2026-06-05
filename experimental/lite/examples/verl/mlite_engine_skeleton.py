"""Minimal VERL external-engine skeleton for Megatron Lite.

This mirrors the config mapping used by ``verl-recipe/verlbb`` while using the
Megatron Lite public import path. It is intentionally small: a full VERL engine
should keep VERL's ``BaseEngine`` lifecycle and call these helpers during
``initialize()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from megatron.lite.model import resolve_model_type_from_hf
from megatron.lite.runtime import MegatronLiteConfig, RuntimeConfig, create_runtime
from megatron.lite.runtime.contracts import (
    OptimizerConfig as MegatronLiteOptimizerConfig,
)
from megatron.lite.runtime.contracts import ParallelConfig


@dataclass
class VerlMegatronLiteEngineConfig:
    """Subset of a VERL engine config consumed by Megatron Lite."""

    model_name: str = "auto"
    impl: str = "lite"

    tp: int = 1
    etp: int | None = None
    ep: int = 1
    pp: int = 1
    vpp: int = 1
    cp: int = 1

    attention_backend_override: str | None = "flash"
    router_aux_loss_coef: float | None = None
    impl_cfg: dict[str, Any] = field(default_factory=dict)

    forward_only: bool = False
    full_determinism: bool = False


def build_runtime_config(
    *,
    hf_config: Any,
    hf_path: str,
    engine_config: VerlMegatronLiteEngineConfig,
    optimizer_config: Any,
) -> RuntimeConfig:
    """Build a Megatron Lite runtime config from VERL-style config objects."""

    model_name = (
        engine_config.model_name
        if engine_config.model_name != "auto"
        else resolve_model_type_from_hf(hf_config)
    )
    impl_cfg = dict(engine_config.impl_cfg)
    impl_cfg.setdefault("use_thd", True)
    if engine_config.full_determinism:
        impl_cfg.setdefault("deterministic", True)
    if engine_config.forward_only:
        impl_cfg["optimizer"] = None

    backend_cfg = MegatronLiteConfig(
        model_name=model_name,
        impl=engine_config.impl,
        hf_path=hf_path,
        parallel=ParallelConfig(
            tp=engine_config.tp,
            etp=engine_config.etp or 1,
            ep=engine_config.ep,
            pp=engine_config.pp,
            vpp=engine_config.vpp,
            cp=engine_config.cp,
        ),
        optimizer=build_optimizer_config(optimizer_config),
        attention_backend_override=engine_config.attention_backend_override,
        router_aux_loss_coef=engine_config.router_aux_loss_coef,
        impl_cfg=impl_cfg,
    )
    return RuntimeConfig(
        backend="mlite",
        hf_path=hf_path,
        backend_cfg=backend_cfg,
    )


def build_optimizer_config(config: Any) -> MegatronLiteOptimizerConfig:
    """Translate a VERL optimizer config object to Megatron Lite."""

    override = getattr(config, "override_optimizer_config", {}) or {}
    betas = tuple(getattr(config, "betas", (0.9, 0.999)))
    min_lr = getattr(config, "min_lr", None)
    min_lr_ratio = getattr(config, "min_lr_ratio", None)
    if min_lr is None:
        min_lr = 0.0 if min_lr_ratio is None else config.lr * min_lr_ratio

    lr_decay_style = getattr(config, "lr_decay_style", None)
    if lr_decay_style is None:
        lr_decay_style = getattr(config, "lr_scheduler_type", "constant")

    return MegatronLiteOptimizerConfig(
        optimizer=normalize_optimizer_name(config),
        lr=config.lr,
        min_lr=min_lr,
        clip_grad=config.clip_grad,
        weight_decay=config.weight_decay,
        lr_warmup_steps_ratio=config.lr_warmup_steps_ratio,
        total_training_steps=config.total_training_steps,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_warmup_init=getattr(config, "lr_warmup_init", 0.0),
        lr_decay_steps=getattr(config, "lr_decay_steps", None),
        lr_decay_style=lr_decay_style,
        weight_decay_incr_style=getattr(config, "weight_decay_incr_style", "constant"),
        lr_wsd_decay_style=getattr(config, "lr_wsd_decay_style", "exponential"),
        lr_wsd_decay_steps=getattr(config, "lr_wsd_decay_steps", None),
        use_checkpoint_opt_param_scheduler=getattr(
            config,
            "use_checkpoint_opt_param_scheduler",
            False,
        ),
        adam_beta1=betas[0],
        adam_beta2=betas[1],
        adam_eps=override.get("adam_eps", override.get("eps")),
        offload_fraction=override.get(
            "offload_fraction",
            override.get("optimizer_offload_fraction"),
        ),
        use_precision_aware_optimizer=override.get("use_precision_aware_optimizer"),
        decoupled_weight_decay=override.get("decoupled_weight_decay"),
    )


def initialize_runtime(
    *,
    hf_config: Any,
    hf_path: str,
    engine_config: VerlMegatronLiteEngineConfig,
    optimizer_config: Any,
):
    """Create the runtime and build the Megatron Lite model handle."""

    runtime = create_runtime(
        build_runtime_config(
            hf_config=hf_config,
            hf_path=hf_path,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
        )
    )
    handle = runtime.build_model()
    return runtime, handle


def normalize_optimizer_name(config: Any) -> str:
    """Map common VERL optimizer spellings to Megatron Lite names."""

    name = getattr(config, "optimizer", "adam")
    if name in {"adamw", "fused_adam"}:
        return "adam"
    return name

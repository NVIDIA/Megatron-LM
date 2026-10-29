"""Qwen3.5 lite implementation for the Megatron Lite runtime."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.common import is_expert_param
from megatron.lite.model.qwen3_5.lite.checkpoint import (
    load_hf_weights as _load_hf_weights_impl,
)
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "mc"
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    use_thd: bool = False
    hf_path: str = ""
    attention_backend_override: str | None = None
    router_aux_loss_coef: float | None = None
    router_bias_rate: float = 0.0
    deterministic: bool = True
    optimizer_config: OptimizerConfig | None = None
    mtp_enable: bool = False
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool | None = None
    deterministic_embedding: bool | None = None


MODULE_MAP = {
    "core_attn": lambda layer: layer.self_attention.core_attention,
    "experts":   lambda layer: layer.mlp.experts,
    "moe":       lambda layer: layer.mlp,
    "router":    lambda layer: layer.mlp.router,
    "mlp_norm":  lambda layer: layer.pre_mlp_layernorm,
}


def build_model_config(source: str | Path | dict, **overrides) -> Qwen35Config:
    """Build Qwen3.5 architecture config from HF source."""
    if isinstance(source, dict):
        cfg = Qwen35Config._from_hf_dict(source)
    else:
        cfg = Qwen35Config.from_hf(str(source))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _forward_step(model: nn.Module, batch: dict) -> dict:
    kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
    }
    if "position_ids" in batch:
        kwargs["position_ids"] = batch["position_ids"]
    if "packed_seq_params" in batch:
        kwargs["packed_seq_params"] = batch["packed_seq_params"]
    for key in ("loss_mask", "temperature", "use_fused_kernels", "calculate_entropy"):
        if key in batch:
            kwargs[key] = batch[key]
    if kwargs["input_ids"].dim() == 1:
        kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
    return model(**kwargs)


def _make_aux_loss_hook():
    from megatron.core.transformer.moe.moe_utils import (  # pyright: ignore[reportMissingImports]
        MoEAuxLossAutoScaler as _MCMoEAuxLossAutoScaler,
    )

    from megatron.lite.primitive.modules.moe import (
        MoEAuxLossAutoScaler as _LiteMoEAuxLossAutoScaler,
    )
    from megatron.lite.model.qwen3_5.lite.model import MTPLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        _MCMoEAuxLossAutoScaler.set_loss_scale(scale)
        _LiteMoEAuxLossAutoScaler.set_loss_scale(scale)
        MTPLossAutoScaler.set_loss_scale(scale)

    return hook


def build_model(model_cfg: Qwen35Config, *, impl_cfg: ImplConfig) -> ModelBundle:
    from megatron.lite.model.qwen3_5.lite.model import Qwen35Model

    p = impl_cfg.parallel

    if impl_cfg.use_deepep and (p.etp is not None and p.etp > 1):
        raise ValueError("use_deepep and etp>1 are mutually exclusive")

    if impl_cfg.router_aux_loss_coef is not None:
        model_cfg.router_aux_loss_coef = impl_cfg.router_aux_loss_coef
    mtp_enable = bool(impl_cfg.mtp_enable)
    mtp_enable_train = mtp_enable and bool(impl_cfg.mtp_enable_train)
    if mtp_enable:
        if model_cfg.num_nextn_predict_layers <= 0:
            raise ValueError("mtp_enable=True but HF config has no num_nextn_predict_layers.")
        model_cfg.mtp_loss_scaling_factor = impl_cfg.mtp_loss_scaling_factor
        if impl_cfg.mtp_use_repeated_layer is not None:
            model_cfg.mtp_use_repeated_layer = impl_cfg.mtp_use_repeated_layer
    else:
        model_cfg.num_nextn_predict_layers = 0

    ps = init_parallel(p)
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    vpp = None if p.vpp == 1 else p.vpp
    deterministic = impl_cfg.deterministic
    if impl_cfg.use_thd and deterministic and "linear_attention" in model_cfg.layer_types:
        deterministic = False
    effective_impl_cfg = (
        impl_cfg if deterministic == impl_cfg.deterministic else replace(impl_cfg, deterministic=deterministic)
    )
    train_cfg = SimpleNamespace(
        tp=ps.tp_size,
        ep=ps.ep_size,
        etp=ps.etp_size,
        pp=ps.pp_size,
        cp=ps.cp_size,
        vpp=vpp,
        use_deepep=impl_cfg.use_deepep,
        fp8=False,
        recompute_modules=recompute_spec,
        deterministic=deterministic,
        deterministic_embedding=impl_cfg.deterministic_embedding,
    )
    model_kwargs: dict[str, Any] = dict(
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        hf_path=impl_cfg.hf_path,
        attention_backend_override=impl_cfg.attention_backend_override,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable_train,
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
    )

    if vpp is None:
        chunks = [
            Qwen35Model(model_cfg, train_cfg, ps, **model_kwargs)
            .to(torch.bfloat16)
            .cuda()
        ]
    else:
        chunks = [
            Qwen35Model(
                model_cfg,
                train_cfg,
                ps,
                vpp_chunk_id=i,
                **model_kwargs,
            )
            .to(torch.bfloat16)
            .cuda()
            for i in range(vpp)
        ]

    if recompute_spec:
        for chunk in chunks:
            apply_recompute(chunk.layers, recompute_spec, MODULE_MAP)

    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(chunk.layers, impl_cfg.offload, MODULE_MAP)

    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    optimizer_backend = "none"
    if impl_cfg.optimizer == "mc":
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_mc_training_optimizer,
        )

        optimizer, finalize_grads = build_mc_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=effective_impl_cfg,
            ps=ps,
            model_name="qwen3_5",
            is_expert=is_expert_param,
        )
        optimizer_backend = "mc"

        from megatron.lite.runtime.megatron_utils import (
            register_training_hooks,
        )

        register_training_hooks(chunks, optimizer)
    elif impl_cfg.optimizer is not None:
        raise ValueError(f"Unknown qwen3_5 lite optimizer: {impl_cfg.optimizer!r}.")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras={
            "model_cfg": model_cfg,
            "optimizer_backend": optimizer_backend,
            "post_model_load_hook": post_model_load_hook,
            "pre_forward_hook": _make_aux_loss_hook(),
        },
    )


def load_hf_weights(
    chunk: nn.Module, hf_path: str, model_cfg: Qwen35Config, ps: ParallelState
) -> None:
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(
    chunks: list[nn.Module],
    model_cfg: Qwen35Config,
    ps: ParallelState,
    **kwargs,
):
    from megatron.lite.model.qwen3_5.lite.checkpoint import export_weights as _export_weights

    limit = kwargs.pop("limit", None)
    if limit is not None and limit <= 0:
        limit = None
    hf_flat = _export_weights(chunks, model_cfg, ps, **kwargs)
    for idx, item in enumerate(hf_flat.items(), start=1):
        yield item
        if limit is not None and idx >= limit:
            break


def vocab_size(model_cfg) -> int | None:
    cfg = getattr(model_cfg, "text_config", model_cfg)
    return getattr(cfg, "vocab_size", None)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 lite impl — native model protocol for Megatron Lite runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from megatron.lite.model.protocol_utils import (
    add_cross_entropy_fusion,
    add_loss_context_kwargs,
    pack_thd_forward_kwargs,
    set_cross_entropy_fusion,
    unpack_thd_forward_output,
)
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.lite.checkpoint import EXPERT_CLASSIFIER, PLACEMENT_FN
from megatron.lite.model.qwen3_5.lite.checkpoint import export_hf_weights as _export_hf_weights_impl
from megatron.lite.model.qwen3_5.lite.checkpoint import load_hf_weights as _load_hf_weights_impl
from megatron.lite.model.qwen3_5.lite.checkpoint import save_hf_weights as _save_hf_weights_impl
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig
from megatron.lite.runtime.contracts.data import PackedBatch

__all__ = [
    "EXPERT_CLASSIFIER",
    "ImplConfig",
    "PLACEMENT_FN",
    "build_model",
    "build_model_config",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
    "vocab_size",
]


def is_expert_param(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


@dataclass(frozen=True)
class ImplConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "dist_opt"
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    use_thd: bool = False
    cross_entropy_fusion: bool = False
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
    mount_vision_model: bool = False
    gdn_cp_mode: str = "fla_allgather"


def _full_attn_module(layer, name: str):
    full_attn = getattr(layer, "full_attn", None)
    return getattr(full_attn, name, None) if full_attn is not None else None


MODULE_MAP = {
    "core_attn": lambda layer: _full_attn_module(layer, "core_attn"),
    "experts": lambda layer: layer.moe.experts,
    "moe": lambda layer: layer.moe,
    "router": lambda layer: layer.moe.router,
    "mlp_norm": lambda layer: layer.mlp_norm,
    "attn_proj": lambda layer: _full_attn_module(layer, "proj"),
    "linear_attn": lambda layer: layer.linear_attn,
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


def _forward_step(model: nn.Module, batch: PackedBatch) -> dict:
    kwargs = pack_thd_forward_kwargs(model, batch)
    add_loss_context_kwargs(kwargs)
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def _forward_step_bshd(model: nn.Module, batch: PackedBatch) -> dict:
    """Dense [b=1, s] forward for a single packed sequence (no THD packing).

    Used for deterministic parity comparison vs a dense Megatron-Core reference:
    the THD GatedDeltaNet kernel is non-deterministic, whereas the dense path is
    deterministic. CP=1 only (single unpadded sequence => dense == THD tokens).
    """
    input_ids = batch.input_ids.reshape(1, -1)
    labels = batch.labels.reshape(1, -1) if batch.labels is not None else None
    kwargs: dict[str, Any] = {"input_ids": input_ids, "labels": labels, "packed_seq_params": None}
    add_cross_entropy_fusion(kwargs, model)
    return model(**kwargs)


def unpack_forward_output(model: nn.Module, batch: PackedBatch, output) -> Any:
    return unpack_thd_forward_output(model, batch, output)


def _make_aux_loss_hook():
    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    def hook(scale: torch.Tensor) -> None:
        MoEAuxLossAutoScaler.set_loss_scale(scale)
        MTPLossAutoScaler.set_loss_scale(scale)

    return hook


def _build_dist_opt_optimizer(
    chunks, model_cfg: Qwen35Config, impl_cfg: ImplConfig, ps: ParallelState
):
    from megatron.lite.primitive.optimizers.megatron_wrap import (
        build_dist_opt_training_optimizer,
    )

    return build_dist_opt_training_optimizer(
        chunks,
        model_cfg=model_cfg,
        impl_cfg=impl_cfg,
        ps=ps,
        is_expert=is_expert_param,
        model_name="qwen3_5",
        deterministic=impl_cfg.deterministic,
    )


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
    )
    model_kwargs: dict[str, Any] = dict(
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        hf_path=impl_cfg.hf_path,
        attention_backend_override=impl_cfg.attention_backend_override,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable_train,
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
        mount_vision_model=impl_cfg.mount_vision_model,
        gdn_cp_mode=impl_cfg.gdn_cp_mode,
    )

    if vpp is None:
        chunks = [Qwen35Model(model_cfg, train_cfg, ps, **model_kwargs).to(torch.bfloat16).cuda()]
    else:
        chunks = [
            Qwen35Model(model_cfg, train_cfg, ps, vpp_chunk_id=i, **model_kwargs)
            .to(torch.bfloat16)
            .cuda()
            for i in range(vpp)
        ]
    set_cross_entropy_fusion(chunks, impl_cfg.cross_entropy_fusion)

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
    if impl_cfg.optimizer == "dist_opt":
        optimizer, finalize_grads = _build_dist_opt_optimizer(chunks, model_cfg, impl_cfg, ps)
        from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
        from megatron.lite.runtime.megatron_utils import register_training_hooks

        attach_model_sharded_state_dict(
            chunks, ps, get_placements=PLACEMENT_FN, is_expert=is_expert_param
        )
        register_training_hooks(chunks, optimizer)
        optimizer_backend = "dist_opt"
    elif impl_cfg.optimizer == "fsdp2":
        optimizer_backend = "fsdp2"

        def _post_model_load_hook():
            from megatron.lite.model.qwen3_5.lite.model import Qwen35Layer
            from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(Qwen35Layer,),
                    expert_classifier=is_expert_param,
                    deterministic=deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    leaf_module_names=(),
                )
            }

        post_model_load_hook = _post_model_load_hook
    elif impl_cfg.optimizer is not None:
        raise ValueError(f"Unknown qwen3_5 lite optimizer: {impl_cfg.optimizer!r}.")

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step if impl_cfg.use_thd else _forward_step_bshd,
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
    chunks: list[nn.Module], model_cfg: Qwen35Config, ps: ParallelState, **kwargs
):
    yield from _export_hf_weights_impl(chunks, model_cfg, ps, **kwargs)


def save_hf_weights(
    chunks: list[nn.Module], path: str, model_cfg: Qwen35Config, ps: ParallelState
) -> None:
    _save_hf_weights_impl(chunks, path, model_cfg, ps)


def vocab_size(model_cfg) -> int | None:
    cfg = getattr(model_cfg, "text_config", model_cfg)
    return getattr(cfg, "vocab_size", None)

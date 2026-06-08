"""Qwen3MoE lite impl — model protocol for Megatron Lite runtime.

This file is the reference implementation of the Megatron Lite model protocol.
New model authors: copy this file and adapt.

Protocol convention (what runtime calls):
  Required:
    ImplConfig                                      — @dataclass, per-impl knobs
    build_model_config(source, **overrides)          → ModelConfig
    build_model(model_cfg, *, impl_cfg)              → ModelBundle
  Optional (in ModelBundle.extras or module-level):
    load_hf_weights(chunk, hf_path, model_cfg, ps)  — HF weight loading
    export_hf_weights(chunks, model_cfg, ps)         — HF weight export
    vocab_size(model_cfg) -> int                     — benchmark metadata
  Escape hatch:
    create_runtime(hf_path, cfg) -> Runtime          — fully override runtime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.model.qwen3_moe.common import is_expert_param
from megatron.lite.model.qwen3_moe.lite.checkpoint import (
    EXPERT_CLASSIFIER,
    PLACEMENT_FN,
    load_hf_weights as _load_hf_weights_impl,
)
from megatron.lite.model.qwen3_moe.lite.model import MTPLossAutoScaler, Qwen3MoEModel
from megatron.lite.primitive.bundle import ModelBundle
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    freeze_non_lora_params,
    normalize_lora_config,
    trainable_param_stats,
)
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.primitive.recompute import apply_recompute, parse_recompute_spec
from megatron.lite.runtime.contracts import OptimizerConfig, ParallelConfig


# ---------------------------------------------------------------------------
# ImplConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImplConfig:
    """Lite impl knobs. Constructed by runtime from user config."""

    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: str | None = "mc"  # None = no optimizer (inference)
    recompute: list[str] = field(default_factory=list)
    offload: list[str] = field(default_factory=list)
    use_deepep: bool = False
    use_thd: bool = False
    router_aux_loss_coef: float | None = None
    router_bias_rate: float = 0.0
    # User-level OptimizerConfig threaded through the runtime.
    optimizer_config: OptimizerConfig | None = None
    mtp_enable: bool = False
    mtp_enable_train: bool = False
    mtp_detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool | None = None
    deterministic: bool = True
    lora: LoraConfig | dict | None = None


# ---------------------------------------------------------------------------
# Module map for recompute/offload
# ---------------------------------------------------------------------------

MODULE_MAP = {
    "core_attn": lambda layer: layer.attn.core_attn,
    "experts": lambda layer: layer.moe.experts,
    "moe": lambda layer: layer.moe,
    "router": lambda layer: layer.moe.router,
    "mlp_norm": lambda layer: layer.mlp_norm,
    "attn_proj": lambda layer: layer.attn.proj,
}


# ---------------------------------------------------------------------------
# Required: build_model_config
# ---------------------------------------------------------------------------


def build_model_config(source: str | Path | dict, **overrides) -> Qwen3MoEConfig:
    """Build Qwen3MoE architecture config from HF source."""
    if isinstance(source, dict):
        cfg = Qwen3MoEConfig._from_hf_dict(source)
    else:
        cfg = Qwen3MoEConfig.from_hf(str(source))
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Required: build_model
# ---------------------------------------------------------------------------


def _forward_step(model: nn.Module, batch: dict) -> dict:
    kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}
    if "packed_seq_params" in batch:
        kwargs["packed_seq_params"] = batch["packed_seq_params"]
    if "position_ids" in batch:
        kwargs["position_ids"] = batch["position_ids"]
    for key in (
        "loss_mask",
        "temperature",
        "use_fused_kernels",
        "calculate_entropy",
        "return_log_probs",
    ):
        if key in batch:
            kwargs[key] = batch[key]
    if kwargs["input_ids"].dim() == 1:
        kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
    return model(**kwargs)


def build_model(model_cfg: Qwen3MoEConfig, *, impl_cfg: ImplConfig) -> ModelBundle:
    """Build lite Qwen3MoE: model, parallel state, optimizer — everything.

    Model owns all construction. Runtime just consumes the ModelBundle.
    """
    p = impl_cfg.parallel
    lora_config = normalize_lora_config(impl_cfg.lora)

    # ── validation ──
    if impl_cfg.use_deepep and (p.etp is not None and p.etp > 1):
        raise ValueError("use_deepep and etp>1 are mutually exclusive")

    # ── override model config from impl_cfg ──
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

    # ── parallel state (model creates its own) ──
    ps = init_parallel(p)
    deterministic = impl_cfg.deterministic

    # ── build chunks ──
    recompute_spec = parse_recompute_spec(impl_cfg.recompute)
    model_kwargs: dict[str, Any] = dict(
        use_deepep=impl_cfg.use_deepep,
        fp8=False,
        recompute_modules=recompute_spec,
        router_bias_rate=impl_cfg.router_bias_rate,
        use_thd=impl_cfg.use_thd,
        mtp_enable=mtp_enable,
        mtp_enable_train=mtp_enable_train,
        mtp_detach_encoder=impl_cfg.mtp_detach_encoder,
        lora_config=lora_config,
    )

    vpp = None if p.vpp == 1 else p.vpp
    if vpp is None:
        chunks = [Qwen3MoEModel(model_cfg, ps, **model_kwargs).to(torch.bfloat16).cuda()]
    else:
        chunks = []
        for i in range(vpp):
            chunks.append(
                Qwen3MoEModel(model_cfg, ps, vpp=vpp, vpp_chunk_id=i, **model_kwargs)
                .to(torch.bfloat16).cuda()
            )

    # ── recompute ──
    if recompute_spec:
        for chunk in chunks:
            apply_recompute(chunk.layers, recompute_spec, MODULE_MAP)

    # ── offload ──
    if impl_cfg.offload:
        from megatron.lite.primitive.recompute import apply_offload

        for chunk in chunks:
            apply_offload(chunk.layers, impl_cfg.offload, MODULE_MAP)

    lora_stats = None
    if lora_config.enabled:
        lora_stats = {"chunks": []}
        for chunk in chunks:
            freeze_stats = freeze_non_lora_params(chunk)
            trainable_stats = trainable_param_stats(chunk)
            lora_stats["chunks"].append({**freeze_stats, **trainable_stats})

    # ── optimizer (model chooses which primitive) ──
    optimizer = None
    finalize_grads = None
    post_model_load_hook = None
    if impl_cfg.optimizer == "mc":
        from megatron.lite.primitive.optimizers.megatron_wrap import (
            build_mc_training_optimizer,
        )

        optimizer, finalize_grads = build_mc_training_optimizer(
            chunks,
            model_cfg=model_cfg,
            impl_cfg=impl_cfg,
            ps=ps,
            model_name="qwen3_moe",
            is_expert=is_expert_param,
            deterministic=deterministic,
        )
        optimizer_backend = "mc"
    elif impl_cfg.optimizer == "fsdp2":
        optimizer_backend = "fsdp2"

        def _post_model_load_hook():
            from megatron.lite.model.qwen3_moe.lite.model import TransformerLayer
            from megatron.lite.primitive.optimizers.fsdp2 import (
                build_fsdp2_training_optimizer,
            )

            return {
                "optimizer": build_fsdp2_training_optimizer(
                    chunks,
                    impl_cfg.optimizer_config,
                    ps,
                    unit_modules=(TransformerLayer,),
                    expert_classifier=is_expert_param,
                    deterministic=deterministic,
                    vpp=impl_cfg.parallel.vpp,
                    # Non-layer params stay under the root FSDP2 unit. The fused
                    # CE path reads head.col.linear.weight directly, and the
                    # embedding path is also driven from model.forward().
                    leaf_module_names=(),
                ),
            }

        post_model_load_hook = _post_model_load_hook
    elif impl_cfg.optimizer is None:
        optimizer_backend = "none"
    else:
        raise ValueError(f"Unknown qwen3_moe lite optimizer: {impl_cfg.optimizer!r}.")

    from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler

    def _pre_forward_hook(loss_scale):
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        MTPLossAutoScaler.set_loss_scale(loss_scale)

    return ModelBundle(
        chunks=chunks,
        parallel_state=ps,
        optimizer=optimizer,
        finalize_grads=finalize_grads,
        forward_step=_forward_step,
        extras={
            "model_cfg": model_cfg,
            # Lite's router uses megatron.lite's MoEAuxLossAutoScaler; hand the
            # classmethod directly as the per-microbatch hook.
            "pre_forward_hook": _pre_forward_hook,
            "optimizer_backend": optimizer_backend,
            "post_model_load_hook": post_model_load_hook,
            "lora_config": lora_config,
            "lora_stats": lora_stats,
        },
    )


# ---------------------------------------------------------------------------
# Optional: load_hf_weights
# ---------------------------------------------------------------------------


def load_hf_weights(
    chunk: nn.Module,
    hf_path: str,
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
) -> None:
    """Load HF pretrained weights into model chunk."""
    if not hf_path:
        return
    _load_hf_weights_impl(chunk, hf_path, model_cfg, ps)


def export_hf_weights(
    chunks: list[nn.Module],
    model_cfg: Qwen3MoEConfig,
    ps: ParallelState,
    **kwargs,
):
    """Export HF weights from model chunks."""
    from megatron.lite.model.qwen3_moe.lite.checkpoint import export_hf_weights as _export

    for chunk in chunks:
        yield from _export(chunk, model_cfg, ps, **kwargs)


# ---------------------------------------------------------------------------
# Tooling metadata (benchmark / debug)
# ---------------------------------------------------------------------------


def vocab_size(model_cfg: Qwen3MoEConfig) -> int | None:
    return getattr(model_cfg, "vocab_size", None)

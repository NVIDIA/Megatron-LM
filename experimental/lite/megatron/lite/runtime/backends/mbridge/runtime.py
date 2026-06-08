"""Runtime backend backed by the legacy ``mbridge`` package."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.runtime.backends.bridge.config import BridgeConfig
from megatron.lite.runtime.backends.bridge.runtime import (
    BridgeRuntime,
    _MpuParallelState,
    _build_lr_scheduler,
    _build_optimizer,
    _lower_transformer_overrides,
)
from megatron.lite.runtime.contracts.handle import ModelHandle
from megatron.lite.runtime.megatron_utils import (
    load_model_to_gpu,
    offload_model_to_cpu,
    offload_optimizer,
    register_training_hooks,
)

logger = logging.getLogger(__name__)


def _build_mbridge(hf_path: str, cfg: BridgeConfig):
    """Build the legacy mbridge AutoBridge lazily from an HF model path."""
    from mbridge import AutoBridge
    from megatron.lite.primitive.deterministic import deterministic_requested

    bridge = AutoBridge.from_pretrained(hf_path, trust_remote_code=True)
    bridge.set_extra_args(sequence_parallel=cfg.parallel.tp > 1)

    transformer_overrides = _lower_transformer_overrides(cfg)
    if transformer_overrides:
        bridge.set_extra_args(**transformer_overrides)

    if not hasattr(bridge.hf_config, "rope_theta"):
        bridge.hf_config.rope_theta = bridge.hf_config.to_dict().get("rope_theta", 1000000.0)

    bridge.set_extra_args(bf16=True, fp16=False)

    if cfg.model_name == "qwen3_5":
        bridge.set_extra_args(
            deterministic_mode=deterministic_requested(),
            fused_single_qkv_rope=False,
        )

    if callable(cfg.bridge_post_init):
        cfg.bridge_post_init(bridge)

    return bridge


def _resolve_mbridge_benchmark_protocol(cfg: BridgeConfig, bridge) -> Any | None:
    """Best-effort protocol lookup for model stats used by bench examples."""
    from megatron.lite.model.registry import get_train_runtime_module, resolve_model_type_from_hf

    model_name = cfg.model_name
    if model_name == "auto":
        try:
            model_name = resolve_model_type_from_hf(bridge.hf_config)
        except ValueError:
            return None

    try:
        return get_train_runtime_module(model_name)
    except ValueError:
        return None


class MBridgeRuntime(BridgeRuntime):
    """mbridge training backend using Megatron-Core optimizer state."""

    def build_model(
        self,
        hf_path: str | None = None,
        cfg: BridgeConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> ModelHandle:
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.enums import ModelType

        if cfg is None:
            rt_cfg = self._cfg
        elif isinstance(cfg, BridgeConfig):
            rt_cfg = cfg
        else:
            rt_cfg = BridgeConfig.from_dict(cfg)
        hf_path = hf_path or self._hf_path

        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(minutes=10))
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

        p = rt_cfg.parallel
        init_kwargs = dict(
            tensor_model_parallel_size=p.tp,
            pipeline_model_parallel_size=p.pp,
            expert_model_parallel_size=p.ep,
            context_parallel_size=p.cp,
        )
        if p.etp is not None:
            init_kwargs["expert_tensor_parallel_size"] = p.etp
        if p.vpp > 1:
            init_kwargs["virtual_pipeline_model_parallel_size"] = p.vpp

        if not mpu.model_parallel_is_initialized():
            mpu.initialize_model_parallel(**init_kwargs)
        model_parallel_cuda_manual_seed(rt_cfg.seed)

        bridge = _build_mbridge(hf_path, rt_cfg)

        ddp_config = {
            "use_distributed_optimizer": True,
            "overlap_grad_reduce": False,
            "grad_reduce_in_fp32": True,
        }
        ddp_config.update(rt_cfg.override_ddp_config)

        model_list = bridge.get_model(
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            ddp_config=ddp_config,
        )
        if rt_cfg.load_hf_weights:
            bridge.load_weights(model_list, hf_path, memory_efficient=True)

        optimizer = _build_optimizer(model_list, rt_cfg) if rt_cfg.build_optimizer else None
        lr_scheduler = _build_lr_scheduler(optimizer, rt_cfg) if optimizer is not None else None
        register_training_hooks(model_list, optimizer)

        if self._offload_param:
            offload_model_to_cpu(model_list)
        if self._offload_optimizer and optimizer is not None:
            offload_optimizer(optimizer)

        logger.info(
            "MBridgeRuntime: model built, tp=%d ep=%d pp=%d cp=%d",
            p.tp,
            p.ep,
            p.pp,
            p.cp,
        )

        return ModelHandle(
            model=model_list[0],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallel_state=_MpuParallelState(mpu),
            config=rt_cfg,
            _extras={
                "bridge": bridge,
                "model_list": model_list,
                "mpu": mpu,
                "model_cfg": bridge.hf_config,
                "protocol": _resolve_mbridge_benchmark_protocol(rt_cfg, bridge),
                "optimizer_backend": "distopt" if optimizer is not None else "none",
                "world_size": dist.get_world_size(),
            },
        )

    def export_weights(self, handle: ModelHandle, **kwargs) -> Iterator[tuple[str, torch.Tensor]]:
        bridge = handle._extras["bridge"]
        model_list = handle._extras["model_list"]
        load_model_to_gpu(model_list, load_grad=False)
        return bridge.export_weights(model_list)


__all__ = ["MBridgeRuntime"]

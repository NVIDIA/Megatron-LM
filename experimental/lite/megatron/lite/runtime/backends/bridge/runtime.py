"""Runtime backend backed by Megatron-Bridge."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.primitive.optimizers.megatron_wrap import build_mc_optimizer_config
from megatron.lite.runtime.backends import Runtime as RuntimeBase
from megatron.lite.runtime.backends.bridge.config import BridgeConfig
from megatron.lite.runtime.contracts.data import Batch, ForwardResult, ModelOutputs
from megatron.lite.runtime.contracts.handle import ModelHandle
from megatron.lite.runtime.megatron_utils import (
    build_sharded_state_dict,
    is_mp_src_rank_with_outputs,
    load_model_to_gpu,
    load_optimizer,
    offload_model_to_cpu,
    offload_optimizer,
    register_training_hooks,
)

logger = logging.getLogger(__name__)


class _MpuParallelState:
    """Adapter exposing Megatron-Core mpu through ``ModelHandle`` properties."""

    def __init__(self, mpu):
        self._mpu = mpu

    @property
    def dp_rank(self) -> int:
        return self._mpu.get_data_parallel_rank()

    @property
    def dp_size(self) -> int:
        return self._mpu.get_data_parallel_world_size()

    @property
    def dp_group(self):
        return self._mpu.get_data_parallel_group()

    @property
    def tp_rank(self) -> int:
        return self._mpu.get_tensor_model_parallel_rank()

    @property
    def tp_size(self) -> int:
        return self._mpu.get_tensor_model_parallel_world_size()

    @property
    def pp_rank(self) -> int:
        return self._mpu.get_pipeline_model_parallel_rank()

    @property
    def pp_size(self) -> int:
        return self._mpu.get_pipeline_model_parallel_world_size()

    @property
    def cp_rank(self) -> int:
        return self._mpu.get_context_parallel_rank()

    @property
    def cp_size(self) -> int:
        return self._mpu.get_context_parallel_world_size()

    @property
    def cp_group(self):
        return self._mpu.get_context_parallel_group()


def _lower_transformer_overrides(cfg: BridgeConfig) -> dict[str, Any]:
    overrides = {"attention_backend": "flash"}
    overrides.update(cfg.override_transformer_config)
    return overrides


def _bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is None:
        return None
    return getattr(hf_pretrained, "config", hf_pretrained)


def _lower_provider_value(key: str, value: Any) -> Any:
    if key == "attention_backend" and isinstance(value, str):
        from megatron.core.transformer.enums import AttnBackend

        return AttnBackend[value]
    return value


def _configure_provider(provider, cfg: BridgeConfig) -> None:
    p = cfg.parallel
    provider.tensor_model_parallel_size = p.tp
    provider.pipeline_model_parallel_size = p.pp
    provider.context_parallel_size = p.cp
    provider.expert_model_parallel_size = p.ep
    if p.etp is not None:
        provider.expert_tensor_parallel_size = p.etp
    if p.vpp > 1:
        provider.virtual_pipeline_model_parallel_size = p.vpp
    provider.sequence_parallel = p.tp > 1
    provider.bf16 = True
    provider.fp16 = False

    for key, value in _lower_transformer_overrides(cfg).items():
        setattr(provider, key, _lower_provider_value(key, value))


def _register_bridge_compat_aliases() -> None:
    """Register local Megatron-Bridge aliases for supported checkpoint variants."""
    from megatron.bridge.models.conversion import model_bridge
    from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
    from megatron.core.models.gpt.gpt_model import GPTModel

    registry = getattr(model_bridge.get_model_bridge, "_exact_types", {})
    for source in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"):
        if source not in registry:
            model_bridge.register_bridge_implementation(
                source=source,
                target=GPTModel,
                bridge_class=Qwen3MoEBridge,
            )


def _build_bridge(hf_path: str, cfg: BridgeConfig):
    """Build Megatron-Bridge AutoBridge lazily from an HF model path."""
    from megatron.bridge import AutoBridge
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    _register_bridge_compat_aliases()
    bridge = AutoBridge.from_hf_config(hf_config)
    hf_config = _bridge_hf_config(bridge)
    if hf_config is not None and not hasattr(hf_config, "rope_theta"):
        hf_config.rope_theta = hf_config.to_dict().get("rope_theta", 1000000.0)

    if callable(cfg.bridge_post_init):
        cfg.bridge_post_init(bridge)

    return bridge


def _build_optimizer(model_list: list, cfg: BridgeConfig):
    from megatron.core.optimizer import get_megatron_optimizer

    return get_megatron_optimizer(
        config=build_mc_optimizer_config(
            cfg.optimizer,
            override_optimizer_config=cfg.override_optimizer_config,
        ),
        model_chunks=model_list,
    )


def _build_lr_scheduler(optimizer, cfg: BridgeConfig):
    opt = cfg.optimizer
    total_steps = opt.total_training_steps
    if total_steps <= 0:
        return None

    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    warmup_steps = opt.lr_warmup_steps
    if warmup_steps <= 0 and opt.lr_warmup_steps_ratio > 0:
        warmup_steps = int(opt.lr_warmup_steps_ratio * total_steps)
    warmup_steps = max(warmup_steps, 0)
    decay_steps = opt.lr_decay_steps if opt.lr_decay_steps is not None else total_steps

    return OptimizerParamScheduler(
        optimizer,
        init_lr=opt.lr_warmup_init,
        max_lr=opt.lr,
        min_lr=opt.min_lr,
        lr_warmup_steps=warmup_steps,
        lr_decay_steps=decay_steps,
        lr_decay_style=opt.lr_decay_style,
        start_wd=opt.weight_decay,
        end_wd=opt.weight_decay,
        wd_incr_steps=total_steps,
        wd_incr_style=opt.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=opt.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=not opt.use_checkpoint_opt_param_scheduler,
        wsd_decay_steps=opt.lr_wsd_decay_steps,
        lr_wsd_decay_style=opt.lr_wsd_decay_style,
    )


def _resolve_benchmark_protocol(cfg: BridgeConfig, bridge) -> Any | None:
    """Best-effort protocol lookup for model stats used by bench examples."""
    from megatron.lite.model.registry import get_train_runtime_module, resolve_model_type_from_hf

    model_name = cfg.model_name
    if model_name == "auto":
        try:
            model_name = resolve_model_type_from_hf(_bridge_hf_config(bridge))
        except ValueError:
            return None

    try:
        return get_train_runtime_module(model_name)
    except ValueError:
        return None


def _as_data_iter(data: Any):
    if hasattr(data, "__next__"):
        return data
    if isinstance(data, list):
        return iter(data)
    return iter([data])


class BridgeRuntime(RuntimeBase):
    """Megatron-Bridge training backend using Megatron-Core optimizer state."""

    def __init__(self, hf_path: str, cfg: BridgeConfig | dict[str, Any]):
        self._hf_path = hf_path
        self._cfg = cfg if isinstance(cfg, BridgeConfig) else BridgeConfig.from_dict(cfg)
        self._offload_param = self._cfg.param_offload
        self._offload_optimizer = self._cfg.optimizer_offload

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

        bridge = _build_bridge(hf_path, rt_cfg)
        provider = bridge.to_megatron_provider(
            load_weights=rt_cfg.load_hf_weights,
            hf_path=hf_path if rt_cfg.load_hf_weights else None,
        )
        _configure_provider(provider, rt_cfg)
        if hasattr(provider, "finalize"):
            provider.finalize()

        ddp_config = {
            "use_distributed_optimizer": True,
            "overlap_grad_reduce": False,
            "grad_reduce_in_fp32": True,
        }
        ddp_config.update(rt_cfg.override_ddp_config)

        model_list = provider.provide_distributed_model(
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            ddp_config=ddp_config,
            bf16=True,
        )

        optimizer = _build_optimizer(model_list, rt_cfg) if rt_cfg.build_optimizer else None
        lr_scheduler = _build_lr_scheduler(optimizer, rt_cfg) if optimizer is not None else None
        register_training_hooks(model_list, optimizer)

        if self._offload_param:
            offload_model_to_cpu(model_list)
        if self._offload_optimizer and optimizer is not None:
            offload_optimizer(optimizer)

        logger.info(
            "BridgeRuntime: model built, tp=%d ep=%d pp=%d cp=%d",
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
                "provider": provider,
                "model_list": model_list,
                "mpu": mpu,
                "model_cfg": _bridge_hf_config(bridge),
                "protocol": _resolve_benchmark_protocol(rt_cfg, bridge),
                "optimizer_backend": "mc" if optimizer is not None else "none",
                "world_size": dist.get_world_size(),
            },
        )

    def forward_backward(
        self,
        handle: ModelHandle,
        data: Any,
        loss_fn,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
    ) -> ForwardResult:
        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        if num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")

        model_list = handle._extras["model_list"]
        data_iter = _as_data_iter(data)
        last_loss: list[float | None] = [None]

        def _fwd_step(data_iterator, model):
            sample = next(data_iterator)
            if isinstance(sample, Batch):
                sample = {
                    "input_ids": sample["input_ids"],
                    "labels": sample["labels"],
                    "position_ids": getattr(sample, "position_ids", None),
                }
            if not isinstance(sample, dict):
                raise TypeError(
                    f"BridgeRuntime expected dict or Batch data, got {type(sample).__name__}."
                )

            output_tensor = model(
                input_ids=sample["input_ids"],
                position_ids=sample.get("position_ids"),
                attention_mask=sample.get("attention_mask"),
                labels=sample["labels"],
                packed_seq_params=sample.get("packed_seq_params"),
            )
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]

            def _mc_loss_fn(output_tensor, non_loss_data=False):
                if loss_fn is not None:
                    loss, _metrics = loss_fn({"output_tensor": output_tensor}, sample)
                else:
                    loss = output_tensor.mean()
                last_loss[0] = float(loss.detach().item())
                return loss, {}

            return output_tensor, _mc_loss_fn

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            batches = [next(data_iter) for _ in range(num_microbatches)]
            batch_generator = [iter(batches) for _ in range(vpp_size)]
        else:
            batch_generator = data_iter

        get_forward_backward_func()(
            forward_step_func=_fwd_step,
            data_iterator=batch_generator,
            model=model_list,
            num_microbatches=num_microbatches,
            forward_only=forward_only,
            seq_length=1,
            micro_batch_size=1,
        )

        if not forward_only:
            from megatron.core.distributed.finalize_model_grads import finalize_model_grads

            finalize_model_grads(model_list)

        loss_val = last_loss[0]
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            loss_t = torch.tensor([loss_val or 0.0], device="cuda")
            dist.broadcast(loss_t, src=mpu.get_pipeline_model_parallel_last_rank())
            loss_val = float(loss_t.item())

        result_loss = torch.tensor(loss_val or 0.0)
        return ForwardResult(
            model_output=ModelOutputs(loss=result_loss),
            metrics={"loss": loss_val if loss_val is not None else 0.0},
        )

    def zero_grad(self, handle: ModelHandle) -> None:
        if handle._optimizer is not None:
            handle._optimizer.zero_grad()
        for model in handle._extras["model_list"]:
            if handle._optimizer is None:
                model.zero_grad(set_to_none=True)
            if hasattr(model, "zero_grad_buffer"):
                model.zero_grad_buffer()

    def optimizer_step(self, handle: ModelHandle) -> tuple[bool, float, int | None]:
        if handle._optimizer is None:
            return True, 0.0, 0
        update_successful, grad_norm, num_zeros = handle._optimizer.step()
        return update_successful, float(grad_norm), num_zeros

    def lr_scheduler_step(self, handle: ModelHandle) -> float | list[float]:
        if handle._lr_scheduler is not None:
            handle._lr_scheduler.step(1)
            return handle._optimizer.param_groups[0]["lr"]
        return 0.0

    def is_mp_src_rank_with_outputs(self, handle: ModelHandle) -> bool:
        return is_mp_src_rank_with_outputs()

    def to(
        self,
        handle: ModelHandle,
        device: str,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        model_list = handle._extras["model_list"]
        opt = handle._optimizer
        if device == "cuda":
            if model:
                load_model_to_gpu(model_list, load_grad=grad)
            if optimizer and opt is not None:
                load_optimizer(opt)
        elif device == "cpu":
            if model:
                offload_model_to_cpu(model_list)
            if optimizer and opt is not None:
                offload_optimizer(opt)
        else:
            raise ValueError(f"BridgeRuntime.to supports only 'cpu' or 'cuda', got {device!r}.")

    def train_mode(self, handle: ModelHandle):
        return _BridgeTrainCtx(self, handle)

    def eval_mode(self, handle: ModelHandle):
        return _BridgeEvalCtx(self, handle)

    def export_weights(self, handle: ModelHandle, **kwargs) -> Iterator[tuple[str, torch.Tensor]]:
        bridge = handle._extras["bridge"]
        model_list = handle._extras["model_list"]
        load_model_to_gpu(model_list, load_grad=False)
        return bridge.export_hf_weights(model_list, cpu=bool(kwargs.get("cpu", False)))

    def save_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> None:
        model_list = handle._extras["model_list"]
        opt = handle._optimizer
        load_model_to_gpu(model_list, load_grad=True)

        from megatron.core import dist_checkpointing

        state = build_sharded_state_dict(model_list, opt, handle._lr_scheduler)
        os.makedirs(path, exist_ok=True)
        dist_checkpointing.save(state, path)
        dist.barrier()

        if self._offload_param:
            offload_model_to_cpu(model_list)

    def load_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> None:
        model_list = handle._extras["model_list"]
        opt = handle._optimizer
        load_model_to_gpu(model_list, load_grad=True)

        from megatron.core import dist_checkpointing

        state = build_sharded_state_dict(model_list, opt, handle._lr_scheduler)
        dist_checkpointing.load(state, path)

        if self._offload_param:
            offload_model_to_cpu(model_list)
        if self._offload_optimizer and opt is not None:
            offload_optimizer(opt)


class _BridgeTrainCtx:
    def __init__(self, runtime: BridgeRuntime, handle: ModelHandle):
        self._runtime = runtime
        self._handle = handle

    def __enter__(self):
        if self._runtime._offload_param or self._runtime._offload_optimizer:
            self._runtime.to(
                self._handle,
                "cuda",
                model=self._runtime._offload_param,
                optimizer=self._runtime._offload_optimizer,
                grad=self._runtime._offload_param,
            )
        for model in self._handle._extras["model_list"]:
            model.train()
        return self

    def __exit__(self, *exc):
        self._runtime.zero_grad(self._handle)
        if self._runtime._offload_param or self._runtime._offload_optimizer:
            self._runtime.to(
                self._handle,
                "cpu",
                model=self._runtime._offload_param,
                optimizer=self._runtime._offload_optimizer,
                grad=self._runtime._offload_param,
            )
        return False


class _BridgeEvalCtx:
    def __init__(self, runtime: BridgeRuntime, handle: ModelHandle):
        self._runtime = runtime
        self._handle = handle
        self._prev_grad = torch.is_grad_enabled()

    def __enter__(self):
        if self._runtime._offload_param:
            self._runtime.to(self._handle, "cuda", model=True, optimizer=False, grad=False)
        for model in self._handle._extras["model_list"]:
            model.eval()
        torch.set_grad_enabled(False)
        return self

    def __exit__(self, *exc):
        torch.set_grad_enabled(self._prev_grad)
        if self._runtime._offload_param:
            self._runtime.to(self._handle, "cpu", model=True, optimizer=False, grad=False)
        return False


__all__ = ["BridgeRuntime"]

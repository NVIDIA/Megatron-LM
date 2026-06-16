# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MegatronLiteRuntime — Megatron Lite's default training backend."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import fields as dc_fields
from datetime import timedelta
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.runtime.backends import Runtime as RuntimeBase
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.contracts.data import ForwardResult, ModelOutputs
from megatron.lite.runtime.contracts.handle import ModelHandle


def _build_impl_cfg(proto, rt_cfg: MegatronLiteConfig):
    """Construct typed impl config, backfilling hf_path + optimizer_config."""
    impl_cfg_kwargs = {**rt_cfg.impl_cfg, "parallel": rt_cfg.parallel}
    init_fields = {f.name for f in dc_fields(proto.ImplConfig) if f.init}
    if (
        "attention_backend_override" in init_fields
        and impl_cfg_kwargs.get("attention_backend_override") is None
    ):
        impl_cfg_kwargs["attention_backend_override"] = rt_cfg.attention_backend_override
    if "hf_path" in init_fields and impl_cfg_kwargs.get("hf_path") in (None, "") and rt_cfg.hf_path:
        impl_cfg_kwargs["hf_path"] = rt_cfg.hf_path
    # Thread the user-level OptimizerConfig so the protocol can pass it to
    # optimizer primitives without reading runtime internals.
    if (
        "optimizer_config" in init_fields
        and impl_cfg_kwargs.get("optimizer_config") is None
        and getattr(rt_cfg, "optimizer", None) is not None
    ):
        impl_cfg_kwargs["optimizer_config"] = rt_cfg.optimizer
    return proto.ImplConfig(**impl_cfg_kwargs)


def _apply_attention_backend_env(backend: str | None, *, tag: str) -> None:
    if backend is None:
        return

    env_overrides = {
        "auto": ("1", "1", "1"),
        "flash": ("1", "0", "0"),
        "fused": ("0", "1", "0"),
        "unfused": ("0", "0", "1"),
        "local": ("0", "0", "0"),
    }
    try:
        flash, fused, unfused = env_overrides[backend]
    except KeyError as exc:
        raise ValueError(
            "attention_backend_override must be one of {'auto', 'flash', 'fused', 'unfused', 'local'}"
        ) from exc

    os.environ["NVTE_FLASH_ATTN"] = flash
    os.environ["NVTE_FUSED_ATTN"] = fused
    os.environ["NVTE_UNFUSED_ATTN"] = unfused


def _infer_pipeline_tensor_shape(batch: Any, model_cfg: Any, ps) -> tuple[int, int, int]:
    if model_cfg is None or not hasattr(model_cfg, "hidden_size"):
        raise ValueError("Megatron Lite pipeline runtime requires model_cfg.hidden_size.")
    if not isinstance(batch, dict) or "input_ids" not in batch:
        raise TypeError("Megatron Lite pipeline runtime requires dict batches with input_ids.")

    input_ids = batch["input_ids"]
    if input_ids.dim() == 1:
        batch_size = 1
        local_seq_len = int(input_ids.size(0))
    elif input_ids.dim() == 2:
        batch_size = int(input_ids.size(0))
        local_seq_len = int(input_ids.size(1))
    else:
        raise ValueError(f"Unsupported input_ids rank for pipeline runtime: {input_ids.dim()}.")

    if local_seq_len < 1:
        raise ValueError("Pipeline tensor shape requires non-empty sequence.")

    tp_size = int(getattr(ps, "tp_size", 1) or 1)
    if tp_size > 1:
        if local_seq_len % tp_size != 0:
            raise ValueError(
                f"Pipeline tensor sequence length {local_seq_len} is not divisible by TP={tp_size}."
            )
        # Megatron Lite Qwen3.5 scatters embeddings into Megatron sequence-parallel form
        # before the first layer, so PP activations carry S / (CP * TP).
        local_seq_len //= tp_size

    return (local_seq_len, batch_size, int(model_cfg.hidden_size))


def _last_loss_output(outputs: list[dict]) -> dict:
    for output in reversed(outputs):
        if output.get("loss") is not None:
            return output
    return {}


def _checkpoint_module(model: Any) -> torch.nn.Module:
    if isinstance(model, torch.nn.Module):
        return model
    if isinstance(model, list | tuple):
        return torch.nn.ModuleList(model)
    raise TypeError(
        f"Checkpoint model must be an nn.Module or sequence of modules, got {type(model).__name__}."
    )


class MegatronLiteRuntime(RuntimeBase):
    """Megatron Lite default training backend (Megatron-style 5D parallel)."""

    def __init__(self, hf_path: str, cfg: MegatronLiteConfig | dict[str, Any]):
        self._hf_path = hf_path
        self._cfg = (
            cfg
            if isinstance(cfg, MegatronLiteConfig)
            else MegatronLiteConfig.from_dict(hf_path, cfg)
        )

    # ── build_model ──

    def build_model(
        self,
        hf_path: str | None = None,
        cfg: MegatronLiteConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> ModelHandle:
        if cfg is not None and isinstance(cfg, dict):
            rt_cfg = MegatronLiteConfig.from_dict(hf_path or self._hf_path, cfg)
        elif cfg is not None and isinstance(cfg, MegatronLiteConfig):
            rt_cfg = cfg
        else:
            rt_cfg = self._cfg

        # ── init distributed ──
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(minutes=10))
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        torch.cuda.manual_seed(42)

        # ── load model protocol module ──
        proto = self._load_protocol(rt_cfg)

        _apply_attention_backend_env(
            rt_cfg.attention_backend_override, tag=f"{rt_cfg.model_name}:{rt_cfg.impl}"
        )

        # ── escape hatch: model takes over ──
        if hasattr(proto, "create_runtime"):
            return proto.create_runtime(rt_cfg.hf_path, rt_cfg).build_model()

        # ── construct impl_cfg (parallel injected) ──
        impl_cfg = _build_impl_cfg(proto, rt_cfg)

        # ── build model config ──
        model_cfg = proto.build_model_config(rt_cfg.hf_path)
        if callable(rt_cfg.model_config_hook):
            model_cfg = rt_cfg.model_config_hook(model_cfg)

        # ── build model (model owns ps + optimizer + everything) ──
        bundle = proto.build_model(model_cfg, impl_cfg=impl_cfg)

        # ── load HF weights (optional) ──
        loaded_hf_weights = False
        if rt_cfg.load_hf_weights and rt_cfg.hf_path and hasattr(proto, "load_hf_weights"):
            for chunk in bundle.chunks:
                proto.load_hf_weights(chunk, rt_cfg.hf_path, model_cfg, bundle.parallel_state)
            loaded_hf_weights = True

        post_load_hook = bundle.extras.pop("post_model_load_hook", None)
        if callable(post_load_hook):
            post_load_updates = post_load_hook()
            if post_load_updates is not None:
                if not isinstance(post_load_updates, dict):
                    raise TypeError("post_model_load_hook must return a dict or None.")
                if "optimizer" in post_load_updates:
                    bundle.optimizer = post_load_updates["optimizer"]
                if "finalize_grads" in post_load_updates:
                    bundle.finalize_grads = post_load_updates["finalize_grads"]
                extra_updates = post_load_updates.get("extras")
                if extra_updates:
                    if not isinstance(extra_updates, dict):
                        raise TypeError("post_model_load_hook extras update must be a dict.")
                    bundle.extras.update(extra_updates)

        if loaded_hf_weights and bundle.optimizer is not None:
            reload_model_params = getattr(bundle.optimizer, "reload_model_params", None)
            if callable(reload_model_params):
                reload_model_params()

        # ── forward_step default ──
        forward_fn = bundle.forward_step or (lambda m, b: m(**b))

        p = rt_cfg.parallel
        model = bundle.chunks[0] if len(bundle.chunks) == 1 else bundle.chunks
        return ModelHandle(
            model=model,
            optimizer=bundle.optimizer,
            lr_scheduler=None,
            parallel_state=bundle.parallel_state,
            config=rt_cfg,
            _extras={
                "model_chunks": bundle.chunks,
                "model_cfg": model_cfg,
                "forward_step": forward_fn,
                "protocol": proto,
                "finalize_grads": bundle.finalize_grads,
                "world_size": dist.get_world_size(),
                "cp_range": (p.cp, p.cp),
                **bundle.extras,
            },
        )

    def _load_protocol(self, rt_cfg: MegatronLiteConfig):
        """Load and return the model protocol module."""
        from megatron.lite.model.registry import TRAIN_RUNTIME_MODULES, resolve_runtime_model_name

        try:
            runtime_key = resolve_runtime_model_name(rt_cfg.model_name, rt_cfg.impl)
        except ValueError as exc:
            raise ValueError(
                f"No protocol registered for model={rt_cfg.model_name!r}, "
                f"impl={rt_cfg.impl!r}. Register with register_model(...)."
            ) from exc

        mod_path = TRAIN_RUNTIME_MODULES.get(runtime_key)
        if mod_path is None:
            raise ValueError(f"No protocol module for runtime key {runtime_key!r}")

        import importlib

        proto = importlib.import_module(mod_path)

        for fn_name in ("build_model_config", "build_model"):
            if not callable(getattr(proto, fn_name, None)):
                raise ValueError(f"Protocol module {mod_path} missing required function: {fn_name}")
        if not hasattr(proto, "ImplConfig"):
            raise ValueError(f"Protocol module {mod_path} missing ImplConfig class")

        return proto

    # ── Checkpoint ──

    def save_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> None:
        from megatron.lite.primitive.ckpt import save_training_checkpoint

        step = kwargs.pop("step", None)
        if step is None:
            step = kwargs.pop("iteration", None)
        if step is None:
            step = kwargs.pop("global_step", 0)
        use_dcp = bool(kwargs.pop("use_dcp", True))
        save_rng = bool(kwargs.pop("save_rng", True))
        get_placements, is_expert = _checkpoint_hooks(handle)
        save_training_checkpoint(
            _checkpoint_model(handle, use_dcp=use_dcp),
            handle._optimizer,
            int(step),
            path,
            _checkpoint_parallel_config(handle),
            handle._parallel_state,
            get_placements=kwargs.pop("get_placements", get_placements),
            is_expert=kwargs.pop("is_expert", is_expert),
            use_dcp=use_dcp,
            save_rng=save_rng,
            save_model=kwargs.pop("save_model", True),
            save_optimizer=kwargs.pop("save_optimizer", True),
            **kwargs,
        )

    def load_checkpoint(self, handle: ModelHandle, path: str, **kwargs) -> int:
        from megatron.lite.primitive.ckpt import load_training_checkpoint

        use_dcp = bool(kwargs.pop("use_dcp", True))
        load_rng = bool(kwargs.pop("load_rng", True))
        update_legacy_format = bool(
            kwargs.pop(
                "load_parameter_state_update_legacy_format",
                kwargs.pop("update_legacy_format", False),
            )
        )
        get_placements, is_expert = _checkpoint_hooks(handle)
        return load_training_checkpoint(
            _checkpoint_model(handle, use_dcp=use_dcp),
            handle._optimizer,
            path,
            _checkpoint_parallel_config(handle),
            handle._parallel_state,
            get_placements=kwargs.pop("get_placements", get_placements),
            is_expert=kwargs.pop("is_expert", is_expert),
            use_dcp=use_dcp,
            load_rng=load_rng,
            load_parameter_state_update_legacy_format=update_legacy_format,
            load_model=kwargs.pop("load_model", True),
            load_optimizer=kwargs.pop("load_optimizer", True),
            **kwargs,
        )

    def export_weights(self, handle: ModelHandle, **kwargs) -> Iterator[tuple[str, torch.Tensor]]:
        model_chunks = handle._extras.get("model_chunks", [handle._model])
        proto = handle._extras.get("protocol")
        model_cfg = handle._extras.get("model_cfg")
        ps = handle._parallel_state

        if proto and hasattr(proto, "export_hf_weights"):
            yield from proto.export_hf_weights(model_chunks, model_cfg, ps, **kwargs)
        else:
            for chunk in model_chunks:
                yield from chunk.named_parameters()

    # ── Memory ──

    def to(
        self,
        handle: ModelHandle,
        device: str,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        model_chunks = handle._extras.get("model_chunks", [handle._model])
        from megatron.lite.runtime.megatron_utils import (
            load_model_to_gpu,
            load_optimizer,
            offload_model_to_cpu,
            offload_optimizer,
        )

        if device == "cpu":
            if model:
                offload_model_to_cpu(model_chunks)
            if optimizer and handle._optimizer is not None:
                offload_state = getattr(handle._optimizer, "offload_state_to_cpu", None)
                if callable(offload_state):
                    offload_state()
                else:
                    offload_optimizer(handle._optimizer)
        elif device == "cuda":
            if model:
                load_model_to_gpu(model_chunks, load_grad=grad)
            if optimizer and handle._optimizer is not None:
                load_state = getattr(handle._optimizer, "load_state_to_device", None)
                if callable(load_state):
                    load_state()
                else:
                    load_optimizer(handle._optimizer)

    # ── Mode switching ──

    def train_mode(self, handle: ModelHandle):
        return _TrainModeCtx(handle)

    def eval_mode(self, handle: ModelHandle):
        return _EvalModeCtx(handle)

    # ── Training atoms ──

    def forward_backward(
        self,
        handle: ModelHandle,
        data: Any,
        loss_fn: Callable | None,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
    ) -> ForwardResult:
        from megatron.lite.primitive.train_step import run_microbatch_loop

        forward_step = handle._extras["forward_step"]
        if num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")

        if hasattr(data, "__next__"):
            data_iter = data
        elif hasattr(data, "__iter__"):
            data_iter = iter(data)
        else:
            data_iter = iter([data])

        ps = handle._parallel_state
        if ps.pp_size > 1:
            from types import SimpleNamespace

            from megatron.lite.primitive.parallel.pipeline import forward_backward_pipelining

            first_batch = next(data_iter)
            data_iter = chain([first_batch], data_iter)
            tensor_shape = _infer_pipeline_tensor_shape(
                first_batch, handle._extras.get("model_cfg"), ps
            )
            outputs = forward_backward_pipelining(
                forward_step,
                handle._extras.get("model_chunks", [handle._model]),
                data_iter,
                SimpleNamespace(num_microbatches=num_microbatches),
                ps,
                tensor_shape=tensor_shape,
                pre_forward_hook=handle._extras.get("pre_forward_hook"),
                loss_fn=loss_fn,
                forward_only=forward_only,
            )
            out = _last_loss_output(outputs)
            loss_obj = out.get("loss") if out else None
            if isinstance(loss_obj, torch.Tensor):
                loss_float = float(loss_obj.detach().item())
            elif loss_obj is not None:
                loss_float = float(loss_obj)
            else:
                loss_float = 0.0
            loss_t = torch.tensor([loss_float], device="cuda")
            if ps.pp_group is not None and ps.pp_global_ranks is not None:
                dist.broadcast(loss_t, src=ps.pp_global_ranks[-1], group=ps.pp_group)
            out = {"loss": loss_t.squeeze(0)}
        else:
            out = run_microbatch_loop(
                handle._model,
                data_iter,
                num_microbatches,
                forward_step,
                optimizer=handle._optimizer if not forward_only else None,
                dist_opt=not forward_only,
                pre_forward_hook=handle._extras.get("pre_forward_hook"),
                loss_fn=loss_fn,
            )

        if not forward_only:
            finalize_grads = handle._extras.get("finalize_grads")
            if finalize_grads is not None:
                finalize_grads()

        loss_tensor = out.get("loss") if out else None
        loss_val = (
            loss_tensor.item()
            if isinstance(loss_tensor, torch.Tensor)
            else float(loss_tensor or 0.0)
        )
        metrics: dict = {"loss": loss_val}
        for m in out.get("_loss_fn_metrics", []) if out else []:
            for k, v in m.items():
                if k not in metrics:
                    metrics[k] = v
        if ps.pp_size > 1:
            for item in outputs:
                for k, v in item.get("metrics", {}).items():
                    if k not in metrics:
                        metrics[k] = v
            metrics["_micro_outputs"] = outputs

        return ForwardResult(
            model_output=ModelOutputs(
                loss=loss_tensor,
                vocab_parallel_logits=out.get("logits") if out else None,
                log_probs=out.get("log_probs") if out else None,
                routed_experts=out.get("routed_experts") if out else None,
            ),
            metrics=metrics,
        )

    def is_mp_src_rank_with_outputs(self, handle: ModelHandle) -> bool:
        ps = handle._parallel_state
        return ps.tp_rank == 0 and ps.cp_rank == 0 and ps.pp_rank == ps.pp_size - 1

    def zero_grad(self, handle: ModelHandle) -> None:
        for chunk in handle._extras.get("model_chunks", [handle._model]):
            if hasattr(chunk, "zero_grad_buffer"):
                chunk.zero_grad_buffer()
        if handle._optimizer is not None:
            handle._optimizer.zero_grad()

    def optimizer_step(self, handle: ModelHandle) -> tuple[bool, float, int | None]:
        if handle._optimizer is None:
            return True, 0.0, 0
        update_successful, grad_norm, num_zeros = handle._optimizer.step()
        return update_successful, float(grad_norm), num_zeros

    def lr_scheduler_step(self, handle: ModelHandle) -> float | list[float]:
        if handle._lr_scheduler is not None:
            handle._lr_scheduler.step()
            return handle._lr_scheduler.get_last_lr()
        return 0.0


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


class _TrainModeCtx:
    def __init__(self, handle: ModelHandle):
        self._handle = handle

    def __enter__(self):
        for chunk in self._handle._extras.get("model_chunks", [self._handle._model]):
            chunk.train()
        return self

    def __exit__(self, *exc):
        return False


class _EvalModeCtx:
    def __init__(self, handle: ModelHandle):
        self._handle = handle
        self._prev_grad = torch.is_grad_enabled()

    def __enter__(self):
        for chunk in self._handle._extras.get("model_chunks", [self._handle._model]):
            chunk.eval()
        torch.set_grad_enabled(False)
        return self

    def __exit__(self, *exc):
        torch.set_grad_enabled(self._prev_grad)
        return False


def _checkpoint_parallel_config(handle: ModelHandle):
    cfg = handle.config
    if cfg is None:
        return None
    return getattr(cfg, "parallel", cfg)


def _checkpoint_model(handle: ModelHandle, *, use_dcp: bool):
    model = handle._model
    if not use_dcp or isinstance(model, torch.nn.Module):
        return model
    return torch.nn.ModuleList(handle._extras.get("model_chunks", model))


def _checkpoint_hooks(handle: ModelHandle):
    from megatron.lite.primitive.protocols import default_expert_classifier, default_placement_fn

    proto = handle._extras.get("protocol")
    return (
        getattr(proto, "PLACEMENT_FN", default_placement_fn),
        getattr(proto, "EXPERT_CLASSIFIER", default_expert_classifier),
    )

"""Small pretrain benchmark session composed from runtime atoms."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from megatron.lite.runtime.backends import Runtime
from megatron.lite.runtime.contracts.handle import ModelHandle

from .results import RunResult, StepTrace


@dataclass
class PretrainSessionConfig:
    steps: int = 2
    warmup: int = 0
    num_microbatches: int = 1
    seq_len: int = 2048
    seed: int = 42
    device: str = "cuda"
    use_thd: bool = False
    same_data_across_dp: bool = False
    no_optimizer: bool = False


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def _sync(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize()


def _reset_peak_memory(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_gb(device: str) -> float:
    if _is_cuda_device(device):
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _resolve_vocab_size(handle: ModelHandle) -> int:
    proto = handle._extras.get("protocol")
    model_cfg = handle._extras.get("model_cfg")
    if proto is not None and model_cfg is not None and hasattr(proto, "vocab_size"):
        return int(proto.vocab_size(model_cfg))
    if model_cfg is not None and hasattr(model_cfg, "vocab_size"):
        return int(model_cfg.vocab_size)
    return 151936


def _make_data_iter(handle: ModelHandle, cfg: PretrainSessionConfig):
    data_seed = cfg.seed if cfg.same_data_across_dp else cfg.seed + handle.dp_rank
    vocab_size = _resolve_vocab_size(handle)

    if cfg.use_thd:
        from megatron.lite.primitive.data import infinite_batches_thd

        ps = handle._parallel_state
        return infinite_batches_thd(
            vocab_size,
            cfg.seq_len,
            cp_size=getattr(ps, "cp_size", 1),
            cp_rank=getattr(ps, "cp_rank", 0),
            device=cfg.device,
            seed=data_seed,
        )

    from megatron.lite.primitive.data import infinite_batches

    return infinite_batches(vocab_size, cfg.seq_len, device=cfg.device, seed=data_seed)


def _calc_tflops_per_gpu(
    *,
    num_floating_point_operations: int | None,
    activated_params: int | None,
    tokens_per_step: int,
    step_s: float,
    world_size: int,
) -> float | None:
    if step_s <= 0:
        return None
    if num_floating_point_operations:
        return num_floating_point_operations / (step_s * world_size * 1e12)
    if activated_params:
        return 6 * activated_params * tokens_per_step / (step_s * world_size * 1e12)
    return None


def _resolve_model_stats(config: Any, proto: Any) -> Any | None:
    model_name = getattr(config, "model_name", None)
    if model_name and model_name != "auto":
        stats_module = f"megatron.lite.model.{model_name}.stats"
        try:
            return importlib.import_module(stats_module)
        except ModuleNotFoundError as exc:
            if exc.name is not None and not stats_module.startswith(exc.name):
                raise
    return proto


def _resolve_step_flops(
    handle: ModelHandle, cfg: PretrainSessionConfig
) -> tuple[int | None, int | None]:
    config = handle.config
    proto = handle._extras.get("protocol")
    model_stats = _resolve_model_stats(config, proto)
    model_cfg = handle._extras.get("model_cfg")
    if model_stats is None or model_cfg is None:
        return None, None

    step_flops = None
    if hasattr(model_stats, "num_floating_point_operations"):
        parallel_cfg = getattr(config, "parallel", None)
        tp_size = getattr(parallel_cfg, "tp", 1)
        step_flops = model_stats.num_floating_point_operations(
            model_cfg,
            seq_len=cfg.seq_len,
            global_batch_size=cfg.num_microbatches * handle.dp_size,
            tp_size=tp_size,
        )

    activated_params = None
    if step_flops is None and hasattr(model_stats, "activated_params"):
        activated_params = model_stats.activated_params(model_cfg)

    return step_flops, activated_params


def run_pretrain_session(
    rt: Runtime,
    handle: ModelHandle,
    cfg: PretrainSessionConfig,
    *,
    data_iter: Any = None,
    step_reporter: Callable[[StepTrace], None] | None = None,
) -> RunResult:
    """Run a fixed-shape benchmark loop through the public runtime API."""
    if cfg.steps < 1:
        raise ValueError("steps must be >= 1")
    if cfg.warmup < 0 or cfg.warmup >= cfg.steps:
        raise ValueError("warmup must satisfy 0 <= warmup < steps")
    if cfg.num_microbatches < 1:
        raise ValueError("num_microbatches must be >= 1")

    if data_iter is None:
        data_iter = _make_data_iter(handle, cfg)

    world_size = _world_size()
    tokens_per_step = cfg.num_microbatches * cfg.seq_len * world_size
    step_flops, activated_params = _resolve_step_flops(handle, cfg)

    step_traces: list[StepTrace] = []
    timings: list[float] = []

    _reset_peak_memory(cfg.device)
    with rt.train_mode(handle):
        for step in range(cfg.steps):
            if step == cfg.warmup:
                _reset_peak_memory(cfg.device)

            rt.zero_grad(handle)
            _sync(cfg.device)
            t0 = time.perf_counter()
            result = rt.forward_backward(
                handle, data_iter, loss_fn=None, num_microbatches=cfg.num_microbatches
            )
            if cfg.no_optimizer:
                grad_norm = 0.0
            else:
                _, grad_norm, _ = rt.optimizer_step(handle)
                rt.lr_scheduler_step(handle)
            _sync(cfg.device)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            tflops_per_gpu = _calc_tflops_per_gpu(
                num_floating_point_operations=step_flops,
                activated_params=activated_params,
                tokens_per_step=tokens_per_step,
                step_s=elapsed_ms / 1000,
                world_size=world_size,
            )
            trace = StepTrace(
                step=step,
                loss=float(result.metrics.get("loss", 0.0)),
                grad_norm=float(grad_norm),
                step_ms=elapsed_ms,
                peak_mem_gb=_peak_memory_gb(cfg.device),
                tflops_per_gpu=tflops_per_gpu,
            )
            if step_reporter is not None:
                step_reporter(trace)
            if step >= cfg.warmup:
                timings.append(elapsed_ms)
                trace.step = step - cfg.warmup
                step_traces.append(trace)

    avg_step_ms = sum(timings) / len(timings) if timings else 0.0
    avg_step_s = avg_step_ms / 1000
    tok_per_s = tokens_per_step / avg_step_s if avg_step_s > 0 else 0.0
    avg_tflops = _calc_tflops_per_gpu(
        num_floating_point_operations=step_flops,
        activated_params=activated_params,
        tokens_per_step=tokens_per_step,
        step_s=avg_step_s,
        world_size=world_size,
    )

    config = handle.config
    parallel = config.parallel
    backend = "bridge" if type(config).__name__ == "BridgeConfig" else "mlite"
    return RunResult(
        backend=backend,
        model_name=getattr(config, "model_name", "unknown"),
        impl=getattr(config, "impl", "bridge"),
        optimizer_backend=handle._extras.get(
            "optimizer_backend",
            getattr(handle._optimizer, "name", "none") if handle._optimizer is not None else "none",
        ),
        tp=parallel.tp,
        etp=parallel.etp,
        ep=parallel.ep,
        pp=parallel.pp,
        vpp=parallel.vpp,
        cp=parallel.cp,
        seq_len=cfg.seq_len,
        num_microbatches=cfg.num_microbatches,
        step_traces=step_traces,
        avg_step_ms=avg_step_ms,
        peak_mem_gb=_peak_memory_gb(cfg.device),
        tok_per_s=tok_per_s,
        tok_per_s_per_gpu=tok_per_s / world_size,
        tflops_per_gpu=avg_tflops,
        metadata={"warmup": cfg.warmup, "device": cfg.device, "use_thd": cfg.use_thd},
    )


__all__ = ["PretrainSessionConfig", "run_pretrain_session"]

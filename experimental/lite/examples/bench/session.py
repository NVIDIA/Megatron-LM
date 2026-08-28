# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small pretrain benchmark session composed from runtime atoms."""

from __future__ import annotations

import importlib
import hashlib
import math
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
    forward_only: bool = False
    empty_cache_between_steps: bool = False
    max_steady_peak_growth: float = 0.02
    trace_fingerprints: bool = False


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def _sync(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.synchronize()


def _reset_peak_memory(device: str) -> None:
    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats()


def _memory_snapshot(device: str) -> dict[str, int]:
    if not _is_cuda_device(device):
        return {
            "peak_allocated_bytes": 0,
            "post_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "post_reserved_bytes": 0,
            "active_bytes": 0,
        }
    stats = torch.cuda.memory_stats()
    values = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "post_allocated_bytes": torch.cuda.memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "post_reserved_bytes": torch.cuda.memory_reserved(),
        "active_bytes": stats["active_bytes.all.current"],
    }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        packed = torch.tensor(
            list(values.values()), dtype=torch.int64, device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.MAX)
        values = dict(zip(values, packed.cpu().tolist(), strict=True))
    return values


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _global_grad_norm_without_step(handle: ModelHandle) -> float:
    """Report the runtime-aligned global norm without mutating weights or grads."""
    from megatron.lite.primitive.train_step import compute_global_grad_norm

    ps = handle._parallel_state
    chunks = handle._extras.get("model_chunks", [handle._model])
    protocol = handle._extras.get("protocol")
    is_expert_param = getattr(protocol, "is_expert_param", None)
    if ps is None or is_expert_param is None:
        total = None
        for chunk in chunks:
            for parameter in chunk.parameters():
                if parameter.grad is not None:
                    squared = parameter.grad.detach().float().norm().square()
                    total = squared if total is None else total + squared
        return 0.0 if total is None else float(total.sqrt().item())

    squared_norms = [
        compute_global_grad_norm(chunk, ps, is_expert_param=is_expert_param).square()
        for chunk in chunks
    ]
    return float(torch.stack(squared_norms).sum().sqrt().item())


def _sampled_parameter_fingerprint(
    handle: ModelHandle, *, gradients: bool
) -> dict[str, Any]:
    """Hash bounded endpoint samples from every local parameter in one D2H."""
    digest = hashlib.sha256()
    samples = []
    tensor_count = 0
    chunks = handle._extras.get("model_chunks", [handle._model])
    for chunk_idx, chunk in enumerate(chunks):
        for name, parameter in sorted(chunk.named_parameters(), key=lambda item: item[0]):
            tensor = parameter
            if gradients:
                tensor = parameter.grad
                if tensor is None:
                    tensor = getattr(parameter, "main_grad", None)
                if tensor is None:
                    continue
            if hasattr(tensor, "to_local"):
                tensor = tensor.to_local()
            flat = tensor.detach().reshape(-1)
            if flat.numel() == 0:
                continue
            take = min(8, flat.numel())
            sample = torch.cat((flat[:take], flat[-take:])).float()
            samples.append(sample)
            digest.update(f"{chunk_idx}:{name}:{tuple(tensor.shape)}\0".encode())
            tensor_count += 1
    if samples:
        payload = torch.cat(samples).contiguous().cpu()
        digest.update(payload.view(torch.uint8).numpy().tobytes())
        sample_count = payload.numel()
    else:
        sample_count = 0
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "sample_count": sample_count,
        "rule": "first8_last8_as_float32_per_local_parameter",
    }


def _resolve_vocab_size(handle: ModelHandle) -> int:
    proto = handle._extras.get("protocol")
    model_cfg = handle._extras.get("model_cfg")
    if proto is not None and model_cfg is not None and hasattr(proto, "vocab_size"):
        return int(proto.vocab_size(model_cfg))
    if model_cfg is not None and hasattr(model_cfg, "vocab_size"):
        return int(model_cfg.vocab_size)
    return 151936


def _infinite_packed_batches(
    vocab_size: int, seq_len: int, *, device: str, seed: int
):
    """Yield raw, model-agnostic :class:`PackedBatch` objects for the bench.

    The bench is the single source of truth for one unpadded packed batch (1-D
    ``input_ids``/``labels`` plus true per-sequence ``seq_lens``). Padding, CP
    layout and THD metadata (``packed_seq_params``) are derived by whichever
    runtime/model consumes the batch at the forward boundary, never baked into
    bench data — that is what keeps the mlite-vs-bridge comparison fair.
    """
    from megatron.lite.runtime.contracts.data import PackedBatch

    g = torch.Generator(device=device).manual_seed(seed)
    seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
    while True:
        yield PackedBatch(
            input_ids=torch.randint(0, vocab_size, (seq_len,), device=device, generator=g),
            labels=torch.randint(0, vocab_size, (seq_len,), device=device, generator=g),
            seq_lens=seq_lens.clone(),
        )


def _make_data_iter(handle: ModelHandle, cfg: PretrainSessionConfig):
    data_seed = cfg.seed if cfg.same_data_across_dp else cfg.seed + handle.dp_rank
    vocab_size = _resolve_vocab_size(handle)
    return _infinite_packed_batches(vocab_size, cfg.seq_len, device=cfg.device, seed=data_seed)


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
    if cfg.max_steady_peak_growth < 0:
        raise ValueError("max_steady_peak_growth must be >= 0")

    if data_iter is None:
        data_iter = _make_data_iter(handle, cfg)

    world_size = _world_size()
    tokens_per_step = cfg.num_microbatches * cfg.seq_len * world_size
    step_flops, activated_params = _resolve_step_flops(handle, cfg)

    step_traces: list[StepTrace] = []
    timings: list[float] = []

    _reset_peak_memory(cfg.device)
    mode = rt.eval_mode(handle) if cfg.forward_only else rt.train_mode(handle)
    with mode:
        for step in range(cfg.steps):
            if cfg.empty_cache_between_steps and _is_cuda_device(cfg.device):
                torch.cuda.empty_cache()
            _reset_peak_memory(cfg.device)

            if not cfg.forward_only:
                rt.zero_grad(handle)
            _sync(cfg.device)
            t0 = time.perf_counter()
            result = rt.forward_backward(
                handle,
                data_iter,
                loss_fn=None,
                num_microbatches=cfg.num_microbatches,
                forward_only=cfg.forward_only,
            )
            if not cfg.forward_only and not cfg.no_optimizer:
                _, grad_norm, _ = rt.optimizer_step(handle)
                rt.lr_scheduler_step(handle)
            else:
                grad_norm = 0.0
            _sync(cfg.device)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            memory = _memory_snapshot(cfg.device)
            if cfg.no_optimizer and not cfg.forward_only:
                # Keep this correctness check outside the timed/memory window:
                # it performs benchmark-only reductions but does not mutate grads.
                grad_norm = _global_grad_norm_without_step(handle)
            if not cfg.forward_only and (
                not math.isfinite(float(grad_norm)) or float(grad_norm) <= 0.0
            ):
                raise RuntimeError(
                    f"training benchmark produced invalid grad_norm={grad_norm}"
                )
            trace_fingerprints = cfg.trace_fingerprints and step >= cfg.warmup
            grad_fingerprint = (
                _sampled_parameter_fingerprint(handle, gradients=True)
                if trace_fingerprints and not cfg.forward_only
                else None
            )
            weight_fingerprint = (
                _sampled_parameter_fingerprint(handle, gradients=False)
                if trace_fingerprints
                else None
            )
            tflops_per_gpu = _calc_tflops_per_gpu(
                num_floating_point_operations=step_flops,
                activated_params=activated_params,
                tokens_per_step=tokens_per_step,
                step_s=elapsed_ms / 1000,
                world_size=world_size,
            )
            trace = StepTrace(
                step=step,
                loss=float(
                    (
                        result.metrics.get("loss", result.model_output.loss)
                        or torch.tensor(0.0)
                    ).detach()
                ),
                grad_norm=float(grad_norm),
                step_ms=elapsed_ms,
                peak_mem_gb=memory["peak_allocated_bytes"] / 1e9,
                **memory,
                tflops_per_gpu=tflops_per_gpu,
                grad_fingerprint=grad_fingerprint,
                weight_fingerprint=weight_fingerprint,
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
    measured_peaks = [
        trace.peak_allocated_bytes or 0 for trace in step_traces
    ]
    growth_rates = [
        (current - previous) / previous
        for previous, current in zip(measured_peaks, measured_peaks[1:])
        if previous > 0
    ]
    max_growth = max([0.0, *growth_rates])
    memory_gate = {
        "all_rank_max": True,
        "max_steady_peak_growth": max_growth,
        "limit": cfg.max_steady_peak_growth,
        "passed": len(measured_peaks) >= 2
        and max_growth < cfg.max_steady_peak_growth,
        "empty_cache_between_steps": cfg.empty_cache_between_steps,
    }

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
        peak_mem_gb=max(measured_peaks, default=0) / 1e9,
        tok_per_s=tok_per_s,
        tok_per_s_per_gpu=tok_per_s / world_size,
        tflops_per_gpu=avg_tflops,
        metadata={
            "warmup": cfg.warmup,
            "device": cfg.device,
            "use_thd": cfg.use_thd,
            "memory_gate": memory_gate,
            "trace_fingerprints": cfg.trace_fingerprints,
        },
    )


__all__ = ["PretrainSessionConfig", "run_pretrain_session"]

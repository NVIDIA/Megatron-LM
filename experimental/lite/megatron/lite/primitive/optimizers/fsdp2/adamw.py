# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""AdamW helpers for the FSDP2 optimizer primitive."""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2.grad_clip import fused_sq_sum


@dataclass(slots=True)
class _PipelineFrag:
    """One contiguous piece of a slot: a flattened [start:stop) slice of a single
    param's master/moments/grad/dest, staged at offset `off` in the slot's pinned
    buffers. A param larger than the slot spans several fragments (all sharing its
    per-param `step`)."""

    param: nn.Parameter
    master: torch.Tensor
    ea: torch.Tensor
    es: torch.Tensor
    grad: torch.Tensor
    dest: torch.Tensor
    step: torch.Tensor
    start: int
    stop: int
    off: int
    length: int


@dataclass(slots=True)
class _PipelineSlot:
    """A fixed equal-size tile of fragments sharing (lr, wd, dest device)."""

    lr: float
    wd: float
    device: torch.device
    frags: list[_PipelineFrag] = field(default_factory=list)


@dataclass(slots=True)
class _D2HPlan:
    """A slot with its in-flight grad D2H: the pinned fp32 grad buffer and the event
    marking the copy's completion (the CPU waits on it before the fused kernel)."""

    slot: _PipelineSlot
    grad_buf: torch.Tensor
    event: torch.cuda.Event


def local_grad_sq_sum(
    params: Iterable[nn.Parameter],
    *,
    dtype: torch.dtype,
    default_device: torch.device | None = None,
) -> torch.Tensor:
    grads: list[torch.Tensor] = []
    device: torch.device | None = None
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        grad = to_local_tensor(grad).detach()
        if device is None:
            device = grad.device
        grads.append(grad)
    if device is None:
        return torch.zeros((), device=default_device or torch.device("cpu"), dtype=dtype)
    return fused_sq_sum(grads, dtype=dtype, device=device)


def to_local_tensor(tensor) -> torch.Tensor:
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def fsdp2_model_param_dtype(param: nn.Parameter) -> torch.dtype | None:
    dtype = getattr(param, "_fsdp2_model_param_dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def local_param_shard(param: nn.Parameter) -> torch.Tensor:
    """The param's writable local storage: its DTensor local shard, or the plain
    detached tensor. Master/moments are init'd from this, so shapes always match."""
    return to_local_tensor(param) if is_dtensor_like(param) else param.detach()


def has_dtensor_grad_or_param(param: nn.Parameter) -> bool:
    grad = param.grad
    return is_dtensor_like(param) or (grad is not None and is_dtensor_like(grad))


def is_dtensor_like(tensor: Any) -> bool:
    return (
        callable(getattr(tensor, "to_local", None))
        and hasattr(tensor, "device_mesh")
        and hasattr(tensor, "placements")
    )


def copy_local_tensor_to_param_(param: nn.Parameter, local_tensor: torch.Tensor) -> None:
    # Copy straight into the param's local shard. Reconstructing a DTensor via
    # DTensor.from_local mis-sizes an unevenly-sharded param (it infers global =
    # local * mesh, e.g. a (3,) param over 8 ranks -> 0 or 8), so copy local->local
    # (master is init'd from this same local shard, so shapes match).
    local_param = local_param_shard(param)
    local_param.copy_(local_tensor.to(device=local_param.device, dtype=local_param.dtype))


def all_reduce_grad_(grad: torch.Tensor, *, group: dist.ProcessGroup) -> None:
    # ``to_local_tensor`` returns the DTensor's local shard storage, so the
    # in-place all-reduce updates the grad directly -- no DTensor.from_local
    # round-trip (which mis-sizes unevenly-sharded grads, e.g. (3,) over 8 ranks).
    local_grad = to_local_tensor(grad)
    dist.all_reduce(local_grad, op=dist.ReduceOp.SUM, group=group)


class ChainedOptimizer:
    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "chained_torch_optimizer",
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers")
        if not isinstance(optimizer_states, list) or len(optimizer_states) != len(self.optimizers):
            raise ValueError("Invalid chained torch optimizer state_dict.")
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(optimizer_state)


class FP32AdamW:
    """AdamW with FP32 master params for BF16/DTensor model weights."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        cpu_update: bool = False,
        pipeline: bool = False,
        model_param_dtypes: dict[int, torch.dtype] | None = None,
    ):
        self.param_groups = normalize_param_groups(params, default_weight_decay=weight_decay)
        self.params: list[nn.Parameter] = []
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.cpu_update = bool(cpu_update)
        # Overlap CPU-offload optimizer D2H/H2D with the fused CPU kernel via the
        # pipelined step (see _step_param_groups_pipelined). Opt-in via the optimizer
        # config's overlap_cpu_optimizer_d2h_h2d=True; requires cpu_update.
        self.pipeline = bool(pipeline)
        self.step_count = 0
        # Per-param optimizer state: fp32 master/moments (Tensors) plus the "step"
        # counter (int), so the value type is heterogeneous (Any). The fused kernel
        # builds its own fp32 step tensor per call.
        self.state: dict[nn.Parameter, dict[str, Any]] = {}
        self._master_for_param: dict[nn.Parameter, torch.Tensor] = {}
        # Free-list of pinned host buffers, keyed by (numel, dtype), reused across
        # steps. Bounds resident pinned memory to the pipeline's in-flight window
        # (see _step_param_groups_pipelined) instead of one buffer per param.
        self._pin_pool: dict[tuple[int, torch.dtype], list[torch.Tensor]] = {}
        self._pipeline_logged = False
        # Dedicated CUDA streams so grad D2H (down-lane) and param writeback H2D
        # (up-lane) overlap on full-duplex PCIe; created lazily on first pipelined step.
        self._d2h_stream: torch.cuda.Stream | None = None
        self._h2d_stream: torch.cuda.Stream | None = None
        self._model_param_dtypes_by_id = dict(model_param_dtypes or {})
        self._model_dtype_for_param: dict[nn.Parameter, torch.dtype] = {}

        for group in self.param_groups:
            group.setdefault("lr", lr)
            group.setdefault("wd_mult", 1.0)
            group_weight_decay = float(group.get("weight_decay", weight_decay))
            group["weight_decay"] = group_weight_decay
            for param in group["params"]:
                self.params.append(param)
                model_dtype = self._model_param_dtypes_by_id.get(id(param))
                if model_dtype is not None:
                    self._model_dtype_for_param[param] = model_dtype
                master = self._init_master_param(param)
                self.state[param] = {
                    "master_param": master,
                    "exp_avg": torch.zeros_like(master, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(master, dtype=torch.float32),
                    "step": 0,
                }
                self._master_for_param[param] = master

    def _init_master_param(self, param: nn.Parameter) -> torch.Tensor:
        if self.cpu_update:
            local_fp32 = to_local_tensor(param.detach()).to(device="cpu", dtype=torch.float32)
            # Pin so the pipelined step's writeback can H2D the fp32 master straight to
            # a GPU staging buffer asynchronously (a pageable source forces a sync copy).
            # pin_memory/clone both return independent, writable storage (.to may alias
            # when already CPU fp32); in-place ops (fused kernel, load_state_dict copy_)
            # then preserve the pinning.
            return local_fp32.pin_memory() if self.pipeline else local_fp32.clone()
        if self._model_param_dtype(param) is not None:
            return param.detach().to(dtype=torch.float32).clone()
        return (
            param.detach()
            if param.dtype is torch.float32
            else param.detach().to(dtype=torch.float32).clone()
        )

    def _model_param_dtype(self, param: nn.Parameter) -> torch.dtype | None:
        return self._model_dtype_for_param.get(param) or fsdp2_model_param_dtype(param)

    def zero_grad(self, *args, **kwargs) -> None:
        set_to_none = kwargs.get("set_to_none", False)
        if args:
            set_to_none = bool(args[0])
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self) -> None:
        self._step_param_groups()

    @torch.no_grad()
    def reload_model_params(self) -> None:
        """Refresh FP32 masters after model weights are loaded or initialized."""
        for param in self.params:
            master = to_local_tensor(self.state[param]["master_param"])
            model_param = to_local_tensor(param.detach())
            master.copy_(model_param.to(device=master.device, dtype=master.dtype))

    def _step_param_groups(self) -> None:
        if self.pipeline:
            self._step_param_groups_pipelined()
        else:
            self._step_param_groups_serial()

    def _step_param_groups_serial(self) -> None:
        self.step_count += 1
        beta1, beta2 = self.betas

        for group in self.param_groups:
            group_lr = float(group.get("lr", self.lr))
            group_weight_decay = float(group.get("weight_decay", self.weight_decay))
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                state["step"] = int(state["step"]) + 1
                param_step = int(state["step"])
                bias_correction1 = 1.0 - beta1**param_step
                bias_correction2_sqrt = (1.0 - beta2**param_step) ** 0.5
                group_step_size = group_lr / bias_correction1
                master = state["master_param"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group_weight_decay != 0.0:
                    master.mul_(1.0 - group_lr * group_weight_decay)
                grad = self._prepare_grad(grad, master)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
                master.addcdiv_(exp_avg.to(dtype=torch.float32), denom, value=-group_step_size)
                self._copy_master_to_param(param, master)

    def _acquire_pinned(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """Pop a reusable flat pinned host buffer, or allocate one. Callers view it."""
        pool = self._pin_pool.get((numel, dtype))
        return pool.pop() if pool else torch.empty(numel, dtype=dtype, pin_memory=True)

    def _release_pinned(self, buf: torch.Tensor) -> None:
        self._pin_pool.setdefault((buf.numel(), buf.dtype), []).append(buf.view(-1))

    def _step_param_groups_pipelined(self) -> None:
        """Pipelined CPU-offload fused AdamW (enabled by overlap_cpu_optimizer_d2h_h2d).

        Params are grouped by (lr, wd, dest device) and tiled into fixed equal-size
        slots of MLITE_FSDP2_ADAMW_SLOT_NUMEL elements (a param larger than a slot is
        split across slots). Each slot runs overlapped stages: async grad D2H into one
        pinned fp32 buffer (down-lane), the fused CPU kernel, and async param writeback
        (up-lane). The next slot's D2H is issued before the current slot's CPU kernel,
        so transfers hide under compute. D2H and H2D use dedicated streams so they
        overlap on full-duplex PCIe. Writeback casts fp32 master -> param dtype on the
        GPU (symmetric to the free bf16->fp32 cast on grad D2H): H2D the pinned fp32
        master to a GPU staging buffer and cast there, removing the CPU cast from the
        critical path. CPU-offload only; masters are pinned at init so the master H2D
        is async.

        Memory: because the next D2H is issued before the current slot is released, up
        to DEPTH+1 slots are resident at once, i.e. peak ~= (DEPTH+1) * slot_numel of
        both pinned fp32 grad buffers (host) and fp32 GPU staging buffers, plus the
        pinned fp32 masters. Tail/small buckets still allocate a full slot_numel buffer.
        """
        assert self.cpu_update, "pipelined AdamW requires cpu_update (CPU offload)"

        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream()
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream()
        d2h_stream, h2d_stream = self._d2h_stream, self._h2d_stream

        self.step_count += 1
        beta1, beta2 = self.betas
        slot_numel = max(1, int(os.getenv("MLITE_FSDP2_ADAMW_SLOT_NUMEL", "268435456")))
        depth = max(1, int(os.getenv("MLITE_FSDP2_ADAMW_PIPELINE_DEPTH", "3")))
        free_pin_after_opt = os.getenv(
            "MLITE_FSDP2_ADAMW_FREE_PIN_AFTER_OPT", "0"
        ) in ("1", "true", "True")

        if not self._pipeline_logged:
            self._pipeline_logged = True
            print(
                "[mlite-adamw] pipelined CPU-offload AdamW active | "
                f"MLITE_FSDP2_ADAMW_SLOT_NUMEL={slot_numel} "
                f"MLITE_FSDP2_ADAMW_PIPELINE_DEPTH={depth} "
                f"MLITE_FSDP2_ADAMW_FREE_PIN_AFTER_OPT={int(free_pin_after_opt)} "
                f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', 'unset')} "
                f"torch_threads={torch.get_num_threads()} | tune via these env vars",
                flush=True,
            )

        # Bump each param's step once (a split param must not double-count), and bucket
        # grad-bearing params by (lr, wd, dest device) so a slot's writeback stages on a
        # single GPU. (Per-fragment GPU cast handles mixed dtypes, so dtype needn't key.)
        buckets: dict[tuple[float, float, torch.device], list[nn.Parameter]] = {}
        for group in self.param_groups:
            lr = float(group.get("lr", self.lr))
            wd = float(group.get("weight_decay", self.weight_decay))
            for param in group["params"]:
                if param.grad is None:
                    continue
                self.state[param]["step"] = int(self.state[param]["step"]) + 1
                buckets.setdefault((lr, wd, local_param_shard(param).device), []).append(param)

        # Tile each bucket into fixed slots, splitting params at slot boundaries.
        slots: list[_PipelineSlot] = []
        for (lr, wd, device), params in buckets.items():
            frags: list[_PipelineFrag] = []
            filled = 0
            for param in params:
                state = self.state[param]
                # master/moments are plain CPU fp32 tensors (see __init__); only grad
                # and the param (dest) are DTensors needing their local shard.
                master = state["master_param"].view(-1)
                ea = state["exp_avg"].view(-1)
                es = state["exp_avg_sq"].view(-1)
                grad = to_local_tensor(param.grad)
                assert grad.is_cuda, "pipelined AdamW expects CUDA grads"
                grad = grad.detach().view(-1)
                dest = local_param_shard(param).view(-1)
                step = torch.tensor(float(state["step"]), dtype=torch.float32, device=master.device)
                pos = 0  # left endpoint of the param's not-yet-tiled remainder
                while pos < master.numel():
                    take = min(slot_numel - filled, master.numel() - pos)
                    frags.append(_PipelineFrag(param, master, ea, es, grad, dest, step,
                                               pos, pos + take, filled, take))
                    pos += take
                    filled += take
                    if filled == slot_numel:
                        slots.append(_PipelineSlot(lr, wd, device, frags))
                        frags, filled = [], 0
            if frags:
                slots.append(_PipelineSlot(lr, wd, device, frags))
        if not slots:
            return

        # Grad D2H runs on the down-lane stream; it must not start before the bwd
        # (default) stream has produced the grads. H2D writeback runs on the up-lane.
        default_stream = torch.cuda.current_stream()
        d2h_stream.wait_stream(default_stream)
        gpu_free: list[torch.Tensor] = []  # reusable fp32 GPU staging, bounded by depth

        # Stage 1: async grad D2H of one slot's fragments into a single pinned fp32
        # buffer on the down-lane. The bf16->fp32 cast runs GPU-side in copy_.
        def issue_d2h(index: int) -> _D2HPlan:
            slot = slots[index]
            grad_buf = self._acquire_pinned(slot_numel, torch.float32)
            with torch.cuda.stream(d2h_stream):
                for f in slot.frags:
                    grad_buf[f.off : f.off + f.length].copy_(f.grad[f.start : f.stop], non_blocking=True)
            event = torch.cuda.Event()
            event.record(d2h_stream)
            return _D2HPlan(slot=slot, grad_buf=grad_buf, event=event)

        # Stage 3: writeback with GPU-side cast. H2D the pinned fp32 master slices to a
        # GPU staging buffer on the up-lane, then cast fp32->param dtype on the GPU
        # (same double-track as serial) straight into each param slice -- no CPU cast.
        def issue_h2d(plan: _D2HPlan) -> tuple[torch.cuda.Event, torch.Tensor]:
            slot = plan.slot
            device = slot.device
            assert device.type == "cuda", "pipelined writeback expects CUDA params"
            staging = gpu_free.pop() if gpu_free else torch.empty(
                slot_numel, dtype=torch.float32, device=device
            )
            with torch.cuda.stream(h2d_stream):
                for f in slot.frags:
                    staging[f.off : f.off + f.length].copy_(f.master[f.start : f.stop], non_blocking=True)
                for f in slot.frags:
                    src = staging[f.off : f.off + f.length]
                    f.dest[f.start : f.stop].copy_(self._master_as_param_dtype(src, f.param))
            event = torch.cuda.Event()
            event.record(h2d_stream)
            return event, staging

        num_slots = len(slots)
        prefetched = {i: issue_d2h(i) for i in range(min(depth, num_slots))}
        h2d_inflight: list[tuple[torch.cuda.Event, torch.Tensor]] = []

        for c in range(num_slots):
            plan = prefetched.pop(c)
            # Keep the window full: issue the next D2H before this slot's CPU kernel
            # so it runs concurrently with the (blocking) CPU compute.
            if c + depth < num_slots:
                prefetched[c + depth] = issue_d2h(c + depth)

            plan.event.synchronize()  # CPU must see landed grads before reading
            slot, grad_buf = plan.slot, plan.grad_buf
            frags = slot.frags
            torch._fused_adamw_(  # type: ignore[attr-defined]  # private fused kernel, no public API
                [f.master[f.start : f.stop] for f in frags],
                [grad_buf[f.off : f.off + f.length] for f in frags],
                [f.ea[f.start : f.stop] for f in frags],
                [f.es[f.start : f.stop] for f in frags],
                [], [f.step for f in frags],
                lr=slot.lr, beta1=beta1, beta2=beta2,
                weight_decay=slot.wd, eps=self.eps, amsgrad=False, maximize=False,
            )
            self._release_pinned(grad_buf)  # blocking CPU op done: grad fully read
            h2d_inflight.append(issue_h2d(plan))

            while len(h2d_inflight) > depth:  # bound in-flight GPU staging
                event, staging = h2d_inflight.pop(0)
                event.synchronize()
                gpu_free.append(staging)

        for event, staging in h2d_inflight:  # drain writeback before returning
            event.synchronize()
        # Every h2d_stream copy is covered by a synchronized event above, so the stream
        # is fully drained -- no default_stream.wait_stream(h2d_stream) needed.

        if free_pin_after_opt:
            # Release the pinned pool so host RAM is free for fwd/bwd (re-allocated
            # next step). Opt-in for memory-constrained hosts; costs re-pin latency.
            self._pin_pool.clear()

    def _prepare_grad(self, grad: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        # No .to(float32): add_()/addcmul_() promote into the FP32 accumulators, so a
        # full-size cast here would only cost peak memory.
        if self.cpu_update:
            return to_local_tensor(grad).detach().to(device=master.device)
        return grad.detach()

    def _copy_master_to_param(self, param: nn.Parameter, master: torch.Tensor) -> None:
        value = self._master_as_param_dtype(master, param)
        if not self.cpu_update:
            param.detach().copy_(value)
            return
        copy_local_tensor_to_param_(param, value)

    def _master_as_param_dtype(self, master: torch.Tensor, param: nn.Parameter) -> torch.Tensor:
        """Cast an fp32 master to the param's storage dtype via the FSDP2 double-track
        (fp32 -> logical model dtype -> param dtype). The intermediate rounding, when
        model dtype != param dtype, is load-bearing; serial and pipelined share this."""
        model_dtype = self._model_param_dtype(param)
        if model_dtype is not None:
            master = master.to(dtype=model_dtype)
        return master.to(dtype=param.dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "fp32_adamw",
            "step_count": self.step_count,
            "master_params": [self.state[param]["master_param"] for param in self.params],
            "exp_avgs": [self.state[param]["exp_avg"] for param in self.params],
            "exp_avg_sqs": [self.state[param]["exp_avg_sq"] for param in self.params],
            "steps": [int(self.state[param]["step"]) for param in self.params],
            "weight_decays": [
                float(group.get("weight_decay", self.weight_decay))
                for group in self.param_groups
                for _param in group["params"]
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("type") != "fp32_adamw":
            raise ValueError("Invalid FP32 AdamW state_dict.")
        self.step_count = int(state_dict.get("step_count", 0))
        for target_name, key in (
            ("master_params", "master_param"),
            ("exp_avgs", "exp_avg"),
            ("exp_avg_sqs", "exp_avg_sq"),
        ):
            loaded = state_dict.get(target_name)
            if not isinstance(loaded, list) or len(loaded) != len(self.params):
                raise ValueError(f"Invalid FP32 AdamW {target_name} state.")
            for param, src in zip(self.params, loaded, strict=True):
                target = self.state[param][key]
                local_target = to_local_tensor(target)
                local_src = to_local_tensor(src).to(
                    device=local_target.device, dtype=local_target.dtype
                )
                local_target.copy_(local_src)
        loaded_steps = state_dict.get("steps")
        if loaded_steps is not None:
            if not isinstance(loaded_steps, list) or len(loaded_steps) != len(self.params):
                raise ValueError("Invalid FP32 AdamW steps state.")
            for param, step in zip(self.params, loaded_steps, strict=True):
                self.state[param]["step"] = int(step)
        else:
            for param in self.params:
                self.state[param]["step"] = self.step_count
        loaded_weight_decays = state_dict.get("weight_decays")
        if loaded_weight_decays is not None:
            if not isinstance(loaded_weight_decays, list) or len(loaded_weight_decays) != len(
                self.params
            ):
                raise ValueError("Invalid FP32 AdamW weight_decay state.")
            idx = 0
            for group in self.param_groups:
                if not group["params"]:
                    continue
                group["weight_decay"] = float(loaded_weight_decays[idx])
                idx += len(group["params"])

        for param in self.params:
            self._copy_master_to_param(param, self.state[param]["master_param"])


def build_adamw_optimizer(
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
    *,
    all_params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    foreach: bool | str,
    use_fp32_master: bool,
    cpu_update: bool,
    model_param_dtypes: dict[int, torch.dtype] | None,
    opt,
) -> Any:
    param_groups = normalize_param_groups(params, default_weight_decay=weight_decay)
    fused_adam = maybe_build_te_fused_adam_optimizer(
        param_groups,
        all_params=all_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        opt=opt,
        use_fp32_master=use_fp32_master,
    )
    if fused_adam is not None:
        return fused_adam
    if use_fp32_master:
        return FP32AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            cpu_update=cpu_update,
            # Opt-in: pipelined CPU-offload path (private torch._fused_adamw_, pinned
            # masters, extra CUDA streams + GPU staging). Off unless explicitly enabled
            # via overlap_cpu_optimizer_d2h_h2d=True and torch exposes the CPU fused
            # kernel; otherwise fall back to the serial path.
            pipeline=cpu_update and hasattr(torch, "_fused_adamw_")
            and get_bool_opt(opt, "overlap_cpu_optimizer_d2h_h2d", default=False),
            model_param_dtypes=model_param_dtypes,
        )
    if foreach not in {True, False, "auto"}:
        raise ValueError(f"adamw_foreach must be True, False, or 'auto', got {foreach!r}.")
    if foreach is False:
        return torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=False
        )

    dtensor_param_groups, tensor_param_groups = split_dtensor_and_tensor_param_groups(
        param_groups, default_weight_decay=weight_decay
    )
    split_param_groups = [group for group in (dtensor_param_groups, tensor_param_groups) if group]
    if foreach == "auto" and not dtensor_param_groups:
        return torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=False
        )
    if len(split_param_groups) <= 1:
        return torch.optim.AdamW(
            split_param_groups[0] if split_param_groups else param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            foreach=True,
        )
    return ChainedOptimizer(
        torch.optim.AdamW(
            group, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps, foreach=True
        )
        for group in split_param_groups
    )


def maybe_build_te_fused_adam_optimizer(
    param_groups: list[dict[str, Any]],
    *,
    all_params: Iterable[nn.Parameter],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    opt,
    use_fp32_master: bool,
) -> Any | None:
    if not get_bool_opt(opt, "fsdp2_use_te_fused_adam", default=False):
        return None
    from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam

    all_param_list = list(all_params)
    master_weights = get_bool_opt(
        opt, "master_weights", default=use_fp32_master and should_use_master_weights(all_param_list)
    )
    kwargs = dict(
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        adam_w_mode=True,
        master_weights=master_weights,
        master_weight_dtype=get_dtype_opt(opt, "master_weight_dtype", default=torch.float32),
        store_param_remainders=get_bool_opt(opt, "store_param_remainders", default=master_weights),
        exp_avg_dtype=get_dtype_opt(opt, "exp_avg_dtype", default=torch.float32),
        exp_avg_sq_dtype=get_dtype_opt(opt, "exp_avg_sq_dtype", default=torch.float32),
    )
    return FusedAdam(param_groups, **filter_supported_kwargs(FusedAdam.__init__, kwargs))


def filter_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def should_use_master_weights(params: Iterable[nn.Parameter]) -> bool:
    return any(param.is_floating_point() and param.dtype is not torch.float32 for param in params)


def get_bool_opt(opt, attr: str, *, default: bool) -> bool:
    value = get_opt_value(opt, attr)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_dtype_opt(opt, attr: str, *, default: torch.dtype) -> torch.dtype:
    value = get_opt_value(opt, attr)
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    name = str(value).removeprefix("torch.")
    resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported dtype for FSDP2 TE FusedAdam: {value!r}.")
    return resolved


def get_opt_value(opt, attr: str):
    if opt is None:
        return None
    if isinstance(opt, dict):
        value = opt.get(attr)
        override = opt.get("override_optimizer_config")
    else:
        value = getattr(opt, attr, None)
        override = getattr(opt, "override_optimizer_config", None)
    if value is not None:
        return value
    if isinstance(override, dict):
        return override.get(attr)
    return None


def normalize_param_groups(
    params: Iterable[nn.Parameter] | Iterable[dict[str, Any]], *, default_weight_decay: float
) -> list[dict[str, Any]]:
    items = list(params)
    if not items:
        return []
    if all(isinstance(item, dict) for item in items):
        groups: list[dict[str, Any]] = []
        for item in items:
            group = dict(item)
            group_params = list(group.get("params", ()))
            if not group_params:
                continue
            group["params"] = group_params
            group.setdefault("weight_decay", default_weight_decay)
            groups.append(group)
        return groups
    return [{"params": items, "weight_decay": default_weight_decay}]


def split_dtensor_and_tensor_param_groups(
    param_groups: Iterable[dict[str, Any]], *, default_weight_decay: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dtensor_groups: list[dict[str, Any]] = []
    tensor_groups: list[dict[str, Any]] = []
    for group in param_groups:
        dtensor_params, tensor_params = split_dtensor_and_tensor_params(group["params"])
        metadata = {key: value for key, value in group.items() if key != "params"}
        metadata.setdefault("weight_decay", default_weight_decay)
        if dtensor_params:
            dtensor_groups.append({**metadata, "params": dtensor_params})
        if tensor_params:
            tensor_groups.append({**metadata, "params": tensor_params})
    return dtensor_groups, tensor_groups


def split_dtensor_and_tensor_params(
    params: Iterable[nn.Parameter],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    dtensor_params: list[nn.Parameter] = []
    tensor_params: list[nn.Parameter] = []
    for param in params:
        if is_dtensor_like(param):
            dtensor_params.append(param)
        else:
            tensor_params.append(param)
    return dtensor_params, tensor_params


def iter_torch_optimizers(optimizer: Any) -> Iterable[torch.optim.Optimizer]:
    if isinstance(optimizer, ChainedOptimizer):
        yield from optimizer.optimizers
    else:
        yield optimizer


def dtensor_from_local(
    local_tensor: torch.Tensor,
    device_mesh: Any,
    placements: Any,
    *,
    shape: Any = None,
    stride: Any = None,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    # ``DTensor.from_local`` infers the global shape as local_shard * mesh, which
    # is WRONG for unevenly-sharded params (FSDP2 pads the last shard). Pass the
    # original global shape/stride so the round-trip is exact for any dim not
    # divisible by the mesh size.
    if shape is not None:
        return DTensor.from_local(
            local_tensor, device_mesh, placements, shape=shape, stride=stride
        )
    return DTensor.from_local(local_tensor, device_mesh, placements)


__all__ = [
    "all_reduce_grad_",
    "build_adamw_optimizer",
    "copy_local_tensor_to_param_",
    "dtensor_from_local",
    "filter_supported_kwargs",
    "fsdp2_model_param_dtype",
    "get_bool_opt",
    "get_dtype_opt",
    "get_opt_value",
    "has_dtensor_grad_or_param",
    "is_dtensor_like",
    "iter_torch_optimizers",
    "local_grad_sq_sum",
    "normalize_param_groups",
    "to_local_tensor",
]

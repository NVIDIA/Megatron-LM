"""
DCP (Distributed Checkpoint) framework for training checkpoints.

Model-agnostic: takes a placement function to describe how each parameter is sharded.
HF weight loading/saving is model-specific and lives in models/<name>/checkpoint.py.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.distributed.checkpoint as dcp  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
from torch.distributed.device_mesh import DeviceMesh  # pyright: ignore[reportMissingImports]
from torch.distributed.tensor import DTensor  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.protocols import (
    ExpertClassifierFn,
    PlacementFn,
    default_expert_classifier,
    default_placement_fn,
)


def save_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int | str,
    path: str | None = None,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = None,
    save_rng: bool = True,
) -> None:
    """Save training checkpoint using DTensor + DCP for automatic resharding."""
    if path is None and isinstance(step, str):
        path = step
        step = 0
    if path is None:
        raise ValueError("checkpoint path is required")
    step = int(step)
    if use_dcp is None:
        use_dcp = config is not None and ps is not None
    if not use_dcp:
        _save_local_training_checkpoint(model, optimizer, step, path, save_rng=save_rng)
        return
    if config is None or ps is None:
        raise ValueError("DCP checkpointing requires config and ParallelState.")
    if not isinstance(model, nn.Module):
        raise TypeError("DCP checkpointing currently expects a single nn.Module.")
    dense_mesh, expert_mesh = _build_meshes(config)
    state_dict: dict = {"step": step}

    for name, param in model.named_parameters():
        placements = get_placements(name)
        mesh = expert_mesh if is_expert(name) else dense_mesh
        state_dict[f"model.{name}"] = DTensor.from_local(
            _to_local_tensor(param.data.detach()),
            mesh,
            placements,
        )

    ckpt_path = os.path.join(path, f"step_{step}")
    dcp.save(state_dict, checkpoint_id=ckpt_path)
    if save_rng:
        _save_rng_sidecar(ckpt_path)
    log_rank0(f"Saved training checkpoint at step {step} to {ckpt_path}")


def load_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    config=None,
    ps: ParallelState | None = None,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
    *,
    use_dcp: bool | None = None,
    load_rng: bool = True,
) -> int:
    """Load training checkpoint with automatic resharding across different parallel configs."""
    if use_dcp is None:
        use_dcp = config is not None and ps is not None
    if not use_dcp:
        return _load_local_training_checkpoint(model, optimizer, path, load_rng=load_rng)
    if config is None or ps is None:
        raise ValueError("DCP checkpointing requires config and ParallelState.")
    if not isinstance(model, nn.Module):
        raise TypeError("DCP checkpointing currently expects a single nn.Module.")
    dense_mesh, expert_mesh = _build_meshes(config)

    ckpt_path = path
    step_dirs = sorted(
        [d for d in os.listdir(path) if d.startswith("step_")],
        key=lambda d: int(d.split("_")[1]),
    )
    if step_dirs:
        ckpt_path = os.path.join(path, step_dirs[-1])

    state_dict: dict = {"step": 0}

    for name, param in model.named_parameters():
        placements = get_placements(name)
        mesh = expert_mesh if is_expert(name) else dense_mesh
        state_dict[f"model.{name}"] = DTensor.from_local(
            torch.empty_like(_to_local_tensor(param.data)),
            mesh,
            placements,
        )

    dcp.load(state_dict, checkpoint_id=ckpt_path)

    for name, param in model.named_parameters():
        key = f"model.{name}"
        if key in state_dict:
            t = state_dict[key]
            with torch.no_grad():
                _copy_tensor_(param.data, t.to_local() if isinstance(t, DTensor) else t)

    step = state_dict.get("step", 0)
    if load_rng:
        _load_rng_sidecar(ckpt_path)
    log_rank0(f"Loaded training checkpoint from {path} at step {step}")
    return step


def _model_chunks(model: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    if isinstance(model, nn.Module):
        return [model]
    chunks = list(model)
    if not all(isinstance(chunk, nn.Module) for chunk in chunks):
        raise TypeError("checkpoint model chunks must be nn.Module instances.")
    return chunks


def _to_local_tensor(tensor: Any) -> torch.Tensor:
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _copy_tensor_(target: torch.Tensor, src: torch.Tensor) -> None:
    local_target = _to_local_tensor(target)
    local_src = _to_local_tensor(src).to(
        device=local_target.device,
        dtype=local_target.dtype,
    )
    if isinstance(local_target, torch.Tensor) and local_target is not target:
        local_target.copy_(local_src)
    else:
        target.copy_(local_src)


def _chunk_tensor_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        state[f"param.{name}"] = _to_local_tensor(param.detach()).cpu().clone()
    for name, buffer in module.named_buffers():
        state[f"buffer.{name}"] = _to_local_tensor(buffer.detach()).cpu().clone()
    return state


def _load_chunk_tensor_state(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    missing: list[str] = []
    for key, src in state.items():
        kind, name = key.split(".", 1)
        if kind == "param" and name in params:
            with torch.no_grad():
                _copy_tensor_(params[name], src)
        elif kind == "buffer" and name in buffers:
            with torch.no_grad():
                _copy_tensor_(buffers[name], src)
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(f"checkpoint contains unknown tensor keys: {missing}")


def _local_checkpoint_file(path: str | os.PathLike[str]) -> Path:
    ckpt_path = Path(path)
    if ckpt_path.is_dir() or ckpt_path.suffix == "":
        return ckpt_path / "training_state.pt"
    return ckpt_path


def _local_optimizer_parameter_state_file(ckpt_file: Path) -> Path:
    return ckpt_file.with_name(f"{ckpt_file.stem}.optimizer_parameter_state{ckpt_file.suffix}")


def _rank_suffix() -> str:
    if dist.is_available() and dist.is_initialized():
        return f"rank_{dist.get_rank():05d}"
    return "rank_00000"


def _rng_sidecar_file(path: str | os.PathLike[str]) -> Path:
    return Path(path) / f"rng_state_{_rank_suffix()}.pt"


def _cpu_clone(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().clone()


def _get_cuda_rng_state() -> torch.Tensor | None:
    if not torch.cuda.is_initialized():
        return None
    return _cpu_clone(torch.cuda.get_rng_state())


def _get_cuda_rng_tracker_states() -> dict[str, torch.Tensor]:
    if not torch.cuda.is_initialized():
        return {}
    try:
        from megatron.core import tensor_parallel

        states = tensor_parallel.get_cuda_rng_tracker().get_states()
    except Exception:
        return {}
    return {name: _cpu_clone(state) for name, state in states.items() if state is not None}


def _get_rng_state() -> dict[str, Any]:
    return {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": _cpu_clone(torch.get_rng_state()),
        "cuda_rng_state": _get_cuda_rng_state(),
        "rng_tracker_states": _get_cuda_rng_tracker_states(),
    }


def _restore_cuda_rng_tracker_states(states: dict[str, torch.Tensor]) -> None:
    if not states or not torch.cuda.is_initialized():
        return
    try:
        from megatron.core import tensor_parallel

        tracker = tensor_parallel.get_cuda_rng_tracker()
        graph_safe = tensor_parallel.is_graph_safe_cuda_rng_tracker(tracker)
        restored = {
            name: tensor_parallel.convert_cuda_rng_state(state, to_graphable=graph_safe)
            for name, state in states.items()
        }
        tracker.set_states(restored)
    except Exception as exc:
        raise RuntimeError("Failed to restore Megatron tensor-parallel RNG tracker state.") from exc


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_state = state.get("cuda_rng_state")
    if cuda_rng_state is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state(cuda_rng_state)
    _restore_cuda_rng_tracker_states(state.get("rng_tracker_states", {}))


def _save_rng_sidecar(path: str | os.PathLike[str]) -> None:
    rng_file = _rng_sidecar_file(path)
    rng_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_get_rng_state(), rng_file)


def _load_rng_sidecar(path: str | os.PathLike[str]) -> None:
    rng_file = _rng_sidecar_file(path)
    if not rng_file.exists():
        log_rank0(f"RNG sidecar not found at {rng_file}; skipping RNG restore.")
        return
    _restore_rng_state(torch.load(rng_file, map_location="cpu", weights_only=False))


def _save_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    step: int,
    path: str,
    *,
    save_rng: bool = True,
) -> None:
    chunks = _model_chunks(model)
    ckpt_file = _local_checkpoint_file(path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    save_parameter_state = getattr(optimizer, "save_parameter_state", None)
    optimizer_parameter_state_file = (
        _local_optimizer_parameter_state_file(ckpt_file) if callable(save_parameter_state) else None
    )
    state = {
        "format": "megatron_lite.local_training.v1",
        "step": int(step),
        "model": [_chunk_tensor_state(chunk) for chunk in chunks],
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_parameter_state": (
            optimizer_parameter_state_file.name
            if optimizer_parameter_state_file is not None
            else None
        ),
        "rng_state": _get_rng_state() if save_rng else None,
    }
    torch.save(state, ckpt_file)
    if optimizer_parameter_state_file is not None:
        save_parameter_state(str(optimizer_parameter_state_file))
    log_rank0(f"Saved local training checkpoint at step {step} to {ckpt_file}")


def _load_local_training_checkpoint(
    model: nn.Module | Iterable[nn.Module],
    optimizer,
    path: str,
    *,
    load_rng: bool = True,
) -> int:
    ckpt_file = _local_checkpoint_file(path)
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if state.get("format") != "megatron_lite.local_training.v1":
        raise RuntimeError(f"Unsupported local checkpoint format in {ckpt_file}")
    chunks = _model_chunks(model)
    chunk_states = state.get("model")
    if not isinstance(chunk_states, list) or len(chunk_states) != len(chunks):
        raise RuntimeError("Checkpoint model chunk count does not match target model.")
    for chunk, chunk_state in zip(chunks, chunk_states, strict=True):
        _load_chunk_tensor_state(chunk, chunk_state)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
        parameter_state_name = state.get("optimizer_parameter_state")
        load_parameter_state = getattr(optimizer, "load_parameter_state", None)
        if parameter_state_name is not None and callable(load_parameter_state):
            load_parameter_state(str(ckpt_file.with_name(parameter_state_name)))
        else:
            reload_model_params = getattr(optimizer, "reload_model_params", None)
            if callable(reload_model_params):
                reload_model_params()
    if load_rng:
        _restore_rng_state(state.get("rng_state"))
    step = int(state.get("step", 0))
    log_rank0(f"Loaded local training checkpoint from {ckpt_file} at step {step}")
    return step


def _build_meshes(config):
    """Build separate meshes for dense and expert parameters.

    Dense mesh  [PP, DP, CP, TP]  — matches init_parallel dense decomposition.
    Expert mesh [PP, EDP, EP, ETP] — matches init_parallel expert decomposition.

    Both meshes use C-order layout so the innermost (rightmost) dimension
    corresponds to the fastest-changing rank index, consistent with
    init_parallel's rank = (...) * inner_size + inner_rank formula.
    """
    ws = dist.get_world_size()
    tp = int(config.tp or 1)
    ep = int(config.ep or 1)
    etp = max(int(config.etp or 1), 1)
    cp = max(int(config.cp or 1), 1)
    pp = max(int(config.pp or 1), 1)

    dense_dp = ws // (tp * cp * pp)
    expert_dp = ws // (etp * ep * pp)

    ranks = torch.arange(ws)
    dense_mesh = DeviceMesh("cuda", ranks.reshape(pp, dense_dp, cp, tp))
    expert_mesh = DeviceMesh("cuda", ranks.reshape(pp, expert_dp, ep, etp))
    return dense_mesh, expert_mesh


def log_rank0(msg: str) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[megatron.lite] {msg}", flush=True)

# ======================================================================
# QKV / FC1 canonicalize for DCP (interleaved-TP ↔ canonical layout)
# ======================================================================


def _ag(data, size, group, dim=0):
    from megatron.lite.primitive.ckpt.hf_weights import allgather_concat

    return allgather_concat(data, size, group, dim)


def canonicalize_qkv_for_dcp(model, num_attention_heads, num_key_value_heads, head_dim, ps):
    """Rearrange fused QKV from interleaved-TP to canonical (Q|K|V) for DCP save."""
    if ps.tp_size <= 1:
        return
    from megatron.lite.primitive.utils import ensure_divisible

    nq = ensure_divisible(num_attention_heads, ps.tp_size) * head_dim
    nkv = ensure_divisible(num_key_value_heads, ps.tp_size) * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        cs = param.data.shape[0]
        q, k, v = [], [], []
        for r in range(ps.tp_size):
            s = full[r * cs : (r + 1) * cs]
            q.append(s[:nq])
            k.append(s[nq : nq + nkv])
            v.append(s[nq + nkv :])
        canon = torch.cat([torch.cat(q), torch.cat(k), torch.cat(v)], dim=0)
        param.data.copy_(canon.chunk(ps.tp_size, dim=0)[ps.tp_rank])


def decanon_qkv_after_dcp(model, num_attention_heads, num_key_value_heads, head_dim, ps):
    """Reverse of canonicalize_qkv_for_dcp."""
    if ps.tp_size <= 1:
        return
    qs = num_attention_heads * head_dim
    kvs = num_key_value_heads * head_dim
    for name, param in model.named_parameters():
        if "qkv" not in name or "layer_norm" in name:
            continue
        full = _ag(param.data, ps.tp_size, ps.tp_group)
        ql = full[:qs].chunk(ps.tp_size)[ps.tp_rank]
        kl = full[qs : qs + kvs].chunk(ps.tp_size)[ps.tp_rank]
        vl = full[qs + kvs :].chunk(ps.tp_size)[ps.tp_rank]
        param.data.copy_(torch.cat([ql, kl, vl], dim=0))


def canonicalize_fc1_for_dcp(model, ps):
    """Rearrange fused gate-up FC1 from interleaved-ETP to canonical for DCP save."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        cs = param.data.shape[0]
        ffn = cs // 2
        g, u = [], []
        for r in range(ps.etp_size):
            s = full[r * cs : (r + 1) * cs]
            g.append(s[:ffn])
            u.append(s[ffn:])
        canon = torch.cat([torch.cat(g), torch.cat(u)], dim=0)
        param.data.copy_(canon.chunk(ps.etp_size, dim=0)[ps.etp_rank])


def decanon_fc1_after_dcp(model, ps):
    """Reverse of canonicalize_fc1_for_dcp."""
    if ps.etp_size <= 1:
        return
    for name, param in model.named_parameters():
        if "experts" not in name or "fc1" not in name:
            continue
        full = _ag(param.data, ps.etp_size, ps.etp_group)
        ffn = full.shape[0] // 2
        gl = full[:ffn].chunk(ps.etp_size)[ps.etp_rank]
        ul = full[ffn:].chunk(ps.etp_size)[ps.etp_rank]
        param.data.copy_(torch.cat([gl, ul], dim=0))


__all__ = [
    "canonicalize_fc1_for_dcp",
    "canonicalize_qkv_for_dcp",
    "decanon_fc1_after_dcp",
    "decanon_qkv_after_dcp",
    "load_training_checkpoint",
    "save_training_checkpoint",
]

"""
DCP (Distributed Checkpoint) framework for training checkpoints.

Model-agnostic: takes a placement function to describe how each parameter is sharded.
HF weight loading/saving is model-specific and lives in models/<name>/checkpoint.py.
"""

from __future__ import annotations

import os

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
    model: nn.Module,
    optimizer,
    step: int,
    path: str,
    config,
    ps: ParallelState,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
) -> None:
    """Save training checkpoint using DTensor + DCP for automatic resharding."""
    dense_mesh, expert_mesh = _build_meshes(config)
    state_dict: dict = {"step": step}

    for name, param in model.named_parameters():
        placements = get_placements(name)
        mesh = expert_mesh if is_expert(name) else dense_mesh
        state_dict[f"model.{name}"] = DTensor.from_local(param.data.detach(), mesh, placements)

    ckpt_path = os.path.join(path, f"step_{step}")
    dcp.save(state_dict, checkpoint_id=ckpt_path)
    log_rank0(f"Saved training checkpoint at step {step} to {ckpt_path}")


def load_training_checkpoint(
    model: nn.Module,
    optimizer,
    path: str,
    config,
    ps: ParallelState,
    get_placements: PlacementFn = default_placement_fn,
    is_expert: ExpertClassifierFn = default_expert_classifier,
) -> int:
    """Load training checkpoint with automatic resharding across different parallel configs."""
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
        state_dict[f"model.{name}"] = DTensor.from_local(torch.empty_like(param.data), mesh, placements)

    dcp.load(state_dict, checkpoint_id=ckpt_path)

    for name, param in model.named_parameters():
        key = f"model.{name}"
        if key in state_dict:
            t = state_dict[key]
            param.data.copy_(t.to_local() if isinstance(t, DTensor) else t)

    step = state_dict.get("step", 0)
    log_rank0(f"Loaded training checkpoint from {path} at step {step}")
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
    tp, ep = config.tp, config.ep
    etp = max(config.etp, 1)
    cp = max(config.cp, 1)
    pp = max(config.pp, 1)

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
    from megatron.lite.primitive.ckpt.hf_bridge import allgather_concat

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

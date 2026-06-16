# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF ↔ Megatron Lite weight conversion.

Everything needed for HuggingFace safetensors ↔ Megatron Lite model conversion:
- HFWeights protocol (model implements this)
- SafeTensorReader / save_safetensors (file I/O)
- Tensor utilities (split_dim, allgather_concat, remap_layer_index, ...)
- Generic load_hf_weights / export_hf_weights / save_hf_weights (orchestration)
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file as _safe_save


@runtime_checkable
class HFWeights(Protocol):
    """Protocol for HF ↔ Megatron Lite weight conversion.

    Model-specific implementations only do tensor math, never distributed comm.
    """

    def weight_map(self) -> dict[str, list[str]]:
        """Megatron Lite param name → [HF param names]. Multiple = concat (QKV, gate+up)."""
        ...

    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        """Convert HF tensors → single Megatron Lite tensor (e.g. merge QKV)."""
        ...

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert Megatron Lite tensor → [(hf_name, hf_tensor)] (e.g. split QKV back)."""
        ...

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        """TP sharding: ``(split_dim, 0=TP|1=ETP)``, or None if replicated."""
        ...

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        """If native_name is a fused QKV weight, return (num_q_heads, num_kv_heads, head_dim).

        Needed for correct GQA TP sharding — Q/K/V must be split independently.
        Return None for non-QKV parameters.
        """
        return None

    @property
    def num_experts(self) -> int:
        """Total number of experts (needed for EP gather index math)."""
        ...

    def is_expert(self, native_name: str) -> bool:
        """Whether this param belongs to an expert (for EP sharding)."""
        ...

    def expert_global_id(self, native_name: str) -> int | None:
        """Global expert ID from synthetic name. None if not expert."""
        ...

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        """Synthetic expert name → actual model param name."""
        ...


# ======================================================================
# SafeTensors I/O
# ======================================================================


class SafeTensorReader:
    """Read individual tensors from an HF safetensors directory."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.index = self._load_index()

    def _load_index(self) -> dict[str, str]:
        idx_file = self.path / "model.safetensors.index.json"
        if idx_file.exists():
            with open(idx_file) as f:
                return json.load(f)["weight_map"]
        return {}

    def get_tensor(self, name: str) -> torch.Tensor:
        if self.index:
            filepath = self.path / self.index[name]
        else:
            filepath = self.path / "model.safetensors"
        with safe_open(str(filepath), framework="pt", device="cpu") as f:
            return f.get_tensor(name)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip nested wrapper modules like DDP -> model."""
    base_model = model
    seen: set[int] = set()
    while hasattr(base_model, "module"):
        ident = id(base_model)
        if ident in seen:
            break
        seen.add(ident)
        next_model = base_model.module
        if not isinstance(next_model, nn.Module) or next_model is base_model:
            break
        base_model = next_model
    return base_model


def save_safetensors(
    tensors: dict[str, torch.Tensor], path: str, filename: str = "model.safetensors"
) -> None:
    os.makedirs(path, exist_ok=True)
    _safe_save(tensors, os.path.join(path, filename))


def _resolve_export_dtype(export_dtype: str | torch.dtype | None) -> torch.dtype | None:
    if export_dtype is None:
        return None
    if isinstance(export_dtype, torch.dtype):
        return export_dtype
    normalized = str(export_dtype).lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported export_dtype={export_dtype!r}")
    return aliases[normalized]


def _cast_export_tensor(tensor: torch.Tensor, export_dtype: torch.dtype | None) -> torch.Tensor:
    if export_dtype is None or not tensor.is_floating_point():
        return tensor
    return tensor.to(dtype=export_dtype)


# ======================================================================
# Tensor utilities
# ======================================================================


def split_dim(tensor: torch.Tensor, rank: int, world: int, dim: int = 0) -> torch.Tensor:
    if world <= 1:
        return tensor
    return tensor.chunk(world, dim=dim)[rank].contiguous()


def split_qkv(
    tensor: torch.Tensor, rank: int, world: int, num_q_heads: int, num_kv_heads: int, head_dim: int
) -> torch.Tensor:
    """TP-shard a fused [Q, K, V] weight, splitting Q/K/V heads independently.

    Naive ``split_dim`` would slice across the Q/K/V boundary incorrectly
    when num_q_heads != num_kv_heads (GQA).
    """
    if world <= 1:
        return tensor
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q = tensor[:q_size]
    k = tensor[q_size : q_size + kv_size]
    v = tensor[q_size + kv_size :]
    q_shard = q.chunk(world, dim=0)[rank]
    k_shard = k.chunk(world, dim=0)[rank]
    v_shard = v.chunk(world, dim=0)[rank]
    return torch.cat([q_shard, k_shard, v_shard], dim=0).contiguous()


def split_gate_up(tensor: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if world <= 1:
        return tensor
    ffn = tensor.shape[0] // 2
    gate = tensor[:ffn].chunk(world, dim=0)[rank]
    up = tensor[ffn:].chunk(world, dim=0)[rank]
    return torch.cat([gate, up], dim=0).contiguous()


def allgather_concat(
    tensor: torch.Tensor, world_size: int, group: dist.ProcessGroup | None, dim: int
) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def remap_layer_index(name: str, global_to_local: dict[int, int]) -> str | None:
    if not global_to_local:
        return name
    m = re.match(r"(layers\.)(\d+)(\..*)", name)
    if not m:
        return name
    gidx = int(m.group(2))
    if gidx not in global_to_local:
        return None
    return f"{m.group(1)}{global_to_local[gidx]}{m.group(3)}"


def extract_layer_idx(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else 0


def parse_expert_idx(name: str) -> int:
    m = re.search(r"weight(\d+)$", name)
    return int(m.group(1)) if m else 0


def set_expert_idx(name: str, idx: int) -> str:
    return re.sub(r"weight\d+$", f"weight{idx}", name)


def to_global_layer_name(name: str, layer_map: dict[int, int]) -> str:
    if not layer_map:
        return name

    def _replace(m: re.Match) -> str:
        return f"layers.{layer_map.get(int(m.group(1)), int(m.group(1)))}."

    return re.sub(r"layers\.(\d+)\.", _replace, name)


def gather_gate_up(tensor: torch.Tensor, world_size: int, group: dist.ProcessGroup) -> torch.Tensor:
    ffn_local = tensor.shape[0] // 2
    gate_full = allgather_concat(tensor[:ffn_local], world_size, group, dim=0)
    up_full = allgather_concat(tensor[ffn_local:], world_size, group, dim=0)
    return torch.cat([gate_full, up_full], dim=0)


# ======================================================================
# Generic load / export / save using HFWeights
# ======================================================================


def load_hf_weights(
    model: nn.Module, hf_path: str, spec: HFWeights, ps, *, vocab_size: int | None = None
) -> None:
    """Load HF safetensors into a Megatron Lite model using HFWeights.

    Handles PP layer filtering, TP split, EP shard assignment.
    ``ps`` is a ParallelState (lazy import to avoid GPU dep at module level).
    """
    from megatron.lite.primitive.parallel import pad_vocab_for_tp
    from megatron.lite.primitive.utils import log_rank0

    base_model = unwrap_model(model)
    reader = SafeTensorReader(hf_path)
    wmap = spec.weight_map()

    global_to_local: dict[int, int] = (
        {gi: li for li, gi in enumerate(base_model.layer_indices)}
        if hasattr(base_model, "layer_indices")
        else {}
    )

    state = base_model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    num_experts_total = getattr(spec, "num_experts", None)
    expert_shard = None
    if num_experts_total is None:
        expert_ids = [spec.expert_global_id(name) for name in wmap]
        expert_ids = [expert_id for expert_id in expert_ids if expert_id is not None]
        if expert_ids:
            num_experts_total = max(expert_ids) + 1
    if num_experts_total:
        from megatron.lite.primitive.utils import ensure_divisible

        experts_per_rank = ensure_divisible(num_experts_total, ps.ep_size)
        local_start = ps.ep_rank * experts_per_rank
        expert_shard = (experts_per_rank, local_start)

    for native_name, hf_names in wmap.items():
        mapped = remap_layer_index(native_name, global_to_local)
        if mapped is None:
            continue

        expert_gid = spec.expert_global_id(mapped)
        if expert_gid is not None:
            _load_expert_weight(
                mapped, hf_names, reader, spec, ps, loaded, expert_gid, expert_shard
            )
            continue

        hf_tensors = [reader.get_tensor(n) for n in hf_names]
        tensor = spec.hf_to_native(mapped, hf_tensors)

        tp_info = spec.tp_spec(mapped)
        if tp_info is not None:
            split_d, tp_or_etp = tp_info
            if tp_or_etp == 0:
                if vocab_size is not None and ("embed" in mapped or "head" in mapped):
                    padded = pad_vocab_for_tp(vocab_size, ps.tp_size)
                    if tensor.size(0) < padded:
                        pad = torch.zeros(
                            padded - tensor.size(0), *tensor.shape[1:], dtype=tensor.dtype
                        )
                        tensor = torch.cat([tensor, pad], dim=0)
                qkv = spec.qkv_spec(mapped) if hasattr(spec, "qkv_spec") else None
                if qkv is not None:
                    tensor = split_qkv(tensor, ps.tp_rank, ps.tp_size, *qkv)
                else:
                    tensor = split_dim(tensor, ps.tp_rank, ps.tp_size, dim=split_d)
            else:
                tensor = split_dim(tensor, ps.etp_rank, ps.etp_size, dim=split_d)

        actual = _resolve_param_name(mapped, state)
        if actual:
            loaded[actual] = tensor.to(dtype=torch.bfloat16)

    for name, param in base_model.named_parameters():
        if name in loaded:
            param.data.copy_(loaded[name])
        elif "lora" in name.lower() or "adapter" in name.lower():
            continue
        else:
            log_rank0(f"WARNING: {name} not loaded from checkpoint")


def _load_expert_weight(native_name, hf_names, reader, spec, ps, loaded, expert_gid, expert_shard):
    if expert_shard is None:
        raise RuntimeError("Expert weight encountered but expert shard metadata is unavailable.")
    experts_per_rank, local_start = expert_shard
    if expert_gid < local_start or expert_gid >= local_start + experts_per_rank:
        return

    hf_tensors = [reader.get_tensor(n) for n in hf_names]
    tensor = spec.hf_to_native(native_name, hf_tensors)

    if ps.etp_size > 1:
        tp_info = spec.tp_spec(native_name)
        if tp_info is not None:
            split_d, _ = tp_info
            if "fc1" in native_name:
                tensor = split_gate_up(tensor, ps.etp_rank, ps.etp_size)
            else:
                tensor = split_dim(tensor, ps.etp_rank, ps.etp_size, dim=split_d)

    loaded[spec.expert_local_name(native_name, expert_gid - local_start)] = tensor.to(
        dtype=torch.bfloat16
    )


def _resolve_param_name(name: str, state_dict: dict) -> str | None:
    if name in state_dict:
        return name
    for key in state_dict:
        if name in key:
            return key
    return None


def export_hf_weights(
    model: nn.Module | list[nn.Module],
    spec: HFWeights,
    ps,
    *,
    vocab_size: int | None = None,
    limit: int | None = None,
    rank0_only: bool = False,
    export_dtype: str | torch.dtype | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Export model weights as HF-format (name, tensor) pairs.

    Gathers across TP/ETP/EP/PP so the output is the full unsharded HF state on
    every participating rank. RL weight sync needs every colocated rollout rank
    to receive weights; save paths can pass ``rank0_only=True`` to avoid
    materializing duplicate writers.
    """
    if isinstance(model, nn.ModuleList):
        chunks: list[nn.Module] = list(model)
    elif isinstance(model, list):
        chunks = model
    else:
        chunks = [model]

    rank = dist.get_rank() if dist.is_initialized() else 0
    resolved_export_dtype = _resolve_export_dtype(export_dtype)

    if ps.pp_size <= 1:
        exported_params = 0
        expert_groups: dict[str, list[tuple[int, str, torch.Tensor]]] = {}
        for chunk in chunks:
            base_chunk = unwrap_model(chunk)
            layer_map = (
                {i: base_chunk.layer_indices[i] for i in range(len(base_chunk.layer_indices))}
                if hasattr(base_chunk, "layer_indices")
                else {}
            )
            for name, param in base_chunk.named_parameters():
                gname = to_global_layer_name(name, layer_map)
                tensor = param.data.detach()

                gathered_one: dict[str, torch.Tensor] = {}
                if spec.is_expert(gname):
                    if limit is None:
                        expert_groups.setdefault(_expert_group_key(gname), []).append(
                            (parse_expert_idx(gname), gname, tensor)
                        )
                        exported_params += 1
                        continue
                    _gather_expert(gname, tensor, spec, ps, gathered_one)
                else:
                    gathered_one[gname] = _gather_dense(gname, tensor, spec, ps)

                exported_params += 1
                if not rank0_only or rank == 0:
                    for native_name, gathered_tensor in gathered_one.items():
                        if vocab_size is not None and (
                            "embed" in native_name or "head" in native_name
                        ):
                            gathered_tensor = gathered_tensor[:vocab_size]
                        for hf_name, hf_tensor in spec.native_to_hf(native_name, gathered_tensor):
                            yield hf_name, _cast_export_tensor(hf_tensor, resolved_export_dtype)

                if limit is not None and exported_params >= limit:
                    return

        for group_key in sorted(expert_groups):
            gathered_group: dict[str, torch.Tensor] = {}
            _gather_expert_group(expert_groups[group_key], spec, ps, gathered_group)
            if not rank0_only or rank == 0:
                for native_name in sorted(gathered_group, key=parse_expert_idx):
                    gathered_tensor = gathered_group[native_name]
                    for hf_name, hf_tensor in spec.native_to_hf(native_name, gathered_tensor):
                        yield hf_name, _cast_export_tensor(hf_tensor, resolved_export_dtype)
        return

    gathered: dict[str, torch.Tensor] = {}
    for chunk in chunks:
        base_chunk = unwrap_model(chunk)
        # Map local layer indices to global for PP
        layer_map = (
            {i: base_chunk.layer_indices[i] for i in range(len(base_chunk.layer_indices))}
            if hasattr(base_chunk, "layer_indices")
            else {}
        )
        for name, param in base_chunk.named_parameters():
            gname = to_global_layer_name(name, layer_map)
            t = param.data.detach()

            if spec.is_expert(gname):
                _gather_expert(gname, t, spec, ps, gathered)
            else:
                gathered[gname] = _gather_dense(gname, t, spec, ps)

    # PP gather
    if ps.pp_size > 1:
        all_states: list[dict | None] = [None] * ps.pp_size
        dist.all_gather_object(all_states, gathered, group=ps.pp_group)
        gathered = {}
        for s in all_states:
            if s is not None:
                gathered.update(s)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return

    # Vocab trim
    if vocab_size is not None:
        for key in list(gathered.keys()):
            if "embed" in key or "head" in key:
                gathered[key] = gathered[key][:vocab_size]

    # Convert Megatron Lite names → HF names via spec
    for native_name, tensor in gathered.items():
        for hf_name, hf_tensor in spec.native_to_hf(native_name, tensor):
            yield hf_name, _cast_export_tensor(hf_tensor, resolved_export_dtype)


def _gather_dense(name: str, tensor: torch.Tensor, spec: HFWeights, ps) -> torch.Tensor:
    """Gather a dense (non-expert) param across TP."""
    custom_gather = getattr(spec, "gather_dense", None)
    if callable(custom_gather):
        gathered = custom_gather(name, tensor, ps)
        if gathered is not None:
            return gathered.cpu()

    tp_info = spec.tp_spec(name)
    if tp_info is not None and ps.tp_size > 1:
        split_d, tp_or_etp = tp_info
        if tp_or_etp == 0:
            tensor = allgather_concat(tensor, ps.tp_size, ps.tp_group, dim=split_d)
    return tensor.cpu()


def _gather_expert(
    name: str, tensor: torch.Tensor, spec: HFWeights, ps, out: dict[str, torch.Tensor]
) -> None:
    """Gather an expert param across ETP + EP."""
    tensor = _gather_expert_etp(name, tensor, spec, ps)

    # EP gather: global_id = ep_rank * n_local + local_id.
    local_idx = parse_expert_idx(name)
    if ps.ep_size > 1 and ps.ep_group is not None:
        n_local = spec.num_experts // ps.ep_size
        ep_gathered = [torch.empty_like(tensor) for _ in range(ps.ep_size)]
        dist.all_gather(ep_gathered, tensor.contiguous(), group=ps.ep_group)
        for ep_rank, ep_tensor in enumerate(ep_gathered):
            global_idx = ep_rank * n_local + local_idx
            out[set_expert_idx(name, global_idx)] = ep_tensor.cpu()
    else:
        out[name] = tensor.cpu()


def _gather_expert_etp(name: str, tensor: torch.Tensor, spec: HFWeights, ps) -> torch.Tensor:
    # ETP gather
    if ps.etp_size > 1 and ps.etp_group is not None:
        tp_info = spec.tp_spec(name)
        if tp_info is not None:
            split_d, _ = tp_info
            if "fc1" in name:
                return gather_gate_up(tensor, ps.etp_size, ps.etp_group)
            return allgather_concat(tensor, ps.etp_size, ps.etp_group, dim=split_d)
    return tensor


def _expert_group_key(name: str) -> str:
    return re.sub(r"weight\d+$", "weight", name)


def _gather_expert_group(
    entries: list[tuple[int, str, torch.Tensor]], spec: HFWeights, ps, out: dict[str, torch.Tensor]
) -> None:
    """Gather local experts in one EP collective per layer/kind."""
    prepared = [
        (local_idx, name, _gather_expert_etp(name, tensor, spec, ps))
        for local_idx, name, tensor in sorted(entries)
    ]
    packed_group_name = getattr(spec, "packed_expert_group_name", None)
    if callable(packed_group_name):
        packed_name = packed_group_name(prepared[0][1])
        if packed_name is not None:
            if ps.ep_size <= 1 or ps.ep_group is None:
                out[packed_name] = torch.stack(
                    [tensor.contiguous() for _, _, tensor in prepared], dim=0
                ).cpu()
                return

            stacked = torch.stack([tensor.contiguous() for _, _, tensor in prepared], dim=0)
            ep_gathered = [torch.empty_like(stacked) for _ in range(ps.ep_size)]
            dist.all_gather(ep_gathered, stacked, group=ps.ep_group)
            out[packed_name] = torch.cat(ep_gathered, dim=0).cpu()
            return

    if ps.ep_size <= 1 or ps.ep_group is None:
        for _, name, tensor in prepared:
            out[name] = tensor.cpu()
        return

    n_local = spec.num_experts // ps.ep_size
    stacked = torch.stack([tensor.contiguous() for _, _, tensor in prepared], dim=0)
    ep_gathered = [torch.empty_like(stacked) for _ in range(ps.ep_size)]
    dist.all_gather(ep_gathered, stacked, group=ps.ep_group)
    for ep_rank, ep_tensor in enumerate(ep_gathered):
        for slot, (local_idx, name, _) in enumerate(prepared):
            global_idx = ep_rank * n_local + local_idx
            out[set_expert_idx(name, global_idx)] = ep_tensor[slot].cpu()


def save_hf_weights(
    model: nn.Module | list[nn.Module],
    hf_path: str,
    spec: HFWeights,
    ps,
    *,
    vocab_size: int | None = None,
) -> None:
    """Export + write to safetensors."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    out = dict(export_hf_weights(model, spec, ps, vocab_size=vocab_size, rank0_only=True))
    if rank == 0 and out:
        save_safetensors(out, hf_path)
    if dist.is_initialized():
        dist.barrier()

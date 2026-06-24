# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Qwen3.5 lite native checkpoint mapping.

The loader reads HF safetensors directly into lite's native module names.
It intentionally does not require wrapper-specific state on the model.
"""

from __future__ import annotations

import re

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader, unwrap_model
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible, log_rank0


def EXPERT_CLASSIFIER(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


def PLACEMENT_FN(param_name: str) -> list:
    if "experts" in param_name and "router" not in param_name and "shared" not in param_name:
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    if "in_proj" in param_name and "layer_norm" not in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "qkv" in param_name and "layer_norm" not in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if (
        ("proj" in param_name or "o_proj" in param_name)
        and ("full_attn" in param_name or "linear_attn" in param_name)
        and "layer_norm" not in param_name
    ):
        # Row-parallel output proj weight: TP-shard on dim 1. Exclude layer_norm_weight (1-D, replicated
        # under TP) which otherwise matches here ("in_proj" contains "proj") and gets an invalid Shard(1).
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "gate_up" in param_name and "shared" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "down" in param_name and "shared" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "conv1d" in param_name or "dt_bias" in param_name or "A_log" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


def _tp(tensor: torch.Tensor, rank: int, size: int, dim: int = 0) -> torch.Tensor:
    return tensor if size <= 1 else tensor.chunk(size, dim=dim)[rank].contiguous()


def _split_gate_up(tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    if size <= 1:
        return tensor
    ffn = tensor.shape[0] // 2
    gate = tensor[:ffn].chunk(size, dim=0)[rank]
    up = tensor[ffn:].chunk(size, dim=0)[rank]
    return torch.cat([gate, up], dim=0).contiguous()


def _tp_linear_attn_in_proj(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    *,
    ps: ParallelState,
) -> torch.Tensor:
    """Shard each GDN projection independently, then pack the local layout."""
    return torch.cat(
        [
            _tp(q, ps.tp_rank, ps.tp_size),
            _tp(k, ps.tp_rank, ps.tp_size),
            _tp(v, ps.tp_rank, ps.tp_size),
            _tp(z, ps.tp_rank, ps.tp_size),
            _tp(b, ps.tp_rank, ps.tp_size),
            _tp(a, ps.tp_rank, ps.tp_size),
        ],
        dim=0,
    ).contiguous()


def _tp_linear_attn_conv1d(
    tensor: torch.Tensor, *, cfg: Qwen35Config, ps: ParallelState
) -> torch.Tensor:
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    q, k, v = tensor.split([qk_dim, qk_dim, v_dim], dim=0)
    return torch.cat(
        [
            _tp(q, ps.tp_rank, ps.tp_size),
            _tp(k, ps.tp_rank, ps.tp_size),
            _tp(v, ps.tp_rank, ps.tp_size),
        ],
        dim=0,
    ).contiguous()


def _zero_centered_gamma_from_hf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - 1


def _get(reader: SafeTensorReader, name: str) -> torch.Tensor:
    return reader.get_tensor(name)


def _has(reader: SafeTensorReader, name: str) -> bool:
    if reader.index:
        return name in reader.index
    try:
        reader.get_tensor(name)
    except Exception:
        return False
    return True


def _load_vocab(
    reader: SafeTensorReader, name: str, cfg: Qwen35Config, ps: ParallelState
) -> torch.Tensor:
    from megatron.lite.primitive.parallel import pad_vocab_for_tp

    tensor = _get(reader, name)
    padded = pad_vocab_for_tp(cfg.vocab_size, ps.tp_size)
    if tensor.size(0) < padded:
        pad = torch.zeros(padded - tensor.size(0), tensor.size(1), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    return _tp(tensor, ps.tp_rank, ps.tp_size)


def _load_full_attn(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_prefix: str,
    input_ln: torch.Tensor,
    cfg: Qwen35Config,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    out[f"{local_prefix}.full_attn.qkv.linear.layer_norm_weight"] = input_ln
    q = _get(reader, f"{hf_prefix}.q_proj.weight")
    k = _get(reader, f"{hf_prefix}.k_proj.weight")
    v = _get(reader, f"{hf_prefix}.v_proj.weight")
    out[f"{local_prefix}.full_attn.qkv.linear.weight"] = _tp(
        _merge_full_attn_qkvg(q, k, v, cfg=cfg), ps.tp_rank, ps.tp_size
    )
    out[f"{local_prefix}.full_attn.q_norm.weight"] = _get(reader, f"{hf_prefix}.q_norm.weight")
    out[f"{local_prefix}.full_attn.k_norm.weight"] = _get(reader, f"{hf_prefix}.k_norm.weight")
    out[f"{local_prefix}.full_attn.proj.linear.weight"] = _tp(
        _get(reader, f"{hf_prefix}.o_proj.weight"), ps.tp_rank, ps.tp_size, dim=1
    )


def _merge_full_attn_qkvg(
    q_gate: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, cfg: Qwen35Config
) -> torch.Tensor:
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    hidden = q_gate.shape[1]
    q_gate = q_gate.reshape(cfg.num_attention_heads, 2 * head_dim, hidden)
    query = q_gate.narrow(1, 0, head_dim).reshape(cfg.num_attention_heads * head_dim, hidden)
    gate = q_gate.narrow(1, head_dim, head_dim).reshape(cfg.num_attention_heads * head_dim, hidden)
    q_heads_per_group = ensure_divisible(cfg.num_attention_heads, cfg.num_key_value_heads)
    q_group_width = q_heads_per_group * head_dim
    query = query.reshape(kv_heads, q_group_width, hidden)
    gate = gate.reshape(kv_heads, q_group_width, hidden)
    key = key.reshape(kv_heads, head_dim, hidden)
    value = value.reshape(kv_heads, head_dim, hidden)
    return torch.cat([query, gate, key, value], dim=1).reshape(-1, hidden).contiguous()


def _unmerge_full_attn_qkvg(
    tensor: torch.Tensor, *, cfg: Qwen35Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert Qwen35 lite's full-attention q/g/k/v packing."""
    q_heads_per_group = ensure_divisible(cfg.num_attention_heads, cfg.num_key_value_heads)
    group_width = (2 * q_heads_per_group + 2) * cfg.head_dim
    hidden = tensor.shape[-1]
    packed = tensor.reshape(cfg.num_key_value_heads, group_width, hidden)
    query, gate, key, value = packed.split(
        [
            q_heads_per_group * cfg.head_dim,
            q_heads_per_group * cfg.head_dim,
            cfg.head_dim,
            cfg.head_dim,
        ],
        dim=1,
    )
    query = query.reshape(cfg.num_attention_heads, cfg.head_dim, hidden)
    gate = gate.reshape(cfg.num_attention_heads, cfg.head_dim, hidden)
    q_gate = torch.cat([query, gate], dim=1).reshape(
        cfg.num_attention_heads * 2 * cfg.head_dim, hidden
    )
    key = key.reshape(cfg.num_key_value_heads * cfg.head_dim, hidden)
    value = value.reshape(cfg.num_key_value_heads * cfg.head_dim, hidden)
    return q_gate.contiguous(), key.contiguous(), value.contiguous()


def _split_linear_attn_in_proj(
    tensor: torch.Tensor, *, cfg: Qwen35Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    return tensor.split(
        [qk_dim, qk_dim, v_dim, v_dim, cfg.linear_num_value_heads, cfg.linear_num_value_heads],
        dim=0,
    )


def _merge_linear_attn_in_proj_tp_shards(
    shards: list[torch.Tensor], *, cfg: Qwen35Config
) -> torch.Tensor:
    world_size = len(shards)
    qk_dim = ensure_divisible(cfg.linear_num_key_heads * cfg.linear_key_head_dim, world_size)
    v_dim = ensure_divisible(cfg.linear_num_value_heads * cfg.linear_value_head_dim, world_size)
    value_heads = ensure_divisible(cfg.linear_num_value_heads, world_size)

    parts: list[list[torch.Tensor]] = [[] for _ in range(6)]
    for shard in shards:
        for bucket, part in zip(
            parts,
            shard.split([qk_dim, qk_dim, v_dim, v_dim, value_heads, value_heads], dim=0),
            strict=True,
        ):
            bucket.append(part)

    return torch.cat([torch.cat(bucket, dim=0) for bucket in parts], dim=0).contiguous()


def _merge_linear_attn_conv1d_tp_shards(
    shards: list[torch.Tensor], *, cfg: Qwen35Config
) -> torch.Tensor:
    world_size = len(shards)
    qk_dim = ensure_divisible(cfg.linear_num_key_heads * cfg.linear_key_head_dim, world_size)
    v_dim = ensure_divisible(cfg.linear_num_value_heads * cfg.linear_value_head_dim, world_size)

    parts: list[list[torch.Tensor]] = [[] for _ in range(3)]
    for shard in shards:
        for bucket, part in zip(parts, shard.split([qk_dim, qk_dim, v_dim], dim=0), strict=True):
            bucket.append(part)

    return torch.cat([torch.cat(bucket, dim=0) for bucket in parts], dim=0).contiguous()


def _merge_gate_up_tp_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    gates: list[torch.Tensor] = []
    ups: list[torch.Tensor] = []
    for shard in shards:
        gate, up = shard.chunk(2, dim=0)
        gates.append(gate)
        ups.append(up)
    return torch.cat([torch.cat(gates, dim=0), torch.cat(ups, dim=0)], dim=0).contiguous()


def _allgather_tp_shards(tensor: torch.Tensor, ps: ParallelState) -> list[torch.Tensor]:
    shards = [torch.empty_like(tensor) for _ in range(ps.tp_size)]
    dist.all_gather(shards, tensor.contiguous(), group=ps.tp_group)
    return shards


class Qwen35WeightSpec:
    """Export Qwen35 lite weights to HF checkpoint or vLLM runtime names."""

    def __init__(self, config: Qwen35Config, target: str = "hf"):
        if target not in {"hf", "vllm"}:
            raise ValueError(f"Unsupported Qwen3.5 export target: {target!r}")
        self.config = config
        self.target = target
        self._expert_export_buffers: dict[tuple[int, str], dict[int, torch.Tensor]] = {}

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def weight_map(self) -> dict[str, list[str]]:
        return {}

    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        del native_name
        return hf_tensors[0]

    def gather_dense(
        self, native_name: str, tensor: torch.Tensor, ps: ParallelState
    ) -> torch.Tensor | None:
        if ps.tp_size <= 1:
            return None
        if native_name.endswith(".linear_attn.in_proj.linear.weight"):
            return _merge_linear_attn_in_proj_tp_shards(
                _allgather_tp_shards(tensor, ps), cfg=self.config
            )
        if native_name.endswith(".linear_attn.conv1d.weight"):
            return _merge_linear_attn_conv1d_tp_shards(
                _allgather_tp_shards(tensor, ps), cfg=self.config
            )
        if native_name.endswith(".moe.shared_expert.gate_up.linear.weight"):
            return _merge_gate_up_tp_shards(_allgather_tp_shards(tensor, ps))
        return None

    def packed_expert_group_name(self, native_name: str) -> str | None:
        if re.fullmatch(r"layers\.\d+\.moe\.experts\.fc[12]\.weight\d+", native_name) is None:
            return None
        return re.sub(r"\.weight\d+$", ".packed", native_name)

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if self.target == "vllm":
            return self._native_to_vllm(native_name, tensor)

        if native_name == "embed.embedding.weight":
            return [("model.language_model.embed_tokens.weight", tensor)]
        if native_name == "norm.weight":
            return [("model.language_model.norm.weight", tensor)]
        if native_name == "head.col.linear.weight":
            return [("lm_head.weight", tensor)]
        if native_name == "mtp_embed.embedding.weight" or native_name.startswith("mtp."):
            return []

        match = re.match(r"layers\.(\d+)\.(.*)", native_name)
        if match is None:
            return []

        layer_idx = int(match.group(1))
        suffix = match.group(2)
        prefix = f"model.language_model.layers.{layer_idx}"

        if suffix == "full_attn.qkv.linear.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", tensor)]
        if suffix == "full_attn.qkv.linear.weight":
            q_gate, key, value = _unmerge_full_attn_qkvg(tensor, cfg=self.config)
            return [
                (f"{prefix}.self_attn.q_proj.weight", q_gate),
                (f"{prefix}.self_attn.k_proj.weight", key),
                (f"{prefix}.self_attn.v_proj.weight", value),
            ]
        if suffix == "full_attn.q_norm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", tensor)]
        if suffix == "full_attn.k_norm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", tensor)]
        if suffix == "full_attn.proj.linear.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", tensor)]

        if suffix == "linear_attn.in_proj.linear.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", tensor)]
        if suffix == "linear_attn.in_proj.linear.weight":
            q, k, value, z, b, a = _split_linear_attn_in_proj(tensor, cfg=self.config)
            return [
                (
                    f"{prefix}.linear_attn.in_proj_qkv.weight",
                    torch.cat([q, k, value], dim=0).contiguous(),
                ),
                (f"{prefix}.linear_attn.in_proj_z.weight", z.contiguous()),
                (f"{prefix}.linear_attn.in_proj_b.weight", b.contiguous()),
                (f"{prefix}.linear_attn.in_proj_a.weight", a.contiguous()),
            ]
        if suffix == "linear_attn.conv1d.weight":
            return [(f"{prefix}.linear_attn.conv1d.weight", tensor)]
        if suffix == "linear_attn.dt_bias":
            return [(f"{prefix}.linear_attn.dt_bias", tensor)]
        if suffix == "linear_attn.A_log":
            return [(f"{prefix}.linear_attn.A_log", tensor)]
        if suffix == "linear_attn.norm.weight":
            return [(f"{prefix}.linear_attn.norm.weight", tensor + 1)]
        if suffix == "linear_attn.o_proj.linear.weight":
            return [(f"{prefix}.linear_attn.out_proj.weight", tensor)]

        if suffix == "mlp_norm.weight":
            return [(f"{prefix}.post_attention_layernorm.weight", tensor)]
        if suffix == "moe.router.gate.weight":
            return [(f"{prefix}.mlp.gate.weight", tensor)]
        if suffix == "moe.shared_expert.gate_up.linear.weight":
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate.contiguous()),
                (f"{prefix}.mlp.shared_expert.up_proj.weight", up.contiguous()),
            ]
        if suffix == "moe.shared_expert.down.linear.weight":
            return [(f"{prefix}.mlp.shared_expert.down_proj.weight", tensor)]
        if suffix == "moe.shared_expert.shared_gate.weight":
            return [(f"{prefix}.mlp.shared_expert_gate.weight", tensor)]

        expert_match = re.fullmatch(r"moe\.experts\.fc([12])\.weight(\d+)", suffix)
        if expert_match is not None:
            kind, expert_idx = expert_match.groups()
            buffer_key = (layer_idx, "gate_up" if kind == "1" else "down")
            buffer = self._expert_export_buffers.setdefault(buffer_key, {})
            buffer[int(expert_idx)] = tensor.contiguous()
            if len(buffer) < self.config.num_experts:
                return []
            packed = torch.stack(
                [buffer[i] for i in range(self.config.num_experts)], dim=0
            ).contiguous()
            del self._expert_export_buffers[buffer_key]
            if kind == "1":
                return [(f"{prefix}.mlp.experts.gate_up_proj", packed)]
            return [(f"{prefix}.mlp.experts.down_proj", packed)]

        packed_expert_match = re.fullmatch(r"moe\.experts\.fc([12])\.packed", suffix)
        if packed_expert_match is not None:
            kind = packed_expert_match.group(1)
            if kind == "1":
                return [(f"{prefix}.mlp.experts.gate_up_proj", tensor.contiguous())]
            return [(f"{prefix}.mlp.experts.down_proj", tensor.contiguous())]

        return []

    def _native_to_vllm(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if native_name == "embed.embedding.weight":
            return [("language_model.model.embed_tokens.weight", tensor)]
        if native_name == "norm.weight":
            return [("language_model.model.norm.weight", tensor)]
        if native_name == "head.col.linear.weight":
            return [("language_model.lm_head.weight", tensor)]
        if native_name == "mtp_embed.embedding.weight" or native_name.startswith("mtp."):
            return []

        match = re.match(r"layers\.(\d+)\.(.*)", native_name)
        if match is None:
            return []

        layer_idx = int(match.group(1))
        suffix = match.group(2)
        prefix = f"language_model.model.layers.{layer_idx}"

        if suffix == "full_attn.qkv.linear.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", tensor)]
        if suffix == "full_attn.qkv.linear.weight":
            q_gate, key, value = _unmerge_full_attn_qkvg(tensor, cfg=self.config)
            return [
                (f"{prefix}.self_attn.q_proj.weight", q_gate),
                (f"{prefix}.self_attn.k_proj.weight", key),
                (f"{prefix}.self_attn.v_proj.weight", value),
            ]
        if suffix == "full_attn.q_norm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", tensor)]
        if suffix == "full_attn.k_norm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", tensor)]
        if suffix == "full_attn.proj.linear.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", tensor)]

        if suffix == "linear_attn.in_proj.linear.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", tensor)]
        if suffix == "linear_attn.in_proj.linear.weight":
            q, k, value, z, b, a = _split_linear_attn_in_proj(tensor, cfg=self.config)
            return [
                (
                    f"{prefix}.linear_attn.in_proj_qkv.weight",
                    torch.cat([q, k, value], dim=0).contiguous(),
                ),
                (f"{prefix}.linear_attn.in_proj_z.weight", z.contiguous()),
                (f"{prefix}.linear_attn.in_proj_b.weight", b.contiguous()),
                (f"{prefix}.linear_attn.in_proj_a.weight", a.contiguous()),
            ]
        if suffix == "linear_attn.conv1d.weight":
            return [(f"{prefix}.linear_attn.conv1d.weight", tensor)]
        if suffix == "linear_attn.dt_bias":
            return [(f"{prefix}.linear_attn.dt_bias", tensor)]
        if suffix == "linear_attn.A_log":
            return [(f"{prefix}.linear_attn.A_log", tensor)]
        if suffix == "linear_attn.norm.weight":
            return [(f"{prefix}.linear_attn.norm.weight", tensor + 1)]
        if suffix == "linear_attn.o_proj.linear.weight":
            return [(f"{prefix}.linear_attn.out_proj.weight", tensor)]

        if suffix == "mlp_norm.weight":
            return [(f"{prefix}.post_attention_layernorm.weight", tensor)]
        if suffix == "moe.router.gate.weight":
            return [(f"{prefix}.mlp.gate.weight", tensor)]
        if suffix == "moe.shared_expert.gate_up.linear.weight":
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate.contiguous()),
                (f"{prefix}.mlp.shared_expert.up_proj.weight", up.contiguous()),
            ]
        if suffix == "moe.shared_expert.down.linear.weight":
            return [(f"{prefix}.mlp.shared_expert.down_proj.weight", tensor)]
        if suffix == "moe.shared_expert.shared_gate.weight":
            return [(f"{prefix}.mlp.shared_expert_gate.weight", tensor)]

        expert_match = re.fullmatch(r"moe\.experts\.fc([12])\.weight(\d+)", suffix)
        if expert_match is not None:
            kind, expert_idx = expert_match.groups()
            buffer_key = (layer_idx, "vllm_gate_up" if kind == "1" else "vllm_down")
            buffer = self._expert_export_buffers.setdefault(buffer_key, {})
            buffer[int(expert_idx)] = tensor.contiguous()
            if len(buffer) < self.config.num_experts:
                return []
            packed = torch.stack(
                [buffer[i] for i in range(self.config.num_experts)], dim=0
            ).contiguous()
            del self._expert_export_buffers[buffer_key]
            if kind == "1":
                return [(f"{prefix}.mlp.experts.gate_up_proj", packed)]
            return [(f"{prefix}.mlp.experts.down_proj", packed)]

        packed_expert_match = re.fullmatch(r"moe\.experts\.fc([12])\.packed", suffix)
        if packed_expert_match is not None:
            kind = packed_expert_match.group(1)
            if kind == "1":
                return [(f"{prefix}.mlp.experts.gate_up_proj", tensor.contiguous())]
            return [(f"{prefix}.mlp.experts.down_proj", tensor.contiguous())]

        return []

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        del native_name
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        if self.is_expert(native_name):
            if ".fc1." in native_name:
                return (0, 1)
            if ".fc2." in native_name:
                return (1, 1)
            return None
        if native_name in {"embed.embedding.weight", "head.col.linear.weight"}:
            return (0, 0)
        if native_name.endswith(".full_attn.qkv.linear.weight"):
            return (0, 0)
        if native_name.endswith(".full_attn.proj.linear.weight"):
            return (1, 0)
        if native_name.endswith(".linear_attn.in_proj.linear.weight"):
            return (0, 0)
        if native_name.endswith(".linear_attn.o_proj.linear.weight"):
            return (1, 0)
        if native_name.endswith(".moe.shared_expert.gate_up.linear.weight"):
            return (0, 0)
        if native_name.endswith(".moe.shared_expert.down.linear.weight"):
            return (1, 0)
        if any(
            native_name.endswith(suffix)
            for suffix in (
                ".linear_attn.conv1d.weight",
                ".linear_attn.dt_bias",
                ".linear_attn.A_log",
            )
        ):
            return (0, 0)
        return None

    def is_expert(self, native_name: str) -> bool:
        return (
            ".moe.experts." in native_name
            and ".router." not in native_name
            and ".shared" not in native_name
        )

    def expert_global_id(self, native_name: str) -> int | None:
        match = re.search(r"\.weight(\d+)$", native_name)
        return int(match.group(1)) if match is not None else None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        return re.sub(r"\.weight\d+$", f".weight{local_idx}", native_name)


def _load_linear_attn(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_prefix: str,
    input_ln: torch.Tensor,
    cfg: Qwen35Config,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    qk_dim, v_dim = nk * dk, nv * dv
    qkv = _get(reader, f"{hf_prefix}.in_proj_qkv.weight")
    q, k, v = qkv.split([qk_dim, qk_dim, v_dim], dim=0)
    z = _get(reader, f"{hf_prefix}.in_proj_z.weight")
    b = _get(reader, f"{hf_prefix}.in_proj_b.weight")
    a = _get(reader, f"{hf_prefix}.in_proj_a.weight")

    out[f"{local_prefix}.linear_attn.in_proj.linear.weight"] = _tp_linear_attn_in_proj(
        q, k, v, z, b, a, ps=ps
    )
    out[f"{local_prefix}.linear_attn.in_proj.linear.layer_norm_weight"] = input_ln
    out[f"{local_prefix}.linear_attn.conv1d.weight"] = _tp_linear_attn_conv1d(
        _get(reader, f"{hf_prefix}.conv1d.weight"), cfg=cfg, ps=ps
    )
    out[f"{local_prefix}.linear_attn.dt_bias"] = _tp(
        _get(reader, f"{hf_prefix}.dt_bias"), ps.tp_rank, ps.tp_size
    )
    out[f"{local_prefix}.linear_attn.A_log"] = _tp(
        _get(reader, f"{hf_prefix}.A_log"), ps.tp_rank, ps.tp_size
    )
    out[f"{local_prefix}.linear_attn.norm.weight"] = _zero_centered_gamma_from_hf(
        _get(reader, f"{hf_prefix}.norm.weight")
    )
    out[f"{local_prefix}.linear_attn.o_proj.linear.weight"] = _tp(
        _get(reader, f"{hf_prefix}.out_proj.weight"), ps.tp_rank, ps.tp_size, dim=1
    )


def _load_shared_expert(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_mlp_prefix: str,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    shared = f"{hf_mlp_prefix}.shared_expert"
    gate_up = torch.cat(
        [_get(reader, f"{shared}.gate_proj.weight"), _get(reader, f"{shared}.up_proj.weight")],
        dim=0,
    )
    out[f"{local_prefix}.moe.shared_expert.gate_up.linear.weight"] = _split_gate_up(
        gate_up, ps.tp_rank, ps.tp_size
    )
    out[f"{local_prefix}.moe.shared_expert.down.linear.weight"] = _tp(
        _get(reader, f"{shared}.down_proj.weight"), ps.tp_rank, ps.tp_size, dim=1
    )
    out[f"{local_prefix}.moe.shared_expert.shared_gate.weight"] = _get(
        reader, f"{hf_mlp_prefix}.shared_expert_gate.weight"
    )


def _load_experts(
    out: dict[str, torch.Tensor],
    *,
    local_prefix: str,
    hf_mlp_prefix: str,
    cfg: Qwen35Config,
    ps: ParallelState,
    reader: SafeTensorReader,
) -> None:
    num_local = ensure_divisible(cfg.num_experts, ps.ep_size)
    local_start = ps.ep_rank * num_local
    packed_gate_up = f"{hf_mlp_prefix}.experts.gate_up_proj"
    packed_down = f"{hf_mlp_prefix}.experts.down_proj"
    if _has(reader, packed_gate_up) and _has(reader, packed_down):
        gate_up_all = _get(reader, packed_gate_up)
        down_all = _get(reader, packed_down)
        for local_idx in range(num_local):
            global_idx = local_start + local_idx
            fc1 = gate_up_all[global_idx]
            fc2 = down_all[global_idx]
            if ps.etp_size > 1:
                fc1 = _split_gate_up(fc1, ps.etp_rank, ps.etp_size)
                fc2 = _tp(fc2, ps.etp_rank, ps.etp_size, dim=1)
            out[f"{local_prefix}.moe.experts.fc1.weight{local_idx}"] = fc1
            out[f"{local_prefix}.moe.experts.fc2.weight{local_idx}"] = fc2
        return

    for local_idx in range(num_local):
        global_idx = local_start + local_idx
        ep = f"{hf_mlp_prefix}.experts.{global_idx}"
        fc1 = torch.cat(
            [_get(reader, f"{ep}.gate_proj.weight"), _get(reader, f"{ep}.up_proj.weight")], dim=0
        )
        fc2 = _get(reader, f"{ep}.down_proj.weight")
        if ps.etp_size > 1:
            fc1 = _split_gate_up(fc1, ps.etp_rank, ps.etp_size)
            fc2 = _tp(fc2, ps.etp_rank, ps.etp_size, dim=1)
        out[f"{local_prefix}.moe.experts.fc1.weight{local_idx}"] = fc1
        out[f"{local_prefix}.moe.experts.fc2.weight{local_idx}"] = fc2


def _copy_loaded_state(model: nn.Module, loaded: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    resolved: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        actual = name if name in state else None
        if actual is None:
            for key in state:
                if name in key:
                    actual = key
                    break
        if actual is not None:
            resolved[actual] = tensor
        else:
            log_rank0(f"WARNING: lite checkpoint tensor has no target param: {name}")

    fp32_names = ("A_log", "dt_bias")
    for name, param in model.named_parameters():
        if name not in resolved:
            log_rank0(f"WARNING: {name} not loaded from checkpoint")
            continue
        tensor = resolved[name].to(device=param.device)
        if any(k in name for k in fp32_names):
            param.data.copy_(tensor.float())
        else:
            param.data.copy_(tensor.to(dtype=param.dtype))


def load_hf_weights(model: nn.Module, path: str, config: Qwen35Config, ps: ParallelState) -> None:
    base_model = unwrap_model(model)
    reader = SafeTensorReader(path)
    out: dict[str, torch.Tensor] = {}

    prefix = "model.language_model"
    if getattr(base_model, "embed", None) is not None:
        out["embed.embedding.weight"] = _load_vocab(
            reader, f"{prefix}.embed_tokens.weight", config, ps
        )
    if getattr(base_model, "norm", None) is not None:
        out["norm.weight"] = _get(reader, f"{prefix}.norm.weight")
    if getattr(base_model, "head", None) is not None:
        out["head.col.linear.weight"] = _load_vocab(reader, "lm_head.weight", config, ps)

    for local_idx, global_idx in enumerate(base_model.layer_indices):
        lp = f"layers.{local_idx}"
        hp = f"{prefix}.layers.{global_idx}"
        input_ln = _get(reader, f"{hp}.input_layernorm.weight")
        if config.layer_type_at(global_idx) == "full_attention":
            _load_full_attn(
                out,
                local_prefix=lp,
                hf_prefix=f"{hp}.self_attn",
                input_ln=input_ln,
                cfg=config,
                ps=ps,
                reader=reader,
            )
        else:
            _load_linear_attn(
                out,
                local_prefix=lp,
                hf_prefix=f"{hp}.linear_attn",
                input_ln=input_ln,
                cfg=config,
                ps=ps,
                reader=reader,
            )
        out[f"{lp}.mlp_norm.weight"] = _get(reader, f"{hp}.post_attention_layernorm.weight")
        out[f"{lp}.moe.router.gate.weight"] = _get(reader, f"{hp}.mlp.gate.weight")[
            : config.num_experts
        ]
        _load_shared_expert(out, local_prefix=lp, hf_mlp_prefix=f"{hp}.mlp", ps=ps, reader=reader)
        _load_experts(
            out, local_prefix=lp, hf_mlp_prefix=f"{hp}.mlp", cfg=config, ps=ps, reader=reader
        )

    _copy_loaded_state(base_model, out)


def export_hf_weights(
    model: nn.Module | list[nn.Module], config: Qwen35Config, ps: ParallelState, **kwargs
):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as _export

    include_mtp_only = kwargs.pop("include_mtp_only", False)
    kwargs.pop("include_local_prefixes", None)
    target = kwargs.pop("target", "hf")
    if include_mtp_only:
        return
    yield from _export(
        model, Qwen35WeightSpec(config, target=target), ps, vocab_size=config.vocab_size, **kwargs
    )


def save_hf_weights(
    model: nn.Module | list[nn.Module], path: str, config: Qwen35Config, ps: ParallelState
) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import save_hf_weights as _save

    _save(model, path, Qwen35WeightSpec(config), ps, vocab_size=config.vocab_size)


__all__ = [
    "EXPERT_CLASSIFIER",
    "PLACEMENT_FN",
    "Qwen35WeightSpec",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]

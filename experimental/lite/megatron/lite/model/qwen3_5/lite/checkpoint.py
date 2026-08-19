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
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization.mxfp4 import MXFP4_BLOCK_SIZE, quantize_mxfp4
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.runtime.contracts.weights import ResyncFormat
from torch.distributed.tensor import Replicate, Shard


def EXPERT_CLASSIFIER(name: str) -> bool:
    return "experts" in name and "router" not in name and "shared" not in name


def PLACEMENT_FN(param_name: str) -> list:
    if (
        "experts" in param_name
        and "router" not in param_name
        and "shared" not in param_name
    ):
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
    if "conv1d" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "dt_bias" in param_name or "A_log" in param_name:
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


def _tp(tensor: torch.Tensor, rank: int, size: int, dim: int = 0) -> torch.Tensor:
    return tensor if size <= 1 else tensor.chunk(size, dim=dim)[rank].contiguous()


def _tp_linear_attn_in_proj(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    *,
    cfg: Qwen35Config,
    ps: ParallelState,
) -> torch.Tensor:
    """Shard each GDN projection independently, then pack the local layout."""
    if _linear_attn_head_replication(cfg, ps.tp_size):
        return _tp(torch.cat([q, k, v, z, b, a], dim=0), ps.tp_rank, ps.tp_size)
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
    if _linear_attn_head_replication(cfg, ps.tp_size):
        return _tp(tensor, ps.tp_rank, ps.tp_size)
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


def _linear_attn_head_replication(cfg: Qwen35Config, tp_size: int) -> bool:
    return cfg.linear_num_key_heads < tp_size or cfg.linear_num_value_heads < tp_size


def _tp_linear_attn_state(
    tensor: torch.Tensor, *, cfg: Qwen35Config, ps: ParallelState
) -> torch.Tensor:
    """Keep GDN state whole only when its heads are physically replicated."""
    if _linear_attn_head_replication(cfg, ps.tp_size):
        return tensor
    return _tp(tensor, ps.tp_rank, ps.tp_size)


def _merge_full_attn_qkvg(
    q_gate: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, cfg: Qwen35Config
) -> torch.Tensor:
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    hidden = q_gate.shape[1]
    q_gate = q_gate.reshape(cfg.num_attention_heads, 2 * head_dim, hidden)
    query = q_gate.narrow(1, 0, head_dim).reshape(
        cfg.num_attention_heads * head_dim, hidden
    )
    gate = q_gate.narrow(1, head_dim, head_dim).reshape(
        cfg.num_attention_heads * head_dim, hidden
    )
    q_heads_per_group = ensure_divisible(
        cfg.num_attention_heads, cfg.num_key_value_heads
    )
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
    q_heads_per_group = ensure_divisible(
        cfg.num_attention_heads, cfg.num_key_value_heads
    )
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
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    qk_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    return tensor.split(
        [
            qk_dim,
            qk_dim,
            v_dim,
            v_dim,
            cfg.linear_num_value_heads,
            cfg.linear_num_value_heads,
        ],
        dim=0,
    )


def _merge_linear_attn_in_proj_tp_shards(
    shards: list[torch.Tensor], *, cfg: Qwen35Config
) -> torch.Tensor:
    world_size = len(shards)
    if _linear_attn_head_replication(cfg, world_size):
        return torch.cat(shards, dim=0).contiguous()
    qk_dim = ensure_divisible(
        cfg.linear_num_key_heads * cfg.linear_key_head_dim, world_size
    )
    v_dim = ensure_divisible(
        cfg.linear_num_value_heads * cfg.linear_value_head_dim, world_size
    )
    value_heads = ensure_divisible(cfg.linear_num_value_heads, world_size)

    parts: list[list[torch.Tensor]] = [[] for _ in range(6)]
    for shard in shards:
        for bucket, part in zip(
            parts,
            shard.split(
                [qk_dim, qk_dim, v_dim, v_dim, value_heads, value_heads], dim=0
            ),
            strict=True,
        ):
            bucket.append(part)

    return torch.cat([torch.cat(bucket, dim=0) for bucket in parts], dim=0).contiguous()


def _merge_linear_attn_conv1d_tp_shards(
    shards: list[torch.Tensor], *, cfg: Qwen35Config
) -> torch.Tensor:
    world_size = len(shards)
    if _linear_attn_head_replication(cfg, world_size):
        return torch.cat(shards, dim=0).contiguous()
    qk_dim = ensure_divisible(
        cfg.linear_num_key_heads * cfg.linear_key_head_dim, world_size
    )
    v_dim = ensure_divisible(
        cfg.linear_num_value_heads * cfg.linear_value_head_dim, world_size
    )

    parts: list[list[torch.Tensor]] = [[] for _ in range(3)]
    for shard in shards:
        for bucket, part in zip(
            parts, shard.split([qk_dim, qk_dim, v_dim], dim=0), strict=True
        ):
            bucket.append(part)

    return torch.cat([torch.cat(bucket, dim=0) for bucket in parts], dim=0).contiguous()


def _merge_gate_up_tp_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    gates: list[torch.Tensor] = []
    ups: list[torch.Tensor] = []
    for shard in shards:
        gate, up = shard.chunk(2, dim=0)
        gates.append(gate)
        ups.append(up)
    return torch.cat(
        [torch.cat(gates, dim=0), torch.cat(ups, dim=0)], dim=0
    ).contiguous()


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
        c = self.config
        weight_map: dict[str, list[str]] = {
            "embed.embedding.weight": ["model.language_model.embed_tokens.weight"],
            "norm.weight": ["model.language_model.norm.weight"],
            "head.col.linear.weight": ["lm_head.weight"],
        }
        for layer_idx in range(c.num_hidden_layers):
            local_prefix = f"layers.{layer_idx}"
            hf_prefix = f"model.language_model.layers.{layer_idx}"
            mlp = f"{hf_prefix}.mlp"
            if c.layer_type_at(layer_idx) == "full_attention":
                attention = f"{hf_prefix}.self_attn"
                weight_map.update(
                    {
                        f"{local_prefix}.full_attn.qkv.linear.layer_norm_weight": [
                            f"{hf_prefix}.input_layernorm.weight"
                        ],
                        f"{local_prefix}.full_attn.qkv.linear.weight": [
                            f"{attention}.q_proj.weight",
                            f"{attention}.k_proj.weight",
                            f"{attention}.v_proj.weight",
                        ],
                        f"{local_prefix}.full_attn.q_norm.weight": [
                            f"{attention}.q_norm.weight"
                        ],
                        f"{local_prefix}.full_attn.k_norm.weight": [
                            f"{attention}.k_norm.weight"
                        ],
                        f"{local_prefix}.full_attn.proj.linear.weight": [
                            f"{attention}.o_proj.weight"
                        ],
                    }
                )
            else:
                attention = f"{hf_prefix}.linear_attn"
                weight_map.update(
                    {
                        f"{local_prefix}.linear_attn.in_proj.linear.layer_norm_weight": [
                            f"{hf_prefix}.input_layernorm.weight"
                        ],
                        f"{local_prefix}.linear_attn.in_proj.linear.weight": [
                            f"{attention}.in_proj_qkv.weight",
                            f"{attention}.in_proj_z.weight",
                            f"{attention}.in_proj_b.weight",
                            f"{attention}.in_proj_a.weight",
                        ],
                        f"{local_prefix}.linear_attn.conv1d.weight": [
                            f"{attention}.conv1d.weight"
                        ],
                        f"{local_prefix}.linear_attn.dt_bias": [f"{attention}.dt_bias"],
                        f"{local_prefix}.linear_attn.A_log": [f"{attention}.A_log"],
                        f"{local_prefix}.linear_attn.norm.weight": [
                            f"{attention}.norm.weight"
                        ],
                        f"{local_prefix}.linear_attn.o_proj.linear.weight": [
                            f"{attention}.out_proj.weight"
                        ],
                    }
                )
            weight_map.update(
                {
                    f"{local_prefix}.mlp_norm.weight": [
                        f"{hf_prefix}.post_attention_layernorm.weight"
                    ],
                    f"{local_prefix}.moe.router.gate.weight": [f"{mlp}.gate.weight"],
                    f"{local_prefix}.moe.shared_expert.gate_up.linear.weight": [
                        f"{mlp}.shared_expert.gate_proj.weight",
                        f"{mlp}.shared_expert.up_proj.weight",
                    ],
                    f"{local_prefix}.moe.shared_expert.down.linear.weight": [
                        f"{mlp}.shared_expert.down_proj.weight"
                    ],
                    f"{local_prefix}.moe.shared_expert.shared_gate.weight": [
                        f"{mlp}.shared_expert_gate.weight"
                    ],
                }
            )
            for expert_idx in range(c.num_experts):
                weight_map[f"{local_prefix}.moe.experts.fc1.weight{expert_idx}"] = [
                    f"{mlp}.experts.{expert_idx}.gate_proj.weight",
                    f"{mlp}.experts.{expert_idx}.up_proj.weight",
                ]
            for expert_idx in range(c.num_experts):
                weight_map[f"{local_prefix}.moe.experts.fc2.weight{expert_idx}"] = [
                    f"{mlp}.experts.{expert_idx}.down_proj.weight"
                ]
        return weight_map

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        if native_name.endswith(".full_attn.qkv.linear.weight"):
            return _merge_full_attn_qkvg(*hf_tensors, cfg=self.config)
        if native_name.endswith(".linear_attn.in_proj.linear.weight"):
            qkv, z, b, a = hf_tensors
            qk_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim
            value_dim = (
                self.config.linear_num_value_heads * self.config.linear_value_head_dim
            )
            q, k, value = qkv.split([qk_dim, qk_dim, value_dim], dim=0)
            return torch.cat([q, k, value, z, b, a], dim=0)
        if native_name.endswith(".linear_attn.norm.weight"):
            return hf_tensors[0] - 1
        if len(hf_tensors) == 2:
            return torch.cat(hf_tensors, dim=0)
        if native_name.endswith(".moe.router.gate.weight"):
            return hf_tensors[0][: self.config.num_experts]
        return hf_tensors[0]

    def hf_name_candidates(self, native_name: str, hf_name: str) -> list[str]:
        candidates = [hf_name]
        expert_id = self.expert_global_id(native_name)
        if expert_id is None:
            return candidates
        mlp_prefix = hf_name.split(".experts.", 1)[0]
        if ".fc1." in native_name:
            candidates.append(f"{mlp_prefix}.experts.gate_up_proj")
        elif ".fc2." in native_name:
            candidates.append(f"{mlp_prefix}.experts.down_proj")
        return candidates

    def transform_hf_source(
        self,
        native_name: str,
        source_index: int,
        resolved_name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        expert_id = self.expert_global_id(native_name)
        if expert_id is None or ".experts." not in resolved_name:
            return tensor
        if resolved_name.endswith(".experts.gate_up_proj"):
            gate, up = tensor[expert_id].chunk(2, dim=0)
            return gate if source_index == 0 else up
        if resolved_name.endswith(".experts.down_proj"):
            return tensor[expert_id]
        return tensor

    def shard_for_load(
        self, native_name: str, tensor: torch.Tensor, ps: ParallelState
    ) -> torch.Tensor | None:
        if native_name.endswith(".linear_attn.in_proj.linear.weight"):
            return _tp_linear_attn_in_proj(
                *_split_linear_attn_in_proj(tensor, cfg=self.config),
                cfg=self.config,
                ps=ps,
            )
        if native_name.endswith(".linear_attn.conv1d.weight"):
            return _tp_linear_attn_conv1d(tensor, cfg=self.config, ps=ps)
        if native_name.endswith((".linear_attn.dt_bias", ".linear_attn.A_log")):
            return _tp_linear_attn_state(tensor, cfg=self.config, ps=ps)
        return None

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
        if native_name.endswith((".linear_attn.dt_bias", ".linear_attn.A_log")):
            shards = _allgather_tp_shards(tensor, ps)
            if _linear_attn_head_replication(self.config, ps.tp_size):
                return shards[0]
            return torch.cat(shards, dim=0).contiguous()
        if native_name.endswith(".moe.shared_expert.gate_up.linear.weight"):
            return _merge_gate_up_tp_shards(_allgather_tp_shards(tensor, ps))
        return None

    def merge_dense_shards(
        self, native_name: str, shards: list[torch.Tensor]
    ) -> torch.Tensor | None:
        if native_name.endswith(".linear_attn.in_proj.linear.weight"):
            return _merge_linear_attn_in_proj_tp_shards(shards, cfg=self.config)
        if native_name.endswith(".linear_attn.conv1d.weight"):
            return _merge_linear_attn_conv1d_tp_shards(shards, cfg=self.config)
        if native_name.endswith((".linear_attn.dt_bias", ".linear_attn.A_log")):
            if _linear_attn_head_replication(self.config, len(shards)):
                return shards[0]
            return torch.cat(shards, dim=0).contiguous()
        if native_name.endswith(".moe.shared_expert.gate_up.linear.weight"):
            return _merge_gate_up_tp_shards(shards)
        return None

    def packed_expert_group_name(self, native_name: str) -> str | None:
        if (
            re.fullmatch(r"layers\.\d+\.moe\.experts\.fc[12]\.weight\d+", native_name)
            is None
        ):
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
        if native_name == "mtp_embed.embedding.weight" or native_name.startswith(
            "mtp."
        ):
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
        if native_name == "mtp_embed.embedding.weight" or native_name.startswith(
            "mtp."
        ):
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
            native_name.endswith(suffix) for suffix in (".linear_attn.conv1d.weight",)
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


def load_hf_weights(
    model: nn.Module, path: str, config: Qwen35Config, ps: ParallelState
) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        load_hf_weights as _load,
    )

    _load(model, path, Qwen35WeightSpec(config), ps, vocab_size=config.vocab_size)


def export_hf_weights(
    model: nn.Module | list[nn.Module],
    config: Qwen35Config,
    ps: ParallelState,
    **kwargs,
):
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        export_hf_weights as _export,
    )

    include_mtp_only = kwargs.pop("include_mtp_only", False)
    kwargs.pop("include_local_prefixes", None)
    target = kwargs.pop("target", "hf")
    resync_config = kwargs.pop("resync_config", None)
    if target in {"hf", "vllm", ResyncFormat.BF16.value}:
        if resync_config:
            raise ValueError("Qwen3.5 resync_config requires target='mxfp4'")
        if include_mtp_only:
            return
        spec_target = "hf" if target == ResyncFormat.BF16.value else target
        yield from _export(
            model,
            Qwen35WeightSpec(config, target=spec_target),
            ps,
            vocab_size=config.vocab_size,
            **kwargs,
        )
        return
    if ResyncFormat.parse(target) is not ResyncFormat.MXFP4:
        raise ValueError(f"Qwen3.5 does not support resync target {target!r}")
    if resync_config:
        raise ValueError("Qwen3.5 MXFP4 resync does not accept resync_config")
    if include_mtp_only:
        return
    yield from _export_mxfp4_weights(
        _export(
            model,
            Qwen35WeightSpec(config),
            ps,
            vocab_size=config.vocab_size,
            **kwargs,
        )
    )


def _export_mxfp4_weights(weights):
    """Convert the Qwen3.5 HF stream to compressed-tensors MXFP4 tensors."""
    for name, tensor in weights:
        ignored = name.endswith(
            ("embed_tokens.weight", "lm_head.weight", ".mlp.gate.weight")
        )
        if (
            ignored
            or not name.endswith(".weight")
            or tensor.ndim != 2
            or not tensor.dtype.is_floating_point
        ):
            yield name, tensor
            continue
        if tensor.shape[-1] % MXFP4_BLOCK_SIZE:
            raise ValueError(
                f"MXFP4 weight {name!r} has input dimension {tensor.shape[-1]}, "
                f"which is not divisible by {MXFP4_BLOCK_SIZE}"
            )
        packed, scale = quantize_mxfp4(tensor)
        yield name, packed.view(torch.uint8)
        yield f"{name[:-7]}.weight_scale", scale.view(torch.uint8)


def save_hf_weights(
    model: nn.Module | list[nn.Module],
    path: str,
    config: Qwen35Config,
    ps: ParallelState,
) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        save_hf_weights as _save,
    )

    _save(model, path, Qwen35WeightSpec(config), ps, vocab_size=config.vocab_size)


__all__ = [
    "EXPERT_CLASSIFIER",
    "PLACEMENT_FN",
    "Qwen35WeightSpec",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]

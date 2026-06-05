"""Qwen3MoE WeightSpec — name mapping + format conversion.

Orchestration (TP scatter, EP sharding, PP remap) lives in
primitive/ckpt/weight_loader.py. This file only defines what's
model-specific: the weight map and tensor conversions.
"""

from __future__ import annotations

import torch
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig
from megatron.lite.primitive.ckpt.dcp import (  # noqa: F401 — re-export
    canonicalize_fc1_for_dcp,
    canonicalize_qkv_for_dcp,
    decanon_fc1_after_dcp,
    decanon_qkv_after_dcp,
)
from megatron.lite.primitive.ckpt.hf_bridge import extract_layer_idx, parse_expert_idx


def _pack_mcore_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: Qwen3MoEConfig) -> torch.Tensor:
    q_per_group = config.num_attention_heads // config.num_key_value_heads
    q = q.view(config.num_key_value_heads, q_per_group * config.head_dim, -1)
    k = k.view(config.num_key_value_heads, config.head_dim, -1)
    v = v.view(config.num_key_value_heads, config.head_dim, -1)
    return torch.cat([q, k, v], dim=1).reshape(-1, q.shape[-1]).contiguous()


def _unpack_mcore_qkv(tensor: torch.Tensor, config: Qwen3MoEConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_per_group = config.num_attention_heads // config.num_key_value_heads
    group_width = (q_per_group + 2) * config.head_dim
    packed = tensor.view(config.num_key_value_heads, group_width, -1)
    q_end = q_per_group * config.head_dim
    k_end = q_end + config.head_dim
    q = packed[:, :q_end].reshape(config.num_attention_heads * config.head_dim, -1)
    k = packed[:, q_end:k_end].reshape(config.num_key_value_heads * config.head_dim, -1)
    v = packed[:, k_end:].reshape(config.num_key_value_heads * config.head_dim, -1)
    return q, k, v


class Qwen3MoEWeightSpec:
    """WeightSpec for Qwen3MoE native impl."""

    def __init__(self, config: Qwen3MoEConfig):
        self.config = config

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def weight_map(self) -> dict[str, list[str]]:
        c = self.config
        wm: dict[str, list[str]] = {
            "embed.embedding.weight": ["model.embed_tokens.weight"],
            "mtp_embed.embedding.weight": ["model.embed_tokens.weight"],
            "norm.weight": ["model.norm.weight"],
            "head.col.linear.weight": ["lm_head.weight"],
        }
        for li in range(c.num_hidden_layers):
            ap = f"model.layers.{li}.self_attn"
            mp = f"model.layers.{li}.mlp"
            lp = f"layers.{li}"
            wm.update({
                f"{lp}.attn.qkv.linear.layer_norm_weight": [f"model.layers.{li}.input_layernorm.weight"],
                f"{lp}.attn.qkv.linear.weight": [f"{ap}.q_proj.weight", f"{ap}.k_proj.weight", f"{ap}.v_proj.weight"],
                f"{lp}.attn.q_norm.weight": [f"{ap}.q_norm.weight"],
                f"{lp}.attn.k_norm.weight": [f"{ap}.k_norm.weight"],
                f"{lp}.attn.proj.linear.weight": [f"{ap}.o_proj.weight"],
                f"{lp}.mlp_norm.weight": [f"model.layers.{li}.post_attention_layernorm.weight"],
                f"{lp}.moe.router.gate.weight": [f"{mp}.gate.weight"],
            })
            for e in range(c.num_experts):
                wm[f"{lp}.moe.experts._fc1_weight_{e}"] = [
                    f"{mp}.experts.{e}.gate_proj.weight",
                    f"{mp}.experts.{e}.up_proj.weight",
                ]
                wm[f"{lp}.moe.experts._fc2_weight_{e}"] = [
                    f"{mp}.experts.{e}.down_proj.weight",
                ]
        for mi in range(c.num_nextn_predict_layers):
            hf_li = c.num_hidden_layers + mi
            hp = f"model.layers.{hf_li}"
            ap = f"{hp}.self_attn"
            mp = f"{hp}.mlp"
            lp = f"mtp.layers.{mi}"
            tlp = f"{lp}.transformer_layer"
            wm.update({
                f"{lp}.enorm.weight": [f"{hp}.enorm.weight"],
                f"{lp}.hnorm.weight": [f"{hp}.hnorm.weight"],
                f"{lp}.eh_proj.linear.weight": [f"{hp}.eh_proj.weight"],
                f"{lp}.final_layernorm.weight": [f"{hp}.shared_head.norm.weight"],
                f"{tlp}.attn.qkv.linear.layer_norm_weight": [f"{hp}.input_layernorm.weight"],
                f"{tlp}.attn.qkv.linear.weight": [f"{ap}.q_proj.weight", f"{ap}.k_proj.weight", f"{ap}.v_proj.weight"],
                f"{tlp}.attn.q_norm.weight": [f"{ap}.q_norm.weight"],
                f"{tlp}.attn.k_norm.weight": [f"{ap}.k_norm.weight"],
                f"{tlp}.attn.proj.linear.weight": [f"{ap}.o_proj.weight"],
                f"{tlp}.mlp_norm.weight": [f"{hp}.post_attention_layernorm.weight"],
                f"{tlp}.moe.router.gate.weight": [f"{mp}.gate.weight"],
            })
            for e in range(c.num_experts):
                wm[f"{tlp}.moe.experts._fc1_weight_{e}"] = [
                    f"{mp}.experts.{e}.gate_proj.weight",
                    f"{mp}.experts.{e}.up_proj.weight",
                ]
                wm[f"{tlp}.moe.experts._fc2_weight_{e}"] = [
                    f"{mp}.experts.{e}.down_proj.weight",
                ]
        return wm

    def hf_to_native(self, bb_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(hf_tensors) == 3 and "qkv" in bb_name:
            # Match MCore SelfAttention's local qkv packing:
            # [q heads for kv-group 0, k0, v0, q heads for kv-group 1, k1, v1, ...].
            return _pack_mcore_qkv(*hf_tensors, self.config)
        if len(hf_tensors) == 2:
            # gate + up → concat
            return torch.cat(hf_tensors, dim=0)
        t = hf_tensors[0]
        if "router.gate.weight" in bb_name:
            return t[: self.config.num_experts]
        return t

    def native_to_hf(self, bb_name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        c = self.config
        if bb_name == "mtp_embed.embedding.weight":
            return []
        if bb_name.startswith("mtp.layers."):
            parts = bb_name.split(".")
            mtp_idx = int(parts[2])
            hf_li = c.num_hidden_layers + mtp_idx
            hp = f"model.layers.{hf_li}"
            if bb_name.endswith(".enorm.weight"):
                return [(f"{hp}.enorm.weight", tensor)]
            if bb_name.endswith(".hnorm.weight"):
                return [(f"{hp}.hnorm.weight", tensor)]
            if bb_name.endswith(".eh_proj.linear.weight"):
                return [(f"{hp}.eh_proj.weight", tensor)]
            if bb_name.endswith(".final_layernorm.weight"):
                return [(f"{hp}.shared_head.norm.weight", tensor)]
            proxy = bb_name.replace(f"mtp.layers.{mtp_idx}.transformer_layer", f"layers.{hf_li}")
            return self.native_to_hf(proxy, tensor)
        if "embed" in bb_name and "embedding" in bb_name:
            return [("model.embed_tokens.weight", tensor)]
        if bb_name.endswith("norm.weight") and "layers" not in bb_name and "attn" not in bb_name and "mlp" not in bb_name:
            return [("model.norm.weight", tensor)]
        if "head" in bb_name:
            return [("lm_head.weight", tensor)]
        if "layer_norm_weight" in bb_name and "qkv" in bb_name:
            li = extract_layer_idx(bb_name)
            return [(f"model.layers.{li}.input_layernorm.weight", tensor)]
        if "mlp_norm" in bb_name:
            li = extract_layer_idx(bb_name)
            return [(f"model.layers.{li}.post_attention_layernorm.weight", tensor)]
        if "qkv" in bb_name and "layer_norm" not in bb_name:
            li = extract_layer_idx(bb_name)
            ap = f"model.layers.{li}.self_attn"
            q, k, v = _unpack_mcore_qkv(tensor, c)
            return [
                (f"{ap}.q_proj.weight", q),
                (f"{ap}.k_proj.weight", k),
                (f"{ap}.v_proj.weight", v),
            ]
        if "q_norm" in bb_name:
            li = extract_layer_idx(bb_name)
            return [(f"model.layers.{li}.self_attn.q_norm.weight", tensor)]
        if "k_norm" in bb_name:
            li = extract_layer_idx(bb_name)
            return [(f"model.layers.{li}.self_attn.k_norm.weight", tensor)]
        if "proj.linear" in bb_name:
            li = extract_layer_idx(bb_name)
            return [(f"model.layers.{li}.self_attn.o_proj.weight", tensor)]
        if "router.gate" in bb_name:
            li = extract_layer_idx(bb_name)
            return [(f"model.layers.{li}.mlp.gate.weight", tensor)]
        if "experts" in bb_name and "fc1" in bb_name:
            li = extract_layer_idx(bb_name)
            ei = parse_expert_idx(bb_name)
            mp = f"model.layers.{li}.mlp"
            gate, up = tensor.chunk(2, dim=0)
            return [(f"{mp}.experts.{ei}.gate_proj.weight", gate), (f"{mp}.experts.{ei}.up_proj.weight", up)]
        if "experts" in bb_name and "fc2" in bb_name:
            li = extract_layer_idx(bb_name)
            ei = parse_expert_idx(bb_name)
            return [(f"model.layers.{li}.mlp.experts.{ei}.down_proj.weight", tensor)]
        return [(bb_name, tensor)]

    def qkv_spec(self, bb_name: str) -> tuple[int, int, int] | None:
        return None

    def tp_spec(self, bb_name: str) -> tuple[int, int] | None:
        if self.is_expert(bb_name):
            if "fc1" in bb_name:
                return (0, 1)  # ETP dim 0
            if "fc2" in bb_name:
                return (1, 1)  # ETP dim 1
            return None
        if "eh_proj" in bb_name:
            return (0, 0)
        if "qkv" in bb_name and "layer_norm" not in bb_name:
            return (0, 0)
        if "proj" in bb_name and "attn" in bb_name:
            return (1, 0)
        if "embed" in bb_name or "head" in bb_name:
            return (0, 0)
        return None

    def is_expert(self, bb_name: str) -> bool:
        return "experts" in bb_name and "router" not in bb_name

    def expert_global_id(self, bb_name: str) -> int | None:
        if "_fc1_weight_" in bb_name or "_fc2_weight_" in bb_name:
            return int(bb_name.split("_")[-1])
        return None

    def expert_local_name(self, bb_name: str, local_idx: int) -> str:
        prefix = bb_name.rsplit("._fc", 1)[0]
        fc_tag = "fc1" if "_fc1_weight_" in bb_name else "fc2"
        return f"{prefix}.{fc_tag}.weight{local_idx}"


# ---------------------------------------------------------------------------
# Convenience: standalone functions wrapping WeightSpec + generic loader
# ---------------------------------------------------------------------------


def load_hf_weights(model, path: str, config: Qwen3MoEConfig, ps) -> None:
    from megatron.lite.primitive.ckpt.hf_bridge import load_hf_weights as _load

    _load(model, path, Qwen3MoEWeightSpec(config), ps, vocab_size=config.vocab_size)


def export_hf_weights(model, config: Qwen3MoEConfig, ps, **kwargs):
    from megatron.lite.primitive.ckpt.hf_bridge import export_hf_weights as _export

    yield from _export(model, Qwen3MoEWeightSpec(config), ps, vocab_size=config.vocab_size, **kwargs)


def save_hf_weights(model, path: str, config: Qwen3MoEConfig, ps) -> None:
    from megatron.lite.primitive.ckpt.hf_bridge import save_hf_weights as _save

    _save(model, path, Qwen3MoEWeightSpec(config), ps, vocab_size=config.vocab_size)


def EXPERT_CLASSIFIER(name: str) -> bool:
    return "experts" in name and "router" not in name


def PLACEMENT_FN(param_name: str) -> list:
    if "experts" in param_name and "router" not in param_name:
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    if "eh_proj" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "qkv" in param_name and "layer_norm" not in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "proj" in param_name and "attn" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]

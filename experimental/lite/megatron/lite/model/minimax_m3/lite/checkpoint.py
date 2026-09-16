# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite <-> Hugging Face checkpoint mapping.

Primary source spelling is the **on-disk** ``MiniMaxAI/MiniMax-M3`` checkpoint
(sglang-style names). The HF *module* spelling (what ``transformers`` 5.16.1
materialises, and what ``tools/minimax_m3/slice_ckpt.py`` writes for Truncated-M3)
is accepted as a fallback through ``hf_name_candidates`` / ``transform_hf_source``.

    disk                                                  HF module
    language_model.model.embed_tokens.weight              model.embed_tokens.weight
    language_model.lm_head.weight                         lm_head.weight
    ...layers.N.self_attn.{q,k,v,o}_proj / {q,k}_norm      same
    ...layers.N.self_attn.index_{q,k}_{proj,norm}          ...self_attn.indexer.{q,k}_{proj,norm}
    ...layers.N.mlp.{gate,up,down}_proj      (dense)       ...mlp.gate_up_proj (fused) / down_proj
    ...layers.N.block_sparse_moe.gate.weight               ...mlp.gate.weight
    ...layers.N.block_sparse_moe.e_score_correction_bias   ...mlp.gate.e_score_correction_bias
    ...layers.N.block_sparse_moe.shared_experts.*_proj     ...mlp.shared_experts.gate_up_proj / down_proj
    ...layers.N.block_sparse_moe.experts.E.w1/w3/w2        ...mlp.experts.gate_up_proj[E] (w1|w3) / down_proj[E]

Gemma RMSNorm weights are stored zero-centred in HF (``(1 + w)``) and consumed
as ``zero_centered_gamma`` by TE: **no +-1 shift** anywhere.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config
from megatron.lite.primitive.parallel import ParallelState


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".moe.experts." in name and "router" not in name and "shared" not in name


def PLACEMENT_FN(param_name: str) -> list:
    if EXPERT_CLASSIFIER(param_name):
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
        return [Replicate(), Replicate(), Replicate(), Replicate()]
    if ".attn.qkv.linear.weight" in param_name or ".attn.indexer.q_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if ".attn.proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "gate_up.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "down.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


def disk_to_module_name(name: str) -> str:
    """Checkpoint (sglang) spelling -> HF module spelling (mirrors transformers/conversion_mapping.py)."""
    name = re.sub(r"^language_model\.lm_head", "lm_head", name)
    name = re.sub(r"^language_model\.model\.", "model.", name)
    name = name.replace(".block_sparse_moe.", ".mlp.")
    name = name.replace(".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
    name = re.sub(r"\.self_attn\.index_(q|k)_(proj|norm)\.", r".self_attn.indexer.\1_\2.", name)
    return name


class MiniMaxM3WeightSpec:
    """HFWeights implementation for MiniMax-M3 lite."""

    def __init__(self, config: MiniMaxM3Config):
        self.config = config
        self._expert_export_buffers: dict[tuple[int, str], dict[int, torch.Tensor]] = {}

    # ------------------------------------------------------------------ names
    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def weight_map(self) -> dict[str, list[str]]:
        c = self.config
        p = c.hf_text_prefix
        wm: dict[str, list[str]] = {
            "embed.embedding.weight": [f"{p}.embed_tokens.weight"],
            "norm.weight": [f"{p}.norm.weight"],
            "head.col.linear.weight": [f"{c.hf_head_name}.weight"],
        }
        for i in range(c.num_hidden_layers):
            lp = f"layers.{i}"
            hp = f"{p}.layers.{i}"
            attn = f"{hp}.self_attn"
            wm.update(
                {
                    f"{lp}.attn.qkv.linear.layer_norm_weight": [f"{hp}.input_layernorm.weight"],
                    f"{lp}.attn.qkv.linear.weight": [
                        f"{attn}.q_proj.weight",
                        f"{attn}.k_proj.weight",
                        f"{attn}.v_proj.weight",
                    ],
                    f"{lp}.attn.q_norm.weight": [f"{attn}.q_norm.weight"],
                    f"{lp}.attn.k_norm.weight": [f"{attn}.k_norm.weight"],
                    f"{lp}.attn.proj.linear.weight": [f"{attn}.o_proj.weight"],
                }
            )
            if c.is_sparse_attention_layer(i):
                wm.update(
                    {
                        f"{lp}.attn.indexer.q_proj.linear.weight": [f"{attn}.index_q_proj.weight"],
                        f"{lp}.attn.indexer.k_proj.weight": [f"{attn}.index_k_proj.weight"],
                        f"{lp}.attn.indexer.q_norm.weight": [f"{attn}.index_q_norm.weight"],
                        f"{lp}.attn.indexer.k_norm.weight": [f"{attn}.index_k_norm.weight"],
                    }
                )
            if c.is_moe_layer(i):
                moe = f"{hp}.block_sparse_moe"
                wm.update(
                    {
                        f"{lp}.mlp_norm.weight": [f"{hp}.post_attention_layernorm.weight"],
                        f"{lp}.moe.router.gate.weight": [f"{moe}.gate.weight"],
                        f"{lp}.moe.router.expert_bias": [f"{moe}.e_score_correction_bias"],
                        f"{lp}.moe.shared_expert.gate_up.linear.weight": [
                            f"{moe}.shared_experts.gate_proj.weight",
                            f"{moe}.shared_experts.up_proj.weight",
                        ],
                        f"{lp}.moe.shared_expert.down.linear.weight": [f"{moe}.shared_experts.down_proj.weight"],
                    }
                )
                for e in range(c.num_experts):
                    wm[f"{lp}.moe.experts.fc1.weight{e}"] = [
                        f"{moe}.experts.{e}.w1.weight",
                        f"{moe}.experts.{e}.w3.weight",
                    ]
                for e in range(c.num_experts):
                    wm[f"{lp}.moe.experts.fc2.weight{e}"] = [f"{moe}.experts.{e}.w2.weight"]
            else:
                mlp = f"{hp}.mlp"
                wm.update(
                    {
                        f"{lp}.mlp.gate_up.linear.layer_norm_weight": [f"{hp}.post_attention_layernorm.weight"],
                        f"{lp}.mlp.gate_up.linear.weight": [f"{mlp}.gate_proj.weight", f"{mlp}.up_proj.weight"],
                        f"{lp}.mlp.down.linear.weight": [f"{mlp}.down_proj.weight"],
                    }
                )
        return wm

    # ------------------------------------------------------------------ HF-module fallback
    def hf_name_candidates(self, native_name: str, hf_name: str) -> list[str]:
        cands = [hf_name]
        mod = disk_to_module_name(hf_name)
        if mod != hf_name:
            cands.append(mod)
        # fused HF-module tensors
        m = re.match(r"^(.*)\.(gate|up)_proj\.weight$", mod)
        if m and ".experts." not in mod:
            cands.append(f"{m.group(1)}.gate_up_proj.weight")
        m = re.match(r"^(.*\.experts)\.(\d+)\.(w1|w3)\.weight$", mod)
        if m:
            cands.append(f"{m.group(1)}.gate_up_proj")
        m = re.match(r"^(.*\.experts)\.(\d+)\.w2\.weight$", mod)
        if m:
            cands.append(f"{m.group(1)}.down_proj")
        return cands

    def transform_hf_source(self, native_name: str, source_index: int, resolved_name: str, tensor: torch.Tensor):
        if resolved_name.endswith(".experts.gate_up_proj"):
            e = self.expert_global_id(native_name)
            gate, up = tensor[e].chunk(2, dim=0)
            return gate if source_index == 0 else up
        if resolved_name.endswith(".experts.down_proj"):
            return tensor[self.expert_global_id(native_name)]
        if resolved_name.endswith(".gate_up_proj.weight"):
            gate, up = tensor.chunk(2, dim=0)
            return gate if source_index == 0 else up
        return tensor

    # ------------------------------------------------------------------ tensor math
    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(hf_tensors) > 1:
            return torch.cat(hf_tensors, dim=0)  # qkv (flat layout) / gate|up / w1|w3
        return hf_tensors[0]

    def native_to_hf(self, native_name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        c = self.config
        p = c.hf_text_prefix
        if native_name == "embed.embedding.weight":
            return [(f"{p}.embed_tokens.weight", tensor)]
        if native_name == "norm.weight":
            return [(f"{p}.norm.weight", tensor)]
        if native_name == "head.col.linear.weight":
            return [(f"{c.hf_head_name}.weight", tensor)]
        m = re.match(r"layers\.(\d+)\.(.*)", native_name)
        if m is None:
            return []
        i, suffix = int(m.group(1)), m.group(2)
        hp = f"{p}.layers.{i}"
        attn, moe, mlp = f"{hp}.self_attn", f"{hp}.block_sparse_moe", f"{hp}.mlp"
        if suffix == "attn.qkv.linear.layer_norm_weight":
            return [(f"{hp}.input_layernorm.weight", tensor)]
        if suffix == "attn.qkv.linear.weight":
            q_dim = c.num_attention_heads * c.head_dim
            kv_dim = c.num_key_value_heads * c.head_dim
            q, k, v = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
            return [
                (f"{attn}.q_proj.weight", q.contiguous()),
                (f"{attn}.k_proj.weight", k.contiguous()),
                (f"{attn}.v_proj.weight", v.contiguous()),
            ]
        simple = {
            "attn.q_norm.weight": f"{attn}.q_norm.weight",
            "attn.k_norm.weight": f"{attn}.k_norm.weight",
            "attn.proj.linear.weight": f"{attn}.o_proj.weight",
            "attn.indexer.q_proj.linear.weight": f"{attn}.index_q_proj.weight",
            "attn.indexer.k_proj.weight": f"{attn}.index_k_proj.weight",
            "attn.indexer.q_norm.weight": f"{attn}.index_q_norm.weight",
            "attn.indexer.k_norm.weight": f"{attn}.index_k_norm.weight",
            "mlp_norm.weight": f"{hp}.post_attention_layernorm.weight",
            "mlp.gate_up.linear.layer_norm_weight": f"{hp}.post_attention_layernorm.weight",
            "mlp.down.linear.weight": f"{mlp}.down_proj.weight",
            "moe.router.gate.weight": f"{moe}.gate.weight",
            "moe.router.expert_bias": f"{moe}.e_score_correction_bias",
            "moe.shared_expert.down.linear.weight": f"{moe}.shared_experts.down_proj.weight",
        }
        if suffix in simple:
            return [(simple[suffix], tensor)]
        if suffix == "mlp.gate_up.linear.weight":
            g, u = tensor.chunk(2, dim=0)
            return [(f"{mlp}.gate_proj.weight", g.contiguous()), (f"{mlp}.up_proj.weight", u.contiguous())]
        if suffix == "moe.shared_expert.gate_up.linear.weight":
            g, u = tensor.chunk(2, dim=0)
            return [
                (f"{moe}.shared_experts.gate_proj.weight", g.contiguous()),
                (f"{moe}.shared_experts.up_proj.weight", u.contiguous()),
            ]
        em = re.fullmatch(r"moe\.experts\.fc([12])\.weight(\d+)", suffix)
        if em is not None:
            kind, e = em.group(1), int(em.group(2))
            if kind == "1":
                w1, w3 = tensor.chunk(2, dim=0)
                return [
                    (f"{moe}.experts.{e}.w1.weight", w1.contiguous()),
                    (f"{moe}.experts.{e}.w3.weight", w3.contiguous()),
                ]
            return [(f"{moe}.experts.{e}.w2.weight", tensor.contiguous())]
        return []

    # ------------------------------------------------------------------ sharding hooks
    def merge_dense_shards(self, native_name: str, shards: list[torch.Tensor]) -> torch.Tensor | None:
        """Inverse of ``split_qkv`` / ``split_gate_up``: each TP shard is ``[q_i|k_i|v_i]`` (or ``[gate_i|up_i]``)."""
        if len(shards) <= 1:
            return None
        if native_name.endswith(".attn.qkv.linear.weight"):
            c = self.config
            tp = len(shards)
            if c.num_key_value_heads < tp:
                return None  # replicated KV: shards are contiguous slices of [Q | K | V]; the default cat is right
            q_rows = c.num_attention_heads * c.head_dim // tp
            kv_rows = c.num_key_value_heads * c.head_dim // tp
            parts = [shard.split([q_rows, kv_rows, kv_rows], dim=0) for shard in shards]
            return torch.cat([torch.cat([p[i] for p in parts]) for i in range(3)]).contiguous()
        if native_name.endswith((".mlp.gate_up.linear.weight", ".moe.shared_expert.gate_up.linear.weight")):
            halves = [shard.chunk(2, dim=0) for shard in shards]
            return torch.cat([torch.cat([h[0] for h in halves]), torch.cat([h[1] for h in halves])]).contiguous()
        return None

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        if native_name.endswith(".attn.qkv.linear.weight"):
            c = self.config
            return (c.num_attention_heads, c.num_key_value_heads, c.head_dim)
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
        if native_name.endswith((".attn.qkv.linear.weight", ".attn.indexer.q_proj.linear.weight")):
            return (0, 0)
        if native_name.endswith(".attn.proj.linear.weight"):
            return (1, 0)
        if native_name.endswith((".mlp.gate_up.linear.weight", ".moe.shared_expert.gate_up.linear.weight")):
            return (0, 0)
        if native_name.endswith((".mlp.down.linear.weight", ".moe.shared_expert.down.linear.weight")):
            return (1, 0)
        return None  # norms, router, index-K (replicated)

    def is_expert(self, native_name: str) -> bool:
        return EXPERT_CLASSIFIER(native_name)

    def expert_global_id(self, native_name: str) -> int | None:
        m = re.search(r"\.weight(\d+)$", native_name)
        return int(m.group(1)) if m is not None else None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        return re.sub(r"\.weight\d+$", f".weight{local_idx}", native_name)

    # No packed_expert_group_name: the disk format is per-expert (w1/w3/w2), so export emits one tensor per expert.


def load_hf_weights(model: nn.Module, path: str, config: MiniMaxM3Config, ps: ParallelState) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import load_hf_weights as _load  # isort: skip

    _load(model, path, MiniMaxM3WeightSpec(config), ps, vocab_size=config.vocab_size)


def export_hf_weights(model, config: MiniMaxM3Config, ps: ParallelState, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as _export  # isort: skip

    kwargs.pop("include_mtp_only", None)
    kwargs.pop("include_local_prefixes", None)
    kwargs.pop("target", None)
    yield from _export(model, MiniMaxM3WeightSpec(config), ps, vocab_size=config.vocab_size, **kwargs)


def save_hf_weights(model, path: str, config: MiniMaxM3Config, ps: ParallelState, **kwargs) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import save_hf_weights as _save  # isort: skip

    _save(model, path, MiniMaxM3WeightSpec(config), ps, vocab_size=config.vocab_size, **kwargs)

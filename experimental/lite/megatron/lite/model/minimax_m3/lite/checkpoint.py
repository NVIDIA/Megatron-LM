# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 HF weight mapping (``minimax_m3_vl`` checkpoints, text tower only).

HF tensors live under ``language_model.``; MXFP8 (1x32 E8M0) linears and NVFP4
experts are decoded to FP32 by ``SafeTensorReader`` and rounded once to the
parameter dtype. Model layer ``i`` reads HF layer ``config.hf_layer_index(i)``
(layer selection). The fused QKV weight is grouped per KV head:
``[q rows of group g; k_g; v_g]`` for ``g`` in KV groups.
"""

from __future__ import annotations

import logging
import re

import torch
from torch.distributed.tensor import Replicate, Shard

from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

logger = logging.getLogger(__name__)

_LM = "language_model.model"


def _pack_grouped_qkv(q, k, v, config: MiniMaxM3Config) -> torch.Tensor:
    groups = config.num_key_value_heads
    q_rows = config.num_attention_heads // groups * config.head_dim
    q = q.view(groups, q_rows, -1)
    k = k.view(groups, config.head_dim, -1)
    v = v.view(groups, config.head_dim, -1)
    return torch.cat([q, k, v], dim=1).reshape(-1, q.shape[-1]).contiguous()


def _unpack_grouped_qkv(tensor, config: MiniMaxM3Config):
    groups = config.num_key_value_heads
    q_rows = config.num_attention_heads // groups * config.head_dim
    packed = tensor.view(groups, q_rows + 2 * config.head_dim, -1)
    q = packed[:, :q_rows].reshape(-1, tensor.shape[-1])
    k = packed[:, q_rows : q_rows + config.head_dim].reshape(-1, tensor.shape[-1])
    v = packed[:, q_rows + config.head_dim :].reshape(-1, tensor.shape[-1])
    return q, k, v


class MiniMaxM3WeightSpec:
    """Native <-> HF names and conversions for the generic HF loader/exporter."""

    def __init__(self, config: MiniMaxM3Config):
        self.config = config
        self._weight_map: dict[str, list[str]] | None = None

    @property
    def num_experts(self) -> int:
        return self.config.n_routed_experts

    def _layer_map(self, li: int) -> dict[str, list[str]]:
        c = self.config
        hp = f"{_LM}.layers.{c.hf_layer_index(li)}"
        ap = f"{hp}.self_attn"
        lp = f"layers.{li}"
        wm = {
            f"{lp}.input_layernorm.weight": [f"{hp}.input_layernorm.weight"],
            f"{lp}.self_attention.linear_qkv.weight": [
                f"{ap}.q_proj.weight",
                f"{ap}.k_proj.weight",
                f"{ap}.v_proj.weight",
            ],
            f"{lp}.self_attention.linear_proj.weight": [f"{ap}.o_proj.weight"],
            f"{lp}.self_attention.q_norm.weight": [f"{ap}.q_norm.weight"],
            f"{lp}.self_attention.k_norm.weight": [f"{ap}.k_norm.weight"],
            f"{lp}.pre_mlp_layernorm.weight": [f"{hp}.post_attention_layernorm.weight"],
        }
        if c.is_sparse_attention_layer(li):
            for name in ("index_q_proj", "index_k_proj", "index_q_norm", "index_k_norm"):
                wm[f"{lp}.self_attention.{name}.weight"] = [f"{ap}.{name}.weight"]
        if not c.is_moe_layer(li):
            mp = f"{hp}.mlp"
            wm[f"{lp}.mlp.fc1.weight"] = [f"{mp}.gate_proj.weight", f"{mp}.up_proj.weight"]
            wm[f"{lp}.mlp.fc2.weight"] = [f"{mp}.down_proj.weight"]
            return wm
        mp = f"{hp}.block_sparse_moe"
        wm[f"{lp}.mlp.router.gate.weight"] = [f"{mp}.gate.weight"]
        wm[f"{lp}.mlp.router.expert_bias"] = [f"{mp}.e_score_correction_bias"]
        sp = f"{mp}.shared_experts"
        wm[f"{lp}.mlp.shared_expert.fc1.weight"] = [
            f"{sp}.gate_proj.weight",
            f"{sp}.up_proj.weight",
        ]
        wm[f"{lp}.mlp.shared_expert.fc2.weight"] = [f"{sp}.down_proj.weight"]
        for e in range(c.n_routed_experts):
            ep = f"{mp}.experts.{e}"
            wm[f"{lp}.mlp.experts._fc1_weight_{e}"] = [f"{ep}.w1.weight", f"{ep}.w3.weight"]
            wm[f"{lp}.mlp.experts._fc2_weight_{e}"] = [f"{ep}.w2.weight"]
        return wm

    def weight_map(self) -> dict[str, list[str]]:
        if self._weight_map is not None:
            return dict(self._weight_map)
        wm: dict[str, list[str]] = {
            "embed.embedding.weight": [f"{_LM}.embed_tokens.weight"],
            "norm.weight": [f"{_LM}.norm.weight"],
            "head.col.linear.weight": ["language_model.lm_head.weight"],
        }
        for li in range(self.config.num_hidden_layers):
            wm.update(self._layer_map(li))
        self._weight_map = wm
        return dict(wm)

    def hf_to_native(self, native_name: str, hf_tensors: list[torch.Tensor]) -> torch.Tensor:
        if native_name.endswith("linear_qkv.weight"):
            return _pack_grouped_qkv(*hf_tensors, self.config)
        if len(hf_tensors) == 2:
            return torch.cat(hf_tensors, dim=0)
        return hf_tensors[0]

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        for name, hf_names in self._native_to_hf_names(native_name):
            if name.endswith("linear_qkv.weight"):
                return list(zip(hf_names, _unpack_grouped_qkv(tensor, self.config)))
            if len(hf_names) == 2:
                return list(zip(hf_names, tensor.chunk(2, dim=0)))
            if name.endswith("router.expert_bias"):
                return [(hf_names[0], tensor.float())]
            return [(hf_names[0], tensor)]
        return [(native_name, tensor)]

    def _native_to_hf_names(self, native_name: str):
        if self._weight_map is None:
            self.weight_map()
        wm = self._weight_map
        # The exporter names experts by global index: ``...experts.fc1.weight{e}``.
        key = re.sub(r"\.experts\.fc([12])\.weight(\d+)$", r".experts._fc\1_weight_\2", native_name)
        if key in wm:
            yield key, wm[key]

    def qkv_spec(self, native_name: str):
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        # MiniMax-M3 runs with TP = ETP = 1 (MSA kernels fix the head counts).
        return None

    def is_expert(self, native_name: str) -> bool:
        return EXPERT_CLASSIFIER(native_name)

    def expert_global_id(self, native_name: str) -> int | None:
        if "_fc1_weight_" in native_name or "_fc2_weight_" in native_name:
            return int(native_name.rsplit("_", 1)[-1])
        return None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        prefix = native_name.rsplit("._fc", 1)[0]
        fc_tag = "fc1" if "_fc1_weight_" in native_name else "fc2"
        return f"{prefix}.{fc_tag}.weight{local_idx}"


def load_hf_weights(model, path: str, config: MiniMaxM3Config, ps) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import load_hf_weights as _load

    _load(model, path, MiniMaxM3WeightSpec(config), ps, vocab_size=config.vocab_size)


def export_hf_weights(chunks, config: MiniMaxM3Config, ps, **kwargs):
    """Yield HF (name, tensor) pairs (BF16; quantized HF formats are not re-encoded)."""
    from megatron.lite.primitive.ckpt.hf_weights import export_hf_weights as _export

    ignored = {key: kwargs.pop(key) for key in ("target", "resync_config") if key in kwargs}
    if ignored:
        logger.warning("MiniMax-M3 export_hf_weights ignoring unsupported kwargs: %s", ignored)
    yield from _export(
        chunks, MiniMaxM3WeightSpec(config), ps, vocab_size=config.vocab_size, **kwargs
    )


def save_hf_weights(chunks, path: str, config: MiniMaxM3Config, ps, **kwargs) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import save_hf_weights as _save

    if kwargs:
        logger.warning("MiniMax-M3 save_hf_weights ignoring unsupported kwargs: %s", kwargs)
    _save(chunks, path, MiniMaxM3WeightSpec(config), ps, vocab_size=config.vocab_size)


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and "shared" not in name


def PLACEMENT_FN(param_name: str) -> list:
    # Same mesh convention as the other MoE models; TP = ETP = 1 leaves the last dim replicated.
    if EXPERT_CLASSIFIER(param_name):
        if "fc1" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(0)]
        if "fc2" in param_name:
            return [Replicate(), Replicate(), Shard(0), Shard(1)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


__all__ = [
    "EXPERT_CLASSIFIER",
    "MiniMaxM3WeightSpec",
    "PLACEMENT_FN",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]

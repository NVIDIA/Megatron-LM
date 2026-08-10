# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5 (deepseek_v3_2) lite native checkpoint mapping.

Cloned from ``megatron/lite/model/kimi_k2/lite/checkpoint.py`` (deepseek_v3)
and renamed KimiK2 -> Glm5.  The MoE / dense-MLP / embed / head / MTP weight
mapping is structurally identical to Kimi.  The ONLY adaptations are in the
attention block:

  * GLM-5's attention is the DSA primitive ``DynamicSparseAttention`` (wrapped
    by ``Glm5DSAAttention``), whose native parameters live under
    ``self_attention.self_attention.*`` -- i.e. ``q_a_proj`` / ``q_a_layernorm``
    / ``q_b_proj`` / ``kv_a_proj_with_mqa`` / ``kv_a_layernorm`` / ``kv_b_proj``
    / ``o_proj`` plus the indexer (``indexer.wq_b`` / ``indexer.wk`` /
    ``indexer.k_norm`` / ``indexer.weights_proj``).  Kimi's MLA names
    (``linear_q_down_proj`` etc.) are replaced accordingly.
  * The HF weight NAMES match GLM-5's HF model: the DSA submodule names map 1:1
    to ``model.layers.{i}.self_attn.*`` (and ``...self_attn.indexer.*``),
    preserving the names the previous GLM-5 checkpoint used.

GLM-5 is TP=1 (DSA is not tensor-parallel-capable), so the ``_tp`` helpers are
no-ops here; they are kept for structural parity with Kimi.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from megatron.lite.model.glm5.config import Glm5Config
from megatron.lite.primitive.ckpt.hf_weights import parse_expert_idx
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.quantization.mxfp4 import MXFP4_BLOCK_SIZE, quantize_mxfp4
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
    if "eh_proj.linear.weight" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    # GLM-5 DSA is TP=1 (no head sharding of attention projections).
    if "gate_up" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    if "down" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(1)]
    if "embed" in param_name or "head" in param_name:
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Replicate()]


class Glm5WeightSpec:
    """Export GLM-5 lite weights to HF GLM-5 (deepseek_v3_2) names."""

    def __init__(self, config: Glm5Config):
        self.config = config

    @property
    def num_experts(self) -> int:
        return self.config.num_experts

    def weight_map(self) -> dict[str, list[str]]:
        c = self.config
        weight_map: dict[str, list[str]] = {
            "embed.embedding.weight": ["model.embed_tokens.weight"],
            "mtp_embed.embedding.weight": ["model.embed_tokens.weight"],
            "norm.weight": ["model.norm.weight"],
            "head.col.linear.weight": ["lm_head.weight"],
        }
        for global_idx in range(c.num_hidden_layers + c.num_nextn_predict_layers):
            if global_idx < c.num_hidden_layers:
                local_prefix = f"layers.{global_idx}"
            else:
                mtp_idx = global_idx - c.num_hidden_layers
                mtp_prefix = f"mtp.layers.{mtp_idx}"
                local_prefix = f"{mtp_prefix}.transformer_layer"
                hf_prefix = f"model.layers.{global_idx}"
                weight_map.update(
                    {
                        f"{mtp_prefix}.enorm.weight": [f"{hf_prefix}.enorm.weight"],
                        f"{mtp_prefix}.hnorm.weight": [f"{hf_prefix}.hnorm.weight"],
                        f"{mtp_prefix}.eh_proj.linear.weight": [
                            f"{hf_prefix}.eh_proj.weight"
                        ],
                        f"{mtp_prefix}.final_layernorm.weight": [
                            f"{hf_prefix}.shared_head.norm.weight"
                        ],
                    }
                )

            hf_prefix = f"model.layers.{global_idx}"
            attention = f"{hf_prefix}.self_attn"
            mlp = f"{hf_prefix}.mlp"
            weight_map[f"{local_prefix}.input_layernorm.weight"] = [
                f"{hf_prefix}.input_layernorm.weight"
            ]
            for native_suffix, hf_suffix in self._ATTN_SUFFIX_MAP.items():
                if ".indexer." in native_suffix and not c.builds_dsa_indexer(
                    global_idx
                ):
                    continue
                weight_map[f"{local_prefix}.{native_suffix}"] = [
                    f"{attention}.{hf_suffix}"
                ]
            if c.is_moe_layer(global_idx):
                weight_map.update(
                    {
                        f"{local_prefix}.mlp_norm.weight": [
                            f"{hf_prefix}.post_attention_layernorm.weight"
                        ],
                        f"{local_prefix}.moe.router.gate.weight": [
                            f"{mlp}.gate.weight"
                        ],
                        f"{local_prefix}.moe.router.expert_bias": [
                            f"{mlp}.gate.e_score_correction_bias"
                        ],
                        f"{local_prefix}.moe.shared_expert.gate_up.linear.weight": [
                            f"{mlp}.shared_experts.gate_proj.weight",
                            f"{mlp}.shared_experts.up_proj.weight",
                        ],
                        f"{local_prefix}.moe.shared_expert.down.linear.weight": [
                            f"{mlp}.shared_experts.down_proj.weight"
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
            else:
                weight_map.update(
                    {
                        f"{local_prefix}.mlp.gate_up.linear.layer_norm_weight": [
                            f"{hf_prefix}.post_attention_layernorm.weight"
                        ],
                        f"{local_prefix}.mlp.gate_up.linear.weight": [
                            f"{mlp}.gate_proj.weight",
                            f"{mlp}.up_proj.weight",
                        ],
                        f"{local_prefix}.mlp.down.linear.weight": [
                            f"{mlp}.down_proj.weight"
                        ],
                    }
                )
        return weight_map

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        del native_name
        return torch.cat(hf_tensors, dim=0) if len(hf_tensors) == 2 else hf_tensors[0]

    def hf_name_candidates(self, native_name: str, hf_name: str) -> list[str]:
        del native_name
        candidates = [hf_name]
        if hf_name.startswith("model."):
            suffix = hf_name.removeprefix("model.")
            candidates.extend(
                [f"language_model.model.{suffix}", f"model.language_model.{suffix}"]
            )
        if ".shared_experts." in hf_name:
            candidates.extend(
                name.replace(".shared_experts.", ".shared_expert.")
                for name in tuple(candidates)
            )
        if hf_name == "lm_head.weight":
            candidates.extend(
                [
                    "language_model.lm_head.weight",
                    "model.lm_head.weight",
                    "language_model.model.lm_head.weight",
                    "model.language_model.lm_head.weight",
                ]
            )
        if hf_name.endswith(".shared_head.norm.weight"):
            candidates.extend(
                name.replace(".shared_head.norm.weight", ".final_layernorm.weight")
                for name in tuple(candidates)
            )
        return candidates

    # GLM-5 ONLY: DSA attention native suffix -> HF suffix.  The wrapper places
    # the DSA module under `self_attention.self_attention.*`; the HF model uses
    # `self_attn.*` with identical submodule names.
    _ATTN_SUFFIX_MAP: dict[str, str] = {
        "self_attention.self_attention.q_a_proj.weight": "q_a_proj.weight",
        "self_attention.self_attention.q_a_layernorm.weight": "q_a_layernorm.weight",
        "self_attention.self_attention.q_b_proj.weight": "q_b_proj.weight",
        "self_attention.self_attention.kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa.weight",
        "self_attention.self_attention.kv_a_layernorm.weight": "kv_a_layernorm.weight",
        "self_attention.self_attention.kv_b_proj.weight": "kv_b_proj.weight",
        "self_attention.self_attention.o_proj.weight": "o_proj.weight",
        "self_attention.self_attention.indexer.wq_b.weight": "indexer.wq_b.weight",
        "self_attention.self_attention.indexer.wk.weight": "indexer.wk.weight",
        "self_attention.self_attention.indexer.k_norm.weight": "indexer.k_norm.weight",
        "self_attention.self_attention.indexer.k_norm.bias": "indexer.k_norm.bias",
        "self_attention.self_attention.indexer.weights_proj.weight": "indexer.weights_proj.weight",
    }

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        if native_name == "mtp_embed.embedding.weight":
            return []
        if native_name.startswith("mtp.layers."):
            parts = native_name.split(".")
            mtp_idx = int(parts[2])
            hf_layer_idx = self.config.num_hidden_layers + mtp_idx
            hp = f"model.layers.{hf_layer_idx}"
            if native_name.endswith(".enorm.weight"):
                return [(f"{hp}.enorm.weight", tensor)]
            if native_name.endswith(".hnorm.weight"):
                return [(f"{hp}.hnorm.weight", tensor)]
            if native_name.endswith(".eh_proj.linear.weight"):
                return [(f"{hp}.eh_proj.weight", tensor)]
            if native_name.endswith(".final_layernorm.weight"):
                return [(f"{hp}.shared_head.norm.weight", tensor)]
            proxy = native_name.replace(
                f"mtp.layers.{mtp_idx}.transformer_layer", f"layers.{hf_layer_idx}"
            )
            return self.native_to_hf(proxy, tensor)
        if native_name == "embed.embedding.weight":
            return [("model.embed_tokens.weight", tensor)]
        if native_name == "norm.weight":
            return [("model.norm.weight", tensor)]
        if native_name == "head.col.linear.weight":
            return [("lm_head.weight", tensor)]

        parts = native_name.split(".")
        if len(parts) < 3 or parts[0] != "layers":
            return []
        layer_idx = int(parts[1])
        suffix = ".".join(parts[2:])
        hp = f"model.layers.{layer_idx}"
        ap = f"{hp}.self_attn"
        mp = f"{hp}.mlp"

        if suffix == "input_layernorm.weight":
            return [(f"{hp}.input_layernorm.weight", tensor)]

        if suffix.startswith(
            "self_attention.self_attention.indexer."
        ) and not self.config.builds_dsa_indexer(layer_idx):
            return []

        # GLM-5 ONLY: DSA attention (incl. indexer) maps 1:1 onto self_attn.*.
        attn_hf = self._ATTN_SUFFIX_MAP.get(suffix)
        if attn_hf is not None:
            return [(f"{ap}.{attn_hf}", tensor)]

        if suffix == "mlp.gate_up.linear.layer_norm_weight":
            return [(f"{hp}.post_attention_layernorm.weight", tensor)]
        if suffix == "mlp.gate_up.linear.weight":
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{mp}.gate_proj.weight", gate.contiguous()),
                (f"{mp}.up_proj.weight", up.contiguous()),
            ]
        if suffix == "mlp.down.linear.weight":
            return [(f"{mp}.down_proj.weight", tensor)]

        if suffix == "mlp_norm.weight":
            return [(f"{hp}.post_attention_layernorm.weight", tensor)]
        if suffix == "moe.router.gate.weight":
            return [(f"{mp}.gate.weight", tensor)]
        if suffix == "moe.router.expert_bias":
            return [(f"{mp}.gate.e_score_correction_bias", tensor.float())]
        if suffix == "moe.shared_expert.gate_up.linear.weight":
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{mp}.shared_experts.gate_proj.weight", gate.contiguous()),
                (f"{mp}.shared_experts.up_proj.weight", up.contiguous()),
            ]
        if suffix == "moe.shared_expert.down.linear.weight":
            return [(f"{mp}.shared_experts.down_proj.weight", tensor)]

        if ".moe.experts.fc1.weight" in native_name:
            expert_idx = parse_expert_idx(native_name)
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{mp}.experts.{expert_idx}.gate_proj.weight", gate.contiguous()),
                (f"{mp}.experts.{expert_idx}.up_proj.weight", up.contiguous()),
            ]
        if ".moe.experts.fc2.weight" in native_name:
            expert_idx = parse_expert_idx(native_name)
            return [(f"{mp}.experts.{expert_idx}.down_proj.weight", tensor)]

        return []

    def qkv_spec(self, native_name: str) -> tuple[int, int, int] | None:
        del native_name
        return None

    def tp_spec(self, native_name: str) -> tuple[int, int] | None:
        # GLM-5 is TP=1 (DSA not TP-capable); only EP / ETP shard tensors.
        if (
            native_name.startswith("mtp.layers.")
            and ".transformer_layer." in native_name
        ):
            proxy = native_name.replace(".transformer_layer.", ".")
            return self.tp_spec(proxy)
        if native_name.endswith(".eh_proj.linear.weight"):
            return (0, 0)
        if self.is_expert(native_name):
            if ".fc1." in native_name:
                return (0, 1)
            if ".fc2." in native_name:
                return (1, 1)
            return None
        if native_name in {"embed.embedding.weight", "head.col.linear.weight"}:
            return (0, 0)
        if native_name.endswith(".mlp.gate_up.linear.weight"):
            return (0, 0)
        if native_name.endswith(".mlp.down.linear.weight"):
            return (1, 0)
        if native_name.endswith(".moe.shared_expert.gate_up.linear.weight"):
            return (0, 0)
        if native_name.endswith(".moe.shared_expert.down.linear.weight"):
            return (1, 0)
        return None

    def is_expert(self, native_name: str) -> bool:
        return ".moe.experts." in native_name and ".router." not in native_name

    def expert_global_id(self, native_name: str) -> int | None:
        if self.is_expert(native_name):
            return parse_expert_idx(native_name)
        return None

    def expert_local_name(self, native_name: str, local_idx: int) -> str:
        prefix = native_name.rsplit(".weight", 1)[0]
        return f"{prefix}.weight{local_idx}"


def load_hf_weights(
    model: nn.Module, path: str, config: Glm5Config, ps: ParallelState
) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        load_hf_weights as _load,
    )

    _load(model, path, Glm5WeightSpec(config), ps, vocab_size=config.vocab_size)


def export_hf_weights(model, config: Glm5Config, ps: ParallelState, **kwargs):
    from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
        export_hf_weights as _export,
    )

    if config is None:
        raise ValueError("GLM5 HF export requires a non-null model config")
    target = kwargs.pop("target", "hf")
    resync_config = kwargs.pop("resync_config", None)
    spec = Glm5WeightSpec(config)
    weights = _export(model, spec, ps, vocab_size=config.vocab_size, **kwargs)
    if target in {"hf", ResyncFormat.BF16.value}:
        if resync_config:
            raise ValueError("GLM5 resync_config requires target='mxfp4'")
        yield from weights
        return
    if ResyncFormat.parse(target) is not ResyncFormat.MXFP4:
        raise ValueError(f"GLM5 does not support resync target {target!r}")
    if resync_config:
        raise ValueError("GLM5 MXFP4 resync does not accept resync_config")
    yield from _export_mxfp4_weights(weights)


def _export_mxfp4_weights(weights):
    """Convert the GLM5 HF stream to compressed-tensors MXFP4 tensors."""
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
    model, path: str, config: Glm5Config, ps: ParallelState, **kwargs
) -> None:
    """Export + write sharded safetensors via ``stream_export_to_shards``."""
    from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards

    kwargs.pop("cpu", None)
    shard_size_bytes = int(kwargs.pop("shard_size_bytes", 5 * 1024**3))
    stream_export_to_shards(
        export_hf_weights(model, config, ps, rank0_only=True, cpu=True, **kwargs),
        path,
        shard_size_bytes=shard_size_bytes,
    )


__all__ = [
    "EXPERT_CLASSIFIER",
    "Glm5WeightSpec",
    "PLACEMENT_FN",
    "export_hf_weights",
    "load_hf_weights",
    "save_hf_weights",
]

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 / M3.1 language-model configuration (text tower only).

Built from the HF ``minimax_m3_vl`` config's ``text_config``. Every layer has a
sigmoid top-k MoE or a dense FFN (``moe_layer_freq``) and either MiniMax sparse
attention (MSA v1: block top-k indexer + sparse attention) or full causal
attention (``sparse_attention_freq``). The training-only losses are not part of the
HF config and are off by default.

``layer_indices`` selects which HF layers the model's layers are loaded from, so
a shortened model can be built from real weights (e.g. ``[0, 1, 3, 4]``); per-layer
lists are taken at those indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict

_PER_LAYER_FIELDS = ("moe_layer_freq", "sparse_attention_freq")


@dataclass
class MiniMaxM3Config:
    """MiniMax-M3.1-Flash defaults."""

    num_hidden_layers: int = 60
    hidden_size: int = 6144
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 200064
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    rotary_dim: int = 64
    tie_word_embeddings: bool = False

    # FFN: dense layers, routed experts, shared expert. Activation "swigluoai":
    # gate * sigmoid(alpha * gate) * (up + 1), gate clamped above, up clamped both ways.
    hidden_act: str = "swigluoai"
    dense_intermediate_size: int = 12288
    moe_intermediate_size: int = 3072
    shared_intermediate_size: int = 3072
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    n_routed_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "sigmoid"
    routed_scaling_factor: float = 2.0
    moe_layer_freq: list[int] = field(default_factory=lambda: [0] * 3 + [1] * 57)

    # MiniMax sparse attention (MSA v1).
    sparse_attention_freq: list[int] = field(default_factory=lambda: [1] * 60)
    sparse_index_dim: int = 128
    sparse_num_index_heads: int = 4
    sparse_topk_blocks: int = 16
    sparse_block_size: int = 128
    sparse_init_block: int = 0
    sparse_local_block: int = 1
    sparse_score_type: str = "max"

    # Training objective (not in the HF config; all off by default).
    sparse_kl_loss_coeff: float = 0.0
    sparse_kl_temperature: float = 1.0
    aux_loss_alpha: float = 0.0

    # HF layer each model layer is loaded from (None = identity).
    layer_indices: list[int] | None = None

    def __post_init__(self):
        self._validate()

    @property
    def num_experts(self) -> int:
        return self.n_routed_experts

    @property
    def num_query_groups(self) -> int:
        return self.num_key_value_heads

    @property
    def qkv_size(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        return bool(self.moe_layer_freq[layer_idx])

    def is_sparse_attention_layer(self, layer_idx: int) -> bool:
        return bool(self.sparse_attention_freq[layer_idx])

    @property
    def has_full_attention_layers(self) -> bool:
        """Any dense layer (model-wide, so every pipeline stage builds the same runtime key)."""
        return not all(self.sparse_attention_freq)

    def hf_layer_index(self, layer_idx: int) -> int:
        return layer_idx if self.layer_indices is None else self.layer_indices[layer_idx]

    def _validate(self) -> None:
        errors: list[str] = []

        def check(cond: bool, message: str) -> None:
            if not cond:
                errors.append(message)

        n = self.num_hidden_layers
        check(n >= 1, "num_hidden_layers must be >= 1")
        for name in _PER_LAYER_FIELDS:
            check(len(getattr(self, name)) == n, f"len({name}) must equal num_hidden_layers={n}")
        if self.layer_indices is not None:
            check(len(self.layer_indices) == n, f"len(layer_indices) must equal {n}")
            check(len(set(self.layer_indices)) == n, "layer_indices must be distinct")
        check(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "num_attention_heads must be divisible by num_key_value_heads",
        )
        check(0 < self.rotary_dim <= self.head_dim, "rotary_dim must be in (0, head_dim]")
        check(1 <= self.num_experts_per_tok <= self.n_routed_experts, "invalid num_experts_per_tok")
        check(self.scoring_func == "sigmoid", "only sigmoid routing is supported")
        check(self.hidden_act == "swigluoai", "only the swigluoai activation is supported")
        check(self.swiglu_alpha == 1.702, "swigluoai is implemented for swiglu_alpha=1.702")
        check(self.n_shared_experts in (0, 1), "n_shared_experts must be 0 or 1")
        check(self.sparse_score_type == "max", "MSA v1 requires sparse_score_type='max'")
        check(self.sparse_init_block == 0, "MSA v1 requires sparse_init_block=0")
        check(self.sparse_local_block == 1, "MSA v1 requires sparse_local_block=1")
        check(not self.tie_word_embeddings, "tied embeddings are not supported")
        if errors:
            raise ValueError(
                f"Invalid MiniMaxM3Config ({len(errors)} error"
                f"{'s' if len(errors) != 1 else ''}):\n  " + "\n  ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def from_hf(cls, path_or_name: str, **overrides) -> MiniMaxM3Config:
        return cls._from_hf_dict(load_hf_config_dict(path_or_name), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> MiniMaxM3Config:
        hf = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        return cls._from_hf_dict(hf, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> MiniMaxM3Config:
        text = hf.get("text_config", hf)
        if text.get("attention_output_gate"):
            raise ValueError("attention_output_gate is not supported")
        if not text.get("use_qk_norm", True) or text.get("qk_norm_type", "per_head") != "per_head":
            raise ValueError("MiniMax-M3 requires per-head QK RMSNorm")
        if not text.get("use_gemma_norm", True):
            raise ValueError("MiniMax-M3 requires (1 + w) RMSNorm")
        names = {f.name for f in dc_fields(cls)}
        kwargs = {k: v for k, v in text.items() if k in names and v is not None}
        kwargs.pop("layer_indices", None)
        if "num_local_experts" in text:
            kwargs["n_routed_experts"] = text["num_local_experts"]
        if "intermediate_size" in text:
            kwargs["moe_intermediate_size"] = text["intermediate_size"]
        sparse = text.get("sparse_attention_config") or {}
        if sparse.get("use_sparse_attention", False):
            for key in (
                "sparse_index_dim",
                "sparse_num_index_heads",
                "sparse_topk_blocks",
                "sparse_block_size",
                "sparse_score_type",
                "sparse_init_block",
                "sparse_local_block",
                "sparse_attention_freq",
            ):
                if key in sparse:
                    kwargs[key] = sparse[key]
        else:
            kwargs["sparse_attention_freq"] = [0] * int(kwargs.get("num_hidden_layers", 60))
        # HF's canonical per-layer dispatch (transformers configs); must match the legacy fields.
        for field, hf_field, kinds in (
            ("sparse_attention_freq", "layer_types", {"full_attention": 0, "minimax_m3_sparse": 1}),
            ("moe_layer_freq", "mlp_layer_types", {"dense": 0, "sparse": 1}),
        ):
            types = text.get(hf_field)
            if not types:
                continue
            unknown = sorted(set(types) - set(kinds))
            if unknown:
                raise ValueError(f"unsupported {hf_field} entries: {unknown}")
            freq = [kinds[t] for t in types]
            legacy = sparse.get(field) if field == "sparse_attention_freq" else text.get(field)
            if legacy and list(legacy) != freq:
                raise ValueError(f"{hf_field} disagrees with {field}")
            kwargs[field] = freq

        layer_indices = overrides.pop("layer_indices", None)
        if layer_indices is not None:
            layer_indices = [int(i) for i in layer_indices]
            for name in _PER_LAYER_FIELDS:
                if name in overrides:
                    continue
                if name not in kwargs:
                    raise ValueError(f"layer_indices needs an explicit {name} in the HF config")
                kwargs[name] = [kwargs[name][i] for i in layer_indices]
            kwargs["num_hidden_layers"] = len(layer_indices)
            kwargs["layer_indices"] = layer_indices
        kwargs.update(overrides)
        return cls(**kwargs)


__all__ = ["MiniMaxM3Config"]

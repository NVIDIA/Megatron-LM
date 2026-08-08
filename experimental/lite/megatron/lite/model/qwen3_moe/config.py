"""Qwen3MoE model configuration — pure architecture parameters.

Like HuggingFace's model config: only describes the model architecture.
Impl-specific knobs live in protocol.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict


@dataclass
class Qwen3MoEConfig:
    """Pure Qwen3MoE architecture parameters (Qwen3-30B-A3B defaults)."""

    num_hidden_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 151936
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    router_aux_loss_coef: float = 0.001
    num_nextn_predict_layers: int = 0
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool = False
    layer_types: list[str] = field(
        default_factory=lambda: ["full_attention"] * 48
    )

    def __post_init__(self):
        self._validate()

    def _validate(self):
        errors: list[str] = []

        def _check(cond: bool, msg: str):
            if not cond:
                errors.append(msg)

        _check(
            self.hidden_size % self.num_attention_heads == 0,
            f"hidden_size({self.hidden_size}) % num_attention_heads({self.num_attention_heads}) != 0",
        )
        _check(
            self.num_attention_heads % self.num_key_value_heads == 0,
            f"num_attention_heads({self.num_attention_heads}) % num_key_value_heads({self.num_key_value_heads}) != 0",
        )
        _check(self.head_dim > 0, f"head_dim must be > 0, got {self.head_dim}")
        _check(self.num_experts >= 1, f"num_experts must be >= 1, got {self.num_experts}")
        _check(
            1 <= self.num_experts_per_tok <= self.num_experts,
            f"num_experts_per_tok({self.num_experts_per_tok}) not in [1, {self.num_experts}]",
        )
        _check(self.moe_intermediate_size > 0, "moe_intermediate_size must be > 0")
        _check(self.vocab_size > 0, "vocab_size must be > 0")
        _check(self.num_hidden_layers >= 1, "num_hidden_layers must be >= 1")
        _check(self.num_nextn_predict_layers >= 0, "num_nextn_predict_layers must be >= 0")
        _check(
            len(self.layer_types) == self.num_hidden_layers,
            f"len(layer_types)={len(self.layer_types)} != "
            f"num_hidden_layers={self.num_hidden_layers}",
        )
        valid_types = {"full_attention"}
        for i, lt in enumerate(self.layer_types):
            _check(
                lt in valid_types,
                f"layer_types[{i}] must be one of {valid_types}, got '{lt}'",
            )

        if errors:
            raise ValueError(
                f"Invalid Qwen3MoEConfig ({len(errors)} errors):\n  " + "\n  ".join(errors)
            )

    @property
    def qkv_size(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def from_hf(cls, path_or_name: str, **overrides) -> Qwen3MoEConfig:
        hf_dict = load_hf_config_dict(path_or_name)
        return cls._from_hf_dict(hf_dict, **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> Qwen3MoEConfig:
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        return cls._from_hf_dict(hf_dict, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> Qwen3MoEConfig:
        valid_fields = {f.name for f in dc_fields(cls)}
        kwargs = {k: v for k, v in hf.items() if k in valid_fields}

        if "rope_theta" not in kwargs:
            if "rope_parameters" in hf and isinstance(hf["rope_parameters"], dict):
                kwargs["rope_theta"] = float(hf["rope_parameters"].get("rope_theta", 1_000_000.0))

        if "head_dim" not in kwargs or kwargs["head_dim"] is None:
            hs = kwargs.get("hidden_size", 2048)
            nh = kwargs.get("num_attention_heads", 32)
            kwargs["head_dim"] = hs // nh
        if kwargs.get("num_nextn_predict_layers") is None:
            kwargs["num_nextn_predict_layers"] = 0

        if "layer_types" not in kwargs:
            kwargs["layer_types"] = ["full_attention"] * kwargs.get(
                "num_hidden_layers", 48
            )
        kwargs.update(overrides)
        return cls(**kwargs)

"""Qwen3.5 model configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

from megatron.lite.primitive.config import load_hf_config_dict

_HF_FIELDS = frozenset({
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "rms_norm_eps",
    "max_position_embeddings",
    "router_aux_loss_coef",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "layer_types",
    "partial_rotary_factor",
    "mrope_section",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
    "num_nextn_predict_layers",
    "mtp_loss_scaling_factor",
    "mtp_use_repeated_layer",
    "mtp_layer_types",
})


@dataclass
class Qwen35Config:
    """Pure Qwen3.5-35B-A3B architecture parameters."""

    num_hidden_layers: int = 40
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    router_aux_loss_coef: float = 0.001
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    layer_types: list[str] = field(
        default_factory=lambda: (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10
        )
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    mrope_section: list[int] | None = None
    num_nextn_predict_layers: int = 0
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_dedicated_embeddings: bool = False
    mtp_use_repeated_layer: bool = False
    mtp_layer_types: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attn_qkv_size(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    @property
    def linear_conv_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        if (
            self.num_nextn_predict_layers > 0
            and not self.mtp_layer_types
            and len(self.layer_types) == self.num_hidden_layers + self.num_nextn_predict_layers
        ):
            self.mtp_layer_types = self.layer_types[self.num_hidden_layers:]
            self.layer_types = self.layer_types[:self.num_hidden_layers]
        if self.num_nextn_predict_layers > 0 and not self.mtp_layer_types:
            self.mtp_layer_types = ["full_attention"] * self.num_nextn_predict_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(self.layer_types)} != "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        if self.num_nextn_predict_layers > 0 and len(self.mtp_layer_types) != self.num_nextn_predict_layers:
            raise ValueError(
                f"len(mtp_layer_types)={len(self.mtp_layer_types)} != "
                f"num_nextn_predict_layers={self.num_nextn_predict_layers}"
            )
        self._validate()

    def _validate(self):
        errors: list[str] = []

        def _check(cond: bool, msg: str):
            if not cond:
                errors.append(msg)

        _check(self.num_hidden_layers >= 1, f"num_hidden_layers must be >= 1, got {self.num_hidden_layers}")
        _check(
            self.num_nextn_predict_layers >= 0,
            f"num_nextn_predict_layers must be >= 0, got {self.num_nextn_predict_layers}",
        )
        _check(
            not self.mtp_use_dedicated_embeddings,
            "Qwen35Config only supports shared embeddings for MTP "
            "(mtp_use_dedicated_embeddings=False)",
        )
        _check(self.hidden_size > 0, f"hidden_size must be > 0, got {self.hidden_size}")
        _check(self.head_dim > 0, f"head_dim must be > 0, got {self.head_dim}")
        _check(self.vocab_size > 0, f"vocab_size must be > 0, got {self.vocab_size}")
        _check(self.num_attention_heads >= 1, f"num_attention_heads must be >= 1, got {self.num_attention_heads}")
        _check(
            self.num_attention_heads % self.num_key_value_heads == 0,
            f"num_attention_heads({self.num_attention_heads}) must be divisible by "
            f"num_key_value_heads({self.num_key_value_heads})",
        )
        _check(self.num_experts >= 1, f"num_experts must be >= 1, got {self.num_experts}")
        _check(
            1 <= self.num_experts_per_tok <= self.num_experts,
            f"num_experts_per_tok({self.num_experts_per_tok}) must be in "
            f"[1, num_experts({self.num_experts})]",
        )
        _check(self.moe_intermediate_size > 0, f"moe_intermediate_size must be > 0, got {self.moe_intermediate_size}")
        _check(
            self.shared_expert_intermediate_size > 0,
            f"shared_expert_intermediate_size must be > 0, got {self.shared_expert_intermediate_size}",
        )
        _check(self.linear_num_key_heads >= 1, f"linear_num_key_heads must be >= 1, got {self.linear_num_key_heads}")
        _check(self.linear_key_head_dim > 0, f"linear_key_head_dim must be > 0, got {self.linear_key_head_dim}")
        _check(self.linear_num_value_heads >= 1, f"linear_num_value_heads must be >= 1, got {self.linear_num_value_heads}")
        _check(self.linear_value_head_dim > 0, f"linear_value_head_dim must be > 0, got {self.linear_value_head_dim}")
        _check(
            0.0 < self.partial_rotary_factor <= 1.0,
            f"partial_rotary_factor must be in (0, 1], got {self.partial_rotary_factor}",
        )
        if self.mrope_section is not None:
            _check(
                len(self.mrope_section) == 3,
                f"mrope_section must have three entries, got {self.mrope_section}",
            )
            _check(
                all(section >= 0 for section in self.mrope_section),
                f"mrope_section entries must be non-negative, got {self.mrope_section}",
            )
            _check(
                2 * sum(self.mrope_section) == self.rotary_dim,
                f"sum(mrope_section)*2 must equal rotary_dim({self.rotary_dim}), "
                f"got {self.mrope_section}",
            )
        valid_types = {"linear_attention", "full_attention"}
        for i, lt in enumerate(self.layer_types):
            _check(lt in valid_types, f"layer_types[{i}] must be one of {valid_types}, got '{lt}'")
        for i, lt in enumerate(self.mtp_layer_types):
            _check(lt in valid_types, f"mtp_layer_types[{i}] must be one of {valid_types}, got '{lt}'")

        if errors:
            raise ValueError(
                f"Invalid Qwen35Config ({len(errors)} error"
                f"{'s' if len(errors) > 1 else ''}):\n  " + "\n  ".join(errors)
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_hf(cls, path: str, **overrides) -> Qwen35Config:
        hf = load_hf_config_dict(path)
        return cls._from_hf_dict(hf, **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> Qwen35Config:
        return cls._from_hf_dict(hf_config.to_dict(), **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict, **overrides) -> Qwen35Config:
        if "text_config" in hf and isinstance(hf["text_config"], dict):
            hf = hf["text_config"]
        kwargs = {k: v for k, v in hf.items() if k in _HF_FIELDS}
        mtp_num_hidden_layers = kwargs.pop("mtp_num_hidden_layers", None)
        if kwargs.get("num_nextn_predict_layers") is None and mtp_num_hidden_layers is not None:
            kwargs["num_nextn_predict_layers"] = int(mtp_num_hidden_layers)
        if "rope_parameters" in hf and isinstance(hf["rope_parameters"], dict):
            rp = hf["rope_parameters"]
            if "rope_theta" not in kwargs:
                kwargs["rope_theta"] = float(rp.get("rope_theta", 10_000_000.0))
            if "partial_rotary_factor" not in kwargs and "partial_rotary_factor" in rp:
                kwargs["partial_rotary_factor"] = float(rp["partial_rotary_factor"])
            if "mrope_section" not in kwargs and "mrope_section" in rp:
                kwargs["mrope_section"] = list(rp["mrope_section"])
        if "head_dim" not in kwargs or kwargs.get("head_dim") is None:
            kwargs["head_dim"] = kwargs.get("hidden_size", 2048) // kwargs.get("num_attention_heads", 16)
        if kwargs.get("num_nextn_predict_layers") is None:
            kwargs["num_nextn_predict_layers"] = 0
        kwargs.update(overrides)
        return cls(**kwargs)

    def layer_type_at(self, layer_idx: int) -> str:
        if layer_idx < self.num_hidden_layers:
            return self.layer_types[layer_idx]
        mtp_idx = layer_idx - self.num_hidden_layers
        if 0 <= mtp_idx < len(self.mtp_layer_types):
            return self.mtp_layer_types[mtp_idx]
        return "full_attention"

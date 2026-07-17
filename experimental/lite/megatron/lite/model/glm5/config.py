"""GLM-5 architecture config."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict

_HF_FIELDS = frozenset(
    {
        "attention_dropout",
        "calculate_per_token_loss",
        "dsa_indexer_loss_coeff",
        "dsa_indexer_use_sparse_loss",
        "first_k_dense_replace",
        "head_dim",
        "hidden_size",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "indexer_layer_norm_eps",
        "indexer_rope_interleave",
        "indexer_rope_first",
        "indexer_use_hadamard",
        "initializer_range",
        "intermediate_size",
        "kv_lora_rank",
        "max_position_embeddings",
        "mlp_layer_types",
        "moe_intermediate_size",
        "n_group",
        "n_routed_experts",
        "n_shared_experts",
        "norm_topk_prob",
        "mtp_loss_scaling_factor",
        "mtp_use_repeated_layer",
        "num_attention_heads",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "num_nextn_predict_layers",
        "q_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "latent_rms_norm_eps",
        "rms_norm_eps",
        "rope_interleave",
        "rope_theta",
        "routed_scaling_factor",
        "topk_group",
        "v_head_dim",
        "vocab_size",
    }
)


@dataclass
class Glm5Config:
    """Pure GLM-5 MoE + MLA + DSA architecture parameters."""

    num_hidden_layers: int = 78
    hidden_size: int = 6144
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 64
    vocab_size: int = 154880
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_head_dim: int = 256
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256

    index_head_dim: int = 128
    index_n_heads: int = 32
    index_topk: int = 2048
    indexer_layer_norm_eps: float = 1e-6
    indexer_rope_interleave: bool = False
    indexer_rope_first: bool = True
    indexer_use_hadamard: bool = False
    dsa_indexer_loss_coeff: float = 0.0
    dsa_indexer_use_sparse_loss: bool = False
    calculate_per_token_loss: bool = False
    rope_interleave: bool = False
    rope_theta: float = 1_000_000.0
    latent_rms_norm_eps: float = 1e-6

    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    first_k_dense_replace: int = 3
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    num_nextn_predict_layers: int = 1
    mtp_loss_scaling_factor: float = 0.1
    mtp_use_repeated_layer: bool = False
    mlp_layer_types: list[str] | None = None

    def __post_init__(self):
        self._validate()

    @property
    def num_experts(self) -> int:
        return self.n_routed_experts

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.mlp_layer_types is not None and layer_idx < len(self.mlp_layer_types):
            return self.mlp_layer_types[layer_idx] == "sparse"
        return layer_idx >= self.first_k_dense_replace

    def _validate(self) -> None:
        errors: list[str] = []

        def check(cond: bool, message: str) -> None:
            if not cond:
                errors.append(message)

        check(self.num_hidden_layers >= 1, "num_hidden_layers must be >= 1")
        check(self.hidden_size > 0, "hidden_size must be > 0")
        check(self.q_lora_rank > 0, "q_lora_rank must be > 0")
        check(self.kv_lora_rank > 0, "kv_lora_rank must be > 0")
        check(
            self.qk_head_dim == self.qk_nope_head_dim + self.qk_rope_head_dim,
            "qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim",
        )
        check(
            self.index_head_dim >= self.qk_rope_head_dim,
            "index_head_dim must be >= qk_rope_head_dim",
        )
        check(self.dsa_indexer_loss_coeff >= 0.0, "dsa_indexer_loss_coeff must be >= 0")
        check(
            self.num_key_value_heads == self.num_attention_heads,
            "initial GLM5 native path expects MLA heads to be ungrouped",
        )
        check(self.vocab_size > 0, "vocab_size must be > 0")
        check(self.num_nextn_predict_layers >= 0, "num_nextn_predict_layers must be >= 0")
        check(self.n_routed_experts >= 1, "n_routed_experts must be >= 1")
        check(
            1 <= self.num_experts_per_tok <= self.n_routed_experts,
            "num_experts_per_tok must be in [1, n_routed_experts]",
        )
        check(1 <= self.topk_group <= self.n_group, "topk_group must be in [1, n_group]")
        if self.mlp_layer_types is not None:
            expected_layer_type_lengths = {
                self.num_hidden_layers,
                self.num_hidden_layers + self.num_nextn_predict_layers,
            }
            check(
                len(self.mlp_layer_types) in expected_layer_type_lengths,
                "len(mlp_layer_types) must equal num_hidden_layers or "
                "num_hidden_layers + num_nextn_predict_layers",
            )
            for idx, layer_type in enumerate(self.mlp_layer_types):
                check(
                    layer_type in {"dense", "sparse"},
                    f"mlp_layer_types[{idx}] must be 'dense' or 'sparse'",
                )

        if errors:
            raise ValueError(
                f"Invalid Glm5Config ({len(errors)} error"
                f"{'s' if len(errors) != 1 else ''}):\n  " + "\n  ".join(errors)
            )

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def from_hf(cls, path_or_name: str, **overrides) -> Glm5Config:
        return cls._from_hf_dict(load_hf_config_dict(path_or_name), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> Glm5Config:
        hf = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        return cls._from_hf_dict(hf, **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> Glm5Config:
        kwargs = {
            key: value for key, value in hf.items() if key in _HF_FIELDS and value is not None
        }
        if "num_nextn_predict_layers" not in kwargs and hf.get("num_nextn_predict") is not None:
            kwargs["num_nextn_predict_layers"] = int(hf["num_nextn_predict"])
        rope_parameters = hf.get("rope_parameters")
        if isinstance(rope_parameters, dict) and "rope_theta" not in kwargs:
            kwargs["rope_theta"] = float(rope_parameters.get("rope_theta", cls.rope_theta))
        kwargs.update(overrides)
        return cls(**kwargs)

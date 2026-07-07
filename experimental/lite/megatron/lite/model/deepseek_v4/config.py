# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from megatron.lite.primitive.config import load_hf_config_dict


@dataclass
class DeepseekV4Config:
    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 128
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    scoring_func: str = "sqrtsoftplus"
    swiglu_limit: float = 10.0
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    rotary_scaling_factor: float = 40.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    compress_ratios: list[int] = field(default_factory=list)
    sliding_window: int = 128
    num_hash_layers: int = 3
    hc_eps: float = 1e-6
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    mtp_loss_scaling_factor: float = 0.1
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    @property
    def num_experts(self) -> int:
        return self.n_routed_experts

    @classmethod
    def from_hf(cls, path: str, **overrides) -> DeepseekV4Config:
        return cls._from_hf_dict(load_hf_config_dict(path), **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict[str, Any], **overrides) -> DeepseekV4Config:
        hf_fields = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in hf.items() if key in hf_fields and value is not None}
        if "num_nextn_predict_layers" not in kwargs and hf.get("num_nextn_predict") is not None:
            kwargs["num_nextn_predict_layers"] = int(hf["num_nextn_predict"])
        rope_parameters = hf.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            if "rope_theta" not in kwargs:
                kwargs["rope_theta"] = float(rope_parameters.get("rope_theta", cls.rope_theta))
        rope_scaling = hf.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            if rope_scaling.get("factor") is not None:
                kwargs["rotary_scaling_factor"] = float(rope_scaling["factor"])
            if rope_scaling.get("original_max_position_embeddings") is not None:
                kwargs["original_max_position_embeddings"] = int(
                    rope_scaling["original_max_position_embeddings"]
                )
            if rope_scaling.get("beta_fast") is not None:
                kwargs["beta_fast"] = float(rope_scaling["beta_fast"])
            if rope_scaling.get("beta_slow") is not None:
                kwargs["beta_slow"] = float(rope_scaling["beta_slow"])
        kwargs.update(overrides)
        return cls(**kwargs)

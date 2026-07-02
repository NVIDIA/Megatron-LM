"""Model-level Qwen3MoE benchmark statistics."""

from __future__ import annotations

from megatron.lite.model.qwen3_moe.config import Qwen3MoEConfig


def activated_params(model_cfg: Qwen3MoEConfig) -> int | None:
    try:
        h = model_cfg.hidden_size
        n_layers = model_cfg.num_hidden_layers
        n_q = model_cfg.num_attention_heads
        n_kv = model_cfg.num_key_value_heads
        d = model_cfg.head_dim
        attn = h * (n_q + n_kv + n_kv) * d + (n_q * d) * h
        router = h * model_cfg.num_experts
        inter = model_cfg.moe_intermediate_size
        expert = h * (inter * 2) + inter * h
        experts_active = model_cfg.num_experts_per_tok * expert
        return int((attn + router + experts_active) * n_layers)
    except (AttributeError, TypeError, ValueError):
        return None


__all__ = ["activated_params"]

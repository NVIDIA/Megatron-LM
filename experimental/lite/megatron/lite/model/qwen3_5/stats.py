"""Model-level Qwen3.5 benchmark statistics."""

from __future__ import annotations

from megatron.lite.model.qwen3_5.config import Qwen35Config


def num_floating_point_operations(
    model_cfg: Qwen35Config,
    *,
    seq_len: int,
    global_batch_size: int,
    tp_size: int = 1,
) -> int | None:
    """Megatron-aligned FLOPs estimate for one training step."""

    cfg = getattr(model_cfg, "text_config", model_cfg)

    def _get(name: str):
        if isinstance(cfg, dict):
            return cfg.get(name)
        return getattr(cfg, name, None)

    try:
        if seq_len <= 0 or global_batch_size <= 0:
            return None

        hidden_size = int(_get("hidden_size"))
        num_hidden_layers = int(_get("num_hidden_layers"))
        if num_hidden_layers <= 0:
            return None

        layer_types_obj = _get("layer_types")
        if layer_types_obj is not None:
            layer_types = list(layer_types_obj)
            if len(layer_types) < num_hidden_layers:
                return None
            layer_types = layer_types[:num_hidden_layers]
        else:
            full_attention_interval = int(_get("full_attention_interval"))
            if full_attention_interval <= 0:
                return None
            layer_types = [
                "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
                for i in range(num_hidden_layers)
            ]

        num_full_attention_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
        num_linear_attention_layers = sum(layer_type == "linear_attention" for layer_type in layer_types)
        if num_full_attention_layers + num_linear_attention_layers != num_hidden_layers:
            return None

        total_tokens = seq_len * global_batch_size
        total_tokens_squared = seq_len * seq_len * global_batch_size

        num_attention_heads = int(_get("num_attention_heads"))
        num_key_value_heads = int(_get("num_key_value_heads"))
        head_dim = int(_get("head_dim"))
        query_projection_size = head_dim * num_attention_heads
        key_projection_size = head_dim * num_key_value_heads
        value_projection_size = head_dim * num_key_value_heads
        gate_projection_size = query_projection_size
        standard_attention_flops = 6 * (
            total_tokens
            * hidden_size
            * (
                query_projection_size
                + key_projection_size
                + value_projection_size
                + gate_projection_size
            )
            + query_projection_size * total_tokens_squared
            + total_tokens * query_projection_size * hidden_size
        )

        linear_num_key_heads = int(_get("linear_num_key_heads"))
        linear_key_head_dim = int(_get("linear_key_head_dim"))
        linear_num_value_heads = int(_get("linear_num_value_heads"))
        linear_value_head_dim = int(_get("linear_value_head_dim"))
        linear_conv_kernel_dim = int(_get("linear_conv_kernel_dim"))
        qk_dim = linear_key_head_dim * linear_num_key_heads
        v_dim = linear_value_head_dim * linear_num_value_heads
        linear_attention_flops = 6 * total_tokens * (
            hidden_size * (2 * qk_dim + 2 * v_dim + 2 * linear_num_value_heads)
            + linear_conv_kernel_dim * (2 * qk_dim + v_dim)
            + linear_num_value_heads * (linear_value_head_dim**2) * 4
            + hidden_size * v_dim
        )

        moe_intermediate_size = int(_get("moe_intermediate_size"))
        num_experts_per_tok = int(_get("num_experts_per_tok"))
        shared_expert_intermediate_size = int(_get("shared_expert_intermediate_size"))
        moe_flops = (
            18
            * total_tokens
            * hidden_size
            * (
                moe_intermediate_size * num_experts_per_tok
                + shared_expert_intermediate_size
            )
            * num_hidden_layers
        )

        from megatron.lite.primitive.parallel.linear import pad_vocab_for_tp

        padded_vocab_size = pad_vocab_for_tp(int(_get("vocab_size")), max(tp_size, 1))
        logits_flops = 6 * total_tokens * hidden_size * padded_vocab_size

        return int(
            standard_attention_flops * num_full_attention_layers
            + linear_attention_flops * num_linear_attention_layers
            + moe_flops
            + logits_flops
        )
    except (TypeError, ValueError):
        return None


def activated_params(model_cfg: Qwen35Config) -> int | None:
    """Approximate active params/token for benchmark TFLOPS reporting."""

    cfg = getattr(model_cfg, "text_config", model_cfg)

    def _get(name: str):
        if isinstance(cfg, dict):
            return cfg.get(name)
        return getattr(cfg, name, None)

    try:
        hidden_size = int(_get("hidden_size"))
        num_hidden_layers = int(_get("num_hidden_layers"))
        layer_types_obj = _get("layer_types")
        if layer_types_obj is None:
            return None
        layer_types = list(layer_types_obj)
        if not layer_types:
            return None
        if len(layer_types) < num_hidden_layers:
            return None
        layer_types = layer_types[:num_hidden_layers]

        num_attention_heads = int(_get("num_attention_heads"))
        num_key_value_heads = int(_get("num_key_value_heads"))
        head_dim = int(_get("head_dim"))
        full_qkv_dim = (num_attention_heads + 2 * num_key_value_heads) * head_dim
        full_attention = hidden_size * full_qkv_dim + (num_attention_heads * head_dim) * hidden_size

        linear_num_key_heads = int(_get("linear_num_key_heads"))
        linear_key_head_dim = int(_get("linear_key_head_dim"))
        linear_num_value_heads = int(_get("linear_num_value_heads"))
        linear_value_head_dim = int(_get("linear_value_head_dim"))
        linear_conv_kernel_dim = int(_get("linear_conv_kernel_dim"))
        qk_dim = linear_num_key_heads * linear_key_head_dim
        v_dim = linear_num_value_heads * linear_value_head_dim
        gdn_in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * linear_num_value_heads
        gdn_conv_dim = 2 * qk_dim + v_dim
        linear_attention = (
            hidden_size * gdn_in_proj_dim
            + gdn_conv_dim * linear_conv_kernel_dim
            + 2 * linear_num_value_heads
            + linear_value_head_dim
            + v_dim * hidden_size
        )

        num_experts = int(_get("num_experts"))
        num_experts_per_tok = int(_get("num_experts_per_tok"))
        moe_intermediate_size = int(_get("moe_intermediate_size"))
        shared_expert_intermediate_size = int(_get("shared_expert_intermediate_size"))
        router = hidden_size * num_experts
        routed_expert = (
            hidden_size * (2 * moe_intermediate_size)
            + moe_intermediate_size * hidden_size
        )
        shared_expert = (
            hidden_size * (2 * shared_expert_intermediate_size)
            + shared_expert_intermediate_size * hidden_size
            + hidden_size
        )
        moe = router + num_experts_per_tok * routed_expert + shared_expert

        total = 0
        for layer_type in layer_types:
            if layer_type == "full_attention":
                total += full_attention
            elif layer_type == "linear_attention":
                total += linear_attention
            else:
                return None
            total += moe

        return int(total)
    except (TypeError, ValueError):
        return None


__all__ = ["activated_params", "num_floating_point_operations"]

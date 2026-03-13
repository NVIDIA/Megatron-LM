# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inference FLOPs calculator for hybrid Mamba/Attention/MoE models.

Computes forward-pass FLOPs per inference step using model architecture
parameters. Used by the dynamic inference engine to report per-step
FLOPs and MFU (Model FLOPs Utilization).

Reference: nemotron6_3b_moe_flops_equations.md
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InferenceFLOPsConfig:
    """Model architecture parameters needed for FLOPs calculation."""

    hidden_size: int = 0
    padded_vocab_size: int = 0
    num_attention_heads: int = 0
    num_query_groups: int = 0
    kv_channels: int = 128
    mamba_num_heads: int = 0
    mamba_head_dim: int = 64
    mamba_state_dim: int = 128
    mamba_num_groups: int = 8
    d_conv: int = 4
    num_experts: int = 0
    moe_router_topk: int = 1
    moe_ffn_hidden_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    ffn_hidden_size: int = 0
    swiglu: bool = False
    num_mamba_layers: int = 0
    num_attention_layers: int = 0
    num_moe_layers: int = 0
    num_mlp_layers: int = 0
    block_size: int = 256


class InferenceFLOPsCalculator:
    """Computes forward-pass FLOPs per inference step.

    The calculator precomputes constant FLOPs terms at init time and provides
    a fast `compute_step_flops()` method for per-step calculation.
    """

    def __init__(self, config: InferenceFLOPsConfig):
        self.config = config
        h = config.hidden_size

        # Mamba layer FLOPs per token (constant, no seq-length dependence)
        d_inner = config.mamba_num_heads * config.mamba_head_dim
        in_proj_dim = (
            2 * d_inner
            + 2 * config.mamba_num_groups * config.mamba_state_dim
            + config.mamba_num_heads
        )
        conv_channels = d_inner + 2 * config.mamba_num_groups * config.mamba_state_dim

        self.f_mamba_per_token = (
            2 * h * in_proj_dim  # in_proj
            + 2 * conv_channels * config.d_conv  # conv1d
            + 5 * config.mamba_num_heads * config.mamba_state_dim * config.mamba_head_dim  # SSM
            + 2 * d_inner * h  # out_proj
        )

        # Attention layer FLOPs per token (fixed part, excluding Q·K^T and attn·V)
        qkv_dim = (
            config.num_attention_heads * config.kv_channels
            + 2 * config.num_query_groups * config.kv_channels
        )
        q_proj_size = config.num_attention_heads * config.kv_channels
        self.f_attn_fixed_per_token = (
            2 * h * qkv_dim + 2 * q_proj_size * h  # QKV projection  # output projection
        )

        # Attention variable FLOPs coefficient: 4 * n_h * d_h per position
        self.f_attn_per_t = 4 * config.num_attention_heads * config.kv_channels

        # MoE layer FLOPs per token
        scale_factor = 3.0 / 2.0 if config.swiglu else 1.0
        moe_ffn = (
            config.moe_ffn_hidden_size if config.moe_ffn_hidden_size else config.ffn_hidden_size
        )
        self.f_moe_per_token = (
            2 * h * config.num_experts  # router
            + 4 * h * moe_ffn * config.moe_router_topk * scale_factor  # routed experts fc1+fc2
            + 4
            * h
            * config.moe_shared_expert_intermediate_size
            * scale_factor  # shared expert fc1+fc2
        )

        # Dense MLP layer FLOPs per token (for hybrid models with '-' pattern layers)
        self.f_mlp_per_token = 4 * h * config.ffn_hidden_size * scale_factor

        # Output layer FLOPs per token
        self.f_output_per_token = 2 * h * config.padded_vocab_size

        # Total fixed FLOPs per token (no attention variable term)
        self.f_fixed_per_token = (
            config.num_mamba_layers * self.f_mamba_per_token
            + config.num_attention_layers * self.f_attn_fixed_per_token
            + config.num_moe_layers * self.f_moe_per_token
            + config.num_mlp_layers * self.f_mlp_per_token
            + self.f_output_per_token
        )

        # Total attention variable coefficient per token
        self.f_attn_var_coeff = config.num_attention_layers * self.f_attn_per_t

        self.block_size = config.block_size

        logger.info(
            f"InferenceFLOPsCalculator initialized: "
            f"F_fixed={self.f_fixed_per_token/1e9:.2f}B/tok, "
            f"F_attn_var={self.f_attn_var_coeff:,}/t, "
            f"layers: {config.num_mamba_layers}M+{config.num_attention_layers}A+"
            f"{config.num_moe_layers}E+{config.num_mlp_layers}D"
        )

    def compute_step_flops(
        self,
        decode_tokens: int,
        prefill_tokens: int,
        total_tokens: int,
        active_blocks: int,
        active_reqs: int,
        num_prefill_reqs: int = 0,
    ) -> dict:
        """Compute FLOPs for a single inference step.

        Args:
            decode_tokens: Number of decode tokens (= number of decode requests).
            prefill_tokens: Number of prefill tokens (= total_tokens - decode_tokens).
            total_tokens: Total tokens processed this step.
            active_blocks: Number of active KV-cache blocks.
            active_reqs: Number of active requests.
            num_prefill_reqs: Number of prefill requests.

        Returns:
            dict with 'decode_flops', 'prefill_flops', 'total_flops', 't_avg'.
        """
        # Estimate average sequence position from KV-cache blocks
        t_avg = (active_blocks * self.block_size) / max(active_reqs, 1) if active_reqs > 0 else 0

        # Decode FLOPs: each decode token sees t_avg context
        decode_flops = decode_tokens * (self.f_fixed_per_token + self.f_attn_var_coeff * t_avg)

        # Prefill FLOPs: linear term + quadratic attention term
        prefill_flops = 0.0
        if prefill_tokens > 0:
            prefill_flops_linear = prefill_tokens * self.f_fixed_per_token
            if num_prefill_reqs > 0:
                avg_prompt_len = prefill_tokens / num_prefill_reqs
                prefill_attn_quad = (
                    self.config.num_attention_layers
                    * num_prefill_reqs
                    * 2
                    * self.config.num_attention_heads
                    * self.config.kv_channels
                    * avg_prompt_len**2
                )
            else:
                prefill_attn_quad = 0
            prefill_flops = prefill_flops_linear + prefill_attn_quad

        total_flops = decode_flops + prefill_flops
        return {
            'decode_flops': decode_flops,
            'prefill_flops': prefill_flops,
            'total_flops': total_flops,
            't_avg': t_avg,
        }

    @classmethod
    def from_args(cls, args) -> "InferenceFLOPsCalculator":
        """Create calculator from megatron args (get_args()).

        Automatically detects layer counts from hybrid_override_pattern.
        """
        num_attn = 0
        num_mamba = 0
        num_mlp = 0
        num_moe = 0

        if getattr(args, 'hybrid_override_pattern', None):
            from megatron.core.ssm.mamba_hybrid_layer_allocation import parse_hybrid_pattern

            parsed = parse_hybrid_pattern(args.hybrid_override_pattern)
            counts = {'M': 0, '*': 0, '-': 0, 'E': 0}
            if parsed.main_pattern:
                for lt in parsed.main_pattern:
                    if lt in counts:
                        counts[lt] += 1
            num_attn, num_mamba, num_mlp, num_moe = (
                counts['*'],
                counts['M'],
                counts['-'],
                counts['E'],
            )
        elif getattr(args, 'is_hybrid_model', False):
            num_attn = round(args.num_layers * args.hybrid_attention_ratio)
            num_mlp = round(args.num_layers * args.hybrid_mlp_ratio)
            num_mamba = args.num_layers - num_attn - num_mlp
        else:
            num_attn = args.num_layers
            num_mamba = 0
            num_mlp = 0
            num_moe = 0

        block_size = getattr(args, 'inference_dynamic_batching_block_size', 256)

        config = InferenceFLOPsConfig(
            hidden_size=args.hidden_size,
            padded_vocab_size=args.padded_vocab_size,
            num_attention_heads=args.num_attention_heads,
            num_query_groups=getattr(args, 'num_query_groups', args.num_attention_heads),
            kv_channels=getattr(args, 'kv_channels', args.hidden_size // args.num_attention_heads),
            mamba_num_heads=getattr(args, 'mamba_num_heads', 0) or 0,
            mamba_head_dim=getattr(args, 'mamba_head_dim', 64) or 64,
            mamba_state_dim=getattr(args, 'mamba_state_dim', 128) or 128,
            mamba_num_groups=getattr(args, 'mamba_num_groups', 8) or 8,
            num_experts=getattr(args, 'num_experts', 0) or 0,
            moe_router_topk=getattr(args, 'moe_router_topk', 1) or 1,
            moe_ffn_hidden_size=getattr(args, 'moe_ffn_hidden_size', 0) or 0,
            moe_shared_expert_intermediate_size=getattr(
                args, 'moe_shared_expert_intermediate_size', 0
            )
            or 0,
            ffn_hidden_size=args.ffn_hidden_size,
            swiglu=getattr(args, 'swiglu', False),
            num_mamba_layers=num_mamba,
            num_attention_layers=num_attn,
            num_moe_layers=num_moe,
            num_mlp_layers=num_mlp,
            block_size=block_size,
        )
        return cls(config)

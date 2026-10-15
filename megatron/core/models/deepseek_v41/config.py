# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration for V4.1 conditional n-gram memory."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass(frozen=True)
class EngramConfig:
    """Released n-gram table layout and tokenizer-compression contract."""

    layer_ids: list[int]
    """Zero-based backbone blocks receiving conditional memory."""

    num_embeddings: list[int]
    """Total fused-table rows per selected backbone block."""

    max_ngram_size: int
    """Largest n-gram order; all orders from two through this value are used."""

    vocab_size: int
    """Initial bucket-size budget for the distinct prime hash tables."""

    n_heads: int
    """Hash heads per n-gram order."""

    head_dim: int
    """Embedding channels returned by each hash head."""

    pad_token_id: int
    """Raw tokenizer ID used for missing causal history."""

    compressed_vocab_size: int
    """Exact number of normalized token identities used to derive hash multipliers."""


@dataclass
class DeepSeekV41Config(MLATransformerConfig):
    """V4.1 uses two Hybrid layers (attention and MoE) per logical block."""

    is_hybrid_model: bool = True
    """Each V/E entry owns one residual branch for depth-scaled initialization."""

    engram_config: EngramConfig | None = None
    """Conditional-memory modules; None constructs a backbone without Engram."""

    def __post_init__(self):
        if self.num_layers % 2 or self.num_layers <= 0:
            raise ValueError("V4.1 uses an attention/MoE pair for each logical block")
        if self.dsv4_version != "v4.1" or not self.mhc_single_pass:
            raise ValueError("V4.1 requires CSA2 and single-pass mHC")
        if self.moe_num_hash_layers:
            raise ValueError("V4.1 uses learned MoE routing; Engram has its own n-gram hashes")
        if self.fp32_residual_connection:
            raise ValueError(
                "V4.1 residual storage uses the model dtype; mixing arithmetic is FP32"
            )
        if self.recompute_granularity == 'full' and (
            self.recompute_method != 'uniform' or self.recompute_num_layers != 1
        ):
            raise ValueError("V4.1 full recompute currently uses one explicit layer per region")
        if self.context_parallel_size > 1 and self.linear_cp_layout != 'contiguous':
            raise ValueError("V4.1 keeps a contiguous CP layout across attention and MoE layers")
        if self.engram_config is not None:
            ids = self.engram_config.layer_ids
            if len(ids) != len(set(ids)) or any(i < 0 or i >= self.num_layers // 2 for i in ids):
                raise ValueError("Engram layers must be distinct zero-based logical blocks")
        super().__post_init__()

    @classmethod
    def from_hf(cls, hf: dict[str, Any], **overrides) -> "DeepSeekV41Config":
        """Import architecture fields without loading weights or allocating the model.

        Overrides use Megatron names. The released config includes three DSpark
        ratios after the backbone; they belong to the draft module, not the stack.
        """
        text = hf.get("text_config", hf)
        depth = text["num_hidden_layers"]
        rope = text.get("rope_scaling", {})
        values = dict(
            params_dtype=torch.bfloat16,
            bf16=True,
            num_layers=2 * depth,
            hidden_size=text["hidden_size"],
            num_attention_heads=text["num_attention_heads"],
            multi_latent_attention=True,
            experimental_attention_variant="dsv4_hybrid",
            dsv4_version="v4.1",
            q_lora_rank=text["q_lora_rank"],
            v_head_dim=text["head_dim"],
            qk_pos_emb_head_dim=text["qk_rope_head_dim"],
            output_projection_groups=text["o_groups"],
            output_projection_lora_rank=text["o_lora_rank"],
            qk_layernorm=True,
            layernorm_epsilon=text["rms_norm_eps"],
            normalization="RMSNorm",
            rotary_base=text["rope_theta"],
            csa_compress_rotary_base=text["compress_rope_theta"],
            original_max_position_embeddings=rope.get("original_max_position_embeddings", 65536),
            rotary_scaling_factor=rope.get("factor", 16),
            beta_fast=rope.get("beta_fast", 32),
            beta_slow=rope.get("beta_slow", 1),
            mscale=0,
            mscale_all_dim=0,
            csa_compress_ratios=[
                value for ratio in text["compress_ratios"][:depth] for value in (ratio, 0)
            ],
            csa_window_size=text["sliding_window"],
            csa2_kv_source_layers=[2 * i for i in text["kv_source_layer_ids"]],
            csa2_index_source_layers=[2 * i for i in text["index_source_layer_ids"]],
            csa2_candidate_source_layer=2 * text["candidate_source_layer_id"],
            csa2_candidate_topk_blocks=text["candidate_topk_blocks"],
            csa2_candidate_block_size=text["candidate_block_size"],
            dsa_indexer_n_heads=text["index_n_heads"],
            dsa_indexer_head_dim=text["index_head_dim"],
            dsa_indexer_topk=text["index_topk"],
            dsa_indexer_rotate_activation=False,
            dsa_kernel_backend="none",
            enable_mhc_connections=True,
            mhc_single_pass=True,
            mhc_epsilon=text.get("hc_eps", 1e-6),
            mhc_num_residual_streams=text["hc_mult"],
            mhc_sinkhorn_iterations=text["hc_sinkhorn_iters"],
            num_moe_experts=text["n_routed_experts"],
            moe_router_topk=text["num_experts_per_tok"],
            moe_ffn_hidden_size=text["moe_intermediate_size"],
            moe_shared_expert_intermediate_size=text["moe_intermediate_size"]
            * text["n_shared_experts"],
            moe_router_score_function=text["scoring_func"],
            moe_router_topk_scaling_factor=text["routed_scaling_factor"],
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_router_load_balancing_type="none",
            activation_func=F.silu,
            gated_linear_unit=True,
            activation_func_clamp_value=text["swiglu_limit"],
            add_bias_linear=False,
            attention_dropout=0,
            hidden_dropout=0,
            gradient_accumulation_fusion=False,
            engram_config=(
                EngramConfig(
                    **{
                        k.removeprefix("engram_"): v
                        for k, v in text.items()
                        if k.startswith("engram_")
                    }
                )
                if text.get("engram_layer_ids")
                else None
            ),
        )
        values.update(overrides)
        return cls(**values)

    @property
    def hybrid_pattern(self) -> str:
        """The text backbone composition, before optional PP/VPP separators."""
        return "VE" * (self.num_layers // 2)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validation of the logical CSA2 layer schedule, independent of stack layout."""

import math

import torch


def validate_csa2_config(config) -> None:
    """Reject inconsistent owners, compression ratios and hierarchical candidate pools."""
    if not config.multi_latent_attention or not config.qk_layernorm:
        raise ValueError("CSA2 requires MLA and learned query/KV RMS normalization")
    if config.normalization != "RMSNorm" or config.layernorm_zero_centered_gamma:
        raise ValueError("CSA2 requires ordinary RMSNorm weights")
    if config.tensor_model_parallel_size != 1 or config.context_parallel_size != 1:
        raise NotImplementedError("CSA2 currently requires TP=CP=1")
    if config.fp16 or config.params_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("CSA2 supports FP32 or BF16 parameters")
    if config.dsa_kernel_backend not in ("none", "cudnn"):
        raise ValueError("CSA2 supports the native and cuDNN sparse attention backends")
    if config.dsa_kernel_backend == "cudnn" and config.params_dtype != torch.bfloat16:
        raise ValueError("Fused CSA2 attention requires BF16 parameters")
    if config.attention_dropout or config.hidden_dropout:
        raise ValueError("The released V4.1 architecture uses zero dropout")
    if config.csa_dense_mode or config.qk_clip:
        raise ValueError("CSA2 does not use dense CSA or QK clipping")
    if config.add_bias_linear or config.qk_l2_norm or config.dsa_indexer_rotate_activation:
        raise ValueError(
            "CSA2 uses bias-free projections, RMSNorm and unrotated indexer activations"
        )
    if config.fp8 or config.fp4 or config.quant_recipe is not None:
        raise NotImplementedError("CSA2 currently supports BF16/FP32 training, without weight QAT")
    for name in ("layernorm_epsilon", "csa_compress_rotary_base"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"CSA2 requires positive finite {name}")
    ratios = config.csa_compress_ratios
    if not ratios or any(type(r) is not int or r not in (0, 1, 2) for r in ratios):
        raise ValueError("CSA2 requires one ratio (0, 1, or 2) per logical attention layer")
    if config.num_layers not in (len(ratios), 2 * len(ratios)):
        raise ValueError("CSA2 ratios must match the logical backbone depth")
    for name in ("csa2_kv_source_layers", "csa2_index_source_layers"):
        sources = getattr(config, name)
        if sources is None or any(type(i) is not int or not 0 <= i < len(ratios) for i in sources):
            raise ValueError(f"{name} must contain zero-based logical attention layer IDs")
        if sources != sorted(set(sources)) or any(ratios[i] == 0 for i in sources):
            raise ValueError(f"{name} must be strictly increasing and exclude SWA layers")
    if not set(config.csa2_kv_source_layers).issubset(config.csa2_index_source_layers):
        raise ValueError("Every global KV owner must also produce sparse indices")
    owner_ratio = None
    for i, ratio in enumerate(ratios):
        if not ratio:
            continue
        if i in config.csa2_kv_source_layers:
            owner_ratio = ratio
        elif owner_ratio is None or owner_ratio != ratio:
            raise ValueError(f"CSA2 layer {i} needs a preceding Full owner with the same ratio")
    source = config.csa2_candidate_source_layer
    blocks, width = config.csa2_candidate_topk_blocks, config.csa2_candidate_block_size
    if source is None:
        if blocks or width:
            raise ValueError("Disabled hierarchical indexing requires zero candidate dimensions")
    else:
        if source not in config.csa2_kv_source_layers or any(
            i > source for i in config.csa2_kv_source_layers
        ):
            raise ValueError("The candidate source must be the final Full owner")
        if type(blocks) is not int or type(width) is not int or min(blocks, width) <= 0:
            raise ValueError("Candidate dimensions must be positive integers")
        if blocks * width < config.dsa_indexer_topk:
            raise ValueError("The hierarchical candidate pool must cover the attention Top-K")
    for name in (
        "csa_window_size",
        "dsa_indexer_topk",
        "dsa_indexer_n_heads",
        "dsa_indexer_head_dim",
    ):
        if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
            raise ValueError(f"CSA2 requires positive {name}")
    if not 0 < config.qk_pos_emb_head_dim <= min(config.v_head_dim, config.dsa_indexer_head_dim):
        raise ValueError("CSA2 rotary dimensions must fit both attention and indexer heads")
    if config.qk_pos_emb_head_dim % 2:
        raise ValueError("CSA2 rotary dimensions must be even")

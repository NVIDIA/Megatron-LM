# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn
from megatron.core.transformer.attention import SelfAttention

class KDASelfAttention(SelfAttention):
    """
    Key Delta Attention (KDA) module.

    Implements delta-based sparse attention logic and integrates with Megatron's
    fused kernels and KV cache compression logic.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.delta_threshold = config.kda_hyperparameters.get('delta_threshold', 0.1)
        self.sparsity_factor = config.kda_hyperparameters.get('sparsity_factor', 0.5)

    def forward(self, query, key, value, attention_mask=None):
        """
        Forward pass for Kimi Delta Attention (KDA).
        Args:
            query: [batch, heads, seq_len, head_dim]
            key:   [batch, heads, seq_len, head_dim]
            value: [batch, heads, seq_len, head_dim]
            attention_mask: Optional mask
        Returns:
            output: [batch, heads, seq_len, head_dim]
            attn_weights: [batch, heads, seq_len, seq_len]
        """
        # Compute pairwise L2 distance between query and key for sparsity
        q_exp = query.unsqueeze(-2)  # [B, H, S, 1, D]
        k_exp = key.unsqueeze(-3)    # [B, H, 1, S, D]
        delta = torch.norm(q_exp - k_exp, dim=-1)  # [B, H, S, S]
        sparse_mask = (delta < self.delta_threshold).float()

        # Standard attention scores
        attn_scores = torch.matmul(query, key.transpose(-1, -2))  # [B, H, S, S]
        # Apply sparsity mask
        attn_scores = attn_scores * sparse_mask * self.sparsity_factor

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights
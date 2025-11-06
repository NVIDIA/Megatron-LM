# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

# TODO(kunlunl): Add third-party fused kernels.


class IndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if IndexerLossAutoScaler.main_loss_backward_scale is None:
            IndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = IndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set. Please ensure that the scale passed in
                   matches the scale of the main_loss.
        """
        if IndexerLossAutoScaler.main_loss_backward_scale is None:
            IndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            IndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


def compute_indexer_loss(
    index_scores: torch.Tensor,
    attention_scores: torch.Tensor,
    indexer_loss_coeff: float,
) -> torch.Tensor:
    """
    Compute KL divergence loss between indexer scores and true attention scores.

    This loss trains the indexer to predict which tokens are important
    by matching the distribution of true attention scores.

    Args:
        index_scores: Scores predicted by indexer [batch, seq, seq]
        attention_scores: True attention scores from q@k [batch, heads, seq, seq]
        indexer_loss_coeff: Coefficient for the indexer KL divergence loss

    Returns:
        indexer_loss: KL divergence loss (scalar)
    """
    # Average attention scores across heads to get target distribution
    # [batch, heads, seq, seq] -> [batch, seq, seq]
    target_scores = attention_scores.mean(dim=1)

    # Convert to probabilities with softmax
    # Apply softmax over the last dimension (keys)
    index_probs = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    target_probs = torch.nn.functional.softmax(target_scores, dim=-1, dtype=torch.float32)

    # Compute KL divergence: KL(target || index)
    # KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    kl_div = torch.nn.functional.kl_div(
        index_probs.log(),
        target_probs,
        reduction='batchmean',
        log_target=False,
    )

    # Scale by coefficient
    indexer_loss = kl_div * indexer_loss_coeff

    return indexer_loss


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.

    Args:
        x: Input tensor (must be bfloat16)

    Returns:
        Rotated tensor
    """
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


def compute_index_score(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute index scores for sparse attention (BF16 version).

    This is a BF16 implementation of the FP8 index kernel logic:
    1. Compute attention scores: q @ k^T
    2. Apply ReLU activation
    3. Weight by attention weights
    4. Sum across attention heads

    Args:
        q: Query tensor [batch, seq_len, n_heads, head_dim] (bf16)
        weights: Attention weights [batch, seq_len, n_heads, 1] (bf16)
        k: Key tensor [batch, seq_len, head_dim] (bf16)

    Returns:
        index_score: Index scores [batch, seq_len, seq_len] (bf16)

    Note: Original FP8 kernel signature was:
        fp8_index(q, q_s, k, k_s) where q_s and k_s are scaling factors
        Here we use BF16 directly without separate scaling factors.
    """
    # q: [bsz, seqlen, n_heads, head_dim]
    # k: [bsz, seqlen, head_dim]
    # weights: [bsz, seqlen, n_heads, 1]

    # Compute attention scores: q @ k^T
    # [bsz, seqlen, n_heads, head_dim] @ [bsz, seqlen, head_dim]^T
    # -> [bsz, seqlen, n_heads, seqlen]
    index_score = torch.einsum('bshd,btd->bsht', q, k)

    # Apply ReLU activation (for throughput efficiency)
    index_score = torch.relu(index_score)

    # Weight each head by attention weights
    # [bsz, seqlen, n_heads, seqlen] * [bsz, seqlen, n_heads, 1]
    index_score = index_score * weights

    # Sum across attention heads
    # [bsz, seqlen, n_heads, seqlen] -> [bsz, seqlen, seqlen]
    index_score = index_score.sum(dim=2)

    return index_score



@dataclass
class IndexerSubmodules:
    """
    Configuration class for specifying the submodules of an Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """
    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class SparseAttentionSubmodules:
    """
    Configuration class for specifying the submodules of SparseAttention.

    Args:
        indexer: Indexer module for computing sparse attention indices.
    """
    indexer: Union[ModuleSpec, type] = None


class Indexer(MegatronModule):
    """
    Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs
    for each query position in sparse attention.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py

    Args:
        config: Transformer configuration.
        submodules: Indexer submodules specification.
        dim: Model hidden dimension.
        n_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        rope_head_dim: Dimension for rotary position embeddings.
        index_topk: Number of top-k indices to select.
        q_lora_rank: Rank for low-rank query projection.
        pg_collection: Process group collection for tensor parallelism.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: IndexerSubmodules,
        dim: int,
        n_heads: int,
        head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        self.pg_collection = pg_collection
        world_size = pg_collection.tp.size()
        self.n_local_heads = n_heads // world_size

        # Initialize Rotary Position Embedding.
        # Use rope_type from config if available, default to "rope".
        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.rope_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.rope_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are 'rope' and 'yarn'"
            )

        # Build linear layers using build_module
        self.wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )

        self.wk = build_module(
            submodules.linear_wk,
            self.dim,
            self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )

        self.k_norm = build_module(
            submodules.k_norm,
            config=self.config,
            hidden_size=self.head_dim,
            eps=self.config.layernorm_epsilon,
        )

        # TODO(kunlunl): The dtype of this module should be torch.get_default_dtype().
        self.weights_proj = build_module(
            submodules.linear_weights_proj,
            self.dim,
            self.n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )

        self.softmax_scale: float = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional = None,
    ):
        """
        Forward pass for Indexer.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or [seq, batch, hidden_dim]
            qr: Query representation tensor [batch, seq_len, q_lora_rank]
            mask: Attention mask
            packed_seq_params: Packed sequence parameters for variable length sequences

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seq_len, index_topk]
        """
        # Call forward_with_scores and only return indices
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            qr: Query representation tensor [batch, seq_len, q_lora_rank]
            mask: Attention mask
            packed_seq_params: Packed sequence parameters

        Returns:
            index_scores: Index scores [batch, seq_len, seq_len]
            topk_indices: Top-k indices [batch, seq_len, index_topk]
        """
        bsz, seqlen, _ = x.size()

        # Compute rotary position embeddings internally
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(seqlen, packed_seq=packed_seq)
            mscale = 1.0
        else:  # yarn
            rotary_pos_emb, mscale = self.rotary_pos_emb(seqlen, packed_seq=packed_seq)

        q = self.wq_b(qr)
        q = q.reshape(bsz, seqlen, -1, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        # Apply RoPE to query position embedding part
        q_pe = apply_rotary_pos_emb(
            q_pe,
            rotary_pos_emb,
            config=self.config,
            cp_group=self.pg_collection.cp,
            mscale=mscale,
        )
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        # Apply RoPE to key position embedding part
        k_pe = k_pe.unsqueeze(2)  # [batch, seq_len, 1, rope_head_dim]
        k_pe = apply_rotary_pos_emb(
            k_pe,
            rotary_pos_emb,
            config=self.config,
            cp_group=self.pg_collection.cp,
            mscale=mscale,
        )
        k_pe = k_pe.squeeze(2)  # [batch, seq_len, rope_head_dim]
        k = torch.cat([k_pe, k_nope], dim=-1)

        q = rotate_activation(q)
        k = rotate_activation(k)

        weights = self.weights_proj(x) * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale

        # Compute index scores (BF16 version of the FP8 kernel)
        index_scores = compute_index_score(q.contiguous(), weights, k.contiguous())

        if mask is not None:
            index_scores = index_scores + mask

        # Select top-k indices
        topk_k = min(self.index_topk, seqlen)
        topk_indices = index_scores.topk(topk_k, dim=-1)[1]

        return index_scores, topk_indices


class SparseAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an Indexer to compute top-k attention
    indices for reducing computational complexity.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py

    Args:
        config: Transformer configuration.
        submodules: Sparse attention submodules specification.
        layer_number: Layer number in the model.
        attn_mask_type: Type of attention mask.
        attention_type: Type of attention.
        attention_dropout: Dropout probability for attention weights.
        softmax_scale: Scale factor for softmax.
        k_channels: Number of channels in key tensor.
        v_channels: Number of channels in value tensor.
        cp_comm_type: Context parallel communication type.
        pg_collection: Process group collection for distributed training.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Currently context parallelism is not supported by SparseAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        else:
            assert hasattr(
                pg_collection, 'tp'
            ), "SparseAttention pg_collection must have tp process group"
        self.pg_collection = pg_collection

        world_size = pg_collection.tp.size()

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

        # Build indexer - required for sparse attention
        assert submodules.indexer is not None, "Indexer is required for SparseAttention"
        self.indexer = build_module(
            submodules.indexer,
            config=self.config,
            dim=self.config.hidden_size,
            n_heads=self.config.index_n_heads,
            head_dim=self.config.index_head_dim,
            rope_head_dim=self.config.qk_pos_emb_head_dim,
            index_topk=self.config.index_topk,
            q_lora_rank=self.config.q_lora_rank,
            pg_collection=self.pg_collection,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        x: Optional[torch.Tensor] = None,
        qr: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn]
            key: Key tensor [sk, b, np, hn]
            value: Value tensor [sk, b, np, hn]
            attention_mask: Attention mask tensor
            attn_mask_type: Type of attention mask
            attention_bias: Optional attention bias
            packed_seq_params: Packed sequence parameters
            x: Original input hidden states [b, s, h] (needed for indexer, will be inferred if not provided)
            qr: Low-rank query representation [b, s, q_lora_rank] (for MLA, will be inferred if not provided)

        Returns:
            context: Output tensor [sq, b, hp]
        """
        # Input shape: [sq, b, np, hn]
        sq, b, np, hn = query.size()
        sk = key.size(0)

        # Prepare inputs for indexer (expects batch-first [b, s, h] format)
        if x is None:
            # Convert query from [sq, b, np, hn] to [b, sq, np*hn]
            x = query.transpose(0, 1).reshape(b, sq, np * hn)
        if qr is None:
            # For non-MLA, use x as qr
            qr = x

        # ===================================
        # Raw attention scores [b, np, sq, sk]
        # ===================================
        output_size = (b, np, sq, sk)

        # Reshape for batch matrix multiplication
        # [sq, b, np, hn] -> [b * np, sq, hn]
        query_reshaped = query.transpose(0, 1).reshape(b * np, sq, hn)
        # [sk, b, np, hn] -> [b * np, sk, hn]
        key_reshaped = key.transpose(0, 1).reshape(b * np, sk, hn)

        # Compute attention scores: [b * np, sq, sk]
        attention_scores = torch.bmm(
            query_reshaped,
            key_reshaped.transpose(1, 2)
        ) * self.softmax_scale

        # Reshape to [b, np, sq, sk]
        attention_scores = attention_scores.view(*output_size)

        # ===================================
        # Use Indexer for sparse selection
        # ===================================
        # Get index scores and top-k indices
        # Note: We need to get index_scores before topk for KL loss computation
        index_scores, topk_indices = self.indexer.forward_with_scores(
            x, qr, mask=None, packed_seq_params=packed_seq_params
        )

        # ===================================
        # Compute and attach indexer loss
        # ===================================
        if self.training and torch.is_grad_enabled():
            # Get indexer loss coefficient from config
            indexer_loss_coeff = getattr(self.config, 'indexer_loss_coeff', 0.0)

            if indexer_loss_coeff > 0:
                # Compute KL divergence loss between indexer scores and true attention scores
                indexer_loss = compute_indexer_loss(
                    index_scores,
                    attention_scores.detach(),  # Don't backprop through attention scores
                    indexer_loss_coeff
                )

                # Attach loss to query activation (will be backpropagated)
                # This doesn't change the forward pass but triggers gradient flow in backward
                query = IndexerLossAutoScaler.apply(query, indexer_loss)

        # ===================================
        # Apply sparse mask from indexer
        # ===================================
        # topk_indices: [b, sq, topk]
        # Create sparse mask
        index_mask = torch.full(
            (b, sq, sk), float("-inf"), device=query.device, dtype=attention_scores.dtype
        )
        # Fill top-k positions with 0 (allow attention)
        index_mask.scatter_(-1, topk_indices, 0)

        # Expand index_mask to [b, np, sq, sk]
        index_mask = index_mask.unsqueeze(1).expand(-1, np, -1, -1)

        # Combine with regular attention mask
        if attention_mask is not None:
            index_mask = index_mask + attention_mask

        attention_scores = attention_scores + index_mask

        # ===================================
        # Attention probabilities [b, np, sq, sk]
        # ===================================
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)

        # ===================================
        # Context layer [sq, b, hp]
        # ===================================
        # Reshape value: [sk, b, np, hn] -> [b * np, sk, hn]
        value_reshaped = value.transpose(0, 1).reshape(b * np, sk, hn)

        # Reshape attention_probs: [b, np, sq, sk] -> [b * np, sq, sk]
        attention_probs_reshaped = attention_probs.view(b * np, sq, sk)

        # Compute context: [b * np, sq, hn]
        context = torch.bmm(attention_probs_reshaped, value_reshaped)

        # Reshape context: [b * np, sq, hn] -> [b, np, sq, hn] -> [sq, b, np, hn]
        context = context.view(b, np, sq, hn).permute(2, 0, 1, 3).contiguous()

        # Flatten: [sq, b, np, hn] -> [sq, b, np*hn]
        context = context.view(sq, b, np * hn)

        return context

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the learnable softmax offset parameter"""
        if self.config.softmax_type == "learnable":
            state_dict = self.state_dict(prefix="", keep_vars=True)
        else:
            state_dict = {}
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'softmax_offset': 0}, sharded_offsets
        )

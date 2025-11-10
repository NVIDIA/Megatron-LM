# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except:
    fused_apply_mla_rope_for_kv = None
    fused_apply_mla_rope_for_q = None

# TODO(kunlunl): Add third-party fused kernels.
try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16)

    Returns:
        Rotated tensor
    """
    assert x.dtype == torch.bfloat16
    assert hadamard_transform is not None, (
        "fast_hadamard_transform is not installed."
    )
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


def compute_index_score(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Apply ReLU activation;
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q          : BF16 [seqlen_q, bsz, n_heads, head_dim], the query tensor.
        weights    : FP32 [seqlen_q, bsz, n_heads], the attention weights.
        k          : BF16 [seqlen_k, bsz, head_dim], the key tensor.

    Returns:
        index_score: FP32 [bsz, seqlen_q, seqlen_k], the index scores.
    """
    # Compute attention scores: q @ k^T
    # [seqlen_q, bsz, n_heads, head_dim] @ [seqlen_k, bsz, head_dim]^T -> [seqlen_q, bsz, n_heads, seqlen_k]
    index_score = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Apply ReLU activation.
    index_score = torch.relu(index_score)

    # Weight each head by attention weights.
    # [seqlen_q, bsz, n_heads, seqlen_k] * [seqlen_q, bsz, n_heads, 1] -> [seqlen_q, bsz, n_heads, seqlen_k]
    index_score = index_score * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, bsz, n_heads, seqlen_k] -> [seqlen_q, bsz, seqlen_k]
    index_score = index_score.sum(dim=2)

    # Transpose to [bsz, seqlen_q, seqlen_k].
    index_score = index_score.transpose(0, 1)

    return index_score


def compute_indexer_loss(
    index_scores: torch.Tensor,
    attention_scores: torch.Tensor,
    indexer_loss_coeff: float,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [bsz, seqlen_q, seqlen_k].
        attention_scores: True attention scores from q @ k^T [bsz, heads, seqlen_q, seqlen_k].
        indexer_loss_coeff: Coefficient for the indexer KL divergence loss.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    # Sum attention scores across heads.
    # [bsz, heads, seqlen_q, seqlen_k] -> [bsz, seqlen_q, seqlen_k]
    target_scores = attention_scores.sum(dim=1)

    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    target_probs = target_scores / target_scores.sum(dim=-1, keepdim=True)

    # Convert index_scores to probabilities with softmax.
    index_probs = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    kl_per_element = (
        target_probs * (torch.log(target_probs + 1e-10) - torch.log(index_probs + 1e-10))
    )
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * indexer_loss_coeff

    return indexer_loss


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
            scale: The scale value to set.
        """
        if IndexerLossAutoScaler.main_loss_backward_scale is None:
            IndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            IndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


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

    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480

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
        self.softmax_scale: float = self.head_dim ** -0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection
        world_size = pg_collection.tp.size()
        self.n_local_heads = n_heads // world_size

        # Initialize Position Embedding.
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
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.wk = build_module(
            submodules.linear_wk,
            self.dim,
            self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
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
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_dim].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_dim].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen]
            topk_indices: Top-k indices [batch, seqlen, index_topk]
        """
        assert packed_seq_params is None, "Packed sequence is not supported for SparseAttention"
        assert not self.config.apply_rope_fusion, "RoPE fusion is not supported for SparseAttention"

        seqlen, bsz, _ = x.size()

        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        # =========================================
        # Apply RoPE to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, n_heads * head_dim]
        q, _ = self.wq_b(qr)
        # [seqlen, batch, n_heads * head_dim] -> [seqlen, batch, n_heads, head_dim]
        q = q.reshape(seqlen, bsz, self.n_heads, self.head_dim)
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_pos_emb(
            q_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        # [seqlen, batch, n_heads, head_dim]
        q = torch.cat([q_nope, q_pe], dim=-1)

        # =========================================
        # Apply RoPE to k
        # =========================================
        # [seqlen, batch, hidden_dim] -> [seqlen, batch, head_dim]
        k, _ = self.wk(x)
        k = self.k_norm(k)
        # [seqlen, batch, head_dim] -> [seqlen, batch, 1, head_dim]
        k = k.reshape(seqlen, bsz, 1, self.head_dim)
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1
        )
        k_pe = apply_rotary_pos_emb(
            k_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        # [seqlen, batch, 1, head_dim]
        k = torch.cat([k_nope, k_pe], dim=-1)
        # [seqlen, batch, head_dim]
        k = k.reshape(seqlen, bsz, self.head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # Compute index scores
        # =========================================
        # [seqlen, batch, hidden_dim] -> [seqlen, batch, n_heads]
        weights, _ = self.weights_proj(x)
        weights = weights * (self.n_heads ** -0.5) * self.softmax_scale
        # [batcch, seqlen, seqlen]
        index_scores = compute_index_score(q, weights, k)
        if mask is not None:
            assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
            index_scores = index_scores + mask

        # =========================================
        # Select top-k indices
        # =========================================
        topk_k = min(self.index_topk, seqlen)
        # [batch, seqlen, index_topk]
        topk_indices = index_scores.topk(topk_k, dim=-1)[1]

        return index_scores, topk_indices


class SparseAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an Indexer to compute top-k attention
    indices for reducing computational complexity.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597

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
        x: torch.Tensor,
        qr: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [seqlen_q, bsz, n_heads, head_dim].
            key: Key tensor [seqlen_k, bsz, n_heads, head_dim].
            value: Value tensor [seqlen_k, bsz, n_heads, head_dim_v].
            x: Original hidden states [seqlen_q, bsz, hidden_dim].
            qr: Low-rank query representation [seqlen_q, bsz, q_lora_rank].
            attention_mask: Attention mask tensor.
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            context: Output tensor [sq, b, hp]
        """
        sq, b, np, hn = query.size()
        sk = key.size(0)
        # Value head dimension may differ from query/key.
        v_hn = value.size(3)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        # TODO(kunlunl): Should x and qr be detached?
        x = x.detach()
        qr = qr.detach()

        # Get a FP32 mask with -inf for masked positions.
        if attention_mask is not None:
            mask = attention_mask.squeeze()
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                mask, float('-inf')
            )
        else:
            float_mask = None

        # ===================================
        # Get index scores and top-k indices
        # ===================================
        index_scores, topk_indices = self.indexer.forward_with_scores(
            x, qr, mask=float_mask, packed_seq_params=packed_seq_params
        )

        # ===================================
        # Raw attention scores [b, np, sq, sk]
        # ===================================
        # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
        query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
        # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
        key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
        # Compute attention scores: [b * np, sq, sk]
        attention_scores = torch.bmm(
            query_reshaped.float(), key_reshaped.float()
        ) * self.softmax_scale
        # Reshape to [b, np, sq, sk]
        attention_scores = attention_scores.view(b, np, sq, sk)

        # ===================================
        # Apply sparse mask from indexer
        # ===================================
        # index_mask [b, sq, sk]
        index_mask = torch.full((b, sq, sk), float("-inf"), device=x.device)
        index_mask.scatter_(-1, topk_indices, 0)
        if float_mask is not None:
            index_mask += float_mask
        attention_scores += index_mask.unsqueeze(1)

        # ===================================
        # Attention probabilities [b, np, sq, sk]
        # ===================================
        attention_probs_fp32 = torch.nn.functional.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        )
        attention_probs = attention_probs_fp32.to(query.dtype)

        # ===================================
        # Output
        # ===================================
        # [sk, b, np, v_hn] -> [b, np, sk, v_hn] -> [b * np, sk, v_hn]
        value_reshaped = value.permute(1, 2, 0, 3).reshape(b * np, sk, v_hn)
        # Reshape attention_probs: [b, np, sq, sk] -> [b * np, sq, sk]
        attention_probs_reshaped = attention_probs.view(b * np, sq, sk)
        # Compute output: [b * np, sq, v_hn]
        output = torch.bmm(attention_probs_reshaped, value_reshaped)
        # Reshape output: [b * np, sq, v_hn] -> [b, np, sq, v_hn] -> [sq, b, np, v_hn]
        output = output.view(b, np, sq, v_hn).permute(2, 0, 1, 3).contiguous()
        # Flatten: [sq, b, np, v_hn] -> [sq, b, np * v_hn]
        output = output.view(sq, b, np * v_hn)

        # ===================================
        # Attach indexer loss
        # ===================================
        if self.training and torch.is_grad_enabled():
            # Get indexer loss coefficient from config
            indexer_loss_coeff = getattr(self.config, 'indexer_loss_coeff', 0.0)
            # Compute KL divergence loss between indexer scores and true attention scores
            indexer_loss = compute_indexer_loss(
                index_scores,
                attention_probs_fp32.detach(),
                indexer_loss_coeff,
            )
            # Attach loss to output output (will trigger backward through indexer)
            output = IndexerLossAutoScaler.apply(output, indexer_loss)

        return output

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict for the learnable softmax offset parameter"""
        # TODO(kunlunl): Add checkpointing for indexer.
        if self.config.softmax_type == "learnable":
            state_dict = self.state_dict(prefix="", keep_vars=True)
        else:
            state_dict = {}
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {'softmax_offset': 0}, sharded_offsets
        )

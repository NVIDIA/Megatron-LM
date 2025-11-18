# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

# TODO(kunlunl): Add third-party fused kernels.
try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert x.dtype == torch.bfloat16
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def compute_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    attention_scores: torch.Tensor,
    indexer_loss_coeff: float,
    use_sparse_indexer_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        attention_scores: True attention scores from q @ k^T [batch, heads, seqlen_q, seqlen_k].
        indexer_loss_coeff: Coefficient for the indexer KL divergence loss.
        use_sparse_indexer_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    target_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(target_scores.contiguous(), group=pg_collection.tp)

    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    target_probs = target_scores / target_scores.sum(dim=-1, keepdim=True)

    # Convert index_scores to probabilities with softmax.
    index_probs = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    kl_per_element = target_probs * (
        torch.log(target_probs + 1e-10) - torch.log(index_probs + 1e-10)
    )

    if use_sparse_indexer_loss:
        sparse_mask = torch.zeros_like(kl_per_element).scatter_(-1, topk_indices, 1)
        kl_per_element = kl_per_element * sparse_mask

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
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
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

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: IndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (IndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = self.config.q_lora_rank
        self.index_n_heads = self.config.index_n_heads
        self.index_head_dim = self.config.index_head_dim
        self.index_topk = self.config.index_topk
        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
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
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
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
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        # TODO(kunlunl): The dtype of this module should be torch.get_default_dtype().
        self.weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        for param in self.parameters():
            setattr(param, 'sequence_parallel', self.config.sequence_parallel)

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float):
        """Apply RoPE to the input tensor."""
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        x_nope, x_pe = torch.split(
            x, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], dim=-1
        )
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_nope, x_pe], dim=-1)
        return x

    def _compute_index_scores(
        self, q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform index score using BF16 precision.

        Reference:
            https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
        This is a BF16 implementation of the `fp8_index` logic:
            1. Compute attention scores: q @ k^T;
            2. Apply ReLU activation;
            3. Weight by attention weights;
            4. Sum across attention heads.

        Args:
            q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
            weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
            k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

        Returns:
            index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
        """
        # Compute attention scores: q @ k^T
        # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
        #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
        index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

        # Apply ReLU activation.
        index_scores = torch.relu(index_scores)

        # Weight each head by attention weights.
        # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
        #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
        index_scores = index_scores * weights.unsqueeze(-1)

        # Sum across attention heads.
        # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
        index_scores = index_scores.sum(dim=2)

        # Transpose to [batch, seqlen_q, seqlen_k].
        index_scores = index_scores.transpose(0, 1)

        return index_scores

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
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        assert packed_seq_params is None, "Packed sequence is not supported for SparseAttention"
        assert not self.config.apply_rope_fusion, "RoPE fusion is not supported for SparseAttention"

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
        # Gather inputs if sp is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.wk(x)
        k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # Compute index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
        # [batch, seqlen, seqlen]
        index_scores = self._compute_index_scores(q, weights, k)
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
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


class SparseAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an Indexer to compute top-k attention
    indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
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
        assert (
            self.config.context_parallel_size == 1
        ), "Currently context parallelism is not supported by SparseAttention!"

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

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
            query: Query tensor [sq, b, np, hn].
            key: Key tensor [skv, b, np, hn].
            value: Value tensor [skv, b, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        sq, b, np, hn = query.size()
        skv = key.size(0)
        hnv = value.size(3)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        # Get a FP32 mask with -inf for masked positions.
        if attn_mask_type is not None:
            assert attn_mask_type == AttnMaskType.causal, 'Only causal mask is supported for now'
            # Generate upper triangular mask with -inf above diagonal, 0 elsewhere
            # torch.triu with diagonal=1 creates upper triangular matrix (excluding main diagonal)
            float_mask = torch.triu(
                torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=x.device),
                diagonal=1,
            )
        else:
            assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
            # [b, 1, sq, skv] -> [b, sq, skv]
            mask = attention_mask.squeeze()
            # float_mask [b, sq, skv]
            float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                mask, float('-inf')
            )

        # ===================================
        # Get index scores and top-k indices
        # ===================================
        index_scores, topk_indices = self.indexer.forward_with_scores(
            x, qr, mask=float_mask, packed_seq_params=packed_seq_params
        )

        # ===================================
        # Raw attention scores [b, np, sq, skv]
        # ===================================
        # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
        query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
        # [skv, b, np, hn] -> [b, np, hn, skv] -> [b * np, hn, skv]
        key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
        # Compute attention scores [b * np, sq, skv]
        attention_scores = torch.bmm(query.float(), key.float()) * self.softmax_scale
        # Reshape to [b, np, sq, skv]
        attention_scores = attention_scores.reshape(b, np, sq, skv)

        # ===================================
        # Apply sparse mask from indexer
        # ===================================
        # index_mask [b, sq, skv]
        index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
        index_mask.scatter_(-1, topk_indices, 0)
        index_mask += float_mask
        # [b, np, sq, skv] + [b, 1, sq, skv] -> [b, np, sq, skv]
        attention_scores += index_mask.unsqueeze(1)

        # ===================================
        # Attention probabilities [b, np, sq, skv]
        # ===================================
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

        # ===================================
        # Output
        # ===================================
        # [skv, b, np, hnv] -> [b, np, skv, hnv] -> [b * np, skv, hnv]
        value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
        # Reshape attention_probs: [b, np, sq, skv] -> [b * np, sq, skv]
        attention_probs_reshaped = attention_probs.reshape(b * np, sq, skv)
        # Compute output: [b * np, sq, hnv]
        output = torch.bmm(attention_probs_reshaped.to(value.dtype), value)
        # Reshape output: [b * np, sq, hnv] -> [b, np, sq, hnv] -> [sq, b, np, hnv]
        output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
        # Flatten: [sq, b, np, hnv] -> [sq, b, np * hnv]
        output = output.reshape(sq, b, np * hnv)

        # ===================================
        # Attach indexer loss
        # ===================================
        if self.training and torch.is_grad_enabled():
            # Compute KL divergence loss between indexer scores and true attention scores
            indexer_loss = compute_indexer_loss(
                index_scores,
                topk_indices,
                attention_probs.detach(),
                getattr(self.config, 'indexer_loss_coeff', 0.0),
                getattr(self.config, "use_sparse_indexer_loss", False),
                self.indexer.pg_collection,
            )
            # Attach loss to output output
            output = IndexerLossAutoScaler.apply(output, indexer_loss)

        return output

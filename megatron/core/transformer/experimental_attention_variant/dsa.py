# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek Sparse Attention implementation.

File organization:
1. Utilities: loss logging/autoscaling and shared helpers.
2. Unfused functions: autograd reference loss/topk and unfused sparse attention.
3. DSA modules: dataclasses plus DSAIndexer/DSAttention modules.
"""

import copy
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    fused_indexer_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.te_mxfp8_compat import (
    patch_te_mxfp8_view_backward_if_needed,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker(num_layers: Optional[int] = None):
        """Collect and reduce the indexer losses across ranks.

        Cross-PP `all_reduce` must be invoked on every rank in the pipeline-parallel group,
        otherwise ranks without any indexer layer would skip the collective and cause a hang.
        Pass `num_layers` to lazily initialize the tracker on such ranks so they participate
        with a zero-filled tensor.

        Args:
            num_layers: Total number of decoder layers; required to lazily initialize the
                tracker on ranks where no indexer layer ran.
        """
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            if num_layers is None:
                return
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        values = tracker["values"]

        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
        num_layers: Optional[int] = None,
        csa_compress_ratios: Optional[List[int]] = None,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
            num_layers: Total number of decoder layers (including MTP). Required when running
                with hybrid attention layouts where some PP ranks may not own any indexer
                layer; passing it ensures every PP rank participates in the cross-PP
                `all_reduce`.
            csa_compress_ratios: Per-layer compress ratios for compressed sparse attention.
                When provided, the cross-layer average uses the count of layers with
                ``ratio == 4`` (the only ratio that owns an indexer) as the divisor.
                Otherwise (legacy DSA path) every layer is assumed to be an indexer layer
                and the divisor is the tracker tensor size.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(num_layers=num_layers)
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale

        if csa_compress_ratios is not None:
            num_indexer_layers = sum(1 for r in csa_compress_ratios if r == 4)
        else:
            num_indexer_layers = indexer_loss_values.shape[0]

        # Average across layers that actually own an indexer; layers without one
        # contribute zero in `tracker["values"]` so they must not be in the divisor.
        avg_indexer_loss = indexer_loss_values.sum() / max(num_indexer_layers, 1)

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


class DSAIndexerLossAutoScaler(torch.autograd.Function):
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
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


# -----------------------------------------------------------------------------
# Unfused functions
# -----------------------------------------------------------------------------


def unfused_dsa_indexer_loss_and_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    indexer_softmax_scale: float,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    topk: int,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
    calculate_per_token_loss: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute unfused indexer top-k and KL loss with ordinary PyTorch autograd.

    Args:
        index_q: Indexer query [seqlen_q, batch, index_heads, index_head_dim].
        index_k: Indexer key [seqlen_k, batch, index_head_dim].
        index_weights: Indexer head weights [seqlen_q, batch, index_heads].
        indexer_softmax_scale: Scale applied to indexer scores.
        query: Attention query [seqlen_q, batch, heads, dim].
        key: Attention key [seqlen_k, batch, 1, dim] for MQA, or [seqlen_k, batch, heads, dim].
        softmax_scale: Scale applied to attention scores.
        topk: Number of KV positions selected by the indexer.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: Whether to compute KL only over selected top-k positions.
        pg_collection: Process group collection.
        calculate_per_token_loss: If True, return raw local row sum.
        mask: Optional additive mask [seqlen_q, seqlen_k] or [batch, seqlen_q, seqlen_k].

    Returns:
        Tuple of top-k indices [batch, seqlen_q, topk] and scalar indexer loss.
    """
    sq, b, np, _ = query.size()
    sk = index_k.size(0)
    assert key.size(0) == sk, "Attention key and indexer key must have matching sequence length."
    assert pg_collection.tp.size() == 1, "Tensor parallelism is not supported for DSA indexer loss."

    index_scores = torch.einsum('sbhd,tbd->sbht', index_q.float(), index_k.float())
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * index_weights.unsqueeze(-1)
    index_scores = index_scores * indexer_softmax_scale
    index_scores = index_scores.sum(dim=2).transpose(0, 1)

    if mask is None:
        causal_mask = torch.triu(
            torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=index_scores.device),
            diagonal=1,
        )
    else:
        causal_mask = mask.to(dtype=torch.float32)
    if causal_mask.dim() == 3:
        index_scores = index_scores + causal_mask
    else:
        index_scores = index_scores + causal_mask.view(1, sq, sk)

    topk_k = min(topk, sk)
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    if key.size(-2) == 1:
        attention_scores = torch.einsum('sbhd,tbd->bhst', query.float(), key.squeeze(-2).float())
    else:
        assert key.size(-2) == np, "Attention key must be MQA or match query heads."
        attention_scores = torch.einsum('sbhd,tbhd->bhst', query.float(), key.float())
    attention_scores = attention_scores * softmax_scale

    if causal_mask.dim() == 3:
        attention_scores = attention_scores + causal_mask.unsqueeze(1)
    else:
        attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)

    if sparse_loss:
        index_mask = torch.full(
            (b, sq, sk), float('-inf'), dtype=torch.float32, device=index_scores.device
        ).scatter_(-1, topk_indices, 0)
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        index_scores = index_scores + index_mask

    row_valid = (causal_mask > float('-inf')).any(dim=-1)
    if row_valid.dim() == 1:
        attn_row_mask = row_valid.view(1, 1, sq, 1)
        idx_row_mask = row_valid.view(1, sq, 1)
    else:
        attn_row_mask = row_valid.view(b, 1, sq, 1)
        idx_row_mask = row_valid.view(b, sq, 1)

    attention_scores = attention_scores.masked_fill(~attn_row_mask, 0.0)
    index_scores = index_scores.masked_fill(~idx_row_mask, 0.0)

    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    attention_scores = attention_scores * attn_row_mask.float()
    index_scores = index_scores * idx_row_mask.float()

    attention_scores = attention_scores.sum(dim=1)
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(torch.float32).tiny
    )

    eps = torch.finfo(torch.float32).tiny
    target = attention_scores.clamp(min=eps)
    predict = index_scores.clamp(min=eps)
    kl_per_row = (target * (torch.log(target) - torch.log(predict))).sum(dim=-1)
    if row_valid.dim() == 1:
        row_valid_bsq = row_valid.view(1, sq).expand(b, sq)
    else:
        row_valid_bsq = row_valid
    kl_per_row = torch.where(row_valid_bsq, kl_per_row, torch.zeros_like(kl_per_row))
    kl_div = kl_per_row.sum()
    if not calculate_per_token_loss:
        kl_div = kl_div / (b * sq)
    indexer_loss = kl_div * loss_coeff

    return topk_indices, indexer_loss


def unfused_dsa_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    v_channels: int,
) -> torch.Tensor:
    """Unfused sparse attention for absorbed MLA's MQA-shaped KV."""
    sq, b, np, _ = query.size()
    skv = key.size(0)

    kv = key.squeeze(-2)
    value = kv[..., :v_channels]

    attention_scores = torch.einsum('sbhd,tbd->bhst', query.float(), kv.float())
    attention_scores = attention_scores * softmax_scale

    index_mask = torch.full((b, sq, skv), float('-inf'), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    causal_mask = torch.triu(
        torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
        diagonal=1,
    )
    attention_scores = attention_scores + index_mask.unsqueeze(1) + causal_mask.view(1, 1, sq, skv)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    output = torch.einsum('bhst,tbd->sbhd', attention_scores.to(value.dtype), value)
    return output.contiguous().reshape(sq, b, np * v_channels)


# -----------------------------------------------------------------------------
# DSA dataclasses and modules
# -----------------------------------------------------------------------------


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

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
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection
        if getattr(self.config, "tensor_model_parallel_size", 1) != 1:
            raise ValueError("Tensor parallelism is not supported for DSAIndexer.")
        if self.pg_collection.tp.size() > 1:
            raise ValueError("Tensor parallelism is not supported for DSAIndexer.")

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

        self.linear_wq_b = build_module(
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

        self.linear_wk = build_module(
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

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
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

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float):
        """Apply RoPE to the input tensor."""
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            # This flag is for the MLA-style interleaving in RoPE.
            # Set it to False, as indexer does not apply interleaved RoPE.
            mla_rotary_interleaved=False,
        )
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return indexer query, key, and weights for DSA top-k/loss computation."""
        assert packed_seq_params is None, "Packed sequence is not supported for DSAttention"
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
            raise ValueError("Tensor parallelism is not supported for DSAIndexer.")

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
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
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5)

        return q, k, weights


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(config=config)

        assert (
            getattr(config, "context_parallel_size", 1) == 1
        ), "Context parallelism is not supported for DSAttention."
        if getattr(config, "tensor_model_parallel_size", 1) != 1:
            assert False, "Tensor parallelism is not supported for DSAttention."
        if parallel_state.model_parallel_is_initialized():
            if parallel_state.get_tensor_model_parallel_world_size() != 1:
                assert False, "Tensor parallelism is not supported for DSAttention."
        patch_te_mxfp8_view_backward_if_needed(config)

        self.layer_number = layer_number
        if is_mtp_layer:
            self.layer_number = self.layer_number + self.config.num_layers

        self.indexer = build_module(
            submodules.indexer, config=self.config, pg_collection=pg_collection
        )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale
        self.v_channels = v_channels
        self.pg_collection = pg_collection
        self.force_unfused_dsa = getattr(config, 'force_unfused_dsa', False)
        self.apply_dsa_kernel_fusion = getattr(config, 'apply_dsa_kernel_fusion', False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn].
            key: Key tensor [skv, b, 1, hn] for absorbed MLA.
            value: Must be None; absorbed MLA stores values inside key.
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        assert x is not None and qr is not None, "DSAttention requires hidden states and q_lora."
        assert attention_bias is None, "Attention bias is not supported for DSAttention."
        assert packed_seq_params is None, "Packed sequence is not supported for DSAttention."
        assert value is None, "DSAttention expects absorbed MLA with value=None."
        assert key.size(-2) == 1, "DSAttention expects MQA KV with one KV head."

        sq, b, np, _ = query.size()
        skv = key.size(0)
        d_v = self.v_channels or self.config.kv_lora_rank or self.config.v_head_dim
        assert key.size(-1) >= d_v, "DSAttention key must contain the value channels."

        backend = getattr(self.config, "attention_backend", None)
        indexer_sparse_loss = getattr(self.config, "dsa_indexer_use_sparse_loss", False)
        use_fused = (
            self.apply_dsa_kernel_fusion
            and not self.force_unfused_dsa
            and backend != AttnBackend.unfused
            and backend != "unfused"
        )
        use_unfused = not use_fused
        if not use_unfused:
            assert (
                attn_mask_type == AttnMaskType.causal
            ), "Only causal mask is supported for fused DSAttention."

        x = x.detach()
        qr = qr.detach()
        index_q, index_k, index_weights = self.indexer(x, qr, packed_seq_params)

        effective_topk = min(self.indexer.index_topk, skv)
        indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', None) or 0.0

        if not use_unfused:
            kv = key.squeeze(-2)
            attn_sink = torch.full((np,), float("-inf"), dtype=torch.float32, device=query.device)
            window_idxs = torch.empty((b, sq, 0), dtype=torch.int32, device=query.device)
            output, indexer_loss = fused_indexer_sparse_attn(
                query,
                kv,
                attn_sink,
                window_idxs,
                index_q,
                index_k,
                index_weights,
                effective_topk,
                1,
                self.softmax_scale,
                self.indexer.softmax_scale,
                indexer_loss_coeff,
                sparse_loss=indexer_sparse_loss,
                kv_offset=0,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
                d_v=d_v,
            )
        else:
            if attn_mask_type is not None:
                assert (
                    attn_mask_type == AttnMaskType.causal
                ), 'Only causal mask is supported for now'
                float_mask = torch.triu(
                    torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=x.device),
                    diagonal=1,
                )
            else:
                assert attention_mask.shape == (b, 1, sq, skv), 'attention_mask shape mismatch'
                mask = attention_mask.squeeze(1)
                float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                    mask, float('-inf')
                )

            if self.training and torch.is_grad_enabled() and indexer_loss_coeff > 0:
                topk_indices, indexer_loss = unfused_dsa_indexer_loss_and_topk(
                    index_q,
                    index_k,
                    index_weights,
                    self.indexer.softmax_scale,
                    query.detach(),
                    key.detach(),
                    self.softmax_scale,
                    effective_topk,
                    indexer_loss_coeff,
                    indexer_sparse_loss,
                    self.indexer.pg_collection,
                    calculate_per_token_loss=self.config.calculate_per_token_loss,
                    mask=float_mask,
                )
            else:
                index_scores = torch.einsum('sbhd,tbd->sbht', index_q.float(), index_k.float())
                index_scores = torch.relu(index_scores)
                index_scores = index_scores * index_weights.unsqueeze(-1)
                index_scores = index_scores * self.indexer.softmax_scale
                index_scores = index_scores.sum(dim=2).transpose(0, 1)
                if float_mask.dim() == 3:
                    index_scores = index_scores + float_mask
                else:
                    index_scores = index_scores + float_mask.view(1, sq, skv)
                topk_indices = index_scores.topk(effective_topk, dim=-1)[1]
                if self.training and torch.is_grad_enabled():
                    indexer_loss = (
                        index_q.float().sum() + index_k.float().sum() + index_weights.float().sum()
                    ) * 0.0
                else:
                    indexer_loss = torch.zeros((), dtype=torch.float32, device=query.device)
            output = unfused_dsa_fn(query, key, topk_indices, self.softmax_scale, d_v)

        if indexer_loss_coeff > 0:
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=indexer_loss,
                layer_number=self.layer_number,
                num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
            )

        # With zero coefficient this preserves zero gradients for indexer parameters
        # instead of making them unused.
        output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        return output

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None

from megatron.core.transformer.experimental_attention_variant.dsa_fused_kernels import DSAFunction


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
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
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
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]

        # Average across all layers (assuming all layers have sparse attention)
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

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

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
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
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

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
    ):
        super().__init__(config=config)

        self.layer_number = layer_number

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
            query: Query tensor [sq, b, np, hn] or [sq, np, hn] in THD mode.
            key: Key tensor [skv, b, 1, hn] or [skv, 1, hn] in THD mode.
            value: Must be None (value is derived from key).
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, np * v_channels] or [sq, np * v_channels] in THD mode.
        """
        assert value is None
        assert key.size(-2) == 1
        assert attn_mask_type == AttnMaskType.causal, "Only causal mask is supported for now"

        use_unfused = getattr(self.config, "attention_backend", None) == AttnBackend.unfused
        indexer_loss_coeff = getattr(self.config, "dsa_indexer_loss_coeff", 0.0)
        cu_seqlens = getattr(packed_seq_params, "cu_seqlens_q_padded", None)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        q, k, weights = self.indexer(x, qr, packed_seq_params)

        def indexer_loss_logging_helper(loss):
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=loss, layer_number=self.layer_number, num_layers=self.config.num_layers
            )

        output = DSAFunction.apply(
            query,
            key,
            q,
            k,
            weights,
            cu_seqlens,
            self.indexer.index_topk,
            self.v_channels,
            self.softmax_scale,
            indexer_loss_coeff,
            indexer_loss_logging_helper,
            self.pg_collection.tp,
            use_unfused,
        )

        # merge last dimension of output: [sq, b, np, hn] -> [sq, b, np * hn]
        output = output.flatten(start_dim=2)

        return output

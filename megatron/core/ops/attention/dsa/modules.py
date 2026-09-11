# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.ops.attention.dsa import dsa_layout, dsa_masking
from megatron.core.ops.attention.dsa.backends import DSAKernels, select_dsa_kernels
from megatron.core.ops.attention.dsa.kernel_metadata import HADAMARD_ROTATION
from megatron.core.ops.attention.dsa.reference import (
    _FUSED_DSA_INDEXER_LOSS_INPUT_NAMES as _FUSED_DSA_INDEXER_LOSS_INPUT_NAMES,
)
from megatron.core.ops.attention.dsa.reference import FusedDSAIndexerLoss as FusedDSAIndexerLoss
from megatron.core.ops.attention.dsa.reference import _compute_index_scores as _compute_index_scores
from megatron.core.ops.attention.dsa.reference import (
    _compute_indexer_teacher_probabilities as _compute_indexer_teacher_probabilities,
)
from megatron.core.ops.attention.dsa.reference import (
    _normalize_indexer_teacher_target as _normalize_indexer_teacher_target,
)
from megatron.core.ops.attention.dsa.reference import (
    _unfused_absorbed_dsa_fn as _unfused_absorbed_dsa_fn,
)
from megatron.core.ops.attention.dsa.reference import (
    bwd_fused_indexer_loss_naive as bwd_fused_indexer_loss_naive,
)
from megatron.core.ops.attention.dsa.reference import (
    compute_dsa_indexer_loss as compute_dsa_indexer_loss,
)
from megatron.core.ops.attention.dsa.reference import fused_qk_topk_naive as fused_qk_topk_naive
from megatron.core.ops.attention.dsa.reference import (
    fwd_fused_indexer_loss_naive as fwd_fused_indexer_loss_naive,
)
from megatron.core.ops.attention.dsa.reference import unfused_dsa_fn as unfused_dsa_fn
from megatron.core.ops.attention.dsa.rotation import rotate_activation as rotate_activation
from megatron.core.ops.kernel_metadata import DeterminismPolicy, validate_kernel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.dsa_loss import DSAIndexerLossAutoScaler as DSAIndexerLossAutoScaler
from megatron.core.transformer.dsa_loss import (
    DSAIndexerLossLoggingHelper as DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_size

if TYPE_CHECKING:
    from megatron.core.models.backends import BackendSpecProvider

__all__ = [
    "DSAttention",
    "DSAttentionSubmodules",
    "DSAIndexer",
    "DSAIndexerSubmodules",
    "DSAIndexerLossAutoScaler",
    "DSAIndexerLossLoggingHelper",
    "FusedDSAIndexerLoss",
    "bwd_fused_indexer_loss_naive",
    "compute_dsa_indexer_loss",
    "fused_qk_topk_naive",
    "fwd_fused_indexer_loss_naive",
    "is_dsa_skip_topk_layer",
    "rotate_activation",
    "source_dsa_compute_layer",
    "unfused_dsa_fn",
]


def is_dsa_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA top-k result."""
    if layer_number < 1:
        raise ValueError(f"layer_number must be 1-indexed and positive, got {layer_number}.")
    if skip_topk_offset < 0:
        raise ValueError(f"skip_topk_offset must be non-negative, got {skip_topk_offset}.")
    if topk_freq < 1:
        raise ValueError(f"topk_freq must be positive, got {topk_freq}.")
    # Layers are 1-indexed, so the default offset 0 must still start at layer 1.
    skip_topk_offset = max(skip_topk_offset, 1)
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_dsa_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the computing layer whose DSA top-k a skip layer reuses."""
    is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
    skip_topk_offset = max(skip_topk_offset, 1)
    if layer_number <= skip_topk_offset:
        return layer_number
    return layer_number - ((layer_number - skip_topk_offset) % topk_freq)


def _run_sparse_attention(
    *,
    absorbed_mla: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    up_v_weight: Optional[torch.Tensor],
    topk_indices: torch.Tensor,
    softmax_scale: float,
    config: TransformerConfig,
    mask: Optional[torch.Tensor],
    varlen_starts: Optional[torch.Tensor],
    varlen_ends: Optional[torch.Tensor],
    key_positions: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor] = None,
    kernels: DSAKernels | None = None,
) -> torch.Tensor:
    """Run sparse attention for absorbed and non-absorbed MLA paths."""
    if kernels is None:
        # Compatibility for direct helper callers; DSAttention supplies bound hooks.
        kernels = select_dsa_kernels(config)
    if absorbed_mla:
        latent_v_channels = int(getattr(config, "kv_lora_rank", 0) or 0)
        if latent_v_channels <= 0:
            raise RuntimeError(
                "Invalid kv_lora_rank for absorbed-MLA DSAttention sparse attention."
            )
        if up_v_weight is None:
            raise RuntimeError(
                "Absorbed DSAttention requires up_v_weight for latent-to-value projection."
            )
        if value is not None:
            raise RuntimeError(
                "Absorbed DSAttention expects value=None (latent path). "
                "Received absorbed layout with explicit value tensor."
            )
        output = None
        if kernels.run_fused_absorbed_sparse_attention is not None:
            output = kernels.run_fused_absorbed_sparse_attention(
                query, key, topk_indices, softmax_scale, latent_v_channels, topk_length=topk_length
            )
            if output is None:
                kernels.log_declined("run_fused_absorbed_sparse_attention")
        # Fused backends may decline unsupported shapes or layouts by returning
        # None, so keep the absorbed PyTorch path as the authoritative fallback.
        if output is None:
            output = _unfused_absorbed_dsa_fn(
                query,
                key,
                topk_indices,
                softmax_scale,
                latent_v_channels,
                mask=mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )
        assert output is not None
        output = torch.einsum("sbhc,hdc->sbhd", output, up_v_weight).contiguous()
        output = output.view(output.size(0), output.size(1), -1)
        return output

    return unfused_dsa_fn(
        query,
        key,
        value,
        topk_indices,
        softmax_scale,
        mask=mask,
        varlen_starts=varlen_starts,
        varlen_ends=varlen_ends,
        key_positions=key_positions,
    )


def _normalize_dsattention_output_rank(output: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Normalize DSAttention output rank to match caller hidden-state rank."""
    if target_ndim not in (2, 3):
        raise RuntimeError(f"DSAttention expected x.ndim in (2, 3), got {target_ndim}")

    if output.ndim == 4:
        output = output.reshape(output.size(0), output.size(1), -1)
    elif output.ndim not in (2, 3):
        raise RuntimeError(
            f"DSAttention produced unexpected output rank {output.ndim}; expected 2D/3D/4D."
        )

    if target_ndim == 3 and output.ndim == 2:
        output = output.unsqueeze(1)
    elif target_ndim == 2 and output.ndim == 3:
        if output.size(1) != 1:
            raise RuntimeError(
                "DSAttention cannot squeeze non-singleton batch dim for packed output: "
                f"shape={tuple(output.shape)}"
            )
        output = output.squeeze(1)

    if output.ndim != target_ndim:
        raise RuntimeError(
            "DSAttention output rank mismatch after normalization: "
            f"target_ndim={target_ndim}, output_shape={tuple(output.shape)}"
        )
    return output


def _validate_nonpacked_cp_uniform_length(
    sq: int,
    skv: int,
    cp_size: int,
    cp_group: Optional[torch.distributed.ProcessGroup],
    device: torch.device,
) -> None:
    """Validate the uniform-length precondition for non-packed allgather CP."""
    expected_skv = sq * cp_size
    if (
        cp_group is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and get_pg_size(cp_group) == cp_size
    ):
        local_len = torch.tensor([sq], device=device, dtype=torch.int64)
        all_lens = [torch.empty_like(local_len) for _ in range(cp_size)]
        torch.distributed.all_gather(all_lens, local_len, group=cp_group)
        all_lens = torch.cat(all_lens)
        if not torch.all(all_lens == sq):
            raise RuntimeError(
                "Non-packed DSA allgather CP expects uniform per-rank sequence lengths; "
                f"got per-rank lengths {all_lens.tolist()}."
            )
        expected_skv = int(all_lens.sum().item())

    if skv != sq and skv != expected_skv:
        raise RuntimeError(
            "Non-packed DSA allgather CP expects uniform per-rank sequence lengths; "
            f"got local query length {sq} and key length {skv} for cp_size={cp_size}."
        )


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
        if config.dsa_indexer_rotate_activation:
            validate_kernel(
                HADAMARD_ROTATION,
                determinism=(
                    DeterminismPolicy.WARN
                    if config.deterministic_mode
                    else DeterminismPolicy.IGNORE
                ),
            )
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
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
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
        k_norm_eps = (
            self.config.dsa_indexer_k_norm_epsilon
            if self.config.dsa_indexer_k_norm_epsilon is not None
            else self.config.layernorm_epsilon
        )
        self.k_norm = build_module(
            submodules.k_norm, config=k_norm_config, hidden_size=self.index_head_dim, eps=k_norm_eps
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
        # Indexer projections are duplicated across tensor-parallel ranks, so their gradients
        # should be averaged during final gradient synchronization.
        for param in self.parameters():
            setattr(param, "average_gradients_across_tp_domain", True)

    def _apply_rope(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mscale: float,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        """Apply RoPE to the input tensor."""
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        squeezed_batch_dim = False
        if cu_seqlens is not None and cu_seqlens.device != x_pe.device:
            cu_seqlens = cu_seqlens.to(device=x_pe.device)
        # THD RoPE path expects [t, h, d], while indexer tensors are [t, 1, h, d].
        if cu_seqlens is not None and x_pe.ndim == 4 and x_pe.size(1) == 1:
            x_pe = x_pe.squeeze(1)
            squeezed_batch_dim = True
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=cu_seqlens,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            # This flag is for the MLA-style interleaving in RoPE.
            mla_rotary_interleaved=self.config.dsa_indexer_rope_interleaved,
        )
        if squeezed_batch_dim:
            x_pe = x_pe.unsqueeze(1)
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        if packed_seq:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
        else:
            cu_seqlens_q = cu_seqlens_kv = None

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
        q = self._apply_rope(q, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_q)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        if self.config.dsa_indexer_k_norm_fp32:
            k_dtype = k.dtype
            k = self.k_norm(k.float()).to(dtype=k_dtype)
        else:
            k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale, cu_seqlens=cu_seqlens_kv)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        if self.config.dsa_indexer_rotate_activation:
            q = rotate_activation(q)
            k = rotate_activation(k)

        # =========================================
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Optional additive attention mask [seqlen, seqlen] or
                [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(
            q, k, weights, self.index_topk, mask, use_relu=self.config.dsa_indexer_scoring_relu
        )

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

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


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    consumes_absorbed_v_up_projection = True
    requires_dsa_inputs = True
    _HOLDER_ATTR = "_dsa_index_share_topk_holder"
    _LENGTH_HOLDER_ATTR = "_dsa_index_share_topk_length_holder"

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
        kernel_backend: "BackendSpecProvider | None" = None,
    ):
        super().__init__(config=config)

        from megatron.core.models.backends import backend_slot, get_backend_from_config

        self.dsa_kernels = backend_slot(
            backend=(
                kernel_backend if kernel_backend is not None else get_backend_from_config(config)
            ),
            name="dsa_kernels",
            default=lambda: select_dsa_kernels(config),
            config=config,
        )

        self.layer_number = layer_number
        self.index_topk = self.config.dsa_indexer_topk
        self.index_topk_freq = self.config.dsa_indexer_topk_freq or 1
        self.index_skip_topk_offset = self.config.dsa_indexer_skip_topk_offset or 0
        self.index_share = self.index_topk_freq > 1
        self.skip_topk = self.index_share and is_dsa_skip_topk_layer(
            layer_number, self.index_skip_topk_offset, self.index_topk_freq
        )
        self.source_layer = (
            source_dsa_compute_layer(
                layer_number, self.index_skip_topk_offset, self.index_topk_freq
            )
            if self.index_share
            else layer_number
        )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        self.indexer = None
        if not self.skip_topk:
            self.indexer = build_module(
                submodules.indexer, config=self.config, pg_collection=self.pg_collection
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale
        self.cp_comm_type = dsa_layout.normalize_cp_comm_type(cp_comm_type)

    def _get_index_share_carrier(
        self, packed_seq_params: Optional[PackedSeqParams], attention_mask: Optional[torch.Tensor]
    ) -> object:
        """Return the object that carries DSA top-k sharing state for this forward."""
        if packed_seq_params is not None:
            return packed_seq_params
        return attention_mask if attention_mask is not None else self.config

    def _get_index_share_topk_holder(
        self,
        packed_seq_params: Optional[PackedSeqParams],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[int, torch.Tensor]:
        """Return the per-forward top-k holder for DSA index sharing."""
        carrier = self._get_index_share_carrier(packed_seq_params, attention_mask)
        holder = getattr(carrier, self._HOLDER_ATTR, None)
        if holder is None:
            holder = {}
            setattr(carrier, self._HOLDER_ATTR, holder)
        return holder

    def _get_index_share_topk_length_holder(
        self,
        packed_seq_params: Optional[PackedSeqParams],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[int, torch.Tensor]:
        """Return the optional per-forward top-k length holder."""
        carrier = self._get_index_share_carrier(packed_seq_params, attention_mask)
        holder = getattr(carrier, self._LENGTH_HOLDER_ATTR, None)
        if holder is None:
            holder = {}
            setattr(carrier, self._LENGTH_HOLDER_ATTR, holder)
        return holder

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        up_v_weight: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn] or packed [t, np, hn].
            key: Key tensor [skv, b, np, hn] or packed [t, np, hn].
            value: Value tensor [skv, b, np, hnv] or packed [t, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            position_ids: Optional position ids [b, sq], used by allgather CP causal masking.
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        query, _ = dsa_layout.ensure_sbhd(query, "query")
        key, _ = dsa_layout.ensure_sbhd(key, "key")
        if value is not None:
            value, _ = dsa_layout.ensure_sbhd(value, "value")
        if up_v_weight is not None:
            assert up_v_weight.ndim == 3, "up_v_weight must be [heads, v_head_dim, kv_lora_rank]"
            up_v_weight = up_v_weight.to(device=query.device, dtype=query.dtype).contiguous()
            if value is not None:
                raise RuntimeError(
                    "DSAttention received up_v_weight with explicit value tensor. "
                    "For absorbed DSA path, value must be None."
                )

        latent_v_channels = int(getattr(self.config, "kv_lora_rank", 0) or 0)
        qk_pos_dim = int(getattr(self.config, "qk_pos_emb_head_dim", 0) or 0)
        expected_absorbed_dim = latent_v_channels + qk_pos_dim
        absorbed_mla = (
            latent_v_channels > 0
            and expected_absorbed_dim > 0
            and key.size(2) == 1
            and query.size(-1) == key.size(-1) == expected_absorbed_dim
        )
        if value is None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received value=None but query/key are not in absorbed layout. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )
        if up_v_weight is not None and not absorbed_mla:
            raise RuntimeError(
                "DSAttention received up_v_weight but absorbed layout was not detected. "
                f"query_hdim={query.size(-1)}, key_hdim={key.size(-1)}, key_heads={key.size(2)}, "
                f"expected_absorbed_dim={expected_absorbed_dim}"
            )

        sq, b, _, _ = query.size()
        local_sequence_rows = x.size(0)

        cp_group = getattr(self.pg_collection, "cp", None)
        cp_size = get_pg_size(cp_group)
        cp_rank = cp_group.rank() if cp_group is not None else 0
        tp_group = getattr(self.pg_collection, "tp", None)
        tp_size = get_pg_size(tp_group)
        sequence_parallel_tp = self.config.sequence_parallel and tp_size > 1
        sequence_parallel_tp_row_start = 0
        sequence_parallel_tp_full_rows = sq
        sequence_parallel_query_is_local = False
        if sequence_parallel_tp:
            sequence_parallel_tp_full_rows = local_sequence_rows * tp_size
            if sq == local_sequence_rows:
                sequence_parallel_query_is_local = True
                sequence_parallel_tp_row_start = tp_group.rank() * local_sequence_rows
            elif sq != sequence_parallel_tp_full_rows:
                raise RuntimeError(
                    "DSA sequence-parallel query row count mismatch: "
                    f"query_rows={sq}, local_rows={local_sequence_rows}, tp_size={tp_size}"
                )
        packed_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        packed_query_positions = None
        nonpacked_query_positions = None
        kv_reorder_idx = None
        single_packed_thd_sequence = False
        if packed_thd:
            cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(packed_seq_params)
            single_packed_thd_sequence = (
                cp_size > 1 and cu_seqlens_q.numel() == 2 and cu_seqlens_kv.numel() == 2
            )
            packed_query_output_size = (
                sequence_parallel_tp_full_rows if sequence_parallel_tp else sq
            )
            packed_global_output_size = packed_query_output_size * cp_size
            if sequence_parallel_query_is_local and cp_size == 1:
                row_start = sequence_parallel_tp_row_start
                packed_query_positions = torch.arange(
                    row_start, row_start + sq, dtype=torch.int64, device=query.device
                )
            elif sequence_parallel_tp and cp_size > 1:
                packed_query_positions_full = dsa_layout.build_packed_allgather_cp_local_positions(
                    cu_seqlens_q,
                    cp_size,
                    cp_rank,
                    query.device,
                    output_size=packed_query_output_size,
                )
                if sequence_parallel_query_is_local:
                    row_start = sequence_parallel_tp_row_start
                    packed_query_positions = packed_query_positions_full[row_start : row_start + sq]
                else:
                    packed_query_positions = packed_query_positions_full
            elif cp_size > 1:
                # For one sequence, host max-seqlen metadata proves whether cu_seqlens already
                # covers every packed row without synchronizing on the CUDA cu_seqlens tensor.
                query_cu_seqlens_cover_output = (
                    single_packed_thd_sequence
                    and isinstance(packed_seq_params.max_seqlen_q, int)
                    and packed_seq_params.max_seqlen_q == packed_global_output_size
                )
                key_cu_seqlens_cover_output = (
                    single_packed_thd_sequence
                    and isinstance(packed_seq_params.max_seqlen_kv, int)
                    and packed_seq_params.max_seqlen_kv == packed_global_output_size
                )
                packed_query_positions, kv_reorder_idx = (
                    dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        device=query.device,
                        local_output_size=packed_query_output_size,
                        key_local_output_size=packed_query_output_size,
                        global_output_size=packed_global_output_size,
                        query_cu_seqlens_cover_output=query_cu_seqlens_cover_output,
                        key_cu_seqlens_cover_output=key_cu_seqlens_cover_output,
                    )
                )
            if packed_query_positions is not None:
                packed_query_positions = packed_query_positions.contiguous()
        elif cp_size > 1:
            _validate_nonpacked_cp_uniform_length(
                sq=sq, skv=key.size(0), cp_size=cp_size, cp_group=cp_group, device=query.device
            )

        if sequence_parallel_tp:
            if key.size(0) == local_sequence_rows:
                key = gather_from_sequence_parallel_region(key, group=tp_group)
            elif key.size(0) != sequence_parallel_tp_full_rows:
                raise RuntimeError(
                    "DSA sequence-parallel key row count mismatch before CP gather: "
                    f"key_rows={key.size(0)}, local_rows={local_sequence_rows}, "
                    f"full_rows={sequence_parallel_tp_full_rows}, tp_size={tp_size}"
                )
            if value is not None:
                if value.size(0) == local_sequence_rows:
                    value = gather_from_sequence_parallel_region(value, group=tp_group)
                elif value.size(0) != sequence_parallel_tp_full_rows:
                    raise RuntimeError(
                        "DSA sequence-parallel value row count mismatch before CP gather: "
                        f"value_rows={value.size(0)}, local_rows={local_sequence_rows}, "
                        f"full_rows={sequence_parallel_tp_full_rows}, tp_size={tp_size}"
                    )

        local_cp_kv_lens = {sq}
        if sequence_parallel_tp:
            local_cp_kv_lens.add(sequence_parallel_tp_full_rows)
        local_cp_kv_len = None
        if cp_size > 1:
            assert (
                self.cp_comm_type == "allgather"
            ), "DSAttention context parallelism currently supports cp_comm_type=allgather only."

            # For allgather CP, keys/values are expected in full-sequence order.
            # Gather local-sequence tensors, then undo MCore's zigzag rank order.
            def _build_kv_reorder_idx(local_len):
                if packed_thd:
                    _, idx = dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder(
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        device=query.device,
                        local_output_size=local_len,
                        key_local_output_size=local_len,
                        global_output_size=local_len * cp_size,
                    )
                    return idx
                return dsa_layout.build_zigzag_allgather_cp_key_reorder(
                    sq=local_len, cp_size=cp_size, device=query.device
                )

            gathered_cp_key = False
            gathered_cp_value = False
            if key.size(0) in local_cp_kv_lens:
                local_cp_kv_len = key.size(0)
                if kv_reorder_idx is None:
                    kv_reorder_idx = _build_kv_reorder_idx(local_cp_kv_len)
                key = gather_from_sequence_parallel_region(key, group=cp_group)
                gathered_cp_key = True
            if value is not None and value.size(0) in local_cp_kv_lens:
                if local_cp_kv_len is None:
                    local_cp_kv_len = value.size(0)
                    if kv_reorder_idx is None:
                        kv_reorder_idx = _build_kv_reorder_idx(local_cp_kv_len)
                elif value.size(0) != local_cp_kv_len:
                    raise RuntimeError(
                        "DSA local key/value sequence length mismatch before CP gather: "
                        f"key_len={local_cp_kv_len}, value_len={value.size(0)}"
                    )
                value = gather_from_sequence_parallel_region(value, group=cp_group)
                gathered_cp_value = True
            if kv_reorder_idx is not None:
                if gathered_cp_key:
                    if key.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered key length mismatch: "
                            f"key_seqlen={key.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    key = key.index_select(0, kv_reorder_idx)
                if gathered_cp_value:
                    if value.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered value length mismatch: "
                            f"value_seqlen={value.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    value = value.index_select(0, kv_reorder_idx)

        skv = key.size(0)

        if not packed_thd and sequence_parallel_query_is_local:
            nonpacked_query_positions = dsa_layout.extract_query_positions_from_position_ids(
                position_ids, sq, query.device
            )
            if nonpacked_query_positions is None:
                full_query_positions, _ = dsa_layout.get_cp_positions_from_layout(
                    sq=sequence_parallel_tp_full_rows,
                    skv=skv,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    cp_comm_type=self.cp_comm_type,
                    device=query.device,
                    cp_group=cp_group,
                )
                row_start = sequence_parallel_tp_row_start
                nonpacked_query_positions = full_query_positions[
                    row_start : row_start + sq
                ].contiguous()

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
        computes_topk = not self.skip_topk
        use_indexer_loss = (
            self.training and torch.is_grad_enabled() and indexer_loss_coeff > 0 and computes_topk
        )
        if use_indexer_loss and sequence_parallel_query_is_local:
            raise RuntimeError(
                "DSA indexer loss requires TP ranks to own the same query rows; "
                "sequence-local TP query shards cannot form a global-head target."
            )
        float_mask, varlen_params, varlen_is_plain_causal = (
            dsa_masking.build_dsattention_forward_mask(
                sq=sq,
                skv=skv,
                b=b,
                device=x.device,
                cp_size=cp_size,
                cp_rank=cp_rank,
                cp_comm_type=self.cp_comm_type,
                cp_group=cp_group,
                attn_mask_type=attn_mask_type,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
                packed_query_positions=packed_query_positions,
                nonpacked_query_positions=nonpacked_query_positions,
            )
        )
        if varlen_params is not None:
            varlen_starts, varlen_ends, key_positions = varlen_params
        else:
            varlen_starts = varlen_ends = key_positions = None
        query_valid_rows = dsa_masking.extract_query_valid_rows_from_packed_seq_params(
            packed_seq_params, b=b, sq=sq, device=query.device
        )
        use_fused_kernels = self.dsa_kernels.backend != "none"
        sparse_indexer_loss = self.config.dsa_indexer_use_sparse_loss
        use_local_indexer_varlen = (
            packed_thd
            and cp_size > 1
            and attn_mask_type == AttnMaskType.causal
            and varlen_starts is not None
            and varlen_ends is not None
            and key_positions is None
        )
        indexer_reduce_group = (
            cp_group if cp_size > 1 and self.config.calculate_per_token_loss else None
        )
        indexer_avg_group = (
            cp_group if cp_size > 1 and not self.config.calculate_per_token_loss else None
        )

        topk_holder = (
            self._get_index_share_topk_holder(packed_seq_params, attention_mask)
            if self.index_share
            else None
        )
        topk_length_holder = (
            self._get_index_share_topk_length_holder(packed_seq_params, attention_mask)
            if self.index_share
            else None
        )
        topk_indices = None
        topk_length = None
        q = k = weights = None
        local_packed_cp_query_start = 0
        local_packed_cp_query_len = sq
        if sequence_parallel_query_is_local:
            local_packed_cp_query_start = sequence_parallel_tp_row_start
            local_packed_cp_query_len = sequence_parallel_tp_full_rows

        if self.skip_topk:
            assert topk_holder is not None
            if self.source_layer not in topk_holder:
                raise RuntimeError(
                    "DSA index-share skip layer "
                    f"(layer_number={self.layer_number}) needs top-k indices from source "
                    f"computing layer {self.source_layer}, but that layer did not run before it "
                    "in this pipeline stage. Cross-PP top-k sharing is not supported. Ensure each "
                    "pipeline stage starts on a computing layer "
                    f"(dsa_indexer_topk_freq={self.index_topk_freq}, "
                    f"dsa_indexer_skip_topk_offset={self.index_skip_topk_offset}). "
                    f"Holder has layers {sorted(topk_holder)}."
                )
            topk_indices = topk_holder[self.source_layer]
            if topk_length_holder is not None:
                topk_length = topk_length_holder.get(self.source_layer)
        else:
            assert self.indexer is not None
            with torch.enable_grad() if use_indexer_loss else torch.no_grad():
                q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
                if cp_size > 1 and k.size(0) in local_cp_kv_lens:
                    if kv_reorder_idx is None:
                        kv_reorder_idx = _build_kv_reorder_idx(k.size(0))
                    k = gather_from_sequence_parallel_region(k, group=cp_group)
                    if k.size(0) != kv_reorder_idx.numel():
                        raise RuntimeError(
                            "DSA gathered indexer-key length mismatch: "
                            f"k_seqlen={k.size(0)}, expected={kv_reorder_idx.numel()}"
                        )
                    k = k.index_select(0, kv_reorder_idx)
                if sequence_parallel_tp and q.size(0) != sq:
                    if (
                        q.size(0) != sequence_parallel_tp_full_rows
                        or weights.size(0) != sequence_parallel_tp_full_rows
                    ):
                        raise RuntimeError(
                            "DSA sequence-parallel indexer row count mismatch: "
                            f"q_rows={q.size(0)}, weights_rows={weights.size(0)}, "
                            f"query_rows={sq}, full_rows={sequence_parallel_tp_full_rows}, "
                            f"tp_size={tp_size}"
                        )
                    if not sequence_parallel_query_is_local:
                        raise RuntimeError(
                            "DSA indexer produced TP-gathered rows while attention query rows "
                            "were not sequence-local."
                        )
                    row_start = sequence_parallel_tp_row_start
                    row_end = row_start + sq
                    q = q[row_start:row_end].contiguous()
                    weights = weights[row_start:row_end].contiguous()

        def compute_indexer_loss_with_reference_path():
            key_for_loss = key.detach()
            if absorbed_mla and key_for_loss.size(2) == 1 and query.size(2) > 1:
                key_for_loss = key_for_loss.expand(-1, -1, query.size(2), -1)
            return FusedDSAIndexerLoss.apply(
                q,
                weights,
                k,
                query.detach(),
                key_for_loss,
                self.softmax_scale,
                self.index_topk,
                indexer_loss_coeff,
                float_mask,
                sparse_indexer_loss,
                self.pg_collection,
                varlen_starts,
                varlen_ends,
                key_positions,
                query_valid_rows,
                self.config.calculate_per_token_loss,
                self.config.dsa_indexer_scoring_relu,
            )

        fused_output = None
        if self.dsa_kernels.run_fused_dsa_attention is not None and not self.index_share:
            assert q is not None and k is not None and weights is not None
            fused_output = self.dsa_kernels.run_fused_dsa_attention(
                config=self.config,
                query=query,
                key=key,
                value=value,
                up_v_weight=up_v_weight,
                q_indexer=q,
                k_indexer=k,
                indexer_weights=weights,
                indexer_topk=self.index_topk,
                softmax_scale=self.softmax_scale,
                loss_coeff=indexer_loss_coeff if use_indexer_loss else 0.0,
                sparse_loss=sparse_indexer_loss,
                calculate_per_token_loss=self.config.calculate_per_token_loss,
                absorbed_mla=absorbed_mla,
                cp_size=cp_size,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
                query_valid_rows=query_valid_rows,
                varlen_is_plain_causal=varlen_is_plain_causal,
                use_relu=self.config.dsa_indexer_scoring_relu,
                use_local_indexer_varlen=use_local_indexer_varlen,
                single_packed_thd_sequence=single_packed_thd_sequence,
                local_packed_cp_rank=cp_rank,
                local_packed_cp_query_start=local_packed_cp_query_start,
                local_packed_cp_query_len=local_packed_cp_query_len,
                pg_collection=self.pg_collection,
            )
            if fused_output is None:
                self.dsa_kernels.log_declined("run_fused_dsa_attention")
        if fused_output is not None:
            output, indexer_loss = fused_output
            if use_indexer_loss:
                if indexer_loss is None:
                    raise RuntimeError("Fused DSA attention did not produce a valid indexer loss.")
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                    reduce_group=indexer_reduce_group,
                    avg_group=indexer_avg_group,
                )
                output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
            return _normalize_dsattention_output_rank(output, x.ndim)

        fused_bounds = None
        if use_fused_kernels and computes_topk:
            assert q is not None
            fused_bounds = dsa_masking.build_fused_indexer_varlen_bounds(
                sq=sq,
                skv=skv,
                device=q.device,
                mask=float_mask,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=key_positions,
            )

        indexer_loss = None

        def slice_topk_to_local_sequence_parallel_rows():
            nonlocal topk_indices, topk_length
            if topk_indices is None or not sequence_parallel_query_is_local:
                return
            topk_sq = topk_indices.size(1)
            if topk_sq == sq:
                return
            expected_topk_sq = sequence_parallel_tp_full_rows
            if topk_sq != expected_topk_sq:
                raise RuntimeError(
                    "DSA sequence-parallel top-k row count mismatch: "
                    f"topk_rows={topk_sq}, query_rows={sq}, tp_size={tp_size}"
                )
            row_start = sequence_parallel_tp_row_start
            row_end = row_start + sq
            topk_indices = topk_indices[:, row_start:row_end].contiguous()
            if topk_length is not None:
                topk_length = topk_length[:, row_start:row_end].contiguous()

        if use_indexer_loss:
            assert q is not None and k is not None and weights is not None
            # ===================================
            # Attach indexer topk and loss
            # ===================================
            if (
                sparse_indexer_loss
                and fused_bounds is not None
                and self.dsa_kernels.run_fused_qk_topk_with_loss is not None
            ):
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                fused_topk_with_loss = self.dsa_kernels.run_fused_qk_topk_with_loss(
                    q,
                    k,
                    weights,
                    self.index_topk,
                    starts_i32,
                    ends_i32,
                    config=self.config,
                    block_size=max(1, block_size),
                    query=query.detach(),
                    key=key.detach(),
                    softmax_scale=self.softmax_scale,
                    loss_coeff=indexer_loss_coeff,
                    pg_collection=self.pg_collection,
                    query_valid_rows=query_valid_rows,
                    calculate_per_token_loss=self.config.calculate_per_token_loss,
                    use_relu=self.config.dsa_indexer_scoring_relu,
                    use_local_indexer_varlen=use_local_indexer_varlen,
                    single_packed_thd_sequence=single_packed_thd_sequence,
                    local_packed_cp_rank=cp_rank,
                    local_packed_cp_query_start=local_packed_cp_query_start,
                    local_packed_cp_query_len=local_packed_cp_query_len,
                    packed_seq_params=packed_seq_params,
                    cp_size=cp_size,
                )
                if fused_topk_with_loss is None:
                    self.dsa_kernels.log_declined("run_fused_qk_topk_with_loss")
                if fused_topk_with_loss is not None:
                    topk_indices, topk_length, indexer_loss = fused_topk_with_loss

            if topk_indices is None or indexer_loss is None:
                topk_indices, indexer_loss = compute_indexer_loss_with_reference_path()
            # No TP-local top-k slicing here: the guard above forbids the indexer loss
            # under sequence-local TP query shards, so the top-k rows are already global.

            # Save indexer loss for logging.
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_layers,
                    reduce_group=indexer_reduce_group,
                    avg_group=indexer_avg_group,
                )
        elif topk_indices is None:
            assert q is not None and k is not None and weights is not None
            # ===================================
            # Get top-k indices
            # ===================================
            if fused_bounds is not None and self.dsa_kernels.run_fused_qk_topk is not None:
                starts_i32, ends_i32 = fused_bounds
                block_size = int(getattr(self, "fused_indexer_block_size", 8192))
                fused_topk = self.dsa_kernels.run_fused_qk_topk(
                    q,
                    k,
                    weights,
                    self.index_topk,
                    starts_i32,
                    ends_i32,
                    block_size=max(1, block_size),
                    use_relu=self.config.dsa_indexer_scoring_relu,
                    use_local_indexer_varlen=use_local_indexer_varlen,
                    single_packed_thd_sequence=single_packed_thd_sequence,
                    local_packed_cp_rank=cp_rank,
                    local_packed_cp_query_start=local_packed_cp_query_start,
                    local_packed_cp_query_len=local_packed_cp_query_len,
                    packed_seq_params=packed_seq_params,
                    cp_size=cp_size,
                )
                if fused_topk is None:
                    self.dsa_kernels.log_declined("run_fused_qk_topk")
                if fused_topk is not None:
                    topk_indices, topk_length = fused_topk

            if topk_indices is None:
                with torch.no_grad():
                    index_scores, topk_indices = fused_qk_topk_naive(
                        q,
                        k,
                        weights,
                        self.index_topk,
                        mask=float_mask,
                        varlen_starts=varlen_starts,
                        varlen_ends=varlen_ends,
                        key_positions=key_positions,
                        use_relu=self.config.dsa_indexer_scoring_relu,
                    )
                    del index_scores
            slice_topk_to_local_sequence_parallel_rows()

        if self.index_share and computes_topk:
            assert topk_holder is not None and topk_indices is not None
            topk_holder[self.layer_number] = topk_indices
            if topk_length_holder is not None and topk_length is not None:
                topk_length_holder[self.layer_number] = topk_length

        # ===================================
        # Run sparse attention kernel
        # ===================================
        output = _run_sparse_attention(
            kernels=self.dsa_kernels,
            absorbed_mla=absorbed_mla,
            query=query,
            key=key,
            value=value,
            up_v_weight=up_v_weight,
            topk_indices=topk_indices,
            topk_length=topk_length,
            softmax_scale=self.softmax_scale,
            config=self.config,
            mask=float_mask,
            varlen_starts=varlen_starts,
            varlen_ends=varlen_ends,
            key_positions=key_positions,
        )

        if use_indexer_loss:
            if indexer_loss is None:
                raise RuntimeError("Indexer loss path did not produce a valid loss tensor.")
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return _normalize_dsattention_output_rank(output, x.ndim)

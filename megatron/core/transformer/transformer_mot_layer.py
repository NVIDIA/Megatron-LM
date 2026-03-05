# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Transformer MoT (Mixture of Transformers) Layer implementation.

This module implements transformer layers that support separate processing
for understanding (und) and generation (gen) tokens, following the MoT architecture.
"""

import logging
from abc import ABC
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_rank,
    get_pg_size,
    divide,
    log_single_rank,
    nvtx_range_pop,
    nvtx_range_push,
)

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

# Import flex_attention for BlockMask support
try:
    from torch.nn.attention.flex_attention import flex_attention
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.functional import scaled_dot_product_attention
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    flex_attention = torch.compile(flex_attention)
    HAVE_FLEX_ATTENTION = True
except ImportError:
    HAVE_FLEX_ATTENTION = False
    flex_attention = None

try:
    from megatron.core.extensions.transformer_engine import (
        TELinear,
        is_te_min_version,
        set_save_original_input,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
    TELinear = None
    is_te_min_version = lambda version: False
    set_save_original_input = lambda module: None
import numpy as np
import os

logger = logging.getLogger(__name__)


# =============================================================================
# SelfAttentionMoT - Attention with separate projections for und/gen tokens
# =============================================================================


@dataclass
class SelfAttentionMoTSubmodules:
    """
    Configuration class for specifying the submodules of a MoT self-attention.

    This includes separate QKV projections for understanding and generation tokens.
    """

    # Standard (understanding) projections
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None

    # MoT: Generation token projections
    linear_qkv_gen: Union[ModuleSpec, type] = None
    linear_proj_gen: Union[ModuleSpec, type] = None
    q_layernorm_gen: Union[ModuleSpec, type] = None
    k_layernorm_gen: Union[ModuleSpec, type] = None


class SelfAttentionMoT(MegatronModule):
    """
    Self-attention layer with MoT (Mixture of Transformers) support.

    This attention layer supports separate QKV projections for understanding (und)
    and generation (gen) tokens, following the MoT architecture.

    Key features:
    - Separate linear_qkv and linear_proj for und/gen tokens
    - Separate q_layernorm and k_layernorm for und/gen tokens
    - Support for freeze_und to detach und token gradients
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionMoTSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        """
        Initialize SelfAttentionMoT.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
            submodules (SelfAttentionMoTSubmodules): Submodule specifications.
            layer_number (int): Layer number in the transformer.
            attn_mask_type (AttnMaskType): Type of attention mask.
            cp_comm_type (str): Context parallel communication type.
            pg_collection (ProcessGroupCollection): Process group collection.
        """
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = "self"

        # MoT specific: freeze understanding token gradients
        self.freeze_und = getattr(config, 'freeze_und', False)

        # Projection sizes
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        # Per attention head and per partition values
        world_size = get_pg_size(self.pg_collection.tp)
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # Key and value hidden sizes
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        # Core attention
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
            pg_collection=self.pg_collection,
        )

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        # QKV projection output dimension
        self.linear_qkv_out_dim = self.query_projection_size + 2 * self.kv_projection_size
        if self.config.attention_output_gate:
            self.linear_qkv_out_dim += self.config.kv_channels * self.config.num_attention_heads

        # =====================================================================
        # Standard (understanding) projections
        # =====================================================================
        print(f"self.config.add_bias_linear={self.config.add_bias_linear} self.config.add_qkv_bias={self.config.add_qkv_bias}")
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.linear_qkv_out_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.pg_collection.tp,
        )

        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
            tp_group=self.pg_collection.tp,
        )

        # Q/K layer norms for understanding tokens
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        # =====================================================================
        # MoT: Generation token projections
        # =====================================================================
        if submodules.linear_qkv_gen is not None:
            self.linear_qkv_gen = build_module(
                submodules.linear_qkv_gen,
                self.config.hidden_size,
                self.linear_qkv_out_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv_gen',
                tp_group=self.pg_collection.tp,
            )
        else:
            self.linear_qkv_gen = None

        if submodules.linear_proj_gen is not None:
            self.linear_proj_gen = build_module(
                submodules.linear_proj_gen,
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='proj_gen',
                tp_group=self.pg_collection.tp,
            )
        else:
            self.linear_proj_gen = None

        # Q/K layer norms for generation tokens
        if submodules.q_layernorm_gen is not None:
            self.q_layernorm_gen = build_module(
                submodules.q_layernorm_gen,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm_gen = None

        if submodules.k_layernorm_gen is not None:
            self.k_layernorm_gen = build_module(
                submodules.k_layernorm_gen,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm_gen = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with MoT (separate processing for und/gen tokens).

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h].
            attention_mask (Tensor, optional): Attention mask.
            packed_und_token_indexes (Tensor, optional): Indexes of understanding tokens.
            packed_gen_token_indexes (Tensor, optional): Indexes of generation tokens.
            inference_context: Inference context for KV cache.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Rotary embedding cosine.
            rotary_pos_sin (Tensor, optional): Rotary embedding sine.
            rotary_pos_cos_sin (Tensor, optional): Combined rotary embedding.
            attention_bias (Tensor, optional): Attention bias.
            packed_seq_params (PackedSeqParams, optional): Packed sequence parameters.
            sequence_len_offset (Tensor, optional): Sequence length offset.
            mode (str): Processing mode - "und" or "gen".

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Output tensor and optional bias.
        """
        if self.training:
            return self._forward_train(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            return self._forward_inference(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                mode=mode,
            )

    def _forward_train(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Training forward pass with MoT.

        Processes understanding and generation tokens through separate projections.
        """
        seq_len, batch_size, hidden_size = hidden_states.shape

        # =====================================================================
        # QKV projection with MoT (separate for und/gen)
        # =====================================================================
        # Initialize output tensors
        qkv_output = hidden_states.new_zeros(seq_len, batch_size, self.linear_qkv_out_dim)

        # Reshape for token-level indexing: [s, b, h] -> [s*b, h]
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        qkv_output_flat = qkv_output.view(-1, self.linear_qkv_out_dim)

        # Process understanding tokens
        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            und_hidden = hidden_states_flat[packed_und_token_indexes]
            und_qkv, _ = self.linear_qkv(und_hidden)
            qkv_output_flat[packed_und_token_indexes] = und_qkv

            if self.freeze_und:
                qkv_output_flat[packed_und_token_indexes] = (
                    qkv_output_flat[packed_und_token_indexes].detach()
                )

        # Process generation tokens
        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            gen_hidden = hidden_states_flat[packed_gen_token_indexes]
            if self.linear_qkv_gen is not None:
                gen_qkv, _ = self.linear_qkv_gen(gen_hidden)
            else:
                raise ValueError("linear_qkv_gen is None")
            qkv_output_flat[packed_gen_token_indexes] = gen_qkv

        # Reshape back
        qkv_output = qkv_output_flat.view(seq_len, batch_size, self.linear_qkv_out_dim)

        # =====================================================================
        # Split QKV and apply layernorms
        # =====================================================================
        query, key, value = self._split_qkv(qkv_output)
        # Apply Q/K layernorms with MoT
        if self.q_layernorm is not None or self.k_layernorm is not None:
            query, key = self._apply_qk_layernorm_mot(
                query, key, packed_und_token_indexes, packed_gen_token_indexes
            )

        # =====================================================================
        # Apply rotary positional embeddings to query and key
        # =====================================================================
        if rotary_pos_emb is not None:
            # rotary_pos_emb can be a tuple (q_pos_emb, k_pos_emb) or a single tensor
            if isinstance(rotary_pos_emb, tuple):
                q_pos_emb, k_pos_emb = rotary_pos_emb
            else:
                q_pos_emb = rotary_pos_emb
                k_pos_emb = rotary_pos_emb

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                )

        # # =====================================================================
        # # Core attention (with flex_attention support)
        # # =====================================================================
        # # Check if attention_mask is a list (nested attention masks) or BlockMask
        # if HAVE_FLEX_ATTENTION and attention_mask is not None and not isinstance(attention_mask, list):
        #     # Use flex_attention for BlockMask
        #     # query/key/value shape: [seq_len, batch_size, num_heads, head_dim]
        #     # flex_attention expects: [batch, num_heads, seq_len, head_dim]
        #     seq_len, batch_size, num_heads, head_dim = query.shape
        #     num_kv_heads = key.shape[2]

        #     # Get the padded sequence length from BlockMask
        #     # BlockMask shape: (batch, num_heads, padded_seq_len, padded_seq_len)
        #     padded_seq_len = attention_mask.shape[2]
        #     pad_size = padded_seq_len - seq_len
        #     # Transpose to [num_heads, seq_len, head_dim] (squeeze batch dim for padding)
        #     # Following qwen2_navit.py pattern: permute then pad
        #     query_flex = query.squeeze(1).permute(1, 0, 2)  # [num_heads, seq_len, head_dim]
        #     key_flex = key.squeeze(1).permute(1, 0, 2)      # [num_kv_heads, seq_len, head_dim]
        #     value_flex = value.squeeze(1).permute(1, 0, 2)  # [num_kv_heads, seq_len, head_dim]

        #     # Pad sequences to match BlockMask size (pad in seq_len dimension)
        #     if pad_size > 0:
        #         # Pad: [num_heads, seq_len, head_dim] -> [num_heads, padded_seq_len, head_dim]
        #         pad_tensor_q = query_flex.new_zeros((num_heads, pad_size, head_dim))
        #         query_flex = torch.cat([query_flex, pad_tensor_q], dim=1)

        #         pad_tensor_k = key_flex.new_zeros((num_kv_heads, pad_size, head_dim))
        #         key_flex = torch.cat([key_flex, pad_tensor_k], dim=1)

        #         pad_tensor_v = value_flex.new_zeros((num_kv_heads, pad_size, head_dim))
        #         value_flex = torch.cat([value_flex, pad_tensor_v], dim=1)

        #     # Call flex_attention with BlockMask
        #     # flex_attention expects: [batch, num_heads, seq_len, head_dim]
        #     core_attn_out = flex_attention(
        #         query_flex.unsqueeze(0),  # [1, num_heads, padded_seq_len, head_dim]
        #         key_flex.unsqueeze(0),    # [1, num_kv_heads, padded_seq_len, head_dim]
        #         value_flex.unsqueeze(0),  # [1, num_kv_heads, padded_seq_len, head_dim]
        #         enable_gqa=True,
        #         block_mask=attention_mask,
        #     )

        #     # Remove padding from output: [1, num_heads, padded_seq_len, head_dim] -> [num_heads, seq_len, head_dim]
        #     end_index = core_attn_out.shape[2] - pad_size
        #     core_attn_out = core_attn_out[0, :, :end_index, :]  # [num_heads, seq_len, head_dim]
        #     # Transpose back to [seq_len, batch_size, num_heads, head_dim]
        #     core_attn_out = core_attn_out.permute(1, 0, 2).unsqueeze(1)  # [seq_len, 1, num_heads, head_dim]
        #     # Reshape to [seq_len, batch_size, hidden_size]
        #     core_attn_out = core_attn_out.reshape(seq_len, batch_size, -1)
        # elif isinstance(attention_mask, list):
        #     # Use per-sample scaled_dot_product_attention for list of nested masks
        #     # query/key/value shape: [seq_len, batch_size, num_heads, head_dim]
        #     seq_len, batch_size, num_heads, head_dim = query.shape
        #     num_kv_heads = key.shape[2]

        #     # For per-sample attention, we need to handle each sample separately
        #     # First, transpose to [batch, num_heads, seq_len, head_dim]
        #     query_sdpa = query.permute(1, 2, 0, 3)  # [batch, num_heads, seq_len, head_dim]
        #     key_sdpa = key.permute(1, 2, 0, 3)      # [batch, num_kv_heads, seq_len, head_dim]
        #     value_sdpa = value.permute(1, 2, 0, 3)  # [batch, num_kv_heads, seq_len, head_dim]

        #     # Expand key/value to match query heads for non-GQA SDPA
        #     if num_kv_heads != num_heads:
        #         num_groups = num_heads // num_kv_heads
        #         key_sdpa = key_sdpa.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        #         key_sdpa = key_sdpa.reshape(batch_size, num_heads, seq_len, head_dim)
        #         value_sdpa = value_sdpa.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        #         value_sdpa = value_sdpa.reshape(batch_size, num_heads, seq_len, head_dim)

        #     # Apply per-sample attention with nested masks
        #     # attention_mask is a list of masks, one per sample in packed sequence
        #     # For batch_size=1 packed sequence, we squeeze batch dim
        #     query_sdpa = query_sdpa.squeeze(0)  # [num_heads, seq_len, head_dim]
        #     key_sdpa = key_sdpa.squeeze(0)
        #     value_sdpa = value_sdpa.squeeze(0)

        #     # Since packed sequence has batch_size=1, attention_mask list corresponds to
        #     # different samples packed together. We process the full sequence with each mask.
        #     # For simplicity, use the first mask (assuming single sample or concatenate later)
        #     with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        #         attn_output = scaled_dot_product_attention(
        #             query_sdpa.to(torch.bfloat16).unsqueeze(0),
        #             key_sdpa.to(torch.bfloat16).unsqueeze(0),
        #             value_sdpa.to(torch.bfloat16).unsqueeze(0),
        #             attn_mask=attention_mask[0].to(torch.bfloat16).unsqueeze(0) if len(attention_mask) == 1 else None,
        #         )

        #     # Transpose back to [seq_len, batch_size, num_heads, head_dim]
        #     core_attn_out = attn_output.squeeze(0).permute(1, 0, 2).contiguous()
        #     core_attn_out = core_attn_out.view(seq_len, batch_size, -1)
        # else:
        # Fallback to original core_attention
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query, key, value, attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        # =====================================================================
        # Output projection with MoT
        # =====================================================================
        output = self._apply_output_projection_mot(
            core_attn_out, packed_und_token_indexes, packed_gen_token_indexes
        )

        return output

    def _forward_inference(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Inference forward pass with MoT.

        Uses mode to determine which projections to use.
        """
        # Choose projections based on mode
        if mode == "gen" and self.linear_qkv_gen is not None:
            linear_qkv = self.linear_qkv_gen
            linear_proj = self.linear_proj_gen if self.linear_proj_gen is not None else self.linear_proj
            q_layernorm = self.q_layernorm_gen if self.q_layernorm_gen is not None else self.q_layernorm
            k_layernorm = self.k_layernorm_gen if self.k_layernorm_gen is not None else self.k_layernorm
        else:
            linear_qkv = self.linear_qkv
            linear_proj = self.linear_proj
            q_layernorm = self.q_layernorm
            k_layernorm = self.k_layernorm

        # QKV projection
        qkv_output, _ = linear_qkv(hidden_states)

        # Split QKV
        query, key, value = self._split_qkv(qkv_output)

        # Apply Q/K layernorms
        if q_layernorm is not None:
            query = q_layernorm(query)
        if k_layernorm is not None:
            key = k_layernorm(key)

        # Apply rotary positional embeddings to query and key
        if rotary_pos_emb is not None:
            # rotary_pos_emb can be a tuple (q_pos_emb, k_pos_emb) or a single tensor
            if isinstance(rotary_pos_emb, tuple):
                q_pos_emb, k_pos_emb = rotary_pos_emb
            else:
                q_pos_emb = rotary_pos_emb
                k_pos_emb = rotary_pos_emb

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                )

        # Core attention
        core_attn_out = self.core_attention(
            query, key, value, attention_mask,
            attn_mask_type=self.attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        # Output projection
        output, bias = linear_proj(core_attn_out)

        return output, bias

    def _split_qkv(self, qkv: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Split the packed QKV tensor into query, key, and value.

        Uses the same interleaved-by-query-groups layout as attention.py:
        For each query group: [query_heads, key_head, value_head]
        """
        # qkv shape: [s, b, ng * (np/ng + 2) * hn]
        seq_len, batch_size, _ = qkv.shape

        # Calculate dimensions
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        num_qkv_heads_per_group = num_query_heads_per_group + 2

        # Reshape to [sq, b, ng, (np/ng + 2) * hn] - organized by query groups
        new_tensor_shape = qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            num_qkv_heads_per_group * self.hidden_size_per_attention_head,
        )
        qkv = qkv.view(*new_tensor_shape)

        # Split along the last dimension (dim=3)
        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        split_arg_list = [
            num_query_heads_per_group * self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        (query, key, value) = torch.split(qkv, split_arg_list, dim=3)

        # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(seq_len, batch_size, -1, self.hidden_size_per_attention_head)
        return query, key, value

    def _apply_qk_layernorm_mot(
        self,
        query: Tensor,
        key: Tensor,
        packed_und_token_indexes: Optional[Tensor],
        packed_gen_token_indexes: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Apply Q/K layernorms with MoT (separate for und/gen tokens)."""
        seq_len, batch_size, num_heads, head_dim = query.shape

        # Flatten for token-level indexing
        query_flat = query.view(-1, num_heads, head_dim)
        key_flat = key.view(-1, key.shape[2], head_dim)

        query_out = query_flat.clone()
        key_out = key_flat.clone()

        # Apply layernorms to understanding tokens
        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            if self.q_layernorm is not None:
                query_out[packed_und_token_indexes] = self.q_layernorm(
                    query_flat[packed_und_token_indexes]
                )
            if self.k_layernorm is not None:
                key_out[packed_und_token_indexes] = self.k_layernorm(
                    key_flat[packed_und_token_indexes]
                )

            if self.freeze_und:
                query_out[packed_und_token_indexes] = query_out[packed_und_token_indexes].detach()
                key_out[packed_und_token_indexes] = key_out[packed_und_token_indexes].detach()

        # Apply layernorms to generation tokens
        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            q_norm = self.q_layernorm_gen if self.q_layernorm_gen is not None else self.q_layernorm
            k_norm = self.k_layernorm_gen if self.k_layernorm_gen is not None else self.k_layernorm

            if q_norm is not None:
                query_out[packed_gen_token_indexes] = q_norm(query_flat[packed_gen_token_indexes])
            if k_norm is not None:
                key_out[packed_gen_token_indexes] = k_norm(key_flat[packed_gen_token_indexes])

        # Reshape back
        query = query_out.view(seq_len, batch_size, num_heads, head_dim)
        key = key_out.view(seq_len, batch_size, key.shape[2], head_dim)

        return query, key

    def _apply_output_projection_mot(
        self,
        attn_output: Tensor,
        packed_und_token_indexes: Optional[Tensor],
        packed_gen_token_indexes: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply output projection with MoT (separate for und/gen tokens)."""
        seq_len, batch_size, hidden_size = attn_output.shape

        # Flatten for token-level indexing
        attn_output_flat = attn_output.view(-1, hidden_size)
        output_flat = attn_output_flat.new_zeros(attn_output_flat.shape[0], self.config.hidden_size)

        # Process understanding tokens
        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            und_output, _ = self.linear_proj(attn_output_flat[packed_und_token_indexes])
            output_flat[packed_und_token_indexes] = und_output

            if self.freeze_und:
                output_flat[packed_und_token_indexes] = output_flat[packed_und_token_indexes].detach()

        # Process generation tokens
        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            linear_proj = self.linear_proj_gen if self.linear_proj_gen is not None else self.linear_proj
            gen_output, _ = linear_proj(attn_output_flat[packed_gen_token_indexes])
            output_flat[packed_gen_token_indexes] = gen_output

        # Reshape back
        output = output_flat.view(seq_len, batch_size, self.config.hidden_size)

        return output, None

    def _checkpointed_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type=None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Checkpoint the attention computation to save memory."""

        def custom_forward(*inputs):
            q, k, v, mask = inputs[:4]
            return self.core_attention(
                q, k, v, mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        return tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, attention_mask
        )


# =============================================================================
# MoTTransformerLayerSubmodules - Configuration for MoT transformer layer
# =============================================================================


@dataclass
class MoTTransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a MoT transformer layer.

    This includes separate layernorms and MLPs for understanding and generation tokens.
    """

    # Input layernorm (understanding)
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    # Input layernorm for generation tokens (MoT)
    input_layernorm_gen: Union[ModuleSpec, type] = IdentityOp

    # Self attention with MoT support
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Pre-MLP layernorm (understanding)
    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    # Pre-MLP layernorm for generation tokens (MoT)
    pre_mlp_layernorm_gen: Union[ModuleSpec, type] = IdentityOp

    # MLP (understanding)
    mlp: Union[ModuleSpec, type] = IdentityOp
    # MLP for generation tokens (MoT)
    mlp_gen: Union[ModuleSpec, type] = IdentityOp

    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Cross attention (optional, for encoder-decoder)
    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# MoTTransformerLayer - Transformer layer with MoT support
# =============================================================================


class MoTTransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
    """
    A single transformer layer with MoT (Mixture of Transformers) support.

    This layer supports separate processing paths for understanding (und)
    and generation (gen) tokens, with:
    - Separate input layernorms for und/gen
    - Separate pre-MLP layernorms for und/gen
    - Separate MLPs for und/gen
    - Support for freezing und token gradients

    The layer takes input with size [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MoTTransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        """
        Initialize MoTTransformerLayer.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
            submodules (MoTTransformerLayerSubmodules): Submodule specifications.
            layer_number (int): Layer number (1-indexed).
            hidden_dropout (float, optional): Dropout rate for hidden states.
            pg_collection (ProcessGroupCollection, optional): Process group collection.
            vp_stage (int, optional): Virtual pipeline stage.
        """
        super().__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp)
        )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # MoT specific: freeze understanding token gradients
        self.freeze_und = getattr(config, 'freeze_und', False)

        # =====================================================================
        # Build modules
        # =====================================================================

        # [Module 1: Input Layernorm] - Separate for und/gen
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.input_layernorm_gen = build_module(
            submodules.input_layernorm_gen,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Attention optional kwargs
        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
        attention_optional_kwargs["pg_collection"] = pg_collection

        # [Module 2: SelfAttention] with MoT support
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Pre-MLP Layernorm] - Separate for und/gen
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.pre_mlp_layernorm_gen = build_module(
            submodules.pre_mlp_layernorm_gen,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: MLP] - Separate for und/gen
        additional_mlp_kwargs = self._get_mlp_kwargs(submodules.mlp, pg_collection)
        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)

        additional_mlp_gen_kwargs = self._get_mlp_kwargs(submodules.mlp_gen, pg_collection)
        self.mlp_gen = build_module(submodules.mlp_gen, config=self.config, **additional_mlp_gen_kwargs)

        # [Module 6: MLP BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        # [Module 7-9: Cross attention] (optional)
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # Check if this is a MoE layer
        try:
            from megatron.core.transformer.moe.moe_layer import MoELayer
            self.is_moe_layer = isinstance(self.mlp, MoELayer)
        except ImportError:
            self.is_moe_layer = False

    def _get_mlp_kwargs(
        self, mlp_spec: Union[ModuleSpec, type], pg_collection: ProcessGroupCollection
    ) -> dict:
        """Get additional kwargs for MLP initialization."""
        additional_mlp_kwargs = {}

        if not isinstance(mlp_spec, ModuleSpec):
            return additional_mlp_kwargs

        try:
            from megatron.core.extensions.transformer_engine import TEFusedMLP
        except ImportError:
            TEFusedMLP = None

        try:
            from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
            from megatron.core.transformer.moe.moe_layer import MoELayer
            moe_classes = (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP)
        except ImportError:
            moe_classes = ()

        if mlp_spec.module in moe_classes:
            additional_mlp_kwargs["pg_collection"] = pg_collection
        elif mlp_spec.module == MLP:
            additional_mlp_kwargs["tp_group"] = pg_collection.tp
        elif TEFusedMLP is not None and mlp_spec.module == TEFusedMLP:
            additional_mlp_kwargs["tp_group"] = pg_collection.tp

        return additional_mlp_kwargs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the MoT transformer layer.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h].
            attention_mask (Tensor, optional): Attention mask.
            packed_und_token_indexes (Tensor, optional): Indexes of understanding tokens.
            packed_gen_token_indexes (Tensor, optional): Indexes of generation tokens.
            context (Tensor, optional): Context for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Rotary embedding cosine.
            rotary_pos_sin (Tensor, optional): Rotary embedding sine.
            rotary_pos_cos_sin (Tensor, optional): Combined rotary embedding.
            attention_bias (Tensor, optional): Attention bias.
            inference_context: Inference context for KV cache.
            packed_seq_params (PackedSeqParams, optional): Packed sequence parameters.
            sequence_len_offset (Tensor, optional): Sequence length offset.
            mode (str): Processing mode - "und" or "gen".

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Output hidden states and context.
        """
        if self.training:
            return self._forward_train(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            return self._forward_inference(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                mode=mode,
            )

    def _forward_train(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Training forward pass with MoT.

        Processes understanding and generation tokens through separate layernorms and MLPs.
        """
        # =====================================================================
        # Input layernorm with MoT
        # =====================================================================
        residual = hidden_states

        hidden_states = self._apply_input_layernorm_mot(
            hidden_states, packed_und_token_indexes, packed_gen_token_indexes
        )

        # =====================================================================
        # Self attention
        # =====================================================================
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        nvtx_range_pop(suffix="self_attention")
        # Freeze und attention output if configured
        if self.freeze_und and packed_und_token_indexes is not None:
            if isinstance(attention_output_with_bias, tuple):
                attn_out, bias = attention_output_with_bias
                attn_out_flat = attn_out.view(-1, attn_out.shape[-1])
                attn_out_flat[packed_und_token_indexes] = attn_out_flat[packed_und_token_indexes].detach()
                attention_output_with_bias = (attn_out_flat.view_as(attn_out), bias)
            else:
                attn_out_flat = attention_output_with_bias.view(-1, attention_output_with_bias.shape[-1])
                attn_out_flat[packed_und_token_indexes] = attn_out_flat[packed_und_token_indexes].detach()
                attention_output_with_bias = attn_out_flat.view_as(attention_output_with_bias)

        # Bias-dropout-add
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        # =====================================================================
        # Cross attention (if applicable)
        # =====================================================================
        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=None,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # =====================================================================
        # MLP with MoT
        # =====================================================================
        output = self._forward_mlp_mot(
            hidden_states, packed_und_token_indexes, packed_gen_token_indexes
        )

        return output, context

    def _forward_inference(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Inference forward pass with MoT.

        Uses mode to determine which layernorms and MLPs to use.
        """
        # Choose modules based on mode
        if mode == "gen":
            input_layernorm = self.input_layernorm_gen if not isinstance(self.input_layernorm_gen, IdentityOp) else self.input_layernorm
            pre_mlp_layernorm = self.pre_mlp_layernorm_gen if not isinstance(self.pre_mlp_layernorm_gen, IdentityOp) else self.pre_mlp_layernorm
            mlp = self.mlp_gen if not isinstance(self.mlp_gen, IdentityOp) else self.mlp
        else:
            input_layernorm = self.input_layernorm
            pre_mlp_layernorm = self.pre_mlp_layernorm
            mlp = self.mlp

        # =====================================================================
        # Input layernorm
        # =====================================================================
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)

        # =====================================================================
        # Self attention
        # =====================================================================
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            mode=mode,
        )

        # Bias-dropout-add
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # =====================================================================
        # MLP
        # =====================================================================
        residual = hidden_states
        hidden_states = pre_mlp_layernorm(hidden_states)

        mlp_output_with_bias = mlp(hidden_states)

        with self.bias_dropout_add_exec_handler():
            output = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        return output, context

    def _apply_input_layernorm_mot(
        self,
        hidden_states: Tensor,
        packed_und_token_indexes: Optional[Tensor],
        packed_gen_token_indexes: Optional[Tensor],
    ) -> Tensor:
        """Apply input layernorm with MoT (separate for und/gen tokens)."""
        seq_len, batch_size, hidden_size = hidden_states.shape

        # Flatten for token-level indexing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        output_flat = hidden_states_flat.new_zeros(hidden_states_flat.shape)

        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            output_flat[packed_und_token_indexes] = self.input_layernorm(
                hidden_states_flat[packed_und_token_indexes]
            )
        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            layernorm = self.input_layernorm_gen if not isinstance(self.input_layernorm_gen, IdentityOp) else self.input_layernorm
            output_flat[packed_gen_token_indexes] = layernorm(
                hidden_states_flat[packed_gen_token_indexes]
            )
        return output_flat.view(seq_len, batch_size, hidden_size)

    def _forward_mlp_mot(
        self,
        hidden_states: Tensor,
        packed_und_token_indexes: Optional[Tensor],
        packed_gen_token_indexes: Optional[Tensor],
    ) -> Tensor:
        """Forward pass through MLP with MoT (separate for und/gen tokens)."""
        seq_len, batch_size, hidden_size = hidden_states.shape

        # Residual connection
        residual = hidden_states

        # Flatten for token-level indexing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        output_flat = hidden_states_flat.new_zeros(hidden_states_flat.shape)

        # Process understanding tokens
        if packed_und_token_indexes is not None and len(packed_und_token_indexes) > 0:
            und_hidden = hidden_states_flat[packed_und_token_indexes]
            und_layernorm_output = self.pre_mlp_layernorm(und_hidden)

            nvtx_range_push(suffix="mlp_und")
            und_mlp_output = self.mlp(und_layernorm_output)
            nvtx_range_pop(suffix="mlp_und")

            # Handle tuple output (output, bias)
            if isinstance(und_mlp_output, tuple):
                und_mlp_output, _ = und_mlp_output
            output_flat[packed_und_token_indexes] = und_mlp_output

            if self.freeze_und:
                output_flat[packed_und_token_indexes] = output_flat[packed_und_token_indexes].detach()

        # Process generation tokens
        if packed_gen_token_indexes is not None and len(packed_gen_token_indexes) > 0:
            gen_hidden = hidden_states_flat[packed_gen_token_indexes]

            pre_mlp_layernorm = self.pre_mlp_layernorm_gen if not isinstance(self.pre_mlp_layernorm_gen, IdentityOp) else self.pre_mlp_layernorm
            gen_layernorm_output = pre_mlp_layernorm(gen_hidden)

            nvtx_range_push(suffix="mlp_gen")
            mlp = self.mlp_gen if not isinstance(self.mlp_gen, IdentityOp) else self.mlp
            gen_mlp_output = mlp(gen_layernorm_output)
            nvtx_range_pop(suffix="mlp_gen")

            # Handle tuple output (output, bias)
            if isinstance(gen_mlp_output, tuple):
                gen_mlp_output, _ = gen_mlp_output

            output_flat[packed_gen_token_indexes] = gen_mlp_output

        # Reshape and add residual
        output = output_flat.view(seq_len, batch_size, hidden_size)
        output = output + residual

        return output

    def bias_dropout_add_exec_handler(self):
        """Context manager for bias-dropout-add execution."""
        return nullcontext()

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Generate a sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}

        # Get state dict from all submodules
        for name, module in self.named_children():
            if hasattr(module, 'sharded_state_dict'):
                module_sharded_state_dict = module.sharded_state_dict(
                    prefix=f'{prefix}{name}.',
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
                sharded_state_dict.update(module_sharded_state_dict)

        # Apply key mapping if specified
        if self.submodules_config.sharded_state_dict_keys_map:
            sharded_state_dict = apply_prefix_mapping(
                sharded_state_dict, self.submodules_config.sharded_state_dict_keys_map
            )

        return sharded_state_dict


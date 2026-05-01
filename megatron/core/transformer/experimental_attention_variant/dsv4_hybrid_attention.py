# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek-V4 hybrid attention layer.

A self-attention module that wraps the ``CompressedSparseAttention`` core attention
together with MLA-style query/key projections, partial RoPE, and a grouped output
projection.

The same module is used for CSA and HCA layers; the per-layer ``compress_ratio``
(controlled by the layer-allocation pattern + config) selects between behaviours.

This is a CP=1 / TP=1 prototype using the unfused RoPE path.  Inference, MTP, fp8/fp4,
and fine-grained activation offloading are not supported.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig


@torch.compile
def _q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMS normalization for query (no learnable weight)."""
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)


@dataclass
class DSv4HybridSelfAttentionSubmodules:
    """Submodules for the ``DSv4HybridSelfAttention`` layer."""

    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class DSv4HybridSelfAttention(MegatronModule):
    """DeepSeek-V4 hybrid (CSA / HCA) self-attention layer."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSv4HybridSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = "self",
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        compress_ratio: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(config=config)
        self.config: MLATransformerConfig
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection
        assert self.pg_collection.tp.size() == 1, "DSv4 hybrid attention requires TP size 1."
        assert self.pg_collection.cp.size() == 1, "DSv4 hybrid attention requires CP size 1."

        self.num_attention_heads_per_partition = self.config.num_attention_heads
        self.q_head_dim = self.config.v_head_dim
        self.query_projection_size = self.q_head_dim * self.config.num_attention_heads

        # Resolve the per-layer compression ratio.
        if compress_ratio is None:
            compress_ratios = self.config.csa_compress_ratios
            assert compress_ratios is not None and len(compress_ratios) >= layer_number, (
                f"csa_compress_ratios must be set and have length >= num_layers; "
                f"got {compress_ratios} for layer {layer_number}"
            )
            compress_ratio = compress_ratios[layer_number - 1]
        self.compress_ratio = compress_ratio

        # ===========================================================================
        # RoPE
        # ===========================================================================
        rope_base = self.config.rotary_base
        if compress_ratio > 1:
            rope_base = self.config.csa_compress_rotary_base
        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=rope_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=rope_base,
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

        # ===========================================================================
        # QKV projections
        # ===========================================================================
        self.linear_q_down_proj = build_module(
            submodules.linear_q_down_proj,
            self.config.hidden_size,
            self.config.q_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q_down_proj',
            skip_weight_param_allocation=False,
            tp_group=None,
            parallel_mode='duplicated',
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.config.q_lora_rank,
            self.config.num_attention_heads * self.q_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q_up_proj',
            tp_group=self.pg_collection.tp,
        )

        self.linear_kv_proj = build_module(
            submodules.linear_kv_proj,
            self.config.hidden_size,
            self.config.v_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_proj',
            tp_group=self.pg_collection.tp,
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.v_head_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        # ===========================================================================
        # Core attention (CSA / HCA)
        # ===========================================================================
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=None,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
            rotary_pos_emb=self.rotary_pos_emb,
            compress_ratio=compress_ratio,
        )

        # ===========================================================================
        # Grouped output projection
        # ===========================================================================
        self.o_local_groups = self.config.o_groups
        assert (
            self.query_projection_size % self.config.o_groups == 0
        ), "num_attention_heads * v_head_dim must be divisible by o_groups"
        group_proj_in_size = self.query_projection_size // self.config.o_groups
        group_proj_out_size = self.config.o_groups * self.config.o_lora_rank

        _linear_o_group_proj = torch.empty(
            group_proj_out_size,
            group_proj_in_size,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        self.config.init_method(_linear_o_group_proj)
        self.linear_o_group_proj = nn.Parameter(_linear_o_group_proj)

        linear_proj_in_size = self.config.o_groups * self.config.o_lora_rank
        self.linear_proj = build_module(
            submodules.linear_proj,
            linear_proj_in_size,
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

    # ===========================================================================
    # Helpers
    # ===========================================================================

    def _build_rotary_pos_emb(self, seqlen: int, dtype: torch.dtype):
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(seqlen, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(seqlen, packed_seq=False)
        return rotary_pos_emb, mscale

    def _get_qkv(self, hidden_states: torch.Tensor):
        """Compute query, key, value, and the compressed q representation."""
        sq, b, _ = hidden_states.size()
        rotary_pos_emb, mscale = self._build_rotary_pos_emb(sq, hidden_states.dtype)

        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = self.q_layernorm(q_compressed)

        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(sq, b, self.num_attention_heads_per_partition, self.q_head_dim)
        q = _q_rms_norm(q, self.config.layernorm_epsilon)

        kv, _ = self.linear_kv_proj(hidden_states)
        kv = self.kv_layernorm(kv)

        # Partial RoPE on the last qk_pos_emb_head_dim dims of q and kv.
        pos_dim = self.config.qk_pos_emb_head_dim
        q_no_pe, q_pe = torch.split(q, [self.q_head_dim - pos_dim, pos_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(
            q_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        query = torch.cat([q_no_pe, q_pe], dim=-1)

        kv_no_pe, k_pe = torch.split(kv, [self.config.v_head_dim - pos_dim, pos_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(
            k_pe.unsqueeze(-2),
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        ).squeeze(-2)
        kv = torch.cat([kv_no_pe, k_pe], dim=-1).unsqueeze(-2)

        return query.contiguous(), kv.contiguous(), kv.contiguous(), q_compressed

    def _inverse_partial_rope_on_output(self, core_attn_out: torch.Tensor) -> torch.Tensor:
        """Apply RoPE with negated frequencies to the last ``qk_pos_emb_head_dim`` dims.

        Required because the same compressed entries are used as both keys and values,
        which causes an absolute-position bias on the output. Inverse RoPE cancels it.
        """
        seq_len, b, _ = core_attn_out.size()
        n_heads = self.num_attention_heads_per_partition
        pos_dim = self.config.qk_pos_emb_head_dim
        nope_dim = self.config.v_head_dim - pos_dim

        core_attn_out = core_attn_out.view(seq_len, b, n_heads, -1)

        rotary_pos_emb, mscale = self._build_rotary_pos_emb(seq_len, core_attn_out.dtype)
        # Inverse rotation: negate the frequencies.
        rotary_pos_emb = -rotary_pos_emb

        content_part, rot_part = torch.split(core_attn_out, [nope_dim, pos_dim], dim=-1)
        rot_part = apply_rotary_pos_emb(
            rot_part,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
        )
        core_attn_out = torch.cat([content_part, rot_part], dim=-1)
        return core_attn_out.view(seq_len, b, -1)

    # ===========================================================================
    # Forward
    # ===========================================================================

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass.  Returns ``(output, bias)`` matching standard self-attention."""
        assert rotary_pos_emb is None, "DSv4HybridSelfAttention computes RoPE internally."
        assert attention_bias is None, "Attention bias is not supported."
        assert packed_seq_params is None, "Packed sequence is not supported."
        assert (
            inference_context is None and inference_params is None
        ), "Inference is not supported for DSv4HybridSelfAttention."

        query, key, value, q_compressed = self._get_qkv(hidden_states)

        core_attn_out = self.core_attention(
            query, key, value, attention_mask, x=hidden_states, qr=q_compressed
        )

        core_attn_out = self._inverse_partial_rope_on_output(core_attn_out)

        # Grouped output projection: split heads into groups, project per-group, concat.
        sq, b, _ = core_attn_out.size()
        core_attn_out = core_attn_out.view(sq, b, self.o_local_groups, -1)
        wo_a_weight = self.linear_o_group_proj.view(
            self.o_local_groups, self.config.o_lora_rank, -1
        )
        # [...g d] @ [g r d]^T -> [...g r]
        core_attn_out = torch.einsum("...gd,grd->...gr", core_attn_out, wo_a_weight)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        output, bias = self.linear_proj(core_attn_out)
        return output, bias

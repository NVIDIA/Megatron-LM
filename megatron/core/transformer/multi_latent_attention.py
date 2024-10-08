# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


import math
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass
class MLASelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: Union[MLASelfAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
    ) -> None:
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        assert (
            world_size == 1
        ), "MLA is not supported with Tensor Parallelism yet, \
        use Expert Parallelism and Pipeline Parallelism for better performance."

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
        )

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        self.rotary_pos_emb = YarnRotaryEmbedding(
            self.config.qk_pos_emb_head_dim,
            rotary_base=self.config.rotary_base,
            scaling_factor=self.config.rotary_scaling_factor,
            original_max_position_embeddings=self.config.max_position_embeddings,
            beta_fast=self.config.beta_fast,
            beta_slow=self.config.beta_slow,
            mscale=self.config.mscale,
            mscale_all_dim=self.config.mscale_all_dim,
        )

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
        )

        # Output.
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
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        position_ids=None,
    ):
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_params=inference_params,
        )

        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        # rotary_pos_emb = None
        key, value, _, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb=None
        )

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=attn_mask_type,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )

        if self.config.q_lora_rank is None:
            # Not projectiing query
            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        else:

            self.linear_q_down_proj = build_module(
                submodules.linear_q_down_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
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
            )

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        if self.config.q_lora_rank is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        q_len, bsz, _ = hidden_states.size()

        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)
            q_compressed = self.q_layernorm(q_compressed)
            q, _ = self.linear_q_up_proj(q_compressed)
        else:
            # hidden_states:[s, b, 2048], q: [s, b, n * 192]
            q, _ = self.linear_q_proj(hidden_states)

        # q: [s, b, n, 192]
        q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

        # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
        q_no_pe, q_pos_emb = torch.split(
            q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )

        # kv_combined: [s, b, 576]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)

        # kv_compressed:[s, b, 512], k_pos_emb: [s, b, 64]
        kv_compressed, k_pos_emb = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )

        # kv: [s, b, 2048]
        kv, _ = self.linear_kv_up_proj(self.kv_layernorm(kv_compressed))

        # kv: [s, b, n, 256]
        kv = kv.view(
            q_len,
            bsz,
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )

        # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
        k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

        # rotary_pos_emb:[s, b, 1, 64]
        rotary_pos_emb = self.rotary_pos_emb(max_seq_len=self.config.max_position_embeddings)

        if len(rotary_pos_emb) == 2:
            mscale = rotary_pos_emb[1]
            rotary_pos_emb = rotary_pos_emb[0]

        if inference_params is not None:
            # add offset to the sequence start for inference
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + q_len
            rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]

        # [s, b, 64] -> [s, b, 1, 64]
        k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
        q_pos_emb = apply_rotary_pos_emb(
            q_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q, mscale=mscale
        )
        k_pos_emb = apply_rotary_pos_emb(
            k_pos_emb, rotary_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv, mscale=mscale
        )

        # query: [s, b, n, 192]
        query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

        # key: [s, b, n, 192]
        key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        return query, key, value

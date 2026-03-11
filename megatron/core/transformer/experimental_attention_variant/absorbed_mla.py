# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Absorbed Multi-Latent Attention implementation.

This module implements MLA with matrix absorption:
- Absorbs K's up projection into Q: Q' = Q @ K_up_proj^T
- Applies V's up projection after core attention
- Core attention operates in MQA form with KV being single-head.

The absorption is mathematically equivalent to standard MLA but enables MQA-style attention which
can be more efficient for certain attention variants.
"""

import math
from dataclasses import dataclass
from typing import NoReturn, Optional, Union

import torch

from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
    apply_rotary_pos_emb,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import deprecate_inference_params, get_pg_size

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except ImportError:
    fused_apply_mla_rope_for_kv = None
    fused_apply_mla_rope_for_q = None

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        set_save_original_input,
    )
    from megatron.core.post_training.modelopt.layers import Linear

    HAVE_TE = True
except ImportError:
    TEColumnParallelLinear, TELinear, Linear, set_save_original_input = None, None, None, None
    HAVE_TE = False


@dataclass
class AbsorbedMLASelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of absorbed multi-latent self-attention.
    """

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_down_proj: Union[ModuleSpec, type] = None
    linear_k_up_proj: Union[ModuleSpec, type] = None
    linear_v_up_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None


class AbsorbedMLASelfAttention(Attention):
    """Multi-latent self-attention layer with matrix absorption.

    This layer takes input with shape [s, b, h] and returns output of the same shape.

    Compared to standard MLA, this class implements matrix absorption:
      - K's up projection is applied to the query before core attention, not to the compressed KV.
      - V's up projection is applied to the output of core attention, not to the compressed KV.
      - Core attention operates in MQA form with KV being single-head.

    The absorption is mathematically equivalent to standard MLA but enables MQA-style attention
    computation which can be more efficient for certain attention variants.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: AbsorbedMLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            pg_collection=pg_collection,
        )

        assert not config.add_bias_linear, "add_bias_linear is not supported for AbsorbedMLA"

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Inference is currently not supported.
        self.key_hidden_size = None
        self.val_hidden_size = None

        self.recompute_up_proj = (
            self.config.recompute_granularity == 'selective'
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale_all_dim)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)
        self.cache_mla_latents = self.config.cache_mla_latents
        assert not self.cache_mla_latents, "cache_mla_latents is not supported for AbsorbedMLA"

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
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

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type="self",
            softmax_scale=self.softmax_scale,
            k_channels=self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            v_channels=self.config.kv_lora_rank,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
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
            tp_group=self.pg_collection.tp,
        )

        if (
            HAVE_TE
            and isinstance(self.linear_proj, TELinear)
            and (
                (
                    self.config.fp8
                    and self.config.fp8_recipe != 'delayed'
                    and is_te_min_version("2.6.0dev0")
                )
                or (self.config.fp4 and is_te_min_version("2.7.0.dev0"))
            )
        ):
            # For fp8/fp4 training, the output of the fused core_attn is saved by itself, and
            # linear_proj also saves the quantized tensor of this output. Here we set the
            # linear_proj to save the original input tensors to avoid the extra memory usage of
            # the quantized tensor.
            set_save_original_input(self.linear_proj)

        if self.config.q_lora_rank is None:
            # Not projecting query
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
                tp_comm_buffer_name='q_proj',
            )
        else:
            q_down_proj_kwargs = {}
            if submodules.linear_q_down_proj in [TELinear]:
                q_down_proj_kwargs['parallel_mode'] = 'duplicated'
            elif submodules.linear_q_down_proj in [
                Linear,
                TEColumnParallelLinear,
                ColumnParallelLinear,
            ]:
                q_down_proj_kwargs['gather_output'] = False
            else:
                raise ValueError(f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}")

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
                tp_group=(
                    pg_collection.tp
                    if q_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                    else None
                ),
                **q_down_proj_kwargs,
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
                tp_group=pg_collection.tp,
            )

        kv_down_proj_kwargs = {}
        if submodules.linear_kv_down_proj in [TELinear]:
            kv_down_proj_kwargs['parallel_mode'] = 'duplicated'
        elif submodules.linear_kv_down_proj in [
            Linear,
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ]:
            kv_down_proj_kwargs['gather_output'] = False
        else:
            raise ValueError(f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}")

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_down_proj',
            skip_weight_param_allocation=False,
            tp_group=(
                pg_collection.tp
                if kv_down_proj_kwargs.get('parallel_mode') != 'duplicated'
                else None
            ),
            **kv_down_proj_kwargs,
        )

        # Build separate K and V up projections
        self.linear_k_up_proj = build_module(
            submodules.linear_k_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * self.config.qk_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='k_up_proj',
            tp_group=pg_collection.tp,
        )
        self.linear_v_up_proj = build_module(
            submodules.linear_v_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * self.config.v_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='v_up_proj',
            tp_group=pg_collection.tp,
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
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """
        Derives absorbed q, compressed q, and compressed kv tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (
                packed_seq_params.local_cp_size is None
            ), "dynamic context parallel is not supported with MLA yet and is planned for future. \
            Please disable dynamic context parallel."

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # Q down projection
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        # =========================================
        # KV down projection
        # =========================================
        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=self.tp_group)

        if packed_seq_params is not None:
            assert q_compressed.ndim == 3 and q_compressed.size(1) == 1
            assert kv_compressed.ndim == 3 and kv_compressed.size(1) == 1
            assert k_pos_emb.ndim == 3 and k_pos_emb.size(1) == 1
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================
        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_layernorm(q_compressed)

        kv_compressed = self.kv_layernorm(kv_compressed)
        # Because we won't apply V up projection to the compressed KV, so we need to gather it
        # manually.
        if get_pg_size(self.tp_group) > 1 and self.config.sequence_parallel:
            kv_compressed = gather_from_sequence_parallel_region(kv_compressed, group=self.tp_group)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
                q, _ = self.linear_q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            # [num_tokens, kv_lora_rank] -> [num_tokens, 1, kv_lora_rank]
            kv_compressed = torch.unsqueeze(kv_compressed, -2)
            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            # Prepare k_up_weight for absorption
            # k_up_weight: linear_k_up_proj.weight viewed as [n, qk_head_dim, kv_lora_rank]
            assert self.linear_k_up_proj.weight.size(0) == (
                self.num_attention_heads_per_partition * self.config.qk_head_dim
            )
            assert self.linear_k_up_proj.weight.size(1) == self.config.kv_lora_rank
            k_up_weight = self.linear_k_up_proj.weight.view(
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim,
                self.config.kv_lora_rank,
            )

            if self.config.apply_rope_fusion:
                # q_no_pe: [num_tokens, n, qk_head_dim]
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # Absorb k_up_weight into q_no_pe
                # q_absorbed: [num_tokens, n, kv_lora_rank]
                q_absorbed = torch.einsum("...nd,ndk->...nk", q_no_pe, k_up_weight)
                q_absorbed = q_absorbed.contiguous()
                assert q_absorbed.ndim == q.ndim
                assert q_absorbed.shape[:-1] == q.shape[:-1]
                assert q_absorbed.size(-1) == self.config.kv_lora_rank

                # q_absorbed: [num_tokens, n, (kv_lora_rank + qk_pos_emb_head_dim)]
                q_absorbed = torch.cat([q_absorbed, q_pos_emb], dim=-1)
                # kv_compressed: [num_tokens, 1, (kv_lora_rank + qk_pos_emb_head_dim)]
                kv_compressed = torch.cat([kv_compressed, k_pos_emb], dim=-1)

                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                q_absorbed = fused_apply_mla_rope_for_q(
                    q_absorbed,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.kv_lora_rank,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                kv_compressed = fused_apply_mla_rope_for_q(
                    kv_compressed,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.kv_lora_rank,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
            else:
                q_len = q.size()[0]
                if inference_context is not None:
                    # add offset to the sequence start for inference
                    sequence_start = inference_context.sequence_len_offset
                    sequence_end = sequence_start + q_len
                    rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
                elif packed_seq_params is None or self.config.context_parallel_size == 1:
                    # Shorten rotary_pos_emb to the sequence length when inference_params
                    # is not provided. This makes sure we can run forward directly with
                    # any sequence length. During training, the sequence length is always
                    # the full rotary_pos_emb length, except for sequence packing + CP.
                    # When sequence packing and context parallel are both enabled, the
                    # position embedding will not split rotary_pos_emb, so it may exceed
                    # the sequence length on this CP rank, but we need the full rotary_pos_emb
                    # to cover the full sequence, so we do not shorten it here.
                    rotary_pos_emb = rotary_pos_emb[0:q_len]

                # q_no_pe: [num_tokens, n, qk_head_dim]
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )

                # Absorb k_up_weight into q_no_pe
                # q_absorbed: [num_tokens, n, kv_lora_rank]
                q_absorbed = torch.einsum("...nd,ndk->...nk", q_no_pe, k_up_weight)
                q_absorbed = q_absorbed.contiguous()
                assert q_absorbed.ndim == q.ndim
                assert q_absorbed.shape[:-1] == q.shape[:-1]
                assert q_absorbed.size(-1) == self.config.kv_lora_rank

                # Apply RoPE to q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )
                # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )

                # query: [num_tokens, n, (kv_lora_rank + qk_pos_emb_head_dim)]
                q_absorbed = torch.cat([q_absorbed, q_pos_emb], dim=-1)
                # key: [num_tokens, 1, (kv_lora_rank + qk_pos_emb_head_dim)]
                kv_compressed = torch.cat([kv_compressed, k_pos_emb], dim=-1)

            assert q_absorbed.is_contiguous()
            assert kv_compressed.is_contiguous()

            return q_absorbed, kv_compressed

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            assert not quantization, "FP8/FP4 is not supported for AbsorbedMLA"
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            q_absorbed, kv_compressed = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            assert not self.cache_mla_latents, "cache_mla_latents is not supported for AbsorbedMLA"
            q_absorbed, kv_compressed = qkv_up_proj_and_rope_apply(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        return q_absorbed, kv_compressed, q_compressed

    def _checkpointed_attention_forward(
        self,
        q_absorbed,
        k_compressed,
        v_compressed,
        hidden_states,
        q_compressed,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            q_absorbed = inputs[0]
            k_compressed = inputs[1]
            v_compressed = inputs[2]
            hidden_states = inputs[3]
            q_compressed = inputs[4]
            attention_mask = inputs[5]
            attn_mask_type = inputs[7]
            attention_bias = inputs[8]
            packed_seq_params = inputs[9]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                q_absorbed,
                k_compressed,
                v_compressed,
                hidden_states,
                q_compressed,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            q_absorbed,
            k_compressed,
            v_compressed,
            hidden_states,
            q_compressed,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
            attention_bias,
            packed_seq_params,
        )

        return hidden_states

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
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention with matrix absorption"""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and self.cache_mla_latents
        ), "cache_mla_latents conflicts with training."
        assert (
            inference_context is None and inference_params is None
        ), "Inference is not supported for AbsorbedMLA"

        # =====================
        # Query, Key, and Value
        # =====================
        q_absorbed, kv_compressed, q_compressed = self.get_query_key_value_tensors(
            hidden_states, key_value_states, packed_seq_params, inference_context=inference_context
        )

        assert q_absorbed.is_contiguous()
        assert q_compressed.is_contiguous()
        assert kv_compressed.is_contiguous()

        # ==================================
        # Core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                q_absorbed,
                kv_compressed,
                None,
                hidden_states,
                q_compressed,
                attention_mask,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                q_absorbed,
                kv_compressed,
                None,
                hidden_states,
                q_compressed,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.attn_mask_type,
            )

        # ==================================
        # Apply V up projection
        # ==================================
        assert self.linear_v_up_proj.weight.size(0) == (
            self.num_attention_heads_per_partition * self.config.v_head_dim
        )
        assert self.linear_v_up_proj.weight.size(1) == self.config.kv_lora_rank
        v_up_weight = self.linear_v_up_proj.weight.view(
            self.num_attention_heads_per_partition, self.config.v_head_dim, self.config.kv_lora_rank
        )
        core_attn_out = core_attn_out.view(
            *core_attn_out.shape[:-1],
            self.num_attention_heads_per_partition,
            self.config.kv_lora_rank,
        )
        core_attn_out = torch.einsum("...nc,ndc->...nd", core_attn_out, v_up_weight)
        core_attn_out = core_attn_out.contiguous()
        core_attn_out = core_attn_out.view(*core_attn_out.shape[:-2], -1)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            core_attn_out = core_attn_out.unsqueeze(1)

        assert core_attn_out.ndim == hidden_states.ndim
        assert core_attn_out.shape[0] == (
            hidden_states.shape[0] * self.config.tensor_model_parallel_size
        ), (
            f"{core_attn_out.shape[0]} != "
            f"{hidden_states.shape[0]} * "
            f"{self.config.tensor_model_parallel_size}"
        )
        assert core_attn_out.shape[1:-1] == hidden_states.shape[1:-1]
        assert core_attn_out.size(-1) == (
            self.config.v_head_dim * self.num_attention_heads_per_partition
        )

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation."""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers."""
        self.linear_k_up_proj.backward_dw()
        self.linear_v_up_proj.backward_dw()
        self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers."""
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer."""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        from megatron.core.extensions.transformer_engine import set_save_original_input

        if self.config.q_lora_rank is not None:
            set_save_original_input(self.linear_q_down_proj)
        set_save_original_input(self.linear_kv_down_proj)

    def clip_qk(self):
        """
        QK Clipping is a technique to clip the query and key attention logits to prevent the
        attention logits from exploding. Per MuonClip usage, we update the weight by calling this
        function after Muon optimizer step.
        """
        raise NotImplementedError("clip_qk is not implemented for AbsorbedMLA")

    def _combine_kv_weights(self, k_weight, v_weight):
        """Combine separate K and V weights into MLA's interleaved format.

        MLA's linear_kv_up_proj weight layout (per head interleaved):
            [head0_K, head0_V, head1_K, head1_V, ...]

        AbsorbedMLA's separate weights layout:
            K: [head0_K, head1_K, ...]
            V: [head0_V, head1_V, ...]

        This method interleaves K and V per head to match MLA's format.

        Args:
            k_weight: [num_heads_per_partition * qk_head_dim, kv_lora_rank]
            v_weight: [num_heads_per_partition * v_head_dim, kv_lora_rank]

        Returns:
            combined: [num_heads_per_partition * (qk_head_dim + v_head_dim), kv_lora_rank]
        """
        n = self.num_attention_heads_per_partition
        qk_dim = self.config.qk_head_dim
        v_dim = self.config.v_head_dim
        lora_rank = self.config.kv_lora_rank

        # Reshape to per-head format
        k_per_head = k_weight.view(n, qk_dim, lora_rank)
        v_per_head = v_weight.view(n, v_dim, lora_rank)

        # Concatenate K and V for each head along dim=1
        # Result: [n, qk_dim + v_dim, lora_rank]
        combined_per_head = torch.cat([k_per_head, v_per_head], dim=1)

        # Reshape back to linear weight format
        combined_weight = combined_per_head.view(n * (qk_dim + v_dim), lora_rank)

        return combined_weight

    def _split_kv_weights(self, combined_weight):
        """Split MLA's interleaved KV weight into separate K and V weights.

        MLA's linear_kv_up_proj weight layout (per head interleaved):
            [head0_K, head0_V, head1_K, head1_V, ...]

        This method extracts K and V into separate tensors:
            K: [head0_K, head1_K, ...]
            V: [head0_V, head1_V, ...]

        Args:
            combined_weight: [num_heads_per_partition * (qk_head_dim + v_head_dim), kv_lora_rank]

        Returns:
            k_weight: [num_heads_per_partition * qk_head_dim, kv_lora_rank]
            v_weight: [num_heads_per_partition * v_head_dim, kv_lora_rank]
        """
        n = self.num_attention_heads_per_partition
        qk_dim = self.config.qk_head_dim
        v_dim = self.config.v_head_dim
        lora_rank = self.config.kv_lora_rank

        # Reshape to per-head format
        combined_per_head = combined_weight.view(n, qk_dim + v_dim, lora_rank)

        # Split K and V for each head (slicing creates non-contiguous views)
        k_per_head = combined_per_head[:, :qk_dim, :]  # [n, qk_dim, lora_rank]
        v_per_head = combined_per_head[:, qk_dim:, :]  # [n, v_dim, lora_rank]

        # Make contiguous and reshape back to linear weight format
        k_weight = k_per_head.contiguous().view(n * qk_dim, lora_rank)
        v_weight = v_per_head.contiguous().view(n * v_dim, lora_rank)

        return k_weight, v_weight

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Handle loading from checkpoints with combined KV up projection weights.

        This method splits the combined 'linear_kv_up_proj.weight' (which has per-head
        interleaved K and V) into separate 'linear_k_up_proj.weight' and 'linear_v_up_proj.weight'.
        """
        combined_key = f'{prefix}linear_kv_up_proj.weight'
        k_up_key = f'{prefix}linear_k_up_proj.weight'
        v_up_key = f'{prefix}linear_v_up_proj.weight'

        # Split combined KV weights into separate K and V
        if combined_key in state_dict:
            combined_weight = state_dict[combined_key]

            # Split with proper per-head de-interleaving
            k_weight, v_weight = self._split_kv_weights(combined_weight)

            state_dict[k_up_key] = k_weight
            state_dict[v_up_key] = v_weight

            del state_dict[combined_key]

        combined_extra_state_key = f'{prefix}linear_kv_up_proj._extra_state'
        k_up_extra_state_key = f'{prefix}linear_k_up_proj._extra_state'
        v_up_extra_state_key = f'{prefix}linear_v_up_proj._extra_state'

        if combined_extra_state_key in state_dict:
            combined_extra_state = state_dict[combined_extra_state_key]

            assert isinstance(combined_extra_state, torch.Tensor)
            # Now we can only handle the case where the extra state is empty.
            assert combined_extra_state.numel() == 0

            state_dict[k_up_extra_state_key] = combined_extra_state.clone()
            state_dict[v_up_extra_state_key] = combined_extra_state.clone()

            del state_dict[combined_extra_state_key]

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

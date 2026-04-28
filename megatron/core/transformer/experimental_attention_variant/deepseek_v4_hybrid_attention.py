# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


from dataclasses import dataclass
from typing import NoReturn, Optional, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    get_pg_size,
    is_te_min_version,
)

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import (
        fused_mla_rope_inplace,
        fused_mla_rope_kv_split,
    )
except Exception:
    fused_mla_rope_inplace = None
    fused_mla_rope_kv_split = None


if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        set_save_original_input,
    )
    from megatron.core.post_training.modelopt.layers import Linear
else:
    (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        Linear,
        set_save_original_input,
        split_te_layernorm_column_parallel_linear,
    ) = (None, None, None, None, None, None)


@torch.compile
def _q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMS normalization for query tensor (no learnable weight)."""
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)


@dataclass
class DSv4HybridSelfAttentionSubmodules:
    """Submodules for the DSv4HybridAttention layer.
    """

    q_layernorm: LayerNormBuilder
    kv_layernorm: LayerNormBuilder

    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    linear_qkv_down_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class DSv4HybridAttention(Attention):
    """DeepSeek-v4 Hybrid Attention layer.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSv4HybridSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
        )
        self.config: MLATransformerConfig

        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads

        self.q_head_dim = self.config.v_head_dim

        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == 'selective'
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        self.softmax_scale = None

        rope_base = self.config.rotary_base
        compress_ratio = self.config.csa_compress_ratios[layer_number - 1]
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

        core_attn_extra_kwargs = {"rotary_pos_emb": self.rotary_pos_emb}
        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
            **core_attn_extra_kwargs,
        )

        # Output.
        tp_size = get_pg_size(self.pg_collection.tp)
        assert self.config.o_groups % tp_size == 0, (
            "o_groups must be divisible by tp_size"
        )
        self.o_local_groups = self.config.o_groups // tp_size
        assert self.query_projection_size % self.config.o_groups == 0, (
            "num_attention_heads * v_head_dim must be divisible by o_groups"
        )
        group_proj_in_size = self.query_projection_size // self.config.o_groups
        group_proj_out_size = self.config.o_groups * self.config.o_lora_rank

        _linear_o_group_proj = torch.empty(
                group_proj_out_size,
                group_proj_in_size,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
        self.config.init_method(_linear_o_group_proj)
        self.linear_o_group_proj = torch.nn.Parameter(_linear_o_group_proj)

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
        """Forward pass for DeepSeek-v4 Hybrid Attention"""
        assert rotary_pos_emb is None, \
            "Rotary position embeddings should not be passed into DSv4HybridAttention."
        assert attention_bias is None, \
            "Attention bias should not be passed into DSv4HybridAttention."
        assert rotary_pos_cos is None and rotary_pos_sin is None, \
            "DSv4HybridAttention does not support Flash Decoding"
        assert not rotary_pos_cos_sin, \
            "Flash-infer rope has not been tested with DSv4HybridAttention."
        assert inference_context is None and inference_params is None, \
            "Inference is not supported for DSv4HybridAttention."

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        qkv_linear_manager = off_interface(self.offload_qkv_linear, hidden_states, "qkv_linear")
        with qkv_linear_manager as hidden_states:
            query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        query = qkv_linear_manager.group_offload(query, forced_released_tensors=[hidden_states])

        # TODO: Currently, TE can only accept contiguous tensors for MLA
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        core_attn_manager = off_interface(
            self.offload_core_attention and self.training, query, "core_attn"
        )
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            extra_kwargs = {}
            if self.config.experimental_attention_variant in ("dsa", "dsv4_hybrid"):
                # For dsa we need to pass in the original hidden states and the compressed
                # query representation.
                extra_kwargs["x"] = hidden_states
                extra_kwargs["qr"] = q_compressed
            with core_attn_manager as query:
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    **extra_kwargs,
                )
            core_attn_out = core_attn_manager.group_offload(
                core_attn_out, forced_released_tensors=[query, key, value]
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        # inverse RoPE on last qk_pos_emb_head_dim of each head
        seq_len = core_attn_out.size(0)
        n_heads = self.num_attention_heads_per_partition
        pos_dim = self.config.qk_pos_emb_head_dim
        nope_dim = self.config.v_head_dim - pos_dim
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), n_heads, -1)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if packed_seq:
            cu_seqlens_kv = (
                packed_seq_params.cu_seqlens_kv_padded
                if packed_seq_params.cu_seqlens_kv_padded is not None
                else packed_seq_params.cu_seqlens_kv
            )
            rope_seqlen = cu_seqlens_kv[-1].item()
        else:
            cu_seqlens_kv = None
            rope_seqlen = seq_len
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rope_seqlen, packed_seq=packed_seq)
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rope_seqlen, dtype=hidden_states.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                assert (
                    inference_context is None
                ), "Inference with MLA RoPE fusion is not supported"
                assert (
                    fused_mla_rope_inplace is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(rope_seqlen, packed_seq=packed_seq)
        if self.config.apply_rope_fusion:
            core_attn_out = fused_mla_rope_inplace(
                core_attn_out,
                rotary_pos_cos,
                rotary_pos_sin,
                nope_dim,
                pos_dim,
                cu_seqlens_kv,
                self.pg_collection.cp.rank(),
                self.pg_collection.cp.size(),
                inverse=True,
            )
        else:
            content_part, rot_part = torch.split(
                core_attn_out, [core_attn_out.size(-1) - pos_dim, pos_dim], dim=-1
            )
            rot_part = apply_rotary_pos_emb(
                rot_part,
                rotary_pos_emb,
                self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                inverse=True,
            )
            core_attn_out = torch.cat([content_part, rot_part], dim=-1)
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)

        # Grouped output
        core_attn_out = core_attn_out.view(
            core_attn_out.size(0), core_attn_out.size(1), self.o_local_groups, -1
        )
        wo_a_weight = self.linear_o_group_proj.view(
            self.o_local_groups, self.config.o_lora_rank, -1
        )
        core_attn_out = torch.einsum("...gd,grd->...gr", core_attn_out, wo_a_weight)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        # =================
        # Output. [sq, b, h]
        # =================
        attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, "attn_proj")
        with attn_proj_manager as core_attn_out:
            output, bias = self.linear_proj(core_attn_out)
        output = attn_proj_manager.group_offload(output, forced_released_tensors=[core_attn_out])

        return output, bias


class DSv4HybridSelfAttention(DSv4HybridAttention):
    """DSv4Hybrid Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSv4HybridSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

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

        self.linear_kv_down_proj = None
        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.hidden_size,
            self.config.v_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=pg_collection.tp,
        )
        self.kv_layernorm = submodules.kv_layernorm(
            hidden_size=self.config.v_head_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.q_layernorm = submodules.q_layernorm(
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
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
        if packed_seq_params is not None:
            assert (
                packed_seq_params.local_cp_size is None
            ), "dynamic_context_parallel is not supported with MLA yet and is planned for future. \
            Please disable dynamic_context_parallel."

        assert inference_context is None and inference_params is None, \
            "Inference is not supported for DSv4HybridSelfAttention."

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
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
                    fused_mla_rope_inplace is not None
                    and fused_mla_rope_kv_split is not None
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
        # QKV down projection and layernorm
        # =========================================
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

        kv_compressed = hidden_states
        k_pos_emb = None

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

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
            # q_compressed: [num_tokens, q_lora_rank]
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q, _ = self.linear_q_up_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
            q = _q_rms_norm(q, self.config.layernorm_epsilon)

            kv, _ = self.linear_kv_up_proj(kv_compressed)
            kv = self.kv_layernorm(kv)

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            if k_pos_emb is not None:
                k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            # todo add assert about fusions and caching
            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                query = fused_mla_rope_inplace(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_head_dim,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                kv = kv.unsqueeze(-2)
                kv = fused_mla_rope_inplace(
                    kv,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_head_dim,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                key = kv
                value = kv
            else:
                q_len = q.size()[0]
                if packed_seq_params is None or self.config.context_parallel_size == 1:
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

                # RoPE and query (shared for wkv and latent)
                # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                )
                # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                pos_dim = self.config.qk_pos_emb_head_dim
                kv_no_pe, k_pos_emb = torch.split(
                    kv, [kv.size(-1) - pos_dim, pos_dim], dim=-1
                )

                # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                )

                # Single head: key = value = [num_tokens, 1, v_head_dim]
                kv = torch.cat([kv_no_pe, k_pos_emb], dim=-1).unsqueeze(-2)
                key = kv
                value = kv

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        return query, key, value, q_compressed, kv_compressed

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        self.linear_kv_up_proj.backward_dw()
        if self.linear_kv_down_proj is not None:
            self.linear_kv_down_proj.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        if self.config.q_lora_rank is None:
            self.linear_q_proj.backward_dw()
        else:
            self.linear_q_down_proj.backward_dw()
            self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        if self.config.q_lora_rank is not None:
            set_save_original_input(self.linear_q_down_proj)
        if self.linear_kv_down_proj is not None:
            set_save_original_input(self.linear_kv_down_proj)
        elif not self.recompute_up_proj:
            set_save_original_input(self.linear_kv_up_proj)

    def clip_qk(self):
        """
        QK Clipping is a technique to clip the query and key attention logits to prevent the
        attention logits from exploding. Per MuonClip usage, we update the weight by calling this
        function after Muon optimizer step.
        """

        if not self.config.qk_clip:
            raise ValueError("qk_clip option needs to be enabled")

        if self.core_attention.current_max_attn_logits is None:
            raise ValueError("current_max_attn_logits is None")

        assert self.core_attention.current_max_attn_logits.shape == (
            self.num_attention_heads_per_partition,
        ), f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition}, ) \
                    but {self.core_attention.current_max_attn_logits.shape}"

        # only update the weight if any head has
        # current_max_attn_logits > qk_clip_threshold
        if torch.any(self.core_attention.current_max_attn_logits > self.config.qk_clip_threshold):
            # Use num_attention_heads_per_partition for tensor parallel scenarios

            # qk_clip_balancing_eta (n, 1, 1)
            assert self.core_attention.current_max_attn_logits.shape == (
                self.num_attention_heads_per_partition,
            ), f"current_max_attn_logits shape is not ({self.num_attention_heads_per_partition},) \
                but {self.core_attention.current_max_attn_logits.shape}"
            self.qk_clip_balancing_eta = torch.clamp(
                self.config.qk_clip_threshold / self.core_attention.current_max_attn_logits, max=1.0
            ).view(self.num_attention_heads_per_partition, 1, 1)
            assert torch.all(self.qk_clip_balancing_eta <= 1.0)

            # Update q side weight, keep qk_pos_emb_head_dim side weight unchanged
            if self.config.q_lora_rank is None:
                q_proj_weight = self.linear_q_proj.weight
            else:
                q_proj_weight = self.linear_q_up_proj.weight

            # Handle different weight access patterns (main_param vs direct access)
            if hasattr(q_proj_weight, 'main_param'):
                q_proj_weight.main_param.data.copy_(
                    self._clip_q_proj_weight(q_proj_weight.main_param.data)
                )
            q_proj_weight.data.copy_(self._clip_q_proj_weight(q_proj_weight.data))

            # Update k side weight, keep v side weight unchanged
            kv_proj_weight = self.linear_kv_up_proj.weight

            # Handle different weight access patterns
            if hasattr(kv_proj_weight, 'main_param'):
                kv_proj_weight.main_param.data.copy_(
                    self._clip_kv_proj_weight(kv_proj_weight.main_param.data)
                )
            kv_proj_weight.data.copy_(self._clip_kv_proj_weight(kv_proj_weight.data))

        # reset current_max_attn_logits
        self.core_attention.current_max_attn_logits = None

    def _clip_q_proj_weight(self, weight):
        """Clip q_proj_weight"""
        # Reshape to (n, a + b, -1)
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.qk_pos_emb_head_dim,
            -1,
        )

        # Split into qk_head_dim and qk_pos_emb_head_dim parts: (n, a, -1) and (n, b, -1)
        weight_q_nope = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_q_pe = weight_reshaped[:, self.config.qk_head_dim :, :]

        # Clipping
        weight_q_nope.mul_(torch.pow(self.qk_clip_balancing_eta, self.config.qk_clip_alpha))
        weight_q_pe.mul_(self.qk_clip_balancing_eta)

        # Concatenate back and reshape to original shape
        weight_q_updated = torch.cat([weight_q_nope, weight_q_pe], dim=1)
        weight_q_updated = weight_q_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.qk_pos_emb_head_dim),
            -1,
        )

        return weight_q_updated

    def _clip_kv_proj_weight(self, weight):
        """Clip kv_proj_weight"""
        # shape: (n, qk_head_dim + v_head_dim, kv_lora_rank)
        weight_reshaped = weight.view(
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
            -1,
        )

        # Split into qk_head_dim and v_head_dim parts: (n, a, -1) and (n, b, -1)
        weight_k = weight_reshaped[:, : self.config.qk_head_dim, :]
        weight_v = weight_reshaped[:, self.config.qk_head_dim :, :]

        # Clipping
        weight_k.mul_(torch.pow(self.qk_clip_balancing_eta, 1 - self.config.qk_clip_alpha))

        # Concatenate back and reshape to original shape
        weight_kv_updated = torch.cat([weight_k, weight_v], dim=1)
        weight_kv_updated = weight_kv_updated.view(
            self.num_attention_heads_per_partition
            * (self.config.qk_head_dim + self.config.v_head_dim),
            -1,
        )

        return weight_kv_updated

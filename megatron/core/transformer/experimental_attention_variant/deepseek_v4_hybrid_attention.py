# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import os
from dataclasses import dataclass
from typing import NoReturn, Optional, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    fused_mla_rope_inplace,
    fused_mla_rope_out_of_place,
)
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.typed_torch import apply_module
from megatron.core.utils import get_pg_size, is_te_min_version, make_tp_sharded_tensor_for_checkpoint

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, set_save_original_input
else:
    (TEColumnParallelLinear, TELinear, set_save_original_input) = (None, None, None)


@torch.compile
def _q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMS normalization for query tensor (no learnable weight)."""
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)


@dataclass
class DSv4HybridSelfAttentionSubmodules:
    """Submodules for the DSv4HybridAttention layer."""

    q_layernorm: LayerNormBuilder
    kv_layernorm: LayerNormBuilder

    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None


class DSv4HybridAttention(Attention):
    """DeepSeek-v4 Hybrid Attention layer."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSv4HybridSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        compress_ratio: Optional[int] = None,
        name: str | None = None,
    ) -> None:

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        self.config: MLATransformerConfig

        self.tp_size = get_pg_size(self.pg_collection.tp)
        if self.tp_size > 1:
            if not self.config.sequence_parallel:
                raise ValueError("DSv4 Hybrid Attention with TP>1 requires sequence_parallel=True.")
            if self.config.num_attention_heads % self.tp_size != 0:
                raise ValueError(
                    "num_attention_heads must be divisible by tensor model parallel size: "
                    f"{self.config.num_attention_heads} % {self.tp_size} != 0"
                )
            if self.config.o_groups % self.tp_size != 0:
                raise ValueError(
                    "o_groups must be divisible by tensor model parallel size: "
                    f"{self.config.o_groups} % {self.tp_size} != 0"
                )
            # The generic Attention base class treats num_query_groups=1 as
            # GQA and assumes each TP rank initially owns all query heads.
            # DSv4's custom q-up projection is already column-sharded, so its
            # output is local-head shaped and must use the explicit TP-local
            # head count here.
            self.num_attention_heads_per_partition = self.config.num_attention_heads // self.tp_size
            self.num_query_groups_per_partition = 1

        assert (
            not self.checkpoint_core_attention
        ), "Checkpoint core attention is not supported in DSv4 Hybrid Attention."
        assert (
            not self.offload_qkv_linear
        ), "Offload qkv linear is not supported in DSv4 Hybrid Attention."

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

        # Per-layer compress ratio. When set explicitly (e.g. hybrid 'C'/'H' layer symbols
        # pass compress_ratio=4/128 via the spec), use it directly; otherwise fall back to the
        # per-(global)-layer csa_compress_ratios array (GPT-parity / array-driven path).
        _ratio_idx = self.config.num_layers + layer_number - 1 if is_mtp_layer else layer_number - 1
        if compress_ratio is None:
            compress_ratio = self.config.csa_compress_ratios[_ratio_idx]
        # compress_ratio == 0 is a sliding-window-only layer (the 'W' symbol): no compressor /
        # no top-k indexer (see CompressedSparseAttention) AND standard (non-YARN) rope.
        use_compressed_yarn = compress_ratio > 1
        rope_base = (
            self.config.csa_compress_rotary_base if use_compressed_yarn else self.config.rotary_base
        )
        self._dsv4_compress_ratio = compress_ratio
        self._dsv4_rope_base = rope_base
        self._dsv4_uses_yarn_rope = use_compressed_yarn
        if not use_compressed_yarn:
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=rope_base,
                cp_group=self.pg_collection.cp,
            )
        else:
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

        core_attn_extra_kwargs = {
            "rotary_pos_emb": self.rotary_pos_emb,
            "compress_ratio": compress_ratio,
            "is_mtp_layer": is_mtp_layer,
            "name": (name + ".core_attention") if name is not None else None,
        }
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
        # Q heads are column-parallel. The manual grouped output projection
        # therefore owns only the corresponding local groups on each TP rank.
        self.o_local_groups = self.config.o_groups // self.tp_size
        assert (
            self.query_projection_size % self.config.o_groups == 0
        ), "num_attention_heads * v_head_dim must be divisible by o_groups"
        group_proj_in_size = self.query_projection_size // self.config.o_groups
        group_proj_out_size = self.o_local_groups * self.config.o_lora_rank

        _linear_o_group_proj = torch.empty(
            group_proj_out_size,
            group_proj_in_size,
            device=torch.cuda.current_device(),
            dtype=self.config.params_dtype,
        )
        self.config.init_method(_linear_o_group_proj)
        self.linear_o_group_proj = torch.nn.Parameter(_linear_o_group_proj)
        if self.tp_size > 1:
            set_tensor_model_parallel_attributes(self.linear_o_group_proj, True, 0, 1)

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
        assert (
            rotary_pos_emb is None
        ), "Rotary position embeddings should not be passed into DSv4HybridAttention."
        assert (
            attention_bias is None
        ), "Attention bias should not be passed into DSv4HybridAttention."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "DSv4HybridAttention does not support Flash Decoding"
        assert (
            not rotary_pos_cos_sin
        ), "Flash-infer rope has not been tested with DSv4HybridAttention."
        assert (
            inference_context is None and inference_params is None
        ), "Inference is not supported for DSv4HybridAttention."

        # Select this microbatch's dynamic CP group. QKV captures it explicitly
        # for recompute; the rest of this forward reads it from pg_collection.
        # Restore the static group before returning.
        _orig_cp_group = self.pg_collection.cp
        tp_local_csa_requested = (
            os.environ.get('DSV4_TP_LOCAL_CSA', '0').strip().lower()
            in ('1', 'true', 'yes', 'on')
        )
        tp_local_csa = (
            tp_local_csa_requested
            and self.tp_size > 1
            and self.config.sequence_parallel
            and packed_seq_params is not None
            and packed_seq_params.qkv_format == 'thd'
            and _orig_cp_group.size() == 1
        )
        if tp_local_csa and os.environ.get('DSV4_TP_LOCAL_CSA_UNSAFE', '0') != '1':
            raise RuntimeError(
                'DSV4_TP_LOCAL_CSA is disabled: rank-distinct TP correctness failed. '
                'Set DSV4_TP_LOCAL_CSA_UNSAFE=1 only to reproduce the negative experiment.'
            )
        tp_local_indexer_q = (
            os.environ.get('DSV4_TP_LOCAL_INDEXER_Q', '0').strip().lower()
            in ('1', 'true', 'yes', 'on')
            and not tp_local_csa
            and self.tp_size > 1
            and self.config.sequence_parallel
            and packed_seq_params is not None
            and packed_seq_params.qkv_format == 'thd'
            and _orig_cp_group.size() == 1
        )
        if tp_local_indexer_q and (getattr(self.config, 'dsa_indexer_loss_coeff', 0.0) or 0.0) > 0:
            raise RuntimeError(
                'DSV4_TP_LOCAL_INDEXER_Q currently requires dsa_indexer_loss_coeff == 0.'
            )
        cp_group = self.pg_collection.tp if tp_local_csa else _orig_cp_group
        if not tp_local_csa and packed_seq_params is not None and packed_seq_params.local_cp_size is not None:
            assert packed_seq_params.cp_group is not None, "cp_group must be set in dynamic-cp mode"
            cp_group = packed_seq_params.cp_group

        cp_size = cp_group.size()
        qkv_format = packed_seq_params.qkv_format if packed_seq_params is not None else None
        if cp_size > 1 and qkv_format != 'thd':
            raise ValueError("DSv4 Hybrid with CP requires qkv_format='thd'.")
        use_thd_cp = cp_size > 1 and qkv_format == 'thd'
        if use_thd_cp and packed_seq_params.cp_partition_mode != "contiguous":
            raise ValueError("DSv4 THD CP requires a contiguous CP partition.")
        self.pg_collection.cp = cp_group

        sequence_parallel_local_length = hidden_states.size(0)
        core_hidden_states = hidden_states
        if self.tp_size > 1 and self.config.sequence_parallel and not tp_local_csa:
            # q-up uses TE's standard SP all-gather. The duplicated KV path
            # and CSA compressor need the same CP-local sequence explicitly.
            core_hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, group=self.pg_collection.tp
            )

        boundary_hidden = None
        if use_thd_cp:
            boundary_hidden = cp_utils.exchange_cp_boundary_hidden(
                core_hidden_states,
                self._dsv4_compress_ratio,
                self.config.csa_window_size,
                self.pg_collection.cp,
            )

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        qkv = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
            boundary_hidden=boundary_hidden,
            tp_local_csa=tp_local_csa,
        )
        if use_thd_cp:
            query, key, value, q_compressed, kv_compressed, boundary_kv = qkv
        else:
            query, key, value, q_compressed, kv_compressed = qkv
            boundary_kv = None

        if tp_local_csa:
            # TE may return a sequence-gathered q-up result even though its
            # GEMM consumed local rows. The local CSA path only needs this
            # rank's contiguous query rows.
            tp_rank = torch.distributed.get_rank(group=self.pg_collection.tp)
            local_start = tp_rank * sequence_parallel_local_length
            global_rows = sequence_parallel_local_length * self.tp_size

            def _take_tp_local_rows(tensor):
                if tensor is None or tensor.size(0) == sequence_parallel_local_length:
                    return tensor
                if tensor.size(0) != global_rows:
                    raise RuntimeError(
                        "DSv4 TP local CSA received an unexpected sequence length: "
                        f"shape={tuple(tensor.shape)}, local={sequence_parallel_local_length}, "
                        f"tp={self.tp_size}"
                    )
                return tensor.narrow(0, local_start, sequence_parallel_local_length).contiguous()

            query = _take_tp_local_rows(query)
            key = _take_tp_local_rows(key)
            value = _take_tp_local_rows(value)
            q_compressed = _take_tp_local_rows(q_compressed)

        core_q_compressed = q_compressed
        if self.tp_size > 1 and self.config.sequence_parallel and not tp_local_csa:
            # q-down is duplicated and remains TP-local, while CSA's learned
            # indexer consumes the same complete CP-local sequence as the
            # gathered hidden states. Keep the local q-compressed tensor for
            # q-up, and provide a gathered copy to the core-attention path.
            core_q_compressed = tensor_parallel.gather_from_sequence_parallel_region(
                q_compressed, group=self.pg_collection.tp
            )

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
        with core_attn_manager as query:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                x=core_hidden_states,
                qr=core_q_compressed,
                boundary_hidden=boundary_hidden,
                boundary_kv=boundary_kv,
                indexer_x_local=hidden_states if tp_local_indexer_q else None,
                indexer_qr_local=q_compressed if tp_local_indexer_q else None,
                tp_local_indexer_q=tp_local_indexer_q,
            )
        forced_released_tensors = [query, key, value]
        if boundary_kv is not None:
            forced_released_tensors.append(boundary_kv)
        core_attn_out = core_attn_manager.group_offload(
            core_attn_out, forced_released_tensors=forced_released_tensors
        )
        if tp_local_csa:
            # The local CSA path computes only this rank's query rows. Restore
            # the legacy global-row contract before inverse RoPE and the
            # existing output projection; its backward reduce-scatters the
            # output gradient back into the local CSA rows.
            core_attn_out = tensor_parallel.gather_from_sequence_parallel_region(
                core_attn_out, group=self.pg_collection.tp
            )
            self.pg_collection.cp = _orig_cp_group

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
            rope_seqlen = packed_seq_params.max_seqlen_kv
            rope_max_seqlen_kv = packed_seq_params.max_seqlen_kv
        else:
            cu_seqlens_kv = None
            rope_seqlen = seq_len
            rope_max_seqlen_kv = None
        # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
        # concentration factor (mscale) is NOT part of the DSv4 model contract --
        # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0.
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.config.apply_rope_fusion:
            # ``mscale=1.0`` strips yarn's concentration factor from the
            # cached cos/sin so the fused kernel matches the unfused
            # path's forced ``mscale=1.0`` (DSv4 "pure rotation").
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                rope_seqlen, dtype=hidden_states.dtype, packed_seq=packed_seq, mscale=mscale
            )
            rotary_pos_emb = None
            assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        elif self._dsv4_uses_yarn_rope:
            rotary_pos_emb, _ = self.rotary_pos_emb(rope_seqlen, packed_seq=packed_seq)
        else:
            rotary_pos_emb = self.rotary_pos_emb(rope_seqlen, packed_seq=packed_seq)
        if self.config.apply_rope_fusion:
            if use_thd_cp and not tp_local_csa:
                global_start = self.pg_collection.cp.rank() * core_attn_out.shape[0]
                core_attn_out = cp_utils.apply_thd_cp_local_rope_fused(
                    core_attn_out,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    nope_dim,
                    pos_dim,
                    cu_seqlens_kv,
                    global_start,
                    inverse=True,
                )
            else:
                if packed_seq:
                    core_attn_out = core_attn_out.squeeze(1)
                # Fused DSA backward retains the raw attention output O. Applying
                # inverse RoPE to its view in-place corrupts the retained O used by
                # the softmax backward, so this call needs private storage.
                core_attn_out = fused_mla_rope_out_of_place(
                    core_attn_out,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    nope_dim,
                    pos_dim,
                    cu_seqlens_kv,
                    self.pg_collection.cp.rank(),
                    self.pg_collection.cp.size(),
                    inverse=True,
                    remove_interleaving=True,
                )
                if packed_seq:
                    core_attn_out = core_attn_out.unsqueeze(1)
        elif use_thd_cp and not tp_local_csa:
            global_start = self.pg_collection.cp.rank() * core_attn_out.shape[0]
            core_attn_out = cp_utils.apply_thd_cp_local_rope_unfused(
                core_attn_out,
                rotary_pos_emb,
                nope_dim,
                pos_dim,
                cu_seqlens_kv,
                global_start,
                self.config,
                inverse=True,
            )
        else:
            content_part, rot_part = torch.split(
                core_attn_out, [core_attn_out.size(-1) - pos_dim, pos_dim], dim=-1
            )
            # ``_apply_rotary_pos_emb_thd`` documents 3-D ``(total, h, d)`` input
            # and adds its own batch dim internally; drop the dummy ``b=1`` axis
            # for THD before the rope and add it back after.
            if packed_seq:
                rot_part_in = rot_part.squeeze(1)
            else:
                rot_part_in = rot_part
            rot_part_out = apply_rotary_pos_emb(
                rot_part_in,
                rotary_pos_emb,
                self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
                cp_group=self.pg_collection.cp,
                mla_rotary_interleaved=True,
                inverse=True,
                mla_output_remove_interleaving=True,
                max_seqlen=rope_max_seqlen_kv,
            )
            if packed_seq:
                rot_part = rot_part_out.unsqueeze(1)
            else:
                rot_part = rot_part_out
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

        # Some TE versions reduce the row-parallel output across TP but leave
        # the sequence dimension gathered. Restore the standard
        # sequence-parallel module contract (local sequence on return) when
        # that backend behavior is observed; do not double-scatter versions
        # that already return the local shape.
        if (
            self.tp_size > 1
            and self.config.sequence_parallel
            and output.size(0) != sequence_parallel_local_length
            and output.size(0) % self.tp_size == 0
        ):
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output, group=self.pg_collection.tp
            )

        self.pg_collection.cp = _orig_cp_group
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
        is_mtp_layer: bool = False,
        pp_layer_offset: Optional[int] = None,
        compress_ratio: Optional[int] = None,
        name: str | None = None,
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
            is_mtp_layer=is_mtp_layer,
            pp_layer_offset=pp_layer_offset,
            compress_ratio=compress_ratio,
            name=name,
        )

        q_down_proj_kwargs = {}
        if submodules.linear_q_down_proj in [TELinear]:
            q_down_proj_kwargs['parallel_mode'] = 'duplicated'
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
            tp_group=None,
            name=(name + ".linear_q_down_proj") if name is not None else None,
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
            name=(name + ".linear_q_up_proj") if name is not None else None,
        )
        kv_proj_kwargs = {}
        kv_proj_tp_group = pg_collection.tp
        if submodules.linear_kv_proj is TELinear:
            # DSv4 has one shared KV stream. Replicate it across TP ranks while
            # query heads and grouped output weights are sharded by TP.
            kv_proj_kwargs['parallel_mode'] = 'duplicated'
            kv_proj_tp_group = None
        else:
            raise ValueError(
                f"Unsupported linear_kv_proj for TP-aware DSv4: {submodules.linear_kv_proj}"
            )

        self.linear_kv_proj = build_module(
            submodules.linear_kv_proj,
            self.config.hidden_size,
            self.config.v_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            is_expert=False,
            tp_comm_buffer_name='kv_up_proj',
            tp_group=kv_proj_tp_group,
            name=(name + ".linear_kv_proj") if name is not None else None,
            **kv_proj_kwargs,
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

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        """Add TP metadata for the manually-created grouped output projection."""
        state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        # The Bridge replaces this Parameter with TE GroupedLinear for FP8
        # parameter-gather mode; that module supplies its own sharding metadata.
        if not isinstance(self.linear_o_group_proj, torch.Tensor):
            return state_dict
        key = f"{prefix}linear_o_group_proj"
        state_dict[key] = make_tp_sharded_tensor_for_checkpoint(
            tensor=self.linear_o_group_proj,
            key=key,
            tp_axis=0,
            prepend_offsets=sharded_offsets,
        )
        return state_dict

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
        boundary_hidden=None,
        tp_local_csa=False,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.

        Returns:
            Tuple of ``(query, key, value, q_compressed, kv_compressed)``. The THD CP
            path appends ``boundary_kv`` carrying the projected left-boundary rows.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert (
            hidden_states.ndim == 3
        ), f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        assert (
            inference_context is None and inference_params is None
        ), "Inference is not supported for DSv4HybridSelfAttention."

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[s, b, 1, 64]
        # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
        # concentration factor (mscale) is NOT part of the DSv4 model contract --
        # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0.
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.config.apply_rope_fusion:
            # ``mscale=1.0`` strips yarn's concentration factor from the
            # cached cos/sin so the fused kernel matches the unfused
            # path's forced ``mscale=1.0`` (DSv4 "pure rotation").
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq, mscale=mscale
            )
            rotary_pos_emb = None
            assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
            assert (
                fused_mla_rope_inplace is not None
            ), "Fused MLA RoPE apply is not imported successfully"
        elif self._dsv4_uses_yarn_rope:
            rotary_pos_emb, _ = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            rope_max_seqlen_q = packed_seq_params.max_seqlen_q
            rope_max_seqlen_kv = packed_seq_params.max_seqlen_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None
            rope_max_seqlen_q = rope_max_seqlen_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        # q_compressed: [s, b, q_lora_rank]
        q_compressed, _ = self.linear_q_down_proj(hidden_states)

        # Despite their legacy names, these are hidden-state inputs to linear_kv_proj;
        # DSv4's actual compressed KV is produced later by the CSA compressor.
        kv_compressed = hidden_states
        k_pos_emb = None
        boundary_kv_compressed = boundary_hidden

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            if boundary_kv_compressed is not None:
                boundary_kv_compressed = boundary_kv_compressed.squeeze(1)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = apply_module(self.q_layernorm)(q_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(
            q_compressed,
            kv_compressed,
            k_pos_emb,
            rotary_pos_emb,
            cp_group,
            boundary_kv_compressed=None,
        ):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [s, b, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [s, b, ...] or [t, ...] for two cases.
            """
            # q_compressed: [num_tokens, q_lora_rank]
            # q: [num_tokens, n * (qk_head_dim + qk_pos_emb_head_dim)]
            q, _ = self.linear_q_up_proj(q_compressed)
            if tp_local_csa and q.size(0) != q_compressed.size(0):
                local_q_rows = q_compressed.size(0)
                expected_global_rows = local_q_rows * cp_group.size()
                if q.size(0) != expected_global_rows:
                    raise RuntimeError(
                        "DSv4 TP local CSA q-up returned an unexpected sequence length: "
                        f"q_rows={q.size(0)}, local_rows={local_q_rows}, "
                        f"tp={cp_group.size()}"
                    )
                q = q.narrow(0, cp_group.rank() * local_q_rows, local_q_rows).contiguous()

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
            q = _q_rms_norm(q, self.config.layernorm_epsilon)

            # Column-parallel q-up is the first point at which the actual
            # sequence-parallel local length is observable.  Slice the global
            # (or CP-local) RoPE table to that q length; using the incoming
            # hidden-state length is incorrect because q-down/q-up may perform
            # sequence-parallel scatter/gather internally.
            local_rotary_pos_emb = rotary_pos_emb
            local_rotary_pos_cos = rotary_pos_cos
            local_rotary_pos_sin = rotary_pos_sin
            if self.tp_size > 1 and self.config.sequence_parallel and not packed_seq:
                local_seq_len = q.size(0)
                tp_rank = torch.distributed.get_rank(group=self.pg_collection.tp)
                start = tp_rank * local_seq_len

                def _slice_tp_rope(tensor):
                    if tensor is None or tensor.size(0) == local_seq_len:
                        return tensor
                    if start + local_seq_len > tensor.size(0):
                        raise RuntimeError(
                            "DSv4 TP RoPE table is shorter than the local sequence slice: "
                            f"table={tensor.size(0)}, start={start}, local={local_seq_len}"
                        )
                    return tensor.narrow(0, start, local_seq_len)

                local_rotary_pos_emb = _slice_tp_rope(local_rotary_pos_emb)
                local_rotary_pos_cos = _slice_tp_rope(local_rotary_pos_cos)
                local_rotary_pos_sin = _slice_tp_rope(local_rotary_pos_sin)

            boundary_rows = 0
            if boundary_kv_compressed is not None:
                boundary_rows = boundary_kv_compressed.shape[0]
                kv_projection_input = torch.cat([boundary_kv_compressed, kv_compressed], dim=0)
            else:
                kv_projection_input = kv_compressed

            kv, _ = self.linear_kv_proj(kv_projection_input)
            kv = self.kv_layernorm(kv)
            if self.tp_size > 1 and self.config.sequence_parallel and not tp_local_csa:
                # q-up is a standard SP column-parallel projection and
                # therefore sees the complete CP-local sequence. The V4 KV
                # projection is duplicated, so gather its local output to the
                # same sequence space before CSA/attention.
                if boundary_rows:
                    boundary_kv_part = kv[:boundary_rows]
                    local_kv_part = tensor_parallel.gather_from_sequence_parallel_region(
                        kv[boundary_rows:], group=self.pg_collection.tp
                    )
                    kv = torch.cat([boundary_kv_part, local_kv_part], dim=0)
                else:
                    kv = tensor_parallel.gather_from_sequence_parallel_region(
                        kv, group=self.pg_collection.tp
                    )
            boundary_kv = None

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            if k_pos_emb is not None:
                k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            cp_size = cp_group.size()
            if self.config.apply_rope_fusion:
                if cp_size > 1 and packed_seq:
                    cp_rank = cp_group.rank()
                    # Rank r owns global rows [r * local_rows, (r + 1) * local_rows).
                    global_start = cp_rank * q.shape[0]
                    query = cp_utils.apply_thd_cp_local_rope_fused(
                        q,
                        local_rotary_pos_cos,
                        local_rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        global_start,
                    )
                    kv = kv.unsqueeze(-2)
                    kv = cp_utils.apply_thd_cp_local_rope_fused(
                        kv,
                        local_rotary_pos_cos,
                        local_rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        global_start - boundary_rows,
                    )
                    if boundary_kv_compressed is not None:
                        boundary_kv = kv[:boundary_rows]
                        kv = kv[boundary_rows:]
                else:
                    cp_rank = cp_group.rank()
                    query = fused_mla_rope_inplace(
                        q,
                        local_rotary_pos_cos,
                        local_rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        cp_rank,
                        cp_size,
                        remove_interleaving=True,
                    )
                    kv = kv.unsqueeze(-2)
                    kv = fused_mla_rope_inplace(
                        kv,
                        local_rotary_pos_cos,
                        local_rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        cp_rank,
                        cp_size,
                        remove_interleaving=True,
                    )
                key = kv
                value = kv
            else:
                if packed_seq and cp_size > 1:
                    global_start = cp_group.rank() * q.shape[0]
                    query = cp_utils.apply_thd_cp_local_rope_unfused(
                        q,
                        local_rotary_pos_emb,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        global_start,
                        self.config,
                    )
                    kv = cp_utils.apply_thd_cp_local_rope_unfused(
                        kv.unsqueeze(-2),
                        local_rotary_pos_emb,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_kv,
                        global_start - boundary_rows,
                        self.config,
                    )
                    if boundary_kv_compressed is not None:
                        boundary_kv = kv[:boundary_rows]
                        kv = kv[boundary_rows:]
                    key = value = kv
                else:
                    q_len = q.size()[0]
                    # Shorten rotary_pos_emb to the sequence length when inference_params
                    # is not provided so direct forward accepts any sequence length.  A
                    # TP/sequence-parallel projection may append alignment rows after
                    # the table was created from the unpadded input length; regenerate
                    # the table in that case instead of relying on a broadcast that
                    # fails one row short in the BSHD path.
                    if local_rotary_pos_emb is not None and local_rotary_pos_emb.size(0) < q_len:
                        if self._dsv4_uses_yarn_rope:
                            local_rotary_pos_emb, _ = self.rotary_pos_emb(q_len, packed_seq=packed_seq)
                        else:
                            local_rotary_pos_emb = self.rotary_pos_emb(q_len, packed_seq=packed_seq)
                    local_rotary_pos_emb = local_rotary_pos_emb[0:q_len]

                    # q_no_pe: [num_tokens, n, qk_head_dim]
                    # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                    q_no_pe, q_pos_emb = torch.split(
                        q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                    )

                    # RoPE and query (shared for wkv and latent)
                    # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                    q_pos_emb = apply_rotary_pos_emb(
                        q_pos_emb,
                        local_rotary_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        mscale=mscale,
                        cp_group=cp_group,
                        mla_rotary_interleaved=True,
                        mla_output_remove_interleaving=True,
                        max_seqlen=rope_max_seqlen_q,
                    )
                    # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                    query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                    pos_dim = self.config.qk_pos_emb_head_dim
                    kv_no_pe, k_pos_emb = torch.split(kv, [kv.size(-1) - pos_dim, pos_dim], dim=-1)

                    # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                    k_pos_emb = apply_rotary_pos_emb(
                        k_pos_emb,
                        local_rotary_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=mscale,
                        cp_group=cp_group,
                        mla_rotary_interleaved=True,
                        mla_output_remove_interleaving=True,
                        max_seqlen=rope_max_seqlen_kv,
                    )

                    # Single head: key = value = [num_tokens, 1, v_head_dim]
                    kv = torch.cat([kv_no_pe, k_pos_emb], dim=-1).unsqueeze(-2)
                    key = value = kv

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            if boundary_kv is not None:
                boundary_kv = boundary_kv.contiguous()

            if boundary_kv is None:
                return query, key, value
            return query, key, value, boundary_kv

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            if boundary_kv_compressed is None:
                query, key, value = self.qkv_up_checkpoint.checkpoint(
                    qkv_up_proj_and_rope_apply,
                    q_compressed,
                    kv_compressed,
                    k_pos_emb,
                    rotary_pos_emb,
                    self.pg_collection.cp,
                )
                boundary_kv = None
            else:
                query, key, value, boundary_kv = self.qkv_up_checkpoint.checkpoint(
                    qkv_up_proj_and_rope_apply,
                    q_compressed,
                    kv_compressed,
                    k_pos_emb,
                    rotary_pos_emb,
                    self.pg_collection.cp,
                    boundary_kv_compressed,
                )
        else:
            if boundary_kv_compressed is None:
                query, key, value = qkv_up_proj_and_rope_apply(
                    q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb, self.pg_collection.cp
                )
                boundary_kv = None
            else:
                query, key, value, boundary_kv = qkv_up_proj_and_rope_apply(
                    q_compressed,
                    kv_compressed,
                    k_pos_emb,
                    rotary_pos_emb,
                    self.pg_collection.cp,
                    boundary_kv_compressed,
                )

        result = (query, key, value, q_compressed, kv_compressed)
        if boundary_kv is not None:
            return result + (boundary_kv,)
        return result

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        # core_attention is always CompressedSparseAttention for the dsv4_hybrid
        # variant; its compressor/indexer linears defer their wgrads under
        # delay_wgrad_compute and must be flushed here as well.
        self.core_attention.backward_dw()
        self._backward_output_proj()

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        self.linear_kv_proj.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        self.linear_q_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.linear_proj.backward_dw()

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8/fp4."""
        set_save_original_input(self.linear_q_down_proj)
        set_save_original_input(self.linear_kv_proj)

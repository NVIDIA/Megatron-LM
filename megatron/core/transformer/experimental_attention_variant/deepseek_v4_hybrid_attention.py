# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
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
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    _apply_rope_at_positions,
)
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    _SINGLE_RANK_CP_GROUP,
    all_gather_fixed_cp_tensor,
    contiguous_cp_partition,
    exchange_left_boundary_tensor,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import get_pg_size, is_te_min_version

try:
    from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
except Exception:
    fused_mla_rope_inplace = None


def _cp_debug_trace(message: str) -> None:
    if not os.environ.get("DSV4_CP_DEBUG_TRACE"):
        return
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else "?"
    print(f"[dsv4-cp-wrapper rank={rank}] {message}", flush=True)


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
        is_mtp_layer: bool = False,
    ) -> None:

        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        self.config: MLATransformerConfig

        assert (
            get_pg_size(self.pg_collection.tp) == 1
        ), "DSv4 Hybrid Attention only supports TP size 1."

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

        if is_mtp_layer:
            layer_idx = self.config.num_layers + layer_number - 1
            compress_ratio = self.config.csa_compress_ratios[layer_idx]
        else:
            compress_ratio = self.config.csa_compress_ratios[layer_number - 1]
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

    def _dsv4_cp_boundary_window(self) -> int:
        ratios = [ratio for ratio in self.config.csa_compress_ratios if ratio and ratio > 1]
        d_comp = max((8 if ratio == 4 else ratio for ratio in ratios), default=0)
        return max(self.config.csa_window_size, d_comp)

    def _dsv4_cp_seq_positions(
        self, global_ids: torch.Tensor, cu_seqlens: torch.Tensor, clamp: bool
    ) -> torch.Tensor:
        if clamp:
            total_tokens = int(cu_seqlens[-1].item())
            global_ids = global_ids.clamp(min=0, max=max(total_tokens - 1, 0))
        batch_ids = torch.searchsorted(cu_seqlens.to(torch.long), global_ids, right=True) - 1
        batch_ids = batch_ids.clamp(min=0, max=cu_seqlens.numel() - 2)
        return global_ids - cu_seqlens.to(torch.long)[batch_ids]

    def _dsv4_cp_local_seq_positions(
        self, packed_seq_params, local_tokens: int
    ) -> torch.Tensor:
        cu_seqlens_q = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        total_tokens = int(cu_seqlens_q[-1].item())
        cp_size = self.pg_collection.cp.size()
        cp_rank = self.pg_collection.cp.rank()
        if total_tokens % cp_size != 0:
            raise RuntimeError(
                "DSv4 THD CP local RoPE requires padded_total_tokens % cp_size == 0: "
                f"total={total_tokens}, cp_size={cp_size}"
            )
        expected_local_tokens = total_tokens // cp_size
        if local_tokens != expected_local_tokens:
            raise RuntimeError(
                "DSv4 THD CP local RoPE expects contiguous equal local token chunks: "
                f"local={local_tokens}, expected={expected_local_tokens}"
            )
        global_start = cp_rank * expected_local_tokens
        global_ids = torch.arange(
            global_start,
            global_start + expected_local_tokens,
            device=cu_seqlens_q.device,
            dtype=torch.long,
        )
        return self._dsv4_cp_seq_positions(global_ids, cu_seqlens_q, clamp=False)

    def _dsv4_rotary_pos_emb_for_positions(self, positions: torch.Tensor) -> torch.Tensor:
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 0
        rope_result = self.rotary_pos_emb(max_pos, packed_seq=True)
        rotary_pos_emb = rope_result[0] if isinstance(rope_result, tuple) else rope_result
        return rotary_pos_emb.index_select(0, positions.to(torch.long))

    def _build_boundary_key_from_hidden(
        self, boundary_hidden: torch.Tensor, packed_seq_params
    ) -> torch.Tensor:
        cu_seqlens_kv = (
            packed_seq_params.cu_seqlens_kv_padded
            if packed_seq_params.cu_seqlens_kv_padded is not None
            else packed_seq_params.cu_seqlens_kv
        )
        total_tokens = int(cu_seqlens_kv[-1].item())
        cp_size = self.pg_collection.cp.size()
        cp_rank = self.pg_collection.cp.rank()
        if total_tokens % cp_size != 0:
            raise RuntimeError(
                "DSv4 THD CP boundary KV projection requires padded_total_tokens % cp_size == 0: "
                f"total={total_tokens}, cp_size={cp_size}"
            )

        local_tokens = total_tokens // cp_size
        global_start = cp_rank * local_tokens
        d_window = boundary_hidden.shape[0]
        boundary_flat = boundary_hidden.squeeze(1)

        kv, _ = self.linear_kv_proj(boundary_flat)
        kv = self.kv_layernorm(kv)

        global_ids = torch.arange(
            global_start - d_window, global_start, device=boundary_hidden.device, dtype=torch.long
        )
        seq_positions = self._dsv4_cp_seq_positions(global_ids, cu_seqlens_kv, clamp=True)

        pos_dim = self.config.qk_pos_emb_head_dim
        kv = _apply_rope_at_positions(
            kv.unsqueeze(1),
            kv.size(-1) - pos_dim,
            pos_dim,
            self.rotary_pos_emb,
            self.config,
            seq_positions,
        )
        return kv.squeeze(1).unsqueeze(-2)

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

        use_thd_cp = (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == 'thd'
            and self.pg_collection.cp.size() > 1
        )
        use_full_context_qkv = (
            use_thd_cp and getattr(self.core_attention, "compress_ratio", 0) > 1
        )
        qkv_hidden_states = hidden_states
        local_token_slice = None
        cp_global_start = None
        cp_local_tokens = None
        if use_full_context_qkv:
            cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
            cp_global_start, cp_local_tokens = contiguous_cp_partition(
                cu_seqlens_q, self.pg_collection.cp.size(), self.pg_collection.cp.rank()
            )
            if hidden_states.shape[0] != cp_local_tokens:
                raise RuntimeError(
                    "DSv4 THD CP full-context QKV fallback expects contiguous equal chunks: "
                    f"local={hidden_states.shape[0]}, expected={cp_local_tokens}"
                )
            local_token_slice = slice(cp_global_start, cp_global_start + cp_local_tokens)
            _cp_debug_trace("full-context qkv hidden all_gather start")
            qkv_hidden_states = all_gather_fixed_cp_tensor(hidden_states, self.pg_collection.cp)
            _cp_debug_trace("full-context qkv hidden all_gather done")

        boundary_hidden = None
        boundary_key = None
        if use_thd_cp:
            d_window = self._dsv4_cp_boundary_window()
            _cp_debug_trace(f"boundary hidden exchange start D={d_window}")
            boundary_hidden = exchange_left_boundary_tensor(
                hidden_states, d_window, self.pg_collection.cp
            )
            _cp_debug_trace("boundary hidden exchange done")

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value, q_compressed, kv_compressed = self.get_query_key_value_tensors(
            qkv_hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
            force_full_thd_positions=use_full_context_qkv,
        )
        if use_full_context_qkv:
            full_query = query
            full_key = key
            full_value = value
            full_q_compressed = q_compressed
            query = query[local_token_slice]
            key = key[local_token_slice]
            value = value[local_token_slice]
            q_compressed = q_compressed[local_token_slice]
            kv_compressed = kv_compressed[local_token_slice]
        if use_thd_cp:
            _cp_debug_trace("local qkv projection done")
        if use_thd_cp:
            _cp_debug_trace("boundary key projection start")
            if use_full_context_qkv:
                boundary_key = full_key.new_zeros((d_window,) + tuple(full_key.shape[1:]))
                boundary_start = cp_global_start - d_window
                valid_start = max(0, boundary_start)
                if valid_start < cp_global_start:
                    dst_start = valid_start - boundary_start
                    boundary_key[dst_start:] = full_key[valid_start:cp_global_start]
            else:
                boundary_key = self._build_boundary_key_from_hidden(
                    boundary_hidden, packed_seq_params
                )
            _cp_debug_trace("boundary key projection done")

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
            if use_thd_cp:
                _cp_debug_trace("core attention start")
            if use_full_context_qkv:
                saved_cp_group = self.core_attention.pg_collection.cp
                saved_cp_size = saved_cp_group.size()
                had_loss_scale = hasattr(self.core_attention, "_cp_indexer_loss_scale")
                saved_loss_scale = getattr(self.core_attention, "_cp_indexer_loss_scale", None)
                self.core_attention.pg_collection.cp = _SINGLE_RANK_CP_GROUP
                self.core_attention._cp_indexer_loss_scale = 1.0 / saved_cp_size
                try:
                    core_attn_out = self.core_attention(
                        full_query,
                        full_key,
                        full_value,
                        attention_mask,
                        packed_seq_params=packed_seq_params,
                        x=qkv_hidden_states,
                        qr=full_q_compressed,
                        boundary_hidden=None,
                        boundary_key=None,
                    )
                finally:
                    self.core_attention.pg_collection.cp = saved_cp_group
                    if had_loss_scale:
                        self.core_attention._cp_indexer_loss_scale = saved_loss_scale
                    else:
                        delattr(self.core_attention, "_cp_indexer_loss_scale")
                core_attn_out = core_attn_out[local_token_slice]
            else:
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    x=hidden_states,
                    qr=q_compressed,
                    boundary_hidden=boundary_hidden,
                    boundary_key=boundary_key,
                )
            if use_thd_cp:
                _cp_debug_trace("core attention done")
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
            rope_seqlen = packed_seq_params.max_seqlen_kv
        else:
            cu_seqlens_kv = None
            rope_seqlen = seq_len
        # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
        # concentration factor (mscale) is NOT part of the DSv4 model contract --
        # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0.
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        use_contiguous_thd_cp = packed_seq and self.pg_collection.cp.size() > 1
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
        if self.config.apply_rope_fusion and not use_contiguous_thd_cp:
            if packed_seq:
                core_attn_out = core_attn_out.squeeze(1)
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
                remove_interleaving=True,
            )
            if packed_seq:
                core_attn_out = core_attn_out.unsqueeze(1)
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
            if use_contiguous_thd_cp:
                inverse_positions = self._dsv4_cp_local_seq_positions(packed_seq_params, seq_len)
                rot_part_out = _apply_rope_at_positions(
                    rot_part_in.clone().unsqueeze(1),
                    0,
                    pos_dim,
                    self.rotary_pos_emb,
                    self.config,
                    inverse_positions,
                    inverse=True,
                ).squeeze(1)
            else:
                rotary_pos_emb_out = rotary_pos_emb
                cu_seqlens_kv_rope = cu_seqlens_kv
                rot_part_out = apply_rotary_pos_emb(
                    rot_part_in,
                    rotary_pos_emb_out,
                    self.config,
                    cu_seqlens=cu_seqlens_kv_rope,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                    mla_rotary_interleaved=True,
                    inverse=True,
                    mla_output_remove_interleaving=True,
                )
            if packed_seq:
                rot_part = rot_part_out.unsqueeze(1)
            else:
                rot_part = rot_part_out
            core_attn_out = torch.cat([content_part, rot_part], dim=-1)
        core_attn_out = core_attn_out.view(seq_len, core_attn_out.size(1), -1)
        if use_full_context_qkv:
            _cp_debug_trace("full-context output projection all_gather start")
            core_attn_out = all_gather_fixed_cp_tensor(core_attn_out, self.pg_collection.cp)
            seq_len = core_attn_out.size(0)
            _cp_debug_trace("full-context output projection all_gather done")

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
        if use_full_context_qkv:
            output = output[local_token_slice]

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
        force_full_thd_positions: bool = False,
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
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        # q_compressed: [s, b, q_lora_rank]
        q_compressed, _ = self.linear_q_down_proj(hidden_states)

        kv_compressed = hidden_states
        k_pos_emb = None

        if packed_seq_params is not None:
            # If sequence packing, TE expect [t, h, d] shaped qkv input.
            # In Megatron-Core, the qkv shape is [t, 1, h, d].
            # So we need to reshape qkv from [t, 1, h, d] to [t, h, d].
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
        explicit_cp_positions = None
        if packed_seq and self.pg_collection.cp.size() > 1 and not force_full_thd_positions:
            explicit_cp_positions = self._dsv4_cp_local_seq_positions(
                packed_seq_params, kv_compressed.shape[0]
            )
        rope_cp_group = _SINGLE_RANK_CP_GROUP if force_full_thd_positions else self.pg_collection.cp

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

            kv, _ = self.linear_kv_proj(kv_compressed)
            kv = self.kv_layernorm(kv)

            # [num_tokens, qk_pos_emb_head_dim] -> [num_tokens, 1, qk_pos_emb_head_dim]
            if k_pos_emb is not None:
                k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            if self.config.apply_rope_fusion and explicit_cp_positions is None:
                cp_rank = 0 if force_full_thd_positions else self.pg_collection.cp.rank()
                cp_size = 1 if force_full_thd_positions else self.pg_collection.cp.size()
                query = fused_mla_rope_inplace(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
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
                    rotary_pos_cos,
                    rotary_pos_sin,
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
                q_len = q.size()[0]
                if explicit_cp_positions is not None:
                    query = _apply_rope_at_positions(
                        q,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        self.rotary_pos_emb,
                        self.config,
                        explicit_cp_positions,
                    )
                    kv = _apply_rope_at_positions(
                        kv.unsqueeze(-2),
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        self.rotary_pos_emb,
                        self.config,
                        explicit_cp_positions,
                    )
                    key = kv
                    value = kv
                else:
                    rotary_pos_emb_q = rotary_pos_emb
                    rotary_pos_emb_k = rotary_pos_emb
                    cu_seqlens_q_rope = cu_seqlens_q
                    cu_seqlens_kv_rope = cu_seqlens_kv

                    if packed_seq_params is None or self.config.context_parallel_size == 1:
                        # Shorten rotary_pos_emb to the sequence length when inference_params
                        # is not provided. This makes sure we can run forward directly with
                        # any sequence length. During training, the sequence length is always
                        # the full rotary_pos_emb length, except for sequence packing + CP.
                        # When sequence packing and context parallel are both enabled, the
                        # position embedding will not split rotary_pos_emb, so it may exceed
                        # the sequence length on this CP rank, but we need the full rotary_pos_emb
                        # to cover the full sequence, so we do not shorten it here.
                        rotary_pos_emb_q = rotary_pos_emb_q[0:q_len]
                        rotary_pos_emb_k = rotary_pos_emb_k[0:q_len]

                    # q_no_pe: [num_tokens, n, qk_head_dim]
                    # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                    q_no_pe, q_pos_emb = torch.split(
                        q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                    )

                    # RoPE and query (shared for wkv and latent)
                    # q_pos_emb: [num_tokens, n, qk_pos_emb_head_dim]
                    q_pos_emb = apply_rotary_pos_emb(
                        q_pos_emb,
                        rotary_pos_emb_q,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q_rope,
                        mscale=mscale,
                        cp_group=rope_cp_group,
                        mla_rotary_interleaved=True,
                        mla_output_remove_interleaving=True,
                    )
                    # query: [num_tokens, n, (qk_head_dim + v_head_dim)]
                    query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

                    pos_dim = self.config.qk_pos_emb_head_dim
                    kv_no_pe, k_pos_emb = torch.split(kv, [kv.size(-1) - pos_dim, pos_dim], dim=-1)

                    # k_pos_emb:[num_tokens, 1, qk_pos_emb_head_dim]
                    k_pos_emb = apply_rotary_pos_emb(
                        k_pos_emb,
                        rotary_pos_emb_k,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv_rope,
                        mscale=mscale,
                        cp_group=rope_cp_group,
                        mla_rotary_interleaved=True,
                        mla_output_remove_interleaving=True,
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

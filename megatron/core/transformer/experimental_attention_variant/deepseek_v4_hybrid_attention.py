# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import warnings
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, NoReturn, Optional, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
)
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
from megatron.core.transformer.experimental_attention_variant import csa_cp_utils as cp_utils
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import get_pg_size, is_te_min_version

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TELinear, set_save_original_input
else:
    (TEColumnParallelLinear, TELinear, set_save_original_input) = (None, None, None)


@torch.compile
def _q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused RMS normalization for query tensor (no learnable weight)."""
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)


def _initialize_grouped_output_projection(
    weight: torch.Tensor, *, init_method: Callable[[torch.Tensor], None]
) -> None:
    """Initialize a grouped weight with the legacy 2D parameter layout."""
    init_method(weight.view(-1, weight.size(-1)))


@dataclass
class DSv4HybridSelfAttentionSubmodules:
    """Submodules for the DSv4HybridAttention layer."""

    q_layernorm: LayerNormBuilder
    kv_layernorm: LayerNormBuilder

    linear_q_down_proj: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_o_group_proj: Union[ModuleSpec, type] = None
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
        self.o_local_groups = self.config.o_groups
        assert (
            self.query_projection_size % self.config.o_groups == 0
        ), "num_attention_heads * v_head_dim must be divisible by o_groups"
        group_proj_in_size = self.query_projection_size // self.config.o_groups
        device = torch.device("cuda", torch.cuda.current_device())
        self._uses_te_batched_linear = submodules.linear_o_group_proj is not None
        if not self._uses_te_batched_linear:
            warnings.warn(
                "transformer_engine.pytorch.BatchedLinear is unavailable. Please upgrade "
                "Transformer Engine to avoid a performance regression; "
                "DSv4HybridAttention.linear_o_group_proj is falling back to torch.einsum.",
                stacklevel=2,
            )
            group_proj_out_size = self.config.o_groups * self.config.o_lora_rank
            linear_o_group_proj = torch.empty(
                group_proj_out_size,
                group_proj_in_size,
                device=device,
                dtype=self.config.params_dtype,
            )
            self.config.init_method(linear_o_group_proj)
            linear_o_group_proj = linear_o_group_proj.view(
                self.o_local_groups, self.config.o_lora_rank, group_proj_in_size
            )
            self.linear_o_group_proj = torch.nn.Parameter(linear_o_group_proj)
        else:
            self.linear_o_group_proj = build_module(
                submodules.linear_o_group_proj,
                self.o_local_groups,
                group_proj_in_size,
                self.config.o_lora_rank,
                batch_dim=-2,
                device=device,
                params_dtype=self.config.params_dtype,
                bias=False,
                accumulate_into_main_grad=self.config.gradient_accumulation_fusion,
                init_method=partial(
                    _initialize_grouped_output_projection, init_method=self.config.init_method
                ),
                name=(name + ".linear_o_group_proj") if name is not None else None,
            )
        self._register_state_dict_hook(self._linear_o_group_proj_state_dict_hook)

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

    @property
    def _linear_o_group_proj_weight(self) -> torch.nn.Parameter:
        """Return the grouped output-projection weight from either implementation."""
        if self._uses_te_batched_linear:
            return self.linear_o_group_proj.weight
        return self.linear_o_group_proj

    def _linear_o_group_proj_runtime_key(self, prefix: str) -> str:
        """State-dict key used by the initialized runtime implementation."""
        key = f"{prefix}linear_o_group_proj"
        return f"{key}.weight" if self._uses_te_batched_linear else key

    @property
    def _linear_o_group_proj_checkpoint_shape(self) -> tuple[int, int]:
        """Legacy 2D checkpoint shape for the grouped output projection."""
        return (
            self.o_local_groups * self.config.o_lora_rank,
            self._linear_o_group_proj_weight.size(-1),
        )

    def _linear_o_group_proj_state_dict_hook(
        self, module, state_dict, prefix, local_metadata
    ) -> None:
        """Keep the grouped output projection in its legacy 2D checkpoint layout."""
        assert module is self
        checkpoint_key = f"{prefix}linear_o_group_proj"
        runtime_key = self._linear_o_group_proj_runtime_key(prefix)
        state_dict[checkpoint_key] = state_dict.pop(runtime_key).reshape(
            self._linear_o_group_proj_checkpoint_shape
        )
        if self._uses_te_batched_linear:
            state_dict.pop(f"{prefix}linear_o_group_proj._extra_state", None)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Accept both legacy 2D and runtime 3D grouped projection weights."""
        checkpoint_key = f"{prefix}linear_o_group_proj"
        runtime_key = self._linear_o_group_proj_runtime_key(prefix)
        if checkpoint_key != runtime_key and checkpoint_key in state_dict:
            checkpoint_weight = state_dict.pop(checkpoint_key)
            state_dict.setdefault(runtime_key, checkpoint_weight)
        checkpoint_weight = state_dict.get(runtime_key)
        if (
            isinstance(checkpoint_weight, torch.Tensor)
            and tuple(checkpoint_weight.shape) == self._linear_o_group_proj_checkpoint_shape
        ):
            state_dict[runtime_key] = checkpoint_weight.reshape(
                self._linear_o_group_proj_weight.shape
            )
        if self._uses_te_batched_linear:
            state_dict.setdefault(f"{prefix}linear_o_group_proj._extra_state", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Preserve the legacy 2D layout in distributed checkpoints."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        checkpoint_key = f"{prefix}linear_o_group_proj"
        runtime_key = self._linear_o_group_proj_runtime_key(prefix)
        runtime_sharded_weight = sharded_state_dict.pop(runtime_key)
        if self._uses_te_batched_linear:
            sharded_state_dict.pop(f"{prefix}linear_o_group_proj._extra_state", None)
        assert isinstance(runtime_sharded_weight, ShardedTensor)
        runtime_shape = runtime_sharded_weight.local_shape
        assert len(runtime_shape) == 3, runtime_shape

        prepend_axis_num = runtime_sharded_weight.prepend_axis_num
        global_prefix = runtime_sharded_weight.global_shape[:prepend_axis_num]
        offset_prefix = runtime_sharded_weight.global_offset[:prepend_axis_num]
        global_group_shape = runtime_sharded_weight.global_shape[prepend_axis_num:]
        global_group_offset = runtime_sharded_weight.global_offset[prepend_axis_num:]
        assert len(global_group_shape) == len(global_group_offset) == 3

        local_groups, local_rank, local_in_features = runtime_shape
        global_groups, global_rank, global_in_features = global_group_shape
        group_offset, rank_offset, in_features_offset = global_group_offset
        assert global_rank == local_rank and rank_offset == 0, (
            "The grouped output projection may only be sharded along its group "
            "or input-feature dimensions"
        )

        checkpoint_local_shape = (local_groups * local_rank, local_in_features)
        checkpoint_global_shape = (*global_prefix, global_groups * global_rank, global_in_features)
        checkpoint_global_offset = (*offset_prefix, group_offset * global_rank, in_features_offset)
        checkpoint_axis_fragmentations = runtime_sharded_weight.axis_fragmentations
        if checkpoint_axis_fragmentations is not None:
            fragmentation_prefix = checkpoint_axis_fragmentations[:prepend_axis_num]
            group_fragmentation, rank_fragmentation, in_features_fragmentation = (
                checkpoint_axis_fragmentations[prepend_axis_num:]
            )
            assert rank_fragmentation == 1
            checkpoint_axis_fragmentations = (
                *fragmentation_prefix,
                group_fragmentation,
                in_features_fragmentation,
            )

        checkpoint_sharded_weight_no_data = replace(
            runtime_sharded_weight.without_data(),
            local_shape=checkpoint_local_shape,
            global_shape=checkpoint_global_shape,
            global_offset=checkpoint_global_offset,
            axis_fragmentations=checkpoint_axis_fragmentations,
        )

        @torch.no_grad()
        def build_fn(
            key: str, tensor: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
        ) -> ShardedTensor:
            checkpoint_tensor = tensor.reshape(checkpoint_local_shape)
            return replace(
                checkpoint_sharded_weight_no_data,
                key=key,
                data=checkpoint_tensor,
                dtype=checkpoint_tensor.dtype,
                replica_id=replica_id,
                flattened_range=flattened_range,
            )

        def merge_fn(checkpoint_tensor: torch.Tensor) -> torch.Tensor:
            return checkpoint_tensor.reshape(runtime_shape)

        runtime_weight = runtime_sharded_weight.data
        assert runtime_weight is not None
        sharded_state_dict[checkpoint_key] = ShardedTensorFactory(
            checkpoint_key,
            runtime_weight,
            build_fn,
            merge_fn,
            runtime_sharded_weight.replica_id,
            flattened_range=runtime_sharded_weight.flattened_range,
        )
        return sharded_state_dict

    def _apply_linear_o_group_proj(self, input_: torch.Tensor) -> torch.Tensor:
        """Apply TE BatchedLinear when available, otherwise use the legacy einsum."""
        if self._uses_te_batched_linear:
            return self.linear_o_group_proj(input_)
        weight = self._linear_o_group_proj_weight.reshape(
            self.o_local_groups, self.config.o_lora_rank, -1
        )
        return torch.einsum("...gd,grd->...gr", input_, weight)

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
        cp_group = _orig_cp_group
        if packed_seq_params is not None and packed_seq_params.local_cp_size is not None:
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

        boundary_hidden = None
        if use_thd_cp:
            boundary_hidden = cp_utils.exchange_cp_boundary_hidden(
                hidden_states,
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
        )
        if use_thd_cp:
            query, key, value, q_compressed, kv_compressed, boundary_kv = qkv
        else:
            query, key, value, q_compressed, kv_compressed = qkv
            boundary_kv = None

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
                x=hidden_states,
                qr=q_compressed,
                boundary_hidden=boundary_hidden,
                boundary_kv=boundary_kv,
            )
        forced_released_tensors = [query, key, value]
        if boundary_kv is not None:
            forced_released_tensors.append(boundary_kv)
        core_attn_out = core_attn_manager.group_offload(
            core_attn_out, forced_released_tensors=forced_released_tensors
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
            if use_thd_cp:
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
        elif use_thd_cp:
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
        core_attn_out = self._apply_linear_o_group_proj(core_attn_out)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        # =================
        # Output. [sq, b, h]
        # =================
        attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out, "attn_proj")
        with attn_proj_manager as core_attn_out:
            output, bias = self.linear_proj(core_attn_out)
        output = attn_proj_manager.group_offload(output, forced_released_tensors=[core_attn_out])

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
            name=(name + ".linear_kv_proj") if name is not None else None,
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
        boundary_hidden=None,
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

            # q: [num_tokens, n, q_head_dim]
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)
            q = _q_rms_norm(q, self.config.layernorm_epsilon)

            boundary_rows = 0
            if boundary_kv_compressed is not None:
                boundary_rows = boundary_kv_compressed.shape[0]
                kv_projection_input = torch.cat([boundary_kv_compressed, kv_compressed], dim=0)
            else:
                kv_projection_input = kv_compressed

            kv, _ = self.linear_kv_proj(kv_projection_input)
            kv = self.kv_layernorm(kv)
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
                        rotary_pos_cos,
                        rotary_pos_sin,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        global_start,
                    )
                    kv = kv.unsqueeze(-2)
                    kv = cp_utils.apply_thd_cp_local_rope_fused(
                        kv,
                        rotary_pos_cos,
                        rotary_pos_sin,
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
                if packed_seq and cp_size > 1:
                    global_start = cp_group.rank() * q.shape[0]
                    query = cp_utils.apply_thd_cp_local_rope_unfused(
                        q,
                        rotary_pos_emb,
                        self.config.qk_head_dim,
                        self.config.qk_pos_emb_head_dim,
                        cu_seqlens_q,
                        global_start,
                        self.config,
                    )
                    kv = cp_utils.apply_thd_cp_local_rope_unfused(
                        kv.unsqueeze(-2),
                        rotary_pos_emb,
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
                    # is not provided so direct forward accepts any sequence length.
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
                        rotary_pos_emb,
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

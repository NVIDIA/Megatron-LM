# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental MLA self-attention that exchanges latent KV over a P2P CP ring."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import megatron.core.tensor_parallel as mcore_tp
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import mappings as tp_mappings
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module

from . import layout as latent_cp_layout
from . import utils as latent_cp_utils
from .backend import DirectAttentionAdapter, _qualified_backend_adapter
from .layout import AlreadyZigZagTHDAdapter
from .transport import LatentCPTransport, P2PRingTransport
from .utils import LatentCPError, QualifiedBackendTuple, _require

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TELinear,
        TERowParallelLinear,
    )
else:
    TEColumnParallelLinear = None
    TELayerNormColumnParallelLinear = None
    TELinear = None
    TERowParallelLinear = None


def _build_local_latent_norm(
    *, config: MLATransformerConfig, hidden_size: int, eps: float
) -> torch.nn.Module:
    """Build a token-local norm while preserving SP parameter-gradient synchronization."""

    norm_config = copy.copy(config)
    norm_config.sequence_parallel = False
    norm = WrappedTorchNorm(config=norm_config, hidden_size=hidden_size, eps=eps)
    if config.sequence_parallel:
        for parameter in norm.parameters():
            setattr(parameter, "sequence_parallel", True)
    return norm


def _validate_supported_submodules(submodules: MLASelfAttentionSubmodules) -> str:
    """Return the preserved projection-stack kind after fail-closed validation."""

    expected_column = (
        "linear_q_proj",
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_down_proj",
        "linear_kv_up_proj",
    )
    local_stack = (
        all(
            getattr(submodules, name) is ColumnParallelLinear
            for name in expected_column
        )
        and submodules.linear_proj is RowParallelLinear
        and submodules.linear_gate in (None, ColumnParallelLinear)
        and submodules.q_layernorm is _build_local_latent_norm
        and submodules.kv_layernorm is _build_local_latent_norm
    )
    te_stack = bool(HAVE_TE) and (
        submodules.linear_q_proj is TEColumnParallelLinear
        and submodules.linear_q_down_proj is TELinear
        and submodules.linear_q_up_proj is TELayerNormColumnParallelLinear
        and submodules.linear_kv_down_proj is TELinear
        and submodules.linear_kv_up_proj is TELayerNormColumnParallelLinear
        and submodules.linear_proj is TERowParallelLinear
        and submodules.linear_gate in (None, TEColumnParallelLinear)
        and submodules.q_layernorm is IdentityOp
        and submodules.kv_layernorm is IdentityOp
    )
    _require(
        local_stack or te_stack,
        "projection modules must preserve either the supported local MCore stack or the "
        "supported Transformer Engine MLA stack",
    )
    _require(
        submodules.linear_qkv_down_proj is None,
        "fused MLA down projection is unsupported",
    )
    _require(
        submodules.core_attention is IdentityOp,
        "core_attention must be IdentityOp; use make_mla_with_latent_cp_spec",
    )
    return "local" if local_stack else "transformer_engine"


@dataclass(frozen=True)
class _MicrobatchLayoutKey:
    """Host-only identity for one immutable packed microbatch layout."""

    packed_id: int
    cu_q_id: int
    cu_kv_id: int
    cp_group_id: int
    local_tokens: int
    device: torch.device
    max_q: int | None
    max_kv: int | None
    local_cp_size: int | None
    partition_mode: str


class MLAWithLatentCP(MLASelfAttention):
    """Training-only THD MLA whose P2P CP ring exchanges normalized latent KV."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        _require(
            pg_collection is not None, "an explicit ProcessGroupCollection is required"
        )
        _require(
            hasattr(pg_collection, "tp")
            and pg_collection.tp is not None
            and hasattr(pg_collection, "cp")
            and pg_collection.cp is not None,
            "explicit non-null TP and CP process groups are required",
        )
        # TODO(mla-latent-cp): Support MTP before this feature leaves experimental status.
        _require(not is_mtp_layer, "MTP layers are unsupported in v1")
        projection_stack = _validate_supported_submodules(submodules)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        self._cp_comm_type = (
            cp_comm_type if cp_comm_type is not None else config.cp_comm_type
        )
        self._projection_stack = projection_stack
        self._layout_adapter: latent_cp_layout.LatentCPLayoutAdapter = (
            AlreadyZigZagTHDAdapter()
        )
        self._parameter_dtypes_validated = False
        self._validate_initial_config()
        self._validate_projection_groups()
        self._backend_adapter: DirectAttentionAdapter
        self._backend_runtime_tuple: QualifiedBackendTuple
        self._backend_adapter, self._backend_runtime_tuple = _qualified_backend_adapter(
            self.config.attention_backend
        )
        self._prepared_microbatch: (
            tuple[
                _MicrobatchLayoutKey, dist.ProcessGroup, latent_cp_layout.ZigZagLayout
            ]
            | None
        ) = None

    def _validate_initial_config(self) -> None:
        config = self.config
        tp_size = dist.get_world_size(self.pg_collection.tp)
        cp_size = dist.get_world_size(self.pg_collection.cp)
        _require(
            isinstance(config, MLATransformerConfig), "MLATransformerConfig is required"
        )
        _require(
            config.multi_latent_attention, "multi_latent_attention=True is required"
        )
        _require(config.mla_latent_cp, "mla_latent_cp=True is required")
        _require(config.qk_layernorm, "standalone Q/KV layer norms must be enabled")
        _require(
            not config.add_bias_linear, "all MLA projection biases must be disabled"
        )
        _require(
            config.rotary_percent == 1.0, "partial rotary dimensions are unsupported"
        )
        _require(
            self.attn_mask_type is AttnMaskType.causal,
            "only causal self-attention is supported",
        )
        _require(
            cp_size == 1 or self._cp_comm_type == "p2p",
            "CP>1 requires cp_comm_type='p2p'",
        )
        _require(
            config.tensor_model_parallel_size == tp_size,
            "configured TP size disagrees with the injected TP group",
        )
        _require(
            config.context_parallel_size == cp_size,
            "configured CP size disagrees with the injected CP group",
        )
        _require(config.q_lora_rank is not None, "a nonzero q_lora_rank is required")
        _require(config.q_lora_rank > 0, "q_lora_rank must be positive")
        _require(
            config.normalization == "RMSNorm",
            "only RMSNorm projection specs are supported",
        )
        _require(
            tp_size == 1 or config.sequence_parallel,
            "TP>1 requires sequence_parallel=True",
        )
        _require(
            config.num_attention_heads % tp_size == 0, "attention heads must divide TP"
        )
        _require(
            config.num_query_groups == config.num_attention_heads,
            "v1 requires Hq=Hkv (no GQA)",
        )
        _require(
            config.qk_head_dim == 128
            and config.qk_pos_emb_head_dim == 64
            and config.v_head_dim == 128,
            "v1 requires qk content/rope/value dimensions 128/64/128",
        )
        _require(
            config.rope_type in ("rope", "yarn"), "only rope and yarn are supported"
        )
        _require(
            not self.use_rope or not config.apply_rope_fusion,
            "fused RoPE is unsupported when this layer applies RoPE",
        )
        _require(config.attention_dropout == 0.0, "attention dropout must be zero")
        _require(config.bf16 and not config.fp16, "v1 requires BF16 and rejects FP16")
        # TODO(mla-latent-cp): Qualify FP8/FP4, including MXFP8, before merge.
        _require(
            config.fp8 is None and config.fp4 is None, "FP8 and FP4 are unsupported"
        )
        _require(
            not config.cache_mla_latents, "inference latent caching is unsupported"
        )
        # TODO(mla-latent-cp): Define nested-checkpoint semantics for outer/selective
        # recompute before merge.
        _require(
            config.recompute_granularity is None,
            "outer/selective recompute is unsupported; disable recompute_granularity",
        )
        # TODO(mla-latent-cp): Add saved-tensor/offload coverage before enabling
        # fine-grained activation offload.
        _require(
            not config.fine_grained_activation_offloading,
            "fine-grained activation offload is unsupported",
        )
        _require(
            not config.cpu_offloading and config._cpu_offloading_context is None,
            "CPU offloading is unsupported",
        )
        _require(
            config.cuda_graph_impl == "none"
            and not config.enable_cuda_graph
            and not config.external_cuda_graph,
            "CUDA graph execution is unsupported",
        )
        _require(
            config.attention_backend in (AttnBackend.fused, AttnBackend.flash),
            "attention_backend must be fused (cuDNN) or flash (FA4)",
        )

    def _validate_projection_groups(self) -> None:
        def stored_tp_group(module: torch.nn.Module) -> dist.ProcessGroup | None:
            group = getattr(module, "tp_group", None)
            return group if group is not None else getattr(module, "_tp_group", None)

        for name in (
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
            "linear_proj",
        ):
            module = getattr(self, name)
            _require(
                stored_tp_group(module) is self.pg_collection.tp,
                f"{name} does not retain the injected TP process group",
            )

        if self.linear_gate is not None:
            _require(
                stored_tp_group(self.linear_gate) is self.pg_collection.tp,
                "linear_gate does not retain the injected TP process group",
            )
            if self._projection_stack == "local":
                _require(
                    isinstance(self.linear_gate, ColumnParallelLinear)
                    and not self.linear_gate.gather_output
                    and not self.linear_gate.skip_bias_add
                    and self.linear_gate.bias is None
                    and not self.linear_gate.explicit_expert_comm,
                    "linear_gate must be a bias-free non-expert sharded projection",
                )
            else:
                _require(
                    not self.linear_gate.use_bias,
                    "TE linear_gate must be a bias-free column-parallel projection",
                )

        output_projection = self.linear_proj
        if self._projection_stack == "local":
            _require(
                output_projection.input_is_parallel
                and output_projection.skip_bias_add
                and output_projection.sequence_parallel == self.config.sequence_parallel
                and not output_projection.explicit_expert_comm
                and output_projection.bias is None,
                "linear_proj must be a bias-free non-expert row-parallel projection",
            )
        else:
            _require(
                output_projection.sequence_parallel == self.config.sequence_parallel
                and not output_projection.use_bias,
                "TE linear_proj must be a bias-free row-parallel projection",
            )

    def _validate_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        key_value_states: Tensor | None,
        inference_context: Any,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None,
        rotary_pos_cos: Tensor | None,
        rotary_pos_sin: Tensor | None,
        rotary_pos_cos_sin: Tensor | None,
        attention_bias: Tensor | None,
        packed_seq_params: PackedSeqParams | None,
        position_ids: Tensor | None,
        sequence_len_offset: int | None,
        inference_params: Any,
    ) -> PackedSeqParams:
        _require(self.training, "v1 is training-only")
        _require(
            hidden_states.ndim == 3, "hidden_states must have shape [T, 1, hidden]"
        )
        _require(hidden_states.size(1) == 1, "THD requires the singleton batch axis")
        _require(hidden_states.is_cuda, "activations must be CUDA tensors")
        _require(
            hidden_states.device.index == torch.cuda.current_device(),
            "hidden_states must use the current CUDA device",
        )
        _require(hidden_states.dtype == torch.bfloat16, "activations must be BF16")
        _require(attention_mask is None, "explicit attention masks are unsupported")
        _require(key_value_states is None, "cross attention is unsupported")
        _require(
            inference_context is None and inference_params is None,
            "inference is unsupported",
        )
        _require(rotary_pos_emb is None, "external rotary_pos_emb is unsupported")
        _require(
            rotary_pos_cos is None
            and rotary_pos_sin is None
            and rotary_pos_cos_sin is None,
            "flash-decoding/fused rotary inputs are unsupported",
        )
        _require(attention_bias is None, "attention bias is unsupported")
        _require(position_ids is None, "external position_ids are unsupported")
        _require(sequence_len_offset is None, "sequence offsets are unsupported")
        _require(
            isinstance(packed_seq_params, PackedSeqParams),
            "PackedSeqParams is required",
        )
        local_tokens = hidden_states.size(0)
        if self.config.sequence_parallel:
            local_tokens *= dist.get_world_size(self.pg_collection.tp)
        _require(
            packed_seq_params.total_tokens in (None, local_tokens),
            "total_tokens must equal the pre-SP local THD token count",
        )
        _require(
            packed_seq_params.pad_between_seqs in (None, False),
            "inter-sequence padding is unsupported",
        )
        if not self._parameter_dtypes_validated:
            for name, parameter in self.named_parameters():
                if parameter.is_floating_point():
                    _require(
                        parameter.dtype == torch.bfloat16,
                        f"parameter {name} must be BF16",
                    )
            self._parameter_dtypes_validated = True
        return packed_seq_params

    def _microbatch_layout_key(
        self,
        hidden_states: Tensor,
        packed_seq_params: PackedSeqParams,
        cp_group: dist.ProcessGroup,
    ) -> _MicrobatchLayoutKey:
        cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv
        if cu_kv is None:
            cu_kv = cu_q
        return _MicrobatchLayoutKey(
            packed_id=id(packed_seq_params),
            cu_q_id=id(cu_q),
            cu_kv_id=id(cu_kv),
            cp_group_id=id(cp_group),
            local_tokens=hidden_states.size(0),
            device=hidden_states.device,
            max_q=packed_seq_params.max_seqlen_q,
            max_kv=packed_seq_params.max_seqlen_kv,
            local_cp_size=packed_seq_params.local_cp_size,
            partition_mode=packed_seq_params.cp_partition_mode,
        )

    def _microbatch_layout(
        self, hidden_states: Tensor, packed_seq_params: PackedSeqParams
    ) -> tuple[dist.ProcessGroup, latent_cp_layout.ZigZagLayout]:
        """Return a preprocessed layout or build one for direct module callers."""

        cp_group = resolve_cp_group(self.pg_collection.cp, packed_seq_params)
        key = self._microbatch_layout_key(hidden_states, packed_seq_params, cp_group)
        if self._prepared_microbatch is not None:
            prepared_key, prepared_group, prepared_layout = self._prepared_microbatch
            if prepared_key == key:
                return prepared_group, prepared_layout
        layout = self._layout_adapter.prepare(
            hidden_states,
            packed_seq_params,
            cp_group,
            tp_group=self.pg_collection.tp,
            sequence_parallel=self.config.sequence_parallel,
        )
        self._prepared_microbatch = (key, cp_group, layout)
        return cp_group, layout

    def _preprocess_backend(
        self,
        hidden_states: Tensor,
        packed_seq_params: PackedSeqParams,
        prepared_layout: tuple[dist.ProcessGroup, latent_cp_layout.ZigZagLayout]
        | None = None,
    ) -> None:
        """Retain the shared layout and prepare expensive backend bindings."""

        _require(
            isinstance(packed_seq_params, PackedSeqParams),
            "PackedSeqParams is required",
        )
        if prepared_layout is None:
            cp_group, layout = self._microbatch_layout(hidden_states, packed_seq_params)
        else:
            cp_group, layout = prepared_layout
            key = self._microbatch_layout_key(
                hidden_states, packed_seq_params, cp_group
            )
            self._prepared_microbatch = (key, cp_group, layout)
        self._backend_adapter.prepare(
            num_heads=self.num_attention_heads_per_partition,
            qk_dim=self.q_head_dim,
            v_dim=self.config.v_head_dim,
            phases=layout.phases,
            scale=self.softmax_scale,
        )

    def _explicit_output_projection(
        self, core_output: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Apply the preserved old-path output module, with an explicit local fallback."""

        projection = self.linear_proj
        _require(core_output.dtype == torch.bfloat16, "linear_proj input must be BF16")
        _require(
            not self.config.cpu_offloading
            and self.config._cpu_offloading_context is None,
            "CPU offloading is unsupported",
        )
        _require(
            projection.weight.requires_grad,
            "frozen linear_proj weights are unsupported in v1",
        )
        if self._projection_stack == "transformer_engine":
            return apply_module(projection)(core_output)

        _require(projection.bias is None, "linear_proj bias is unsupported")
        output_parallel = mcore_tp.linear_with_grad_accumulation_and_async_allreduce(
            input=core_output,
            weight=projection.weight,
            bias=None,
            gradient_accumulation_fusion=projection.gradient_accumulation_fusion,
            allreduce_dgrad=False,
            sequence_parallel=False,
            grad_output_buffer=None,
            wgrad_deferral_limit=0,
            tp_group=self.pg_collection.tp,
        )
        if self.config.sequence_parallel:
            output = mcore_tp.reduce_scatter_to_sequence_parallel_region(
                output_parallel, group=self.pg_collection.tp
            )
        else:
            output = mcore_tp.reduce_from_tensor_model_parallel_region(
                output_parallel, group=self.pg_collection.tp
            )
        return output, None

    def _latent_cp_down_projection(
        self, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor, bool, bool]:
        """Run the preserved down projections and explicitly gather feature shards."""

        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        expected_kv = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        q_was_sharded = q_compressed.size(-1) != self.config.q_lora_rank
        kv_was_sharded = kv_combined.size(-1) != expected_kv
        if q_was_sharded:
            q_compressed = tp_mappings.gather_from_tensor_model_parallel_region(
                q_compressed, group=self.pg_collection.tp
            )
        if kv_was_sharded:
            kv_combined = tp_mappings.gather_from_tensor_model_parallel_region(
                kv_combined, group=self.pg_collection.tp
            )
        _require(
            q_compressed.size(-1) == self.config.q_lora_rank,
            "Q down-projection gather produced the wrong size",
        )
        _require(
            kv_combined.size(-1) == expected_kv,
            "KV down-projection gather produced the wrong size",
        )
        return q_compressed, kv_combined, q_was_sharded, kv_was_sharded

    def _project_query_and_payload(
        self,
        hidden_states: Tensor,
        packed_seq_params: PackedSeqParams,
        layout: latent_cp_layout.ZigZagLayout,
        cp_group: dist.ProcessGroup | None = None,
    ) -> tuple[Tensor, Tensor]:
        if cp_group is None:
            cp_group = self.pg_collection.cp
        q_compressed, kv_combined, q_was_sharded, kv_was_sharded = (
            self._latent_cp_down_projection(hidden_states)
        )
        q_compressed = q_compressed.squeeze(1)
        kv_combined = kv_combined.squeeze(1)
        kv_compressed, k_rope_raw = torch.split(
            kv_combined,
            [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim],
            dim=-1,
        )
        if self.config.sequence_parallel and q_was_sharded:
            q_compressed = tp_mappings.scatter_to_sequence_parallel_region(
                q_compressed, group=self.pg_collection.tp
            )
        if self.config.sequence_parallel and kv_was_sharded:
            kv_compressed = tp_mappings.scatter_to_sequence_parallel_region(
                kv_compressed, group=self.pg_collection.tp
            )
        elif self.config.sequence_parallel:
            k_rope_raw = tp_mappings.gather_from_sequence_parallel_region(
                # The phase gather supplies the TP gradient sum; this pre-RoPE
                # gather only restores rows, so its backward must only split.
                k_rope_raw,
                tensor_parallel_output_grad=False,
                group=self.pg_collection.tp,
            )
        q_compressed = self.q_layernorm(q_compressed)
        kv_compressed = self.kv_layernorm(kv_compressed)

        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(q.size(0), self.num_attention_heads_per_partition, self.q_head_dim)
        q_content, q_rope = torch.split(
            q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )
        k_rope = k_rope_raw.unsqueeze(1)

        if self.use_rope:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                None, None, hidden_states, self.config, packed_seq_params
            )
            mscale = 1.0
            if self.config.rope_type == "rope":
                rotary = self.rotary_pos_emb(
                    rotary_seq_len, packed_seq=True, cp_group=cp_group
                )
            else:
                rotary, mscale = self.rotary_pos_emb(
                    rotary_seq_len, packed_seq=True, cp_group=cp_group
                )
            q_rope = apply_rotary_pos_emb(
                q_rope,
                rotary,
                config=self.config,
                cu_seqlens=layout.cu_global,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=True,
                max_seqlen=layout.max_global,
            )
            k_rope = apply_rotary_pos_emb(
                k_rope,
                rotary,
                config=self.config,
                cu_seqlens=layout.cu_global,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=True,
                max_seqlen=layout.max_global,
            )
        if self.config.sequence_parallel:
            k_rope = tp_mappings.scatter_to_sequence_parallel_region(
                k_rope, group=self.pg_collection.tp
            )
        query = torch.cat((q_content, q_rope), dim=-1).contiguous()
        payload = torch.cat((kv_compressed, k_rope.squeeze(1)), dim=-1).contiguous()
        _require(query.dtype == torch.bfloat16, "query projection must remain BF16")
        _require(
            payload.dtype == torch.bfloat16, "latent ring payload must remain BF16"
        )
        return query, payload

    @staticmethod
    def _phase_rows(
        tensor: Tensor,
        indices: Tensor,
        row_slice: tuple[int, int] | None,
    ) -> Tensor:
        """Select phase rows as a view when the packed layout is contiguous."""

        if row_slice is None:
            return tensor.index_select(0, indices)
        start, stop = row_slice
        return tensor[start:stop]

    def _expand_phase_kv(
        self,
        payload: Tensor,
        phase: latent_cp_layout.PhaseSpec,
    ) -> tuple[Tensor, Tensor]:
        """Reconstruct one phase's full K/V from the communicated latent payload."""

        latent, k_rope = torch.split(
            payload, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        latent = latent.contiguous()
        k_rope = k_rope.contiguous()
        expanded, _ = self.linear_kv_up_proj(latent)
        if self.config.sequence_parallel:
            k_rope = tp_mappings.gather_from_sequence_parallel_region(
                k_rope, tensor_parallel_output_grad=True, group=self.pg_collection.tp
            )
        expanded = expanded.view(
            expanded.size(0),
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )
        expanded = self._phase_rows(expanded, phase.kv_indices, phase.kv_slice)
        k_rope = self._phase_rows(k_rope, phase.kv_indices, phase.kv_slice)
        k_content, value = torch.split(
            expanded, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
        )
        key = torch.cat(
            (
                k_content,
                k_rope.unsqueeze(1).expand(
                    -1, self.num_attention_heads_per_partition, -1
                ),
            ),
            dim=-1,
        ).contiguous()
        return key, value.contiguous()

    def _phase_attention(
        self,
        query: Tensor,
        payload: Tensor,
        phase: latent_cp_layout.PhaseSpec,
        backend: DirectAttentionAdapter,
    ) -> tuple[Tensor, Tensor]:
        key, value = self._expand_phase_kv(payload, phase)
        output, lse = backend.forward_phase(
            query.contiguous(),
            key,
            value,
            phase.cu_seqlens_q,
            phase.cu_seqlens_kv,
            phase.max_seqlen_q,
            phase.max_seqlen_kv,
            phase.causal,
            self.softmax_scale,
        )
        _require(output.dtype == torch.float32, "backend canonical output must be FP32")
        _require(lse.dtype == torch.float32, "backend canonical LSE must be FP32")
        return output, lse

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        key_value_states: Tensor | None = None,
        inference_context: Any = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        position_ids: Tensor | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: Any = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run latent-P2P context-parallel MLA for one packed THD input."""
        packed_seq_params = self._validate_forward(
            hidden_states,
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            packed_seq_params,
            position_ids,
            sequence_len_offset,
            inference_params,
        )
        effective_cp_group, layout = self._microbatch_layout(
            hidden_states, packed_seq_params
        )
        backend = self._backend_adapter
        query, local_payload = self._project_query_and_payload(
            hidden_states, packed_seq_params, layout, effective_cp_group
        )
        transport: LatentCPTransport = P2PRingTransport(effective_cp_group)

        merged_output: Tensor | None = None
        merged_lse: Tensor | None = None
        lease_count = 0
        leases = transport.iter_payloads(local_payload, layout.phases)
        recomputed_forward = getattr(backend, "forward_recomputed_phase", None)
        projection_parameters: tuple[Tensor, ...] = ()
        if recomputed_forward is not None:
            projection_parameters = tuple(
                parameter
                for parameter in self.linear_kv_up_proj.parameters()
                if parameter.requires_grad
            )
            _require(
                projection_parameters,
                "latent-KV up projection has no trainable parameters",
            )
        for phase, lease in zip(layout.phases, leases, strict=True):
            lease_count += 1
            _require(
                lease.owner == phase.owner, "transport owner order disagrees with plan"
            )
            q_phase = self._phase_rows(query, phase.q_indices, phase.q_slice)
            payload_phase = lease.tensor

            def run_phase(
                q_input: Tensor,
                payload_input: Tensor,
                phase_spec: latent_cp_layout.PhaseSpec = phase,
                phase_backend: DirectAttentionAdapter = backend,
            ) -> tuple[Tensor, Tensor]:
                return self._phase_attention(
                    q_input, payload_input, phase_spec, phase_backend
                )

            if recomputed_forward is None:
                partial_output, partial_lse = checkpoint(
                    run_phase,
                    q_phase,
                    payload_phase,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                partial_output, partial_lse = recomputed_forward(
                    q_phase,
                    payload_phase,
                    phase,
                    self.softmax_scale,
                    self._expand_phase_kv,
                    *projection_parameters,
                )
            if merged_output is None:
                _require(
                    phase.scatter_indices is None,
                    "the first phase must cover every local query row",
                )
                merged_output, merged_lse = partial_output, partial_lse
            else:
                merged_output, merged_lse = (
                    latent_cp_utils.merge_attention_partial_rows(
                        merged_output,
                        merged_lse,
                        partial_output,
                        partial_lse,
                        phase.scatter_indices,
                        phase.scatter_slice,
                    )
                )

        _require(
            lease_count == len(layout.phases),
            "transport did not yield exactly one lease per CP phase",
        )

        if merged_output is None:
            raise LatentCPError(
                "zigzag phase plan unexpectedly produced no attention output"
            )
        # This is the one and only post-backend FP32-to-BF16 cast.
        core_output = merged_output.to(torch.bfloat16).reshape(
            layout.local_tokens,
            1,
            self.num_attention_heads_per_partition * self.config.v_head_dim,
        )
        if self.linear_gate is not None:
            core_output = self._project_and_apply_mla_output_gate(
                core_output, hidden_states
            )
        return self._explicit_output_projection(core_output)


def preprocess_mla_latent_cp(
    block: torch.nn.Module,
    hidden_states: Tensor,
    packed_seq_params: PackedSeqParams | None,
) -> None:
    """Prepare every latent-CP attention layer before a block enters its layer loop.

    Backend qualification is construction-time state. This hook shares immutable layout metadata
    and prepares expensive microbatch-specific backend bindings. It runs no collectives.
    """

    latent_layers = tuple(
        module for module in block.modules() if isinstance(module, MLAWithLatentCP)
    )
    if not latent_layers:
        return
    _require(
        isinstance(packed_seq_params, PackedSeqParams), "PackedSeqParams is required"
    )
    first_layer, *other_layers = latent_layers
    prepared_layout = first_layer._microbatch_layout(hidden_states, packed_seq_params)
    first_layer._preprocess_backend(
        hidden_states, packed_seq_params, prepared_layout=prepared_layout
    )
    for layer in other_layers:
        layer._preprocess_backend(
            hidden_states, packed_seq_params, prepared_layout=prepared_layout
        )

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer layer specialization for manifold-constrained hyper-connections."""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import (
    HyperConnectionExecutionContext,
    HyperConnectionResidualConnection,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionState,
)
from megatron.core.transformer.residual_recompute import ResidualStreamRecomputeContext
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


def _uses_mhc_attention_cuda_graph_split(config: TransformerConfig) -> bool:
    """Whether mHC attention graphs capture only the input layernorm and self-attention.

    ``TransformerConfig`` validates the remaining requirements of
    ``mhc_recompute_attn_cuda_graph_split``.
    """
    return (
        config.mhc_recompute_attn_cuda_graph_split
        and config.cuda_graph_impl == "transformer_engine"
    )


def _validate_mhc_te_graph_inputs(
    kwargs: dict, *, attention_graph_split: bool = False, packed_sequences: bool = False
) -> None:
    """Reject eager-only inputs that have no static TE training sample.

    The attention-only graph split keeps the padding mask and the mHC recompute manager
    outside the graph, where its eager MLP and mHC steps consume them. With scheduler-packed
    sequences, the split graph also takes their metadata as fixed-capacity offsets.
    """
    unsupported = ["packed_sequence_cp_metadata", "inference_context"]
    if not (attention_graph_split and packed_sequences):
        unsupported.append("packed_seq_params")
    if not attention_graph_split:
        unsupported.append("padding_mask")
    for name in unsupported:
        if kwargs.get(name) is not None:
            raise NotImplementedError(
                f"mHC TE graphs do not support {name} here. Packed THD sequences require "
                "mhc_recompute_attn_cuda_graph_split with sequence_packing_scheduler, and "
                "padding_mask requires the attention graph split."
            )
    if attention_graph_split:
        return
    if kwargs.get("mhc_recompute_manager") is not None:
        raise NotImplementedError(
            "mHC TE graphs support mHC selective recompute only with "
            "mhc_recompute_attn_cuda_graph_split."
        )
    kwargs.pop("mhc_recompute_manager", None)


class HyperConnectionTransformerLayer(TransformerLayer):
    """A transformer layer with Manifold-Constrained Hyper-Connections (mHC).

    Extends TransformerLayer by adding hyper connection modules around each active
    self-attention or MLP branch. The n-stream hidden states are aggregated before each
    branch and expanded back afterwards using learned mappings (H_pre, H_post, H_res).

    Replay uses the shared residual-stream recompute contract, accepting
    ``ResidualStreamRecomputeContext`` via ``residual_stream_recompute_context``.
    The deprecated ``mhc_recompute_manager`` keyword is still converted to a context.

    Cross-attention hyper connection is not supported.
    """

    supports_mhc_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            name=name,
        )

        if submodules.cross_attention_hyper_connection is not IdentityOp:
            raise ValueError(
                "HyperConnectionTransformerLayer does not support cross-attention "
                "hyper connections. Use IdentityOp for cross_attention_hyper_connection."
            )

        has_attention, has_mlp = self._validate_residual_branch_layout()
        if not has_attention and not has_mlp:
            raise ValueError("HyperConnectionTransformerLayer requires an attention or MLP branch.")

        for active, owner_spec, stage in (
            (has_attention, submodules.self_attention_hyper_connection, "self-attention"),
            (has_mlp, submodules.mlp_hyper_connection, "MLP"),
        ):
            if active and owner_spec is IdentityOp:
                raise ValueError(f"The active {stage} branch requires a hyper connection module.")
            if not active and owner_spec is not IdentityOp:
                raise ValueError(
                    f"The inactive {stage} branch requires an identity hyper connection."
                )

        unsupported_moe_cuda_graph_modules = {
            CudaGraphModule.moe,
            CudaGraphModule.moe_router,
            CudaGraphModule.moe_preprocess,
        }
        if self.config.cuda_graph_impl == "transformer_engine":
            # The TE path explicitly transports subclass MLP state across the
            # router graph boundary; local graphs retain their existing limits.
            unsupported_moe_cuda_graph_modules = {CudaGraphModule.moe}
        if self.is_moe_layer and unsupported_moe_cuda_graph_modules.intersection(
            self.config.cuda_graph_modules
        ):
            raise NotImplementedError(
                "HyperConnectionTransformerLayer does not support the requested MoE CUDA graph "
                "scopes. TE graphs support moe_router and moe_preprocess; local graphs require "
                "disabling moe, moe_router, and moe_preprocess for mHC MoE layers."
            )

        if _uses_mhc_attention_cuda_graph_split(self.config) and not isinstance(
            self.cross_attention, IdentityOp
        ):
            raise NotImplementedError(
                "mhc_recompute_attn_cuda_graph_split does not support cross-attention."
            )

        # GraphableMegatronModule calls create_mcore_cudagraph_manager before the MLP is
        # built, so repeat the local-graph decision now that self.is_moe_layer is authoritative.
        if self.config.cuda_graph_impl == "local":
            self.create_mcore_cudagraph_manager(self.config)

        self.self_attention_hyper_connection = None
        self.residual_read_self_attn = None
        self.residual_write_self_attn = None
        if has_attention:
            self.self_attention_hyper_connection = build_module(
                submodules.self_attention_hyper_connection,
                config=self.config,
                layer_number=self.layer_number,
            )
            self.residual_read_self_attn = HyperConnectionResidualConnection(
                self.config, operation="read", use_input_residual=True
            )
            self.residual_write_self_attn = HyperConnectionResidualConnection(
                self.config, operation="write"
            )

        self.mlp_hyper_connection = None
        self.residual_read_mlp = None
        self.residual_write_mlp = None
        if has_mlp:
            self.mlp_hyper_connection = build_module(
                submodules.mlp_hyper_connection, config=self.config, layer_number=self.layer_number
            )
            self.residual_read_mlp = HyperConnectionResidualConnection(
                self.config, operation="read", use_input_residual=True
            )
            self.residual_write_mlp = HyperConnectionResidualConnection(
                self.config, operation="write"
            )

        # mHC has always bypassed this legacy hook, which cannot carry residual state.
        self._legacy_forward_post_mlp = None

        # Replay state for the current public call only (set and cleared in __call__). Deferred
        # local CUDA-graph capture bypasses __call__ after the block was finalized, so it sees
        # None and runs mHC eagerly, like TE capture.
        self._mhc_recompute_context: ResidualStreamRecomputeContext | None = None

    def _te_cuda_graph_capture(self, *args, **kwargs):
        if _uses_mhc_attention_cuda_graph_split(self.config):
            self._reconstruct_packed_graph_inputs(kwargs)
            _validate_mhc_te_graph_inputs(
                kwargs,
                attention_graph_split=True,
                packed_sequences=self.config.sequence_packing_scheduler is not None,
            )
            return self._forward_mhc_attention_graph(*args, **kwargs)
        _validate_mhc_te_graph_inputs(kwargs)
        return super()._te_cuda_graph_capture(*args, **kwargs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        if _uses_mhc_attention_cuda_graph_split(self.config):
            _validate_mhc_te_graph_inputs(
                kwargs,
                attention_graph_split=True,
                packed_sequences=self.config.sequence_packing_scheduler is not None,
            )
            return self._replay_mhc_attention_graph_split(*args, **kwargs)
        _validate_mhc_te_graph_inputs(kwargs)
        return super()._te_cuda_graph_replay(*args, **kwargs)

    def _forward_mhc_attention_graph(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> tuple[Tensor, ...]:
        """Captured body of the attention graph split: input layernorm and self-attention.

        Takes the aggregated single-stream input and returns the self-attention output, plus
        its bias when present. The mHC read and write stay outside the graph.
        """
        input_layernorm_output = apply_module(self.input_layernorm)(hidden_states)
        if isinstance(input_layernorm_output, tuple):
            raise ValueError(
                "input_layernorm cannot return a residual when a residual connection "
                "owns the carried state."
            )
        nvtx_range_push(suffix="self_attention")
        attention_output, attention_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        nvtx_range_pop(suffix="self_attention")
        if attention_bias is None:
            return (attention_output,)
        return attention_output, attention_bias

    def _replay_mhc_attention_graph_split(self, *args, **kwargs):
        """Run the eager mHC read, attention graph, eager mHC write and MLP."""
        hidden_states = args[0] if args else kwargs.pop("hidden_states")
        if kwargs.get("context") is not None:
            raise NotImplementedError(
                "mhc_recompute_attn_cuda_graph_split does not support cross-attention."
            )
        # The eager MLP consumes these; the attention graph was captured without them, and
        # takes scheduler-packed metadata as fixed-capacity offsets instead.
        padding_mask = kwargs.pop("padding_mask", None)
        input_ids = kwargs.pop("input_ids", None)
        packed_seq_params = kwargs.get("packed_seq_params")
        kwargs = self._decompose_packed_graph_inputs(kwargs)

        nvtx_range_push(suffix="self_attention_hyper_connection")
        # TE copies the aggregate into the graph's static input and keeps that copy for the
        # graph's backward, so the read is eager rather than checkpointed.
        aggregated, attn_state = apply_module(self.residual_read_self_attn)(
            hidden_states,
            execution_context=HyperConnectionExecutionContext(self.self_attention_hyper_connection),
        )
        nvtx_range_pop(suffix="self_attention_hyper_connection")

        graph_outputs = GraphableMegatronModule._te_cuda_graph_replay(self, aggregated, **kwargs)
        attention_output_with_bias = (
            graph_outputs[0],
            graph_outputs[1] if len(graph_outputs) > 1 else None,
        )
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self._write_residual_connection(
                self.residual_write_self_attn,
                attention_output_with_bias,
                attn_state,
                is_mlp=False,
                recompute_context=None,
            )
        nvtx_range_pop(suffix="self_attn_bda")
        output = self._forward_mlp(
            hidden_states,
            padding_mask=padding_mask,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )
        return output, None

    def _pack_mlp_cuda_graph_state(self, outputs, residual, mlp_state):
        # The read state already carries the residual, H_res and H_post.
        assert len(mlp_state) == 3, "mHC partial MoE graphs require the MLP read state."
        return list(outputs) + list(mlp_state)

    def _unpack_mlp_cuda_graph_state(self, outputs):
        h_post, h_res, residual = outputs.pop(), outputs.pop(), outputs.pop()
        return residual, (residual, h_res, h_post)

    def __call__(self, *args, **kwargs):
        # Remove the context from kwargs because CUDA graphs cannot accept its live manager.
        # The residual and norm hooks read it from the layer while this call runs.
        context = kwargs.pop("residual_stream_recompute_context", None)
        legacy_manager = kwargs.pop("mhc_recompute_manager", None)
        if legacy_manager is not None:
            # Deprecated entry point: snapshot the manager's mutable block-end flag.
            if context is not None:
                raise ValueError(
                    "Pass residual_stream_recompute_context or the deprecated "
                    "mhc_recompute_manager, not both."
                )
            warnings.warn(
                "HyperConnectionTransformerLayer's mhc_recompute_manager keyword is deprecated "
                "and will be removed in a future release. Pass "
                "residual_stream_recompute_context=ResidualStreamRecomputeContext(manager, "
                "is_block_end) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            context = ResidualStreamRecomputeContext(
                legacy_manager, is_block_end=legacy_manager.is_last_layer_in_recompute_block
            )
        self._mhc_recompute_context = context
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._mhc_recompute_context = None

    def _residual_recompute_context(
        self, *, is_mlp: bool = False
    ) -> ResidualStreamRecomputeContext | None:
        """Only the last configured residual write preserves the layer's block-end flag."""
        context = self._mhc_recompute_context
        if context is None or self._is_last_residual_write(is_mlp=is_mlp):
            return context
        return ResidualStreamRecomputeContext(context.manager, is_block_end=False)

    def supports_two_stage_attention(self) -> bool:
        """mHC layers run only through ``__call__``, where their replay state enters.

        The split two-stage entry points bypass it, so attention-only mHC layers must not
        advertise them (for example, to Shortcut MoE grouping).
        """
        return False

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Override to produce n-stream hidden_states of shape [s, b, n*C].

        CUDA graph capture creates static buffers whose shapes are determined by
        this method. The base class returns [s, b, C], but mHC layers operate on
        n-stream hidden states of shape [s, b, n*C]. The attention graph split
        starts from the aggregated [s, b, C] input instead.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if _uses_mhc_attention_cuda_graph_split(self.config):
            # The eager MLP consumes these; the attention graph is captured without them.
            static_inputs.pop("input_ids", None)
            static_inputs.pop("padding_mask", None)
            return static_inputs
        hs = static_inputs["hidden_states"]
        n = self.config.mhc_num_residual_streams
        static_inputs["hidden_states"] = torch.ones(
            (hs.shape[0], hs.shape[1], n * self.config.hidden_size),
            dtype=hs.dtype,
            requires_grad=hs.requires_grad,
            device=hs.device,
        )
        return static_inputs

    def create_mcore_cudagraph_manager(self, config):
        """Create only CUDA graph managers compatible with this mHC layer."""
        # The first call comes from GraphableMegatronModule before TransformerLayer has
        # constructed self.mlp and set self.is_moe_layer.
        if not hasattr(self, "mlp"):
            return

        if self.is_moe_layer:
            # Whole-layer and MLP-scope graphs would capture dynamic MoE dispatch. Mixed
            # models may still request MLP graphs globally, so leave only this MoE layer eager.
            # An MLP-only MoE layer has no attention branch for an attn scope to graph, and
            # the base MLP branch would reject it.
            if (
                not config.cuda_graph_modules
                or (
                    CudaGraphModule.mlp in config.cuda_graph_modules
                    and CudaGraphModule.attn not in config.cuda_graph_modules
                )
                or isinstance(self.self_attention, IdentityOp)
            ):
                return

        super().create_mcore_cudagraph_manager(config)

    def _get_submodules_under_cudagraphs(self):
        """Override to include hyper connection modules.

        The base TransformerLayer._get_submodules_under_cudagraphs does not include
        self_attention_hyper_connection / mlp_hyper_connection. Their learnable
        parameters (mapping_proj, alpha_*, bias) need manual pre-forward hooks
        during CUDA graph replay so that parameter all-gathers are triggered.
        """
        submodules = super()._get_submodules_under_cudagraphs()

        if not self.config.cuda_graph_modules or _uses_mhc_attention_cuda_graph_split(self.config):
            # The attention graph split runs both hyper connections eagerly.
            return submodules

        if (
            CudaGraphModule.attn in self.config.cuda_graph_modules
            and self.self_attention_hyper_connection is not None
        ):
            submodules.append(self.self_attention_hyper_connection)
        if self.mlp_hyper_connection is not None and (
            (CudaGraphModule.mlp in self.config.cuda_graph_modules and not self.is_moe_layer)
            or (self.is_moe_layer and CudaGraphModule.moe_router in self.config.cuda_graph_modules)
        ):
            submodules.append(self.mlp_hyper_connection)
        return submodules

    def _read_residual_connection(
        self,
        connection: ResidualConnection,
        hidden_states: Tensor,
        *,
        is_mlp: bool,
        recompute_context: ResidualStreamRecomputeContext | None,
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Read through the current owner, preserving its activation dtype."""
        if recompute_context is not None:
            raise ValueError(
                "Pass the mHC recompute context through the layer call, not the base schedule."
            )
        owner = self.mlp_hyper_connection if is_mlp else self.self_attention_hyper_connection
        # Pass replay state to the adapter while preserving the shared schedule's
        # existing mHC norm and offload behavior.
        return apply_module(connection)(
            hidden_states,
            execution_context=HyperConnectionExecutionContext(owner),
            recompute_context=self._residual_recompute_context(is_mlp=is_mlp),
        )

    def _write_residual_connection(
        self,
        connection: ResidualConnection,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        is_mlp: bool,
        recompute_context: ResidualStreamRecomputeContext | None,
    ) -> Tensor:
        """Write through the current owner within the base BDA gradient scope."""
        if recompute_context is not None:
            raise ValueError(
                "Pass the mHC recompute context through the layer call, not the base schedule."
            )
        owner = self.mlp_hyper_connection if is_mlp else self.self_attention_hyper_connection
        return apply_module(connection)(
            branch_output,
            state=state,
            dropout_probability=self.hidden_dropout,
            training=self.training,
            execution_context=HyperConnectionExecutionContext(
                owner, self.config.bias_dropout_fusion
            ),
            recompute_context=self._residual_recompute_context(is_mlp=is_mlp),
        )

    def _run_input_layernorm(self, hidden_states):
        """Use the shared attention schedule with the call's mHC norm manager."""
        context = self._mhc_recompute_context
        return super()._run_input_layernorm(
            hidden_states, mhc_recompute_manager=context.manager if context is not None else None
        )

    def _pre_mlp_layernorm_and_residual(self, hidden_states):
        """Use the shared MLP schedule with the call's mHC norm manager."""
        context = self._mhc_recompute_context
        return super()._pre_mlp_layernorm_and_residual(
            hidden_states, mhc_recompute_manager=context.manager if context is not None else None
        )

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer layer specialization for manifold-constrained hyper-connections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor, nvtx_range_pop, nvtx_range_push

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager


def _uses_mhc_attention_cuda_graph_split(config: TransformerConfig) -> bool:
    """Whether mHC attention graphs capture only the input layernorm and self-attention.

    ``TransformerConfig`` validates the remaining requirements of
    ``mhc_recompute_attn_cuda_graph_split``.
    """
    return (
        config.mhc_recompute_attn_cuda_graph_split
        and config.cuda_graph_impl == 'transformer_engine'
    )


def _validate_mhc_te_graph_inputs(
    kwargs: dict, *, attention_graph_split: bool = False, packed_sequences: bool = False
) -> None:
    """Reject eager-only inputs that have no static TE training sample.

    The attention-only graph split keeps the padding mask and the mHC recompute manager
    outside the graph, where its eager MLP and mHC steps consume them. With scheduler-packed
    sequences, the split graph also takes their metadata as fixed-capacity offsets.
    """
    unsupported = ['packed_sequence_cp_metadata', 'inference_context']
    if not (attention_graph_split and packed_sequences):
        unsupported.append('packed_seq_params')
    if not attention_graph_split:
        unsupported.append('padding_mask')
    for name in unsupported:
        if kwargs.get(name) is not None:
            raise NotImplementedError(
                f'mHC TE graphs do not support {name} here. Packed THD sequences require '
                'mhc_recompute_attn_cuda_graph_split with sequence_packing_scheduler, and '
                'padding_mask requires the attention graph split.'
            )
    if attention_graph_split:
        return
    if kwargs.get('mhc_recompute_manager') is not None:
        raise NotImplementedError(
            'mHC TE graphs support mHC selective recompute only with '
            'mhc_recompute_attn_cuda_graph_split.'
        )
    kwargs.pop('mhc_recompute_manager', None)


class HyperConnectionTransformerLayer(TransformerLayer):
    """A transformer layer with Manifold-Constrained Hyper-Connections (mHC).

    Extends TransformerLayer by adding hyper connection modules around self-attention
    and MLP. The n-stream hidden states are aggregated before each sub-layer and
    expanded back afterwards using learned mappings (H_pre, H_post, H_res).

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
    ):
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

        assert submodules.self_attention_hyper_connection is not IdentityOp, (
            "HyperConnectionTransformerLayer requires self_attention_hyper_connection. "
            "Use TransformerLayer instead if hyper connections are not needed."
        )
        assert submodules.mlp_hyper_connection is not IdentityOp, (
            "HyperConnectionTransformerLayer requires mlp_hyper_connection. "
            "Use TransformerLayer instead if hyper connections are not needed."
        )

        unsupported_moe_cuda_graph_modules = {
            CudaGraphModule.moe,
            CudaGraphModule.moe_router,
            CudaGraphModule.moe_preprocess,
        }
        if self.config.cuda_graph_impl == 'transformer_engine':
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

        self.self_attention_hyper_connection = build_module(
            submodules.self_attention_hyper_connection,
            config=self.config,
            layer_number=self.layer_number,
        )

        self.mlp_hyper_connection = build_module(
            submodules.mlp_hyper_connection, config=self.config, layer_number=self.layer_number
        )

        # Set per-call by __call__ from kwargs so forward can read it without re-piping
        # the manager through the CUDA-graph kwarg path (CheckpointWithoutOutputManager
        # is not a CUDA-graph-supported type and gets stripped during capture). Read by
        # _run_input_layernorm, _apply_self_attn_bda_step, _pre_mlp_layernorm_and_residual, and
        # _apply_mlp_bda_step — do not delete; appears unused only at the class level.
        self._mhc_recompute_manager: Optional['CheckpointWithoutOutputManager'] = None

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
        its bias when present. The mHC aggregate and bias-dropout-add stay outside the graph.
        """
        input_layernorm_output = self._reject_residual_returning_norm(
            self.input_layernorm(hidden_states), "input_layernorm"
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
        """Run eager mHC aggregate, attention graph, eager mHC bias-dropout-add and MLP."""
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

        residual = hidden_states
        nvtx_range_push(suffix="self_attention_hyper_connection")
        # TE copies the aggregate into the graph's static input and keeps that copy for the
        # graph's backward, so the aggregate is not checkpointed.
        aggregated, h_res, h_post = self.self_attention_hyper_connection(hidden_states)
        nvtx_range_pop(suffix="self_attention_hyper_connection")

        graph_outputs = GraphableMegatronModule._te_cuda_graph_replay(self, aggregated, **kwargs)
        attention_output_with_bias = (
            graph_outputs[0],
            graph_outputs[1] if len(graph_outputs) > 1 else None,
        )
        hidden_states = self._mhc_self_attention_bda(
            attention_output_with_bias, residual, h_res, h_post
        )
        output = self._forward_mlp(
            hidden_states,
            padding_mask=padding_mask,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )
        return output, None

    def _pack_mlp_cuda_graph_state(self, outputs, residual, mlp_state):
        assert len(mlp_state) == 2, 'mHC partial MoE graphs require H_res and H_post.'
        return list(outputs) + [*mlp_state, residual]

    def _unpack_mlp_cuda_graph_state(self, outputs):
        residual = outputs.pop()
        h_post, h_res = outputs.pop(), outputs.pop()
        return residual, (h_res, h_post)

    def __call__(self, *args, **kwargs):
        # Pull the manager off kwargs before super().__call__ hands them to the
        # CUDA-graph machinery (which can't handle a CheckpointWithoutOutputManager).
        # forward() reads the value back from self.
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        return super().__call__(*args, **kwargs)

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
            if not config.cuda_graph_modules or (
                CudaGraphModule.mlp in config.cuda_graph_modules
                and CudaGraphModule.attn not in config.cuda_graph_modules
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

        if CudaGraphModule.attn in self.config.cuda_graph_modules:
            submodules.append(self.self_attention_hyper_connection)
        if (CudaGraphModule.mlp in self.config.cuda_graph_modules and not self.is_moe_layer) or (
            self.is_moe_layer and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        ):
            submodules.append(self.mlp_hyper_connection)
        return submodules

    def _run_input_layernorm(self, hidden_states):
        """HC input layernorm: hyper-connection pre-wrap + mHC-aware checkpoint.

        Threads ``h_res`` and ``h_post`` (produced by the hyper-connection
        pre-wrap) to ``_apply_self_attn_bda_step`` via the ``attn_state`` slot
        in the return tuple. Also sets
        ``self._input_layernorm_checkpoint_active`` for the post-self-attn
        discard hook.

        Returns ``(input_layernorm_output, residual, (h_res, h_post))`` where
        ``residual`` is the n-stream hidden state captured before
        aggregation — it flows to ``_apply_self_attn_bda_step`` via the base
        skeleton's ``residual`` argument, and ``(h_res, h_post)`` flows via
        ``attn_state``.
        """
        # Capture the n-stream residual BEFORE self_attention_hyper_connection
        # aggregates n-stream -> single-stream. The fused bda kernel needs the
        # original n-stream tensor.
        residual = hidden_states

        nvtx_range_push(suffix="self_attention_hyper_connection")
        hidden_states, h_res, h_post = self.self_attention_hyper_connection(
            hidden_states, mhc_recompute_manager=self._mhc_recompute_manager
        )
        nvtx_range_pop(suffix="self_attention_hyper_connection")

        self.attn_norm_manager = self.off_interface(
            self.offload_attn_norm, hidden_states, "attn_norm"
        )
        self._input_layernorm_checkpoint_active = self.recompute_input_layernorm or (
            self._mhc_recompute_manager is not None and self.mhc_checkpoint_input_layernorm
        )
        if self._input_layernorm_checkpoint_active:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=self._mhc_recompute_manager
            )
            with self.attn_norm_manager as hidden_states:
                input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                    self.input_layernorm, hidden_states
                )
        else:
            with self.attn_norm_manager as hidden_states:
                input_layernorm_output = self.input_layernorm(hidden_states)

        input_layernorm_output = self._reject_residual_returning_norm(
            input_layernorm_output, "input_layernorm"
        )
        return input_layernorm_output, residual, (h_res, h_post)

    @staticmethod
    def _reject_residual_returning_norm(layernorm_output, norm_name):
        """Reject layernorms that also return a residual.

        Base ``TransformerLayer`` accepts a ``(output, residual)`` tuple from a
        layernorm and uses the returned residual. mHC cannot: its residual must
        be the n-stream tensor captured *before* the hyper-connection aggregates
        n-stream -> single-stream, because ``fused_h_res_h_post_bda`` mixes it
        with ``H_res``. A norm-supplied single-stream residual would silently
        produce wrong shapes, so fail fast instead.
        """
        if isinstance(layernorm_output, tuple):
            raise ValueError(
                f"HyperConnectionTransformerLayer does not support a {norm_name} that "
                f"returns a (output, residual) tuple. mHC requires the n-stream residual "
                f"captured before hyper-connection aggregation. Use a layernorm submodule "
                f"that returns only the normalized output."
            )
        return layernorm_output

    def _apply_self_attn_bda_step(self, attention_output_with_bias, residual, attn_state=()):
        """HC fused bias-dropout-add: combines apply_h_res + apply_h_post + bda.

        Unpacks ``h_res`` and ``h_post`` from ``attn_state`` (threaded by
        ``_run_input_layernorm`` via the base skeleton). Keeps the base class's
        default so base-class call sites that pass only two arguments fail with a
        readable message rather than a TypeError.
        """
        if not attn_state:
            raise ValueError(
                "HyperConnectionTransformerLayer._apply_self_attn_bda_step requires the "
                "(h_res, h_post) attn_state produced by _run_input_layernorm. It was called "
                "through a base-class path that does not thread mHC intermediates."
            )
        h_res, h_post = attn_state
        hidden_states = self._mhc_self_attention_bda(
            attention_output_with_bias, residual, h_res, h_post
        )
        # HC omits forced_released_tensors — the n-stream residual is consumed
        # by the fused kernel above, so the base class's "release residual after
        # commit" trick doesn't apply.
        hidden_states = self.attn_norm_manager.group_offload(hidden_states)
        self.attn_norm_manager = None
        return hidden_states

    def _mhc_self_attention_bda(self, attention_output_with_bias, residual, h_res, h_post):
        """Apply H_res, H_post and bias-dropout-add to the self-attention output."""
        nvtx_range_push(suffix="self_attention_fused_h_res_h_post_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attention_hyper_connection.fused_h_res_h_post_bda(
                h_res,
                residual,
                h_post,
                attention_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                self._mhc_recompute_manager,
            )
        nvtx_range_pop(suffix="self_attention_fused_h_res_h_post_bda")
        return hidden_states

    def _pre_mlp_layernorm_and_residual(self, hidden_states):
        """HC pre-mlp layernorm: hyper-connection pre-wrap + mHC-aware checkpoint.

        Threads ``mlp_h_res`` and ``mlp_hc_h_post`` (produced by the
        hyper-connection pre-wrap) to ``_apply_mlp_bda_step`` via the
        ``mlp_state`` slot in the return tuple.

        Returns ``(pre_mlp_layernorm_output, residual, (mlp_h_res, mlp_hc_h_post))``
        where ``residual`` is the n-stream hidden state captured before
        aggregation — it flows to ``_apply_mlp_bda_step`` via the base
        skeleton's ``residual`` argument, and ``(mlp_h_res, mlp_hc_h_post)``
        flows via ``mlp_state``.
        """
        # Capture the n-stream residual BEFORE mlp_hyper_connection
        # aggregates n-stream -> single-stream. The fused bda kernel needs the
        # original n-stream tensor.
        residual = hidden_states

        nvtx_range_push(suffix="mlp_hyper_connection")
        hidden_states, mlp_h_res, mlp_hc_h_post = self.mlp_hyper_connection(
            hidden_states, mhc_recompute_manager=self._mhc_recompute_manager
        )
        nvtx_range_pop(suffix="mlp_hyper_connection")

        self.mlp_norm_manager = self.off_interface(self.offload_mlp_norm, hidden_states, "mlp_norm")
        checkpoint_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm or (
            self._mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        )
        if checkpoint_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=self._mhc_recompute_manager
            )
            with self.mlp_norm_manager as hidden_states:
                pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                    self.pre_mlp_layernorm, hidden_states
                )
        else:
            with self.mlp_norm_manager as hidden_states:
                pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        pre_mlp_layernorm_output = self._reject_residual_returning_norm(
            pre_mlp_layernorm_output, "pre_mlp_layernorm"
        )
        return pre_mlp_layernorm_output, residual, (mlp_h_res, mlp_hc_h_post)

    def _apply_mlp_bda_step(self, mlp_output_with_bias, residual, mlp_state=()):
        """HC fused bias-dropout-add for MLP: combines apply_h_res + apply_h_post + bda.

        Unpacks ``mlp_h_res`` and ``mlp_hc_h_post`` from ``mlp_state`` (threaded
        by ``_pre_mlp_layernorm_and_residual`` via the base skeleton). Computes the
        per-call ``mhc_mlp_bda_manager`` from ``self._mhc_recompute_manager``:
        the last layer of a recompute block does NOT pass the manager into the
        fused-bda checkpoint — the block-end finalize hook handles its output
        discard.

        Keeps the base class's ``mlp_state`` default so base-class call sites that
        pass only two arguments (``_te_cuda_graph_replay_impl``,
        ``MoETransformerLayer._forward_mlp_postprocess``) fail with a readable
        message rather than a TypeError.
        """
        if not mlp_state:
            raise ValueError(
                "HyperConnectionTransformerLayer._apply_mlp_bda_step requires the "
                "(h_res, h_post) mlp_state produced by _pre_mlp_layernorm_and_residual. It "
                "was called through a base-class path that does not thread mHC intermediates."
            )
        mlp_h_res, mlp_hc_h_post = mlp_state

        is_last_in_recompute_block = bool(
            self._mhc_recompute_manager is not None
            and getattr(self._mhc_recompute_manager, "is_last_layer_in_recompute_block", False)
        )
        mhc_mlp_bda_manager = None if is_last_in_recompute_block else self._mhc_recompute_manager

        if self.recompute_pre_mlp_layernorm or (
            mhc_mlp_bda_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        ):
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        nvtx_range_push(suffix="mlp_fused_h_res_h_post_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_hyper_connection.fused_h_res_h_post_bda(
                mlp_h_res,
                residual,
                mlp_hc_h_post,
                mlp_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_mlp_bda_manager,
            )
        nvtx_range_pop(suffix="mlp_fused_h_res_h_post_bda")

        # HC omits forced_released_tensors — the n-stream residual is consumed
        # by the fused kernel above, so the base class's "release residual after
        # commit" trick doesn't apply.
        if self.mlp_norm_manager is not None:
            hidden_states = self.mlp_norm_manager.group_offload(hidden_states)
            self.mlp_norm_manager = None

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output

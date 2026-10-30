# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer layer specialization for manifold-constrained hyper-connections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    deprecate_inference_params,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import MHCCheckpointManager


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
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
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

        self.self_attention_hyper_connection = build_module(
            submodules.self_attention_hyper_connection,
            config=self.config,
            layer_number=self.layer_number,
        )

        self.mlp_hyper_connection = build_module(
            submodules.mlp_hyper_connection, config=self.config, layer_number=self.layer_number
        )

        self._mhc_recompute_manager = None
        self._validate_mhc_recompute_attn_cuda_graph_split()
        if self.config.cuda_graph_impl == "local":
            if self.is_moe_layer and {
                CudaGraphModule.moe,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            }.intersection(self.config.cuda_graph_modules):
                raise NotImplementedError(
                    "HyperConnectionTransformerLayer does not support local partial-MoE CUDA "
                    "graph scopes. Transformer Engine capture retains the mHC MoE paths."
                )
            # GraphableMegatronModule runs before the MLP is built; decide again now
            # that the actual dense/MoE layer type is known.
            self.create_mcore_cudagraph_manager(self.config)

    def _uses_mhc_recompute_attn_cuda_graph_split(self) -> bool:
        """Whether the graph starts after eager mHC aggregation.

        This first production slice deliberately supports one exact scope.  It
        keeps the mHC producer and post-processing eager while capturing only
        the dense attention consumer, so ``CheckpointWithoutOutput`` remains a
        real eager checkpoint rather than being bypassed during graph capture.
        """
        from megatron.core.transformer.mhc_recompute import uses_mhc_recompute_attn_cuda_graph_split

        return uses_mhc_recompute_attn_cuda_graph_split(self.config)

    def _validate_mhc_recompute_attn_cuda_graph_split(self) -> None:
        """Reject the variants the direct-write arena provably cannot represent.

        Deliberately short. Only conditions with a mechanism in this file are
        rejected here; "not yet exercised" is not a reason to fail a user's
        configuration. An earlier revision of this PR rejected FP4, non-BF16,
        non-mxfp8 FP8 recipes, fine-grained activation offloading, delayed
        weight-gradient compute, MoE without EP overlap and an unfused input
        layernorm on exactly that basis, which left the feature reachable by
        almost nothing. Those are left to the guards that already own them --
        the split's own constraints, attention-scope offloading included, live
        with ``mhc_recompute_attn_cuda_graph_split`` in TransformerConfig and
        apply only when that switch is set, and delayed wgrad in particular is
        *supported*: the capture-order bookkeeping extends a sample's liveness
        window across its wgrad entry.

        What actually protects this path is enforced at runtime and fails loudly:
        ``MHCRecomputeArenaSlot.validate_output`` checks that the producer
        direct-wrote the captured address, and
        ``TECudaGraphHelper._validate_mhc_static_hidden_inputs`` checks that
        aliased static inputs have disjoint liveness windows.
        """
        if not self._uses_mhc_recompute_attn_cuda_graph_split():
            return

        if not isinstance(self.cross_attention, IdentityOp):
            # A captured cross-attention adds a context output, and
            # _replay_mhc_attention_consumer accepts exactly one attention output
            # plus an optional bias. Rejecting here turns a mid-capture
            # RuntimeError into a config-time message.
            raise ValueError(
                "mHC recompute with attention-only TE CUDA Graphs cannot represent "
                "cross-attention: its context output exceeds the split's graph "
                "output arity."
            )

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Override to produce n-stream hidden_states of shape [s, b, n*C].

        CUDA graph capture creates static buffers whose shapes are determined by
        this method. The base class returns [s, b, C], but mHC layers operate on
        n-stream hidden states of shape [s, b, n*C].
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if self._uses_mhc_recompute_attn_cuda_graph_split():
            # The captured callable begins at the attention consumer, after the
            # eager mHC producer has reduced n streams to one hidden-size tensor.
            return static_inputs

        hs = static_inputs["hidden_states"]
        n = self.config.num_residual_streams
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
        if self._uses_mhc_recompute_attn_cuda_graph_split():
            # mHC pre/post run eagerly and therefore receive ordinary hooks.
            # Registering them as graph-owned modules would trigger duplicate
            # manual hooks during replay.
            return [self.input_layernorm, self.self_attention]

        submodules = super()._get_submodules_under_cudagraphs()

        if not self.config.cuda_graph_modules:
            return submodules

        if CudaGraphModule.attn in self.config.cuda_graph_modules:
            submodules.append(self.self_attention_hyper_connection)
        if (not self.is_moe_layer and CudaGraphModule.mlp in self.config.cuda_graph_modules) or (
            self.is_moe_layer
            and (
                CudaGraphModule.moe in self.config.cuda_graph_modules
                or CudaGraphModule.moe_router in self.config.cuda_graph_modules
            )
        ):
            submodules.append(self.mlp_hyper_connection)
        return submodules

    def forward(self, *args, **kwargs):
        """Forward pass with MHC recompute manager support."""
        kwargs.pop("dynamic_inference_decode_only", None)
        kwargs.pop("_called_from_hybrid_mhc_wrapper", None)

        mhc_recompute_manager = getattr(self, '_mhc_recompute_manager', None)

        with _otel_managed_span(
            'layer', 'megatron.layer.forward', **{'megatron.layer_number': self.layer_number}
        ):
            hidden_states, context = self._forward_attention(
                *args, mhc_recompute_manager=mhc_recompute_manager, **kwargs
            )

            with _otel_managed_span('layer', 'megatron.layer.mlp'):
                output = self._forward_mlp(
                    hidden_states,
                    kwargs.get("inference_context", None),
                    padding_mask=kwargs.get("padding_mask", None),
                    input_ids=kwargs.get("input_ids", None),
                    packed_seq_params=kwargs.get("packed_seq_params", None),
                    mhc_recompute_manager=mhc_recompute_manager,
                )
            return output, context

    def _forward_mhc_attention_cuda_graph_consumer(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        **_unused_kwargs,
    ):
        """Captured input-layernorm/self-attention consumer for split mHC.

        ``hidden_states`` is the one-stream aggregate produced eagerly by mHC.
        mHC mapping, aggregation, post-processing, and the MLP intentionally stay
        outside this callable so their checkpoint registrations execute at every
        logical microbatch rather than only during CUDA Graph warmup.
        """
        input_layernorm_output = self.input_layernorm(hidden_states)
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        if (
            not isinstance(attention_output_with_bias, tuple)
            or len(attention_output_with_bias) != 2
        ):
            raise TypeError(
                "mHC attention-only CUDA Graph expects self-attention to return "
                "(output, optional_bias)"
            )

        attention_output, output_bias = attention_output_with_bias
        if output_bias is None:
            return (attention_output,)
        return attention_output, output_bias

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """Capture only the attention consumer for the split mHC path.

        This GPT layer is itself the top-level TE graph callable. HybridStack instead wraps
        plain ``TransformerLayer`` instances in ``HyperConnectionHybridLayer``, so no outer
        wrapper calls this subclass's ``_te_cuda_graph_capture_impl``. If that topology changes,
        this subclass must override the implementation entry point as well.
        """
        if not self._uses_mhc_recompute_attn_cuda_graph_split():
            return super()._te_cuda_graph_capture(*args, **kwargs)

        self._reconstruct_packed_seq_params_from_kwargs(kwargs)
        return self._forward_mhc_attention_cuda_graph_consumer(*args, **kwargs)

    def _forward_mhc_attention_post_cuda_graph(
        self,
        attention_output_with_bias,
        self_attn_h_res,
        residual,
        self_attn_hc_h_post,
        context=None,
        context_mask=None,
        inference_context=None,
        mhc_recompute_manager=None,
    ):
        """Run eager mHC post-processing after attention graph replay."""
        nvtx_range_push(suffix="self_attention_fused_h_res_h_post_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attention_hyper_connection.fused_h_res_h_post_bda(
                self_attn_h_res,
                residual,
                self_attn_hc_h_post,
                attention_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_recompute_manager,
            )
        nvtx_range_pop(suffix="self_attention_fused_h_res_h_post_bda")

        # HyperConnectionTransformerLayer does not support a hyper-connected
        # cross-attention branch, but preserve the existing IdentityOp-compatible
        # layer flow here so output semantics remain unchanged.
        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )
        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager: Optional['MHCCheckpointManager'] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """Forward attention with hyper connection pre/post processing on self-attention."""
        inference_context = deprecate_inference_params(inference_context, inference_params)

        nvtx_range_push(suffix="self_attention_hyper_connection")
        hidden_states, self_attn_h_res, self_attn_hc_h_post, residual = (
            self.self_attention_hyper_connection(
                hidden_states, mhc_recompute_manager=mhc_recompute_manager, return_residual=True
            )
        )
        nvtx_range_pop(suffix="self_attention_hyper_connection")

        # Optional Input Layer norm
        checkpoint_input_layernorm = self.recompute_input_layernorm or (
            mhc_recompute_manager is not None and self.mhc_checkpoint_input_layernorm
        )
        attn_norm_manager = self.off_interface(self.offload_attn_norm, hidden_states, "attn_norm")
        if checkpoint_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            with attn_norm_manager as hidden_states:
                input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                    self.input_layernorm, hidden_states
                )
        else:
            with attn_norm_manager as hidden_states:
                input_layernorm_output = self.input_layernorm(hidden_states)

        input_layernorm_output = self._reject_residual_returning_norm(
            input_layernorm_output, "input_layernorm"
        )

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="self_attention")

        if checkpoint_input_layernorm:
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        nvtx_range_push(suffix="self_attention_fused_h_res_h_post_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attention_hyper_connection.fused_h_res_h_post_bda(
                self_attn_h_res,
                residual,
                self_attn_hc_h_post,
                attention_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_recompute_manager,
            )
        nvtx_range_pop(suffix="self_attention_fused_h_res_h_post_bda")

        hidden_states = attn_norm_manager.group_offload(hidden_states)

        # Cross-attention (no hyper connection support).
        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    def _forward_mlp(
        self,
        hidden_states,
        inference_context=None,
        padding_mask=None,
        input_ids=None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        mhc_recompute_manager: Optional['MHCCheckpointManager'] = None,
    ):
        """Forward MLP with hyper connection pre/post processing."""
        is_last_in_recompute_block = bool(
            mhc_recompute_manager is not None
            and getattr(mhc_recompute_manager, "is_last_layer_in_recompute_block", False)
        )
        mhc_mlp_bda_manager = None if is_last_in_recompute_block else mhc_recompute_manager

        nvtx_range_push(suffix="mlp_hyper_connection")
        hidden_states, mlp_h_res, mlp_hc_h_post, residual = self.mlp_hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager, return_residual=True
        )
        nvtx_range_pop(suffix="mlp_hyper_connection")
        # mHC aggregation upcasts the single-stream MLP input to fp32 for numerical
        # stability of the mixing weights, and that fp32 propagates through the residual
        # stream. MoE layers absorb it in their own routing dtype handling, but a dense
        # MLP feeds it straight into a low-precision TE layer whose fused layernorm weight
        # is in params_dtype, so restore params_dtype for dense (non-MoE) layers.
        if not self.is_moe_layer and hidden_states.dtype != self.config.params_dtype:
            hidden_states = hidden_states.to(self.config.params_dtype)

        # Optional Layer norm post the cross-attention.
        checkpoint_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm or (
            mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        )
        self.mlp_norm_manager = self.off_interface(self.offload_mlp_norm, hidden_states, "mlp_norm")
        if checkpoint_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
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
        pre_mlp_layernorm_output, padding_mask, input_ids, moe_unflatten_mbs = (
            self._maybe_unflatten_for_moe(
                pre_mlp_layernorm_output, padding_mask, input_ids, packed_seq_params
            )
        )
        mlp_output_with_bias = self._run_mlp(
            pre_mlp_layernorm_output,
            residual,
            padding_mask,
            inference_context,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )
        if moe_unflatten_mbs is not None:
            output, bias = mlp_output_with_bias
            output = self._maybe_reflatten_from_moe(output, packed_seq_params, moe_unflatten_mbs)
            mlp_output_with_bias = (output, bias)

        # During TE CUDA graph partial MoE capture, skip HC post-processing and return
        # intermediate outputs + HC state. The post-processing will be done during replay.
        if (
            self.is_moe_layer
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and is_graph_capturing()
            and CudaGraphModule.moe_router in self.config.cuda_graph_modules
        ):
            if self.recompute_pre_mlp_layernorm or (
                mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
            ):
                for tensor in mlp_output_with_bias:
                    self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(tensor)
            # Append HC state (mlp_hc_h_post, mlp_h_res, residual) for replay.
            return list(mlp_output_with_bias) + [mlp_hc_h_post, mlp_h_res, residual]

        return self._forward_post_mlp_with_fused_hyper_connection(
            mlp_output_with_bias, mlp_h_res, residual, mlp_hc_h_post, mhc_mlp_bda_manager
        )

    def _forward_mhc_mlp_post_core(
        self,
        mlp_output_with_bias,
        mlp_h_res,
        residual,
        mlp_hc_h_post,
        mhc_mlp_bda_recompute_manager: Optional['MHCCheckpointManager'] = None,
    ):
        """Run the fused mHC post-MLP H_res/H_post/BDA computation."""
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
                mhc_mlp_bda_recompute_manager,
            )
        nvtx_range_pop(suffix="mlp_fused_h_res_h_post_bda")
        return hidden_states

    def _forward_mhc_mlp_post(
        self,
        mlp_output,
        mlp_h_res,
        residual,
        mlp_hc_h_post,
        mhc_mlp_bda_recompute_manager: Optional['MHCCheckpointManager'] = None,
    ):
        """Run mHC post-MLP fused H_res/H_post/BDA without MLP norm offload."""
        return self._forward_mhc_mlp_post_core(
            (mlp_output, None), mlp_h_res, residual, mlp_hc_h_post, mhc_mlp_bda_recompute_manager
        )

    def _forward_post_mlp_with_fused_hyper_connection(
        self,
        mlp_output_with_bias,
        mlp_h_res,
        residual,
        mlp_hc_h_post,
        mhc_mlp_bda_recompute_manager: Optional['MHCCheckpointManager'] = None,
    ):
        """
        Perform operations after the MLP computation with fused hyper connection kernel.

        This method uses the fused kernel combining apply_h_res, apply_h_post and bias-dropout-add.

        Args:
            mlp_output_with_bias (Tensor): Output tensor of the MLP layer with bias.
            mlp_h_res (Tensor): [s, b, n, n] - residual mixing matrix from hyper connection.
            residual (Tensor): [s, b, n*C] - original residual (n-stream hidden states).
            mlp_hc_h_post (Tensor): [s, b, n] - expansion weights from hyper connection.
            mhc_recompute_manager: Optional MHCCheckpointManager for checkpoint management.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        if self.recompute_pre_mlp_layernorm or (
            mhc_mlp_bda_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        ):
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        hidden_states = self._forward_mhc_mlp_post_core(
            mlp_output_with_bias, mlp_h_res, residual, mlp_hc_h_post, mhc_mlp_bda_recompute_manager
        )

        hidden_states = self.mlp_norm_manager.group_offload(hidden_states)

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """Keep packed metadata intact for the split's eager continuation.

        The parent decomposes PackedSeqParams before dispatching replay. The split instead
        decomposes a separate copy for the attention graph, retaining the original object
        (including CP route metadata) for the eager MLP. Preserve the parent's offload lifecycle.
        """
        if not self._uses_mhc_recompute_attn_cuda_graph_split():
            return super()._te_cuda_graph_replay(*args, **kwargs)
        assert (
            kwargs.get("inference_context") is None
        ), "mHC attention-only TE CUDA Graphs do not support inference_context"
        if self.config.delay_offload_until_cuda_graph:
            self.off_interface.enter_replay()
        try:
            return self._te_cuda_graph_replay_impl(args, kwargs, None)
        finally:
            self._te_cuda_graph_route_replay_state = None
            if self.config.delay_offload_until_cuda_graph:
                self.off_interface.exit_replay()

    def _replay_mhc_attention_consumer(self, args, kwargs, context):
        """Eager mHC aggregate -> captured attention replay -> eager post-BDA.

        Shared head of the attention-only split for both the dense 1F1B path and
        the EP-overlap path. Binds the captured static input as an arena slot,
        runs the eager mHC aggregate directly into that slot (and registers its
        checkpoint against the recompute manager), replays the captured
        input-layernorm + self-attention graph, and runs the eager self-attention
        mHC post-processing. Returns the post-attention hidden state, the cross-
        attention context, and the recompute manager for the caller's MLP tail.
        """
        kwargs = kwargs.copy()
        if len(args) > 1:
            raise ValueError(
                "mHC attention-only CUDA Graph expects exactly one positional hidden input"
            )
        if args:
            if "hidden_states" in kwargs:
                raise ValueError("hidden_states must be positional or keyword, not both")
            hidden_states = args[0]
        else:
            hidden_states = kwargs.pop("hidden_states")

        # Forward only the tensor arguments consumed by the captured attention
        # callable. Include unused static kwargs such as padding_mask: TE requires
        # every captured kwarg on replay. Keep the originals for the eager continuation.
        graph_kwargs = {}
        graph_tensor_kwargs = (
            "rotary_pos_emb",
            "rotary_pos_cos",
            "rotary_pos_sin",
            "rotary_pos_cos_sin",
            "attention_bias",
            "sequence_len_offset",
            "padding_mask",
            "input_ids",
        )
        if "attention_mask" in kwargs:
            # TransformerLayer's replay-argument helper turns a None mask into
            # the capture-compatible representation when necessary.
            graph_kwargs["attention_mask"] = kwargs["attention_mask"]
        for name in graph_tensor_kwargs:
            value = kwargs.get(name)
            if value is not None:
                graph_kwargs[name] = value
        if kwargs.get("packed_seq_params") is not None:
            graph_kwargs["packed_seq_params"] = kwargs["packed_seq_params"]
            self._decompose_packed_seq_params_to_kwargs(graph_kwargs)

        # Decomposition also pins graph-dynamic CP route replay to its retained slot.
        # Select the mHC direct-write buffer only after that slot has been resolved.
        mhc_recompute_manager = getattr(self, '_mhc_recompute_manager', None)
        output_slot = None
        if mhc_recompute_manager is not None:
            static_hidden_input = self.get_te_cuda_graph_static_hidden_input()
            output_slot = mhc_recompute_manager.mhc_arena.bind_external_slot(
                ("attention", self.layer_number, "aggregate", 0), static_hidden_input
            )

        nvtx_range_push(suffix="self_attention_hyper_connection")
        aggregated, self_attn_h_res, self_attn_hc_h_post, residual = (
            self.self_attention_hyper_connection(
                hidden_states,
                mhc_recompute_manager=mhc_recompute_manager,
                output_slot=output_slot,
                return_residual=True,
            )
        )
        nvtx_range_pop(suffix="self_attention_hyper_connection")

        cuda_graph_output = GraphableMegatronModule._te_cuda_graph_replay(
            self, aggregated, **graph_kwargs
        )

        # Flush delayed offload groups from previous layers after graph replay, as
        # TransformerLayer._te_cuda_graph_replay_impl does. The split returns out of
        # that method before reaching its flush and exit_replay() only clears the
        # _in_replay flag, so without this the queued groups never run and reset()
        # drops them at the end of the iteration. This is the shared head of both
        # split variants, so one call covers them.
        if self.config.delay_offload_until_cuda_graph:
            self.off_interface.flush_delayed_groups()

        if isinstance(cuda_graph_output, torch.Tensor):
            cuda_graph_output = (cuda_graph_output,)
        else:
            cuda_graph_output = tuple(cuda_graph_output)
        if len(cuda_graph_output) == 1:
            attention_output_with_bias = (cuda_graph_output[0], None)
        elif len(cuda_graph_output) == 2:
            attention_output_with_bias = cuda_graph_output
        else:
            raise RuntimeError(
                "mHC attention-only CUDA Graph must return attention output and optional bias; "
                f"got {len(cuda_graph_output)} tensors"
            )

        hidden_states, context = self._forward_mhc_attention_post_cuda_graph(
            attention_output_with_bias,
            self_attn_h_res,
            residual,
            self_attn_hc_h_post,
            context=kwargs.get("context", context),
            context_mask=kwargs.get("context_mask"),
            inference_context=kwargs.get("inference_context"),
            mhc_recompute_manager=mhc_recompute_manager,
        )
        return hidden_states, context, mhc_recompute_manager, kwargs

    def _te_cuda_graph_replay_mhc_attention_split(self, args, kwargs, context):
        """Replay captured attention between eager mHC pre/post operations."""
        hidden_states, context, mhc_recompute_manager, kwargs = self._replay_mhc_attention_consumer(
            args, kwargs, context
        )
        output = self._forward_mlp(
            hidden_states,
            inference_context=kwargs.get("inference_context"),
            padding_mask=kwargs.get("padding_mask"),
            input_ids=kwargs.get("input_ids"),
            packed_seq_params=kwargs.get("packed_seq_params"),
            mhc_recompute_manager=mhc_recompute_manager,
        )
        return output, context

    def _te_cuda_graph_replay_mhc_attention_split_overlap(self, args, kwargs, context):
        """Attention-only split replay for the EP a2a-overlap schedule.

        Shares the arena/attention head with the dense path, then—instead of
        running the whole MLP synchronously—emits the MoE preprocessing tail
        (mHC MLP aggregate, pre-MLP layernorm checkpoint, shared-expert compute,
        routing, dispatch preprocess) so the fine-grained schedule can overlap
        dispatch/experts/combine. The return contract matches the eager mHC-MoE
        callable in fine_grained_callables.submodule_attn_forward.
        """
        hidden_states, _context, mhc_recompute_manager, kwargs = (
            self._replay_mhc_attention_consumer(args, kwargs, context)
        )
        if not self.is_moe_layer:
            # Dense hyper-connection layers coexist with MoE ones under overlap;
            # they carry no MoE tail (mirror of the eager non-MoE early return).
            return hidden_states, None, None, None

        nvtx_range_push(suffix="mlp_hyper_connection")
        hidden_states, mlp_h_res, mlp_hc_h_post, residual = self.mlp_hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager, return_residual=True
        )
        nvtx_range_pop(suffix="mlp_hyper_connection")

        checkpoint_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm or (
            mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
        )
        if checkpoint_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                apply_module(self.pre_mlp_layernorm), hidden_states
            )
        else:
            pre_mlp_layernorm_output = apply_module(self.pre_mlp_layernorm)(hidden_states)

        # Fused residual norm (e.g. TEFusedResidualRMSNorm) returns (output,
        # residual); the mHC residual comes from mlp_hyper_connection, so the
        # fused residual is dropped here to match the eager mHC-MoE tail.
        if isinstance(pre_mlp_layernorm_output, tuple):
            if len(pre_mlp_layernorm_output) != 2:
                raise ValueError(
                    "When the output of pre_mlp_layernorm is a tuple, it is expected to "
                    f"have 2 elements (output, residual), but got "
                    f"{len(pre_mlp_layernorm_output)}"
                )
            pre_mlp_layernorm_output, _ = pre_mlp_layernorm_output

        shared_expert_output = self.mlp.shared_experts_compute(pre_mlp_layernorm_output)
        # Route with exactly the arguments the eager overlap callable passes
        # (fine_grained_callables.submodule_attn_forward): padding_mask only.
        # The router consumes it in three places -- the z-loss mean, dropless
        # gating, and the expert-load counters behind the aux-loss-free bias --
        # so dropping it would make the graphed path drift from eager on padded
        # batches without failing anything. input_ids and packed_seq_params are
        # deliberately not forwarded: the eager callable does not pass them
        # either, and parity with eager is this method's contract.
        probs, routing_map = self.mlp.route(
            pre_mlp_layernorm_output, padding_mask=kwargs.get("padding_mask")
        )
        local_tokens, probs = self.mlp.preprocess(
            pre_mlp_layernorm_output, probs, routing_map, kwargs.get("padding_mask")
        )
        return (residual, local_tokens, probs, shared_expert_output, mlp_h_res, mlp_hc_h_post)

    def _te_cuda_graph_replay_impl(self, args, kwargs, context):
        """Implementation of _te_cuda_graph_replay with hyper connection support.

        The caller owns the delay_offload_until_cuda_graph lifecycle
        (enter_replay/exit_replay), including on the attention-only split path.

        During MoE partial CUDA graph capture, the graph outputs include HC state
        (mlp_hc_h_post, mlp_h_res) in addition to the base class outputs. This method
        extracts the HC state and uses it for post-processing after resuming the MoE forward.
        """
        if self._uses_mhc_recompute_attn_cuda_graph_split():
            if self.config.overlap_moe_expert_parallel_comm:
                return self._te_cuda_graph_replay_mhc_attention_split_overlap(args, kwargs, context)
            return self._te_cuda_graph_replay_mhc_attention_split(args, kwargs, context)

        # Read, do not pop: on THD, padding_mask IS a capture-time kwarg
        # (get_layer_static_inputs adds it whenever _is_thd_cuda_graph()), and
        # TE's replay raises TypeError for any captured kwarg that is missing.
        # On SBHD it is simply not in the graph's kwargs_keys, and TE reads only
        # the keys it captured -- extra kwargs are ignored, so leaving it in
        # kwargs is safe on both layouts. The routing tail below consumes it.
        padding_mask = kwargs.get("padding_mask", None)

        cuda_graph_output = list(
            GraphableMegatronModule._te_cuda_graph_replay(self, *args, **kwargs)
        )

        # Flush delayed offload groups from previous layers after graph replay.
        if self.config.delay_offload_until_cuda_graph:
            self.off_interface.flush_delayed_groups()

        if kwargs.get('context') is not None:
            context = cuda_graph_output.pop()

        if (
            not self.config.cuda_graph_modules
            or (not self.is_moe_layer and CudaGraphModule.mlp in self.config.cuda_graph_modules)
            or (self.is_moe_layer and CudaGraphModule.moe in self.config.cuda_graph_modules)
        ):
            assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
            output = cuda_graph_output.pop()
            assert not self.config.overlap_moe_expert_parallel_comm, "EP overlap must be \
                disabled when CUDA graph captures the whole MLP/MoE part."
        elif self.is_moe_layer and CudaGraphModule.moe_router in self.config.cuda_graph_modules:
            # Pop HC state (appended during capture in _forward_mlp).
            residual = cuda_graph_output.pop()
            mlp_h_res = cuda_graph_output.pop()
            mlp_hc_h_post = cuda_graph_output.pop()

            shared_expert_output, routing_map = None, None
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                shared_expert_output = cuda_graph_output.pop()

            if CudaGraphModule.moe_preprocess in self.config.cuda_graph_modules:
                (hidden_states, probs), attr_outputs = (
                    cuda_graph_output[:2],
                    cuda_graph_output[2:],
                )
                valid_cudagraph_attrs = self.mlp.token_dispatcher.valid_cudagraph_attrs
                assert len(attr_outputs) == len(
                    valid_cudagraph_attrs
                ), f"attr_outputs: {len(attr_outputs)} != {len(valid_cudagraph_attrs)}"
                for i, attr_name in enumerate(valid_cudagraph_attrs):
                    self.mlp.token_dispatcher.set_cudagraph_attr(attr_name, attr_outputs[i])
            else:
                assert len(cuda_graph_output) == 3, (
                    "CUDA graph output should be [hidden_states, probs, routing_map], "
                    f"but got {len(cuda_graph_output)} elements"
                )
                hidden_states, probs, routing_map = cuda_graph_output

            # Resume the MoELayer forward pass from the end of the CUDA graph scope.
            nvtx_range_push(suffix="mlp")
            self.mlp.cudagraph_tensor_store.set(
                hidden_states=hidden_states,
                probs=probs,
                routing_map=routing_map,
                shared_expert_output=shared_expert_output,
            )
            # If EP overlap is enabled, remaining of mlp will be called as fine_grained_callables
            # and should be skipped here.
            if self.config.overlap_moe_expert_parallel_comm:
                probs, routing_map = self.mlp.route(hidden_states, padding_mask=padding_mask)
                hidden_states, probs = self.mlp.preprocess(
                    hidden_states,
                    probs,
                    routing_map,
                    padding_mask,
                    packed_seq_params=kwargs.get("packed_seq_params"),
                )
                nvtx_range_pop(suffix="mlp")
                return (
                    residual,
                    hidden_states,
                    probs,
                    shared_expert_output,
                    mlp_h_res,
                    mlp_hc_h_post,
                )
            mlp_output_with_bias = self.mlp(hidden_states)
            self.mlp.cudagraph_tensor_store.clear()
            nvtx_range_pop(suffix="mlp")

            # HC post-processing with fused h_res, h_post and BDA.
            recompute_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm
            self.recompute_pre_mlp_layernorm = False
            output = self._forward_post_mlp_with_fused_hyper_connection(
                mlp_output_with_bias, mlp_h_res, residual, mlp_hc_h_post
            )
            self.recompute_pre_mlp_layernorm = recompute_pre_mlp_layernorm
        else:
            if self.config.overlap_moe_expert_parallel_comm:
                assert (
                    len(cuda_graph_output) == 1
                ), "CUDA Graph output should be the attention output."
                hidden_states = cuda_graph_output.pop()
                if not self.is_moe_layer:
                    return hidden_states, None, None, None

                # Whole-attention capture with mHC recompute: the attention-side
                # producer sits inside the graph (its checkpoint cannot register),
                # but the MLP-side mHC group runs eagerly here and must register
                # against the schedule's manager exactly as the split tail does --
                # the schedule builds the manager and calls discard_all_outputs()
                # at group end regardless of which capture shape the layer took.
                mhc_recompute_manager = getattr(self, '_mhc_recompute_manager', None)

                nvtx_range_push(suffix="mlp_hyper_connection")
                hidden_states, mlp_h_res, mlp_hc_h_post, residual = self.mlp_hyper_connection(
                    hidden_states, mhc_recompute_manager=mhc_recompute_manager, return_residual=True
                )
                nvtx_range_pop(suffix="mlp_hyper_connection")

                checkpoint_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm or (
                    mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm
                )
                if checkpoint_pre_mlp_layernorm:
                    self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                        ckpt_manager=mhc_recompute_manager
                    )
                    hidden_states = self.pre_mlp_norm_checkpoint.checkpoint(
                        apply_module(self.pre_mlp_layernorm), hidden_states
                    )
                else:
                    hidden_states = apply_module(self.pre_mlp_layernorm)(hidden_states)
                if isinstance(hidden_states, tuple):
                    if len(hidden_states) != 2:
                        raise ValueError(
                            "When the output of pre_mlp_layernorm is a tuple, it is "
                            f"expected to have 2 elements (output, residual), but "
                            f"got {len(hidden_states)}"
                        )
                    hidden_states, _ = hidden_states

                shared_expert_output = self.mlp.shared_experts_compute(hidden_states)
                # padding_mask mirrors the eager overlap callable and the split
                # tail: the router consumes it in the z-loss mean, dropless
                # gating, and the expert-load counters, so dropping it makes the
                # graphed path drift from eager on padded batches.
                probs, routing_map = self.mlp.route(hidden_states, padding_mask=padding_mask)
                hidden_states, probs = self.mlp.preprocess(
                    hidden_states,
                    probs,
                    routing_map,
                    padding_mask,
                    packed_seq_params=kwargs.get("packed_seq_params"),
                )
                return (
                    residual,
                    hidden_states,
                    probs,
                    shared_expert_output,
                    mlp_h_res,
                    mlp_hc_h_post,
                )

            # Same threading as the dense split tail: the whole-attention capture
            # keeps the MLP-side mHC group eager, so it registers against the
            # manager __call__ stashed on the layer, and the router needs the
            # padding_mask read above.
            output = self._forward_mlp(
                *cuda_graph_output,
                padding_mask=padding_mask,
                input_ids=kwargs.get("input_ids", None),
                packed_seq_params=kwargs.get("packed_seq_params", None),
                mhc_recompute_manager=getattr(self, '_mhc_recompute_manager', None),
            )
        return output, context

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
        hidden_states, h_res, h_post, residual = self.self_attention_hyper_connection(
            hidden_states, mhc_recompute_manager=self._mhc_recompute_manager, return_residual=True
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
        # HC omits forced_released_tensors — the n-stream residual is consumed
        # by the fused kernel above, so the base class's "release residual after
        # commit" trick doesn't apply.
        hidden_states = self.attn_norm_manager.group_offload(hidden_states)
        self.attn_norm_manager = None
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
        hidden_states, mlp_h_res, mlp_hc_h_post, residual = self.mlp_hyper_connection(
            hidden_states, mhc_recompute_manager=self._mhc_recompute_manager, return_residual=True
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

    def supports_two_stage_attention(self) -> bool:
        """Expose the incoming split interface on its supported mHC configurations."""
        return (
            not self.config.fp32_residual_connection
            and not self._uses_mhc_recompute_attn_cuda_graph_split()
            and super().supports_two_stage_attention()
        )

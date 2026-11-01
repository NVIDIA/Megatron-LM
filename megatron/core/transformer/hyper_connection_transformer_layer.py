# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer layer specialization for manifold-constrained hyper-connections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor, nvtx_range_pop, nvtx_range_push

if TYPE_CHECKING:
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager


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
        if self.is_moe_layer and unsupported_moe_cuda_graph_modules.intersection(
            self.config.cuda_graph_modules
        ):
            raise NotImplementedError(
                "HyperConnectionTransformerLayer does not support MoE CUDA graph "
                "scopes. Disable the moe, moe_router, and moe_preprocess CUDA graph modules "
                "when combining mHC with a MoE MLP submodule."
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
        # The mHC weights are replicated across the layer's tensor-parallel group, which sets
        # their checkpoint replica ids.
        self.self_attention_hyper_connection.tp_group = self.tp_group
        self.mlp_hyper_connection.tp_group = self.tp_group

        # Set per-call by __call__ from kwargs so forward can read it without re-piping
        # the manager through the CUDA-graph kwarg path (CheckpointWithoutOutputManager
        # is not a CUDA-graph-supported type and gets stripped during capture). Read by
        # _run_input_layernorm, _apply_self_attn_bda_step, _pre_mlp_layernorm_and_residual, and
        # _apply_mlp_bda_step — do not delete; appears unused only at the class level.
        self._mhc_recompute_manager: Optional['CheckpointWithoutOutputManager'] = None

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
        n-stream hidden states of shape [s, b, n*C].
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
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

        if not self.config.cuda_graph_modules:
            return submodules

        if CudaGraphModule.attn in self.config.cuda_graph_modules:
            submodules.append(self.self_attention_hyper_connection)
        if CudaGraphModule.mlp in self.config.cuda_graph_modules and not self.is_moe_layer:
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

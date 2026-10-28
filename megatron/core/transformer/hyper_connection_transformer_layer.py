# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer layer specialization for manifold-constrained hyper-connections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import (
    HyperConnectionExecutionContext,
    HyperConnectionResidualConnection,
)
from megatron.core.transformer.identity_op import IdentityOp
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

        self.residual_read_self_attn = HyperConnectionResidualConnection(
            self.config, operation="read", use_input_residual=True
        )
        self.residual_write_self_attn = HyperConnectionResidualConnection(
            self.config, operation="write"
        )
        self.residual_read_mlp = HyperConnectionResidualConnection(
            self.config, operation="read", use_input_residual=True
        )
        self.residual_write_mlp = HyperConnectionResidualConnection(self.config, operation="write")

        # mHC has always bypassed this legacy hook, which cannot carry residual state.
        self._legacy_forward_post_mlp = None

        # Keep the manager out of CUDA-graph kwargs; norm wrappers and residual calls
        # read the same per-call manager from this attribute.
        self._mhc_recompute_manager: Optional['CheckpointWithoutOutputManager'] = None

    def __call__(self, *args, **kwargs):
        if kwargs.get("residual_stream_recompute_context") is not None:
            raise ValueError(
                "mHC does not accept residual_stream_recompute_context; use mhc_recompute_manager."
            )
        # Pull the manager off kwargs before super().__call__ hands them to the
        # CUDA-graph machinery (which can't handle a CheckpointWithoutOutputManager).
        # forward() reads the value back from self.
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        return super().__call__(*args, **kwargs)

    def _residual_recompute_context(
        self, *, is_mlp: bool = False
    ) -> ResidualStreamRecomputeContext | None:
        """Share the block manager; only the final MLP write ends a replay block."""
        manager = self._mhc_recompute_manager
        if manager is None:
            return None
        return ResidualStreamRecomputeContext(
            manager,
            is_block_end=is_mlp and getattr(manager, "is_last_layer_in_recompute_block", False),
        )

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
                "mHC does not accept residual_stream_recompute_context; use mhc_recompute_manager."
            )
        owner = self.mlp_hyper_connection if is_mlp else self.self_attention_hyper_connection
        # This operation context must not select WR's norm/offload policy in the base.
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
                "mHC does not accept residual_stream_recompute_context; use mhc_recompute_manager."
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
        return super()._run_input_layernorm(
            hidden_states, mhc_recompute_manager=self._mhc_recompute_manager
        )

    def _pre_mlp_layernorm_and_residual(self, hidden_states):
        """Use the shared MLP schedule with the call's mHC norm manager."""
        return super()._pre_mlp_layernorm_and_residual(
            hidden_states, mhc_recompute_manager=self._mhc_recompute_manager
        )

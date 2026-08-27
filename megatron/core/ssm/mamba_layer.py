# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.residual_connection import (
    ResidualConnectionSpec,
    build_residual_connection,
)
from megatron.core.transformer.residual_recompute import (
    ResidualStreamRecomputeContext,
    checkpoint_residual_read,
    checkpoint_residual_write,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params


@dataclass
class MambaLayerSubmodules:
    """
    Configuration class for specifying the submodules of a Mamba layer.

    This class defines the structure and default implementations for various
    components of a Mamba layer, allowing for flexible customization of the
    layer's architecture.

    Args:
        norm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        mixer (Union[ModuleSpec, type]): Specification for the along-sequence mixing mechanism.
        mamba_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the mixer.
        residual_connection: Optional residual-stream read/write around the Mamba mixer.
    """

    norm: LayerNormBuilder = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)

    # Keep optional architecture seams after existing fields so positional construction remains
    # backward compatible.
    residual_connection: ResidualConnectionSpec = None


class MambaLayer(GraphableMegatronModule):
    """
    A single Mamba layer.

    Mamba layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
        name: str | None = None,
    ):
        """Initialize Mamba Layer.

        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for MambaLayer"
        self.tp_group = pg_collection.tp

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_dropout
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            d_model=self.config.hidden_size,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=(name + f".mixer") if name is not None else None,
        )
        self.norm = submodules.norm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.mamba_bda = build_module(submodules.mamba_bda)
        self.residual_connection = build_residual_connection(
            submodules.residual_connection,
            config=self.config,
            layer_number=self.layer_number,
            branch_name="mamba",
            pg_collection=pg_collection,
            name=(name + ".residual_connection") if name is not None else None,
        )
        self.residual_stream_hidden_size = (
            self.residual_connection.residual_stream_hidden_size
            if self.residual_connection is not None
            else self.config.hidden_size
        )
        if self.config.wide_residual is not None:
            expected_stream_hidden_size = (
                self.config.wide_residual.num_streams * self.config.hidden_size
            )
            if self.residual_connection is None:
                raise ValueError("A wide-residual Mamba layer requires a residual connection.")
            if self.residual_stream_hidden_size != expected_stream_hidden_size:
                raise ValueError(
                    "wide_residual Mamba connections must carry num_streams * hidden_size "
                    f"features, expected {expected_stream_hidden_size}, got "
                    f"{self.residual_stream_hidden_size}."
                )
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def create_mcore_cudagraph_manager(self, config):
        """Register the mamba layer for cudagraphs."""
        assert self.config.cuda_graph_impl == "local"

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if (
            not self.config.cuda_graph_modules
            and self.config.inference_cuda_graph_scope != InferenceCudaGraphScope.block
        ) or CudaGraphModule.mamba in self.config.cuda_graph_modules:
            self.cudagraph_manager = CudaGraphManager(config)

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:
        """Returns the Mamba conv and ssm states shapes per request."""
        return self.mixer.mamba_state_shapes_per_request()

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """Return CUDA-graph inputs using the configured residual-stream width."""

        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)
        if self.residual_connection is not None:
            hidden_states = static_inputs["hidden_states"]
            static_inputs["hidden_states"] = torch.ones(
                (hidden_states.shape[0], hidden_states.shape[1], self.residual_stream_hidden_size),
                dtype=hidden_states.dtype,
                requires_grad=hidden_states.requires_grad,
                device=hidden_states.device,
            )
        return static_inputs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        residual_stream_recompute_context: ResidualStreamRecomputeContext | None = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Perform a forward pass through the Mamba layer.

        This method implements the core computation of a Mamba layer, including
        the convolution and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Whole-layer + mixer lens spans, mirroring transformer_layer.py so the hybrid
        # model's Mamba layers aren't a blind spot in the per-layer breakdown (they were
        # ~34s of uninstrumented first-iteration warmup). No-op unless the 'layer' span
        # group is enabled, so zero cost on normal runs.
        with _otel_managed_span(
            'layer', 'megatron.layer.forward', **{'megatron.layer_number': self.layer_number}
        ):
            residual_connection = self.residual_connection
            recompute_context = (
                residual_stream_recompute_context if residual_connection is not None else None
            )
            connection_state = None
            if residual_connection is not None:
                if recompute_context is None:
                    hidden_states, connection_state = apply_module(residual_connection)(
                        hidden_states,
                        operation="read",
                        fp32_residual_connection=self.config.fp32_residual_connection,
                    )
                else:
                    hidden_states, connection_state = checkpoint_residual_read(
                        residual_connection,
                        hidden_states,
                        recompute_context,
                        fp32_residual_connection=self.config.fp32_residual_connection,
                    )
                residual = residual_connection.residual_stream(connection_state)
            else:
                residual = hidden_states

            if residual_connection is None and self.config.fp32_residual_connection:
                residual = residual.float()

            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
            if recompute_context is not None and not isinstance(self.norm, IdentityOp):
                hidden_states = recompute_context.checkpoint(apply_module(self.norm), hidden_states)
            else:
                hidden_states = apply_module(self.norm)(hidden_states)

            # Mamba mixer: conv + selective SSM/SSD -- the compute block, analog of the
            # transformer layer's self_attention/mlp (this is where the SSD kernel autotune
            # lands on the first pass).
            with _otel_managed_span('layer', 'megatron.layer.mamba'):
                mixer_out_with_bias = self.mixer(
                    hidden_states,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                )

            if residual_connection is not None:
                if connection_state is None:
                    raise RuntimeError("Missing state for the Mamba residual connection.")
                if recompute_context is not None and not recompute_context.is_block_end:
                    hidden_states = checkpoint_residual_write(
                        residual_connection,
                        mixer_out_with_bias,
                        connection_state,
                        recompute_context,
                        dropout_probability=self.hidden_dropout,
                        training=self.training,
                    )
                else:
                    with self.bias_dropout_add_exec_handler():
                        hidden_states = apply_module(residual_connection)(
                            mixer_out_with_bias,
                            operation="write",
                            state=connection_state,
                            dropout_probability=self.hidden_dropout,
                            training=self.training,
                        )
            else:
                with self.bias_dropout_add_exec_handler():
                    hidden_states = self.mamba_bda(
                        training=self.training, fused=self.config.bias_dropout_fusion
                    )(mixer_out_with_bias, residual, self.hidden_dropout)

            return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the mamba layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the mamba layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch `self.current_microbatch` using TE
        interface. TransformerEngine versions>=1.10 allow keyword arguments with CUDA graph.
        However, CUDA graph accepts only Tensor inputs.
        Hence, `inference_context` is excluded from input list.
        """
        assert kwargs.get('inference_context') is None, (
            "CUDA graph accepts only Tensor inputs. inference_context is excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        return super()._te_cuda_graph_replay(*args, **kwargs)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs.
        if (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('inference_context') is None
            and not torch.is_inference_mode_enabled()  # for inference eager dummy_forward
        ):
            return True
        elif InferenceMode.is_active() and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs.get('inference_context') is not None
            and not self.config.cuda_graph_modules  # empty-list = per-layer CUDA graphs
        ):
            context = kwargs['inference_context']
            using_cuda_graph = (context.is_static_batching() and context.is_decode_only()) or (
                not context.is_static_batching() and context.using_cuda_graph_this_step()
            )
            return using_cuda_graph
        return False

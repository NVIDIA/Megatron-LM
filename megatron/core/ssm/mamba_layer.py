# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import functools
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.context_parallel.chunkwise import PackedSequenceCPMetadata
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
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
    """

    norm: LayerNormBuilder = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


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
        self.bias_dropout_add_exec_handler = torch.enable_grad
        self.recompute_mamba_mixer = (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "mamba" in self.config.recompute_modules
        )

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

    def _run_mamba_mixer(
        self,
        hidden_states: Tensor,
        inference_context: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
        packed_sequence_cp_metadata: PackedSequenceCPMetadata | None = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the Mamba mixer, checkpointing it when selective recompute is enabled."""
        mixer_kwargs = {
            "inference_context": inference_context,
            "packed_seq_params": packed_seq_params,
        }
        if packed_sequence_cp_metadata is not None:
            mixer_kwargs["packed_sequence_cp_metadata"] = packed_sequence_cp_metadata

        if self.recompute_mamba_mixer and self.training and inference_context is None:
            # CUDA graphs capture the mixer directly; nested activation checkpointing cannot
            # run inside graph warmup/capture. Keep the TE path consistent with
            # tensor_parallel.checkpoint, which has the same bypass internally.
            from megatron.core.transformer.cuda_graphs import is_graph_capturing, is_graph_warmup

            if is_graph_warmup() or is_graph_capturing():
                return self.mixer(hidden_states, **mixer_kwargs)

            if self.config.fp8 or self.config.fp4:
                # TE checkpointing enters the activation-recompute phase so quantized
                # amax/scaling state is not updated again during the backward recompute.
                # Import here to avoid a circular import.
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    self.mixer,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.tp_group,
                    hidden_states,
                    **mixer_kwargs,
                )

            return tensor_parallel.checkpoint(
                functools.partial(self.mixer, **mixer_kwargs), False, hidden_states
            )

        return self.mixer(hidden_states, **mixer_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        packed_sequence_cp_metadata: PackedSequenceCPMetadata | None = None,
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
            packed_sequence_cp_metadata (PackedSequenceCPMetadata, optional): Rank-local
                packed-sequence metadata for chunkwise CP.

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
            residual = hidden_states
            if self.config.fp32_residual_connection:
                residual = residual.float()

            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
            hidden_states = apply_module(self.norm)(hidden_states)

            # Mamba mixer: conv + selective SSM/SSD -- the compute block, analog of the
            # transformer layer's self_attention/mlp (this is where the SSD kernel autotune
            # lands on the first pass).
            with _otel_managed_span('layer', 'megatron.layer.mamba'):
                mixer_out_with_bias = self._run_mamba_mixer(
                    hidden_states, inference_context, packed_seq_params, packed_sequence_cp_metadata
                )

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

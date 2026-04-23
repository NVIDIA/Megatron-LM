# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, Union

import torch
from torch import Tensor

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.fusions.deferred_add import (
    AbsorbMode,
    DeferMode,
    DeferredAdd,
    add_fusion_disabled_by_env,
    should_absorb_add,
    should_defer_add,
)
from megatron.core.fusions.fused_add_rmsnorm import fused_add_rmsnorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormInterface
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import deprecate_inference_params


class LayerNormBuilder(Protocol):
    """A protocol showing how MambaLayer expects to construct its LayerNorm."""

    def __call__(self, config: TransformerConfig, hidden_size: int, /) -> LayerNormInterface: ...


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
    ):
        """Initialize Mamba Layer."""
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for MambaLayer"

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
        )
        self.norm = submodules.norm(self.config, self.config.hidden_size)
        self.mamba_bda = build_module(submodules.mamba_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

        # Cross-layer deferred-add modes. Set by ``wire_add_fusion`` when
        # this layer is paired with a compatible neighbour. Runtime guards
        # in ``forward`` still fall back to the unfused path under
        # training or grad-enabled modes.
        self._absorb_mode = AbsorbMode.NONE
        self._defer_mode = DeferMode.NONE

    def _entry_norm_module(self):
        """Module owning this layer's entry RMSNorm weight.

        Most ``inference_optimized`` Mamba specs leave ``self.norm`` as
        ``IdentityOp`` and put the entry norm inside
        ``self.mixer.in_proj`` (an ``InferenceLayerNormColumnParallelLinear``).
        Legacy specs with a standalone ``self.norm`` are also supported.
        """
        in_proj = getattr(self.mixer, "in_proj", None)
        if in_proj is not None and hasattr(in_proj, "layer_norm_weight"):
            return in_proj
        if hasattr(self.norm, "weight") and self.norm.weight is not None:
            return self.norm
        return None

    def native_defer_mode(self) -> DeferMode:
        # ``mamba_bda`` is a plain residual-add in inference: no bias, no
        # dropout. Deferring it is a pure transformation.
        if add_fusion_disabled_by_env():
            return DeferMode.NONE
        return DeferMode.MAMBA_EXIT

    def native_absorb_mode(self) -> AbsorbMode:
        if add_fusion_disabled_by_env():
            return AbsorbMode.NONE
        m = self._entry_norm_module()
        # If the norm is fused into in_proj, we need ``skip_input_norm`` so
        # we can tell it not to re-run the norm this call.
        if m is None:
            return AbsorbMode.NONE
        if m is self.norm or hasattr(m, "skip_input_norm"):
            return AbsorbMode.MAMBA_ENTRY
        return AbsorbMode.NONE

    def create_mcore_cudagraph_manager(self, config):
        """Register the mamba layer for cudagraphs."""
        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if not self.config.cuda_graph_scope or CudaGraphScope.mamba in self.config.cuda_graph_scope:
            self.cudagraph_manager = CudaGraphManager(config)

    def mamba_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:
        """Returns the Mamba conv and ssm states shapes per request."""
        return self.mixer.mamba_state_shapes_per_request()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
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

        # --- Entry: absorb a DeferredAdd or run own input norm. ---
        if should_absorb_add(self, hidden_states):
            residual, hidden_states = self._absorb_add_at_entry(hidden_states)
        else:
            if isinstance(hidden_states, DeferredAdd):
                hidden_states = hidden_states.residual + hidden_states.delta
            residual = hidden_states
            if self.config.fp32_residual_connection:
                residual = residual.float()
            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
            hidden_states = apply_module(self.norm)(hidden_states)

        # --- Mixer ---
        mixer_out_with_bias = self.mixer(
            hidden_states, inference_context=inference_context, packed_seq_params=packed_seq_params
        )

        # --- Exit: defer the residual-add or run own BDA. ---
        if should_defer_add(self):
            return DeferredAdd(residual=residual, delta=mixer_out_with_bias[0])
        with self.bias_dropout_add_exec_handler():
            return self.mamba_bda(
                training=self.training, fused=self.config.bias_dropout_fusion
            )(mixer_out_with_bias, residual, self.hidden_dropout)

    def _absorb_add_at_entry(self, incoming: DeferredAdd):
        """Consume ``incoming`` via ``fused_add_rmsnorm``; return
        ``(residual, normed)``.

        If the entry norm is fused inside ``mixer.in_proj``, arms its
        ``skip_input_norm`` so the in_proj doesn't redo the norm.
        """
        norm_module = self._entry_norm_module()
        weight = (
            norm_module.layer_norm_weight
            if norm_module is not self.norm
            else self.norm.weight
        )
        normed, residual = fused_add_rmsnorm(
            incoming.delta, incoming.residual, weight, eps=self.config.layernorm_epsilon
        )
        if norm_module is not self.norm:
            # The consumer (``in_proj``) clears this flag after it fires.
            norm_module.skip_input_norm = True
        return residual, normed

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
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs.get('inference_context') is not None
            and not self.config.cuda_graph_scope  # empty-list = per-layer CUDA graphs
        ):
            context = kwargs['inference_context']
            using_cuda_graph = (context.is_static_batching() and context.is_decode_only()) or (
                not context.is_static_batching() and context.using_cuda_graph_this_step()
            )
            return using_cuda_graph
        return False

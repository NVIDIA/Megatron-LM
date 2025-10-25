# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.cuda_graphs import CudaGraphManager, is_graph_capturing
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params

try:
    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


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

    norm: Union[ModuleSpec, type] = IdentityOp
    mixer: Union[ModuleSpec, type] = IdentityOp
    mamba_bda: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class MambaLayer(MegatronModule):
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
        residual_in_fp32=False,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        """Initialize Mamba Layer."""
        super().__init__(config)
        assert model_comm_pgs is not None, "model_comm_pgs must be provided for MambaLayer"

        # Enable cuda graphs.
        if config.enable_cuda_graph or config.external_cuda_graph:
            assert not (
                config.enable_cuda_graph and config.external_cuda_graph
            ), "Cudagraphs and external cudagraphs cannot be enabled at the same time"
            if config.enable_cuda_graph:
                self.cudagraph_manager = CudaGraphManager(config)
            else:
                # List to store CUDA graphs. A list of `N` CUDA graphs for this layer where N is
                # the number of microbatches. Multiple CUDA graphs per layer is required to support
                # pipelining which requires running FWD graph of multiple microbatches before BWD
                # graph. To enable CUDA graph, this list should be populated in the model training
                # script with the graphs returned by make_graphed_callables API before the first
                # training step.
                self.cuda_graphs = []
                # List to store forward pre-hooks. Forward pre-hooks are not captured into CUDA
                # graphs. Those hooks and args are collected in this list and should be manually
                # triggered before CUDA Graph running. This is required to ensure the correct param
                # all-gather overlap with forward compute.
                self.cuda_graph_manual_hooks = []
                self.current_microbatch = -1
 
        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout
        self.mamba_layer_recompute = (
            config.recompute_granularity == 'selective' and "mamba" in config.recompute_modules
        )
        self.mixer = build_module(
            submodules.mixer,
            self.config,
            d_model=self.config.hidden_size,
            layer_number=layer_number,
            model_comm_pgs=model_comm_pgs,
        )
        self.norm = build_module(submodules.norm, self.config, self.config.hidden_size)
        self.mamba_bda = build_module(submodules.mamba_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        *,
        inference_params: Optional[BaseInferenceContext] = None,
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

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # Mamba forward: norm -> mixer -> bias_dropout_add
        def custom_forward(hidden_states, residual):
            hidden_states = hidden_states.to(dtype=self.config.params_dtype)
            hidden_states = self.norm(hidden_states)

            mixer_out_with_bias = self.mixer(hidden_states, inference_context=inference_context)

            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mamba_bda(
                    training=self.training, fused=self.config.bias_dropout_fusion
                )(mixer_out_with_bias, residual, self.hidden_dropout)

            return hidden_states

        if self.mamba_layer_recompute:
            if self.config.fp8 or self.config.fp4:
                output = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    residual,
                )
            else:
                output = tensor_parallel.checkpoint(custom_forward, False, hidden_states, residual)
        else:
            output = custom_forward(hidden_states, residual)

        return output

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate the inference cache."""
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """
        Get the static inputs for the transformer layer.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        # Calculate data shape related values.
        context_parallel_size = self.config.context_parallel_size
        slen_per_cp = seq_length // context_parallel_size
        sequence_parallel = self.config.sequence_parallel
        tensor_model_parallel_size = self.config.tensor_model_parallel_size
        slen_per_cptp = (
            slen_per_cp // tensor_model_parallel_size if sequence_parallel else slen_per_cp
        )

        static_inputs = {}
        static_inputs["hidden_states"] = torch.ones(
            (slen_per_cptp, micro_batch_size, self.config.hidden_size),
            dtype=torch.bfloat16,
            requires_grad=True,
            device=torch.cuda.current_device(),
        )
        return static_inputs

    def setup_manual_hooks(self, make_hook_func):
        """
        Set CUDA Graph manual hooks for the modules that contain direct parameters and are
        covered by cudagraphs.
        """
        self.cuda_graph_manual_hooks = []

        param_modules = []
        for submodule in self.modules():
            if next(submodule.parameters(recurse=False), None) is not None:
                # Module contains direct parameters.
                param_modules.append(submodule)
        if len(param_modules) > 0:
            for module in param_modules:
                self.cuda_graph_manual_hooks.append((make_hook_func(), (module,)))

    def _cuda_graph_capture(self, *args, **kwargs):
        """
        CUDA Graph capture for this layer. There are some differences from the normal pass:
        1. In some conditions CUDA graph cannot cover the entire layer. The `cuda_graph_scope`
           attribute can be set to control the scope of the CUDA graph.
        2. If context is None, it cannot be returned as output.
        """
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.pop("hidden_states")
        return self.forward(hidden_states, attention_mask=None)

    def _cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch
        `self.current_microbatch`. TransformerEngine versions>=1.10
        allow keyword arguments with CUDA graph. However, CUDA graph
        acccepts only Tensor inputs and Tensor outputs. Hence,
        `inference_context` and `packed_seq_params` are excluded from
        input list while output is limited to `hidden_states`.
        """
        if len(args) == 1:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.pop("hidden_states")
        cudagraph_args = [hidden_states]
        cudagraph_kwargs = {'is_first_microbatch': self.current_microbatch == 0}

        cg_index = self.current_microbatch % len(self.cuda_graphs)
        assert (
            'inference_context' not in kwargs or kwargs['inference_context'] is None
        ), "CUDA graph accepts only Tensor inputs."

        for hook, hook_args in self.cuda_graph_manual_hooks:
            hook(*hook_args)
        return self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)

    def __call__(self, *args, **kwargs):

        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return self.cudagraph_manager(self, args, kwargs)
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs['inference_context'].is_decode_only()
        ):
            return self.cudagraph_manager(self, args, kwargs)
        elif (
            self.config.external_cuda_graph
            and self.training
            and (is_graph_capturing() or self.cuda_graphs)
        ):
            if not self.cuda_graphs:
                # Do CUDA Graphs capture.
                cuda_graph_func = self._cuda_graph_capture
            else:
                # Do CUDA Graphs replay.
                cuda_graph_func = self._cuda_graph_replay
            return cuda_graph_func(*args, **kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)

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

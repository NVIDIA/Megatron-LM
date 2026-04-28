# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import CheckpointManager
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor


@dataclass
class HybridStackSubmodules:
    """
    A class for the module specs for the HybridStack.
    """

    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    gdn_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    dsa_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp
    mtp_block_spec: Optional[ModuleSpec] = None


class HyperConnectionHybridLayer(MegatronModule):
    """Layer-boundary mHC wrapper for HybridStack layers.

    Hybrid layers already own their local residual paths. For this initial
    integration we treat each hybrid layer as a single function by aggregating
    n streams to the layer input, running the existing layer, and feeding only
    the layer delta back through mHC expansion. The expansion path intentionally
    uses zero additional dropout because the wrapped hybrid layer has already
    applied its local dropout/residual update before the delta is computed.
    """

    def __init__(self, config: TransformerConfig, layer: MegatronModule) -> None:
        super().__init__(config=config)
        self.inner_layer = layer
        self.layer_number = layer.layer_number
        self.hyper_connection = HyperConnectionModule(config=config, layer_number=self.layer_number)
        if config.params_dtype is not None:
            self.hyper_connection.to(dtype=config.params_dtype)
        if hasattr(layer, 'tp_group'):
            self.tp_group = layer.tp_group

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """Delegate Mamba inference state shape requests to the wrapped layer."""
        if not hasattr(self.inner_layer, 'mamba_state_shapes_per_request'):
            return None
        return self.inner_layer.mamba_state_shapes_per_request()

    def _call_inner_layer(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext],
        rotary_pos_emb: Optional[Tensor],
        sequence_len_offset: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(self.inner_layer, TransformerLayer):
            output = self.inner_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                _called_from_hybrid_mhc_wrapper=True,
            )
        else:
            # Non-transformer layers (e.g. Mamba, GDN) do not accept
            # rotary_pos_emb / sequence_len_offset / padding_mask — pass only
            # the common arguments. New layer types that consume any of these
            # must add explicit handling here.
            output = self.inner_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )

        if isinstance(output, tuple):
            context = output[1] if len(output) > 1 else None
            return output[0], context
        return output, None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        mhc_recompute_manager=None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the wrapped hybrid layer through one layer-boundary mHC update."""
        residual = hidden_states
        aggregated, h_res, h_post = self.hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager
        )
        layer_output, context = self._call_inner_layer(
            aggregated,
            attention_mask,
            inference_context,
            rotary_pos_emb,
            sequence_len_offset,
            packed_seq_params,
            padding_mask,
        )
        # The inner hybrid layer already applied its own local residual/dropout, so
        # it returns `aggregated + f(aggregated)`. We feed only the function
        # delta `f(aggregated)` into the n-stream BDA so it does not double-count
        # the residual that mHC owns. The temporary [s, b, C] tensor here is the
        # simplest correct form; a future optimization could fuse the subtraction
        # into `h_res_h_post_bda` to avoid the allocation.
        # Sanity check: this contract requires the inner layer to preserve shape;
        # any mismatch indicates a future layer type is breaking the residual
        # assumption and would silently corrupt the n-stream state.
        assert layer_output.shape == aggregated.shape, (
            "HyperConnectionHybridLayer requires inner layers to preserve "
            f"hidden-state shape. Got {tuple(layer_output.shape)} from inner layer "
            f"vs {tuple(aggregated.shape)} input — layer must add its own residual."
        )
        layer_delta = layer_output - aggregated
        # `dropout_prob=0.0` already disables dropout regardless of training mode;
        # `training=self.training` is more semantically accurate than hard-coding
        # False during a training-mode forward.
        hidden_states = self.hyper_connection.h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            (layer_delta, None),
            dropout_prob=0.0,
            training=self.training,
            fused=False,
            manager=mhc_recompute_manager,
        )
        return hidden_states, context


class HybridStack(GraphableMegatronModule, MegatronModule):
    """
    Constructor for the HybridStack class.

    Args:
        config (TransformerConfig): the model configuration
        submodules (HybridStackSubmodules): the submodules for the stack
        pre_process (bool, optional): whether to include an embedding layer.
            Defaults to True.
        layer_type_list (list, optional): pre-computed list of layer type symbols for
            this pipeline segment. When provided (by HybridModel), pipeline stage
            selection has already been done via '|' separators in the pattern.
        pp_layer_offset (int, optional): the global layer offset for this pipeline
            segment. Defaults to 0.
        post_layer_norm (bool, optional): whether to include a final layer norm.
            Defaults to True.
        post_process (bool, optional): whether to include an output layer.
            Defaults to True.
        device (optional): the device to use. Defaults to None.
        dtype (optional): the data type to use. Defaults to None.
        pg_collection (ProcessGroupCollection): the required model communication
            process groups to use.
        is_mtp_layer (bool, optional): whether this is an MTP layer. Defaults to False.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridStackSubmodules,
        pre_process: bool = True,
        layer_type_list: Optional[list[str]] = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.is_mtp_layer = is_mtp_layer

        assert pg_collection is not None, "pg_collection must be provided for HybridStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp

        # Required for pipeline parallel schedules
        self.input_tensor = None
        self.pg_collection = pg_collection

        assert layer_type_list is not None, (
            "layer_type_list must be provided. It should be pre-computed from "
            "--hybrid-layer-pattern by HybridModel."
        )
        self.layer_type_list = layer_type_list

        # Build layers from the pre-selected segment
        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(self.layer_type_list):
            layer_number = i + 1 + pp_layer_offset
            if self.config.fp8:
                quant_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
            elif self.config.fp4:
                quant_init_context = get_fp4_context(self.config, i + pp_layer_offset, is_init=True)
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                if layer_type == LayerSymbols.MAMBA:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pp_layer_offset=pp_layer_offset,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.ATTENTION:
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.DS_ATTENTION:
                    layer = build_module(
                        submodules.dsa_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.MLP:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.GDN:
                    layer = build_module(
                        submodules.gdn_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                    )
                else:
                    raise ValueError("unexpected layer_type")
            if self.config.enable_hyper_connections:
                layer = HyperConnectionHybridLayer(config=self.config, layer=layer)
            self.layers.append(layer)

        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """
        Returns the Mamba conv and ssm states shapes per input sequence
        if this block contains Mamba layers (this may not be the case with PP > 1).
        """
        for layer_type, layer in zip(self.layer_type_list, self.layers):
            if layer_type == LayerSymbols.MAMBA:
                return layer.mamba_state_shapes_per_request()
        return None

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        if (
            not self.training
            and hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                kwargs.get('inference_context') is not None
                or kwargs.get('inference_params') is not None
            )
            and CudaGraphScope.full_iteration_inference in self.config.cuda_graph_scope
        ):
            if kwargs['inference_context'].is_static_batching():
                using_cuda_graph = kwargs['inference_context'].is_decode_only()
            else:
                using_cuda_graph = kwargs['inference_context'].using_cuda_graph_this_step()

            if using_cuda_graph:
                return True
        return False

    def __call__(self, *args, **kwargs):
        if self._should_call_local_cudagraph(*args, **kwargs):
            kwargs['hidden_states'] = (
                kwargs['hidden_states'].unwrap()
                if isinstance(kwargs['hidden_states'], WrappedTensor)
                else kwargs['hidden_states']
            )
            return super().__call__(*args, **kwargs)[0]
        return super().__call__(*args, **kwargs)

    def _build_mhc_recompute_layer_plan(
        self, use_mhc_recompute: bool
    ) -> Tuple[List[Optional[CheckpointManager]], List[bool]]:
        """Pre-build per-layer MHC recompute managers and block-end markers."""
        num_layers = len(self.layers)
        layer_managers: List[Optional[CheckpointManager]] = [None] * num_layers
        is_recompute_block_end: List[bool] = [False] * num_layers

        if not use_mhc_recompute or num_layers == 0:
            return layer_managers, is_recompute_block_end

        mhc_recompute_layer_num = self.config.mhc_recompute_layer_num
        mhc_manager = CheckpointManager()

        for l_no in range(num_layers):
            is_last_in_stack = l_no == num_layers - 1
            is_last_in_recompute_block = is_last_in_stack
            if mhc_recompute_layer_num is not None:
                is_last_in_recompute_block = is_last_in_stack or (
                    (l_no + 1) % mhc_recompute_layer_num == 0
                )

            layer_managers[l_no] = mhc_manager
            is_recompute_block_end[l_no] = is_last_in_recompute_block

            if is_last_in_recompute_block and not is_last_in_stack:
                mhc_manager = CheckpointManager()

        return layer_managers, is_recompute_block_end

    @staticmethod
    def _finalize_mhc_recompute_layer(
        mhc_manager: Optional[CheckpointManager],
        hidden_states: Tensor,
        is_last_in_recompute_block: bool,
    ) -> None:
        """Finalize MHC recompute state for the current layer when a block ends."""
        if mhc_manager is not None and is_last_in_recompute_block:
            mhc_manager.discard_all_outputs_and_register_unified_recompute(hidden_states)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask=None,
    ):
        """
        Forward function of the HybridStack class.

        It either returns the Loss values if labels are given or the
            final hidden units

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): the input tensor.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): the attention mask.
            inference_context (BaseInferenceContext): the inference parameters.
            rotary_pos_emb (Tensor, optional): the rotary positional embeddings.
                Defaults to None.
        Returns:
            Tensor: the output tensor.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if self.config.enable_hyper_connections and self.pre_process:
            hidden_states = HyperConnectionModule.input_expand(
                hidden_states, self.config.num_residual_streams
            )

        if inference_context and inference_context.is_static_batching():
            # NOTE(bnorick): match BaseInferenceContext attributes for
            # mamba_ssm.utils.generation.BaseInferenceContext,
            # this hack supports eval
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            (
                (
                    self.config.cuda_graph_impl == "local"
                    and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
                )
                or self.config.flash_decode
            )
            and inference_context
            and inference_context.is_static_batching()
            and not self.training
        ):
            current_batch_size = hidden_states.shape[1]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device='cuda',
            )
        else:
            sequence_len_offset = None

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        use_fp4_context = self.config.fp4 is not None
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        if use_inner_fp8_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp8_context(config, layer_number)

        elif use_fp4_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp4_context(config, layer_number)

        else:

            def get_inner_quant_context(config, layer_number):
                return nullcontext()

        use_mhc_recompute = (
            self.training
            and self.config.enable_hyper_connections
            and self.config.recompute_granularity == 'selective'
            and "mhc" in self.config.recompute_modules
        )
        mhc_layer_managers, mhc_is_last_in_recompute_block = self._build_mhc_recompute_layer_plan(
            use_mhc_recompute
        )

        with outer_fp8_context:
            for l_no, layer in enumerate(self.layers):
                # Layers have 1-indexed layer numbers attribute.
                inner_quant_context = get_inner_quant_context(self.config, layer.layer_number - 1)
                mhc_manager = mhc_layer_managers[l_no]
                if mhc_manager is not None:
                    mhc_manager.is_last_layer_in_recompute_block = (
                        mhc_is_last_in_recompute_block[l_no]
                    )

                with inner_quant_context:
                    if isinstance(layer, (TransformerLayer, HyperConnectionHybridLayer)):
                        layer_kwargs = dict(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                        if mhc_manager is not None and isinstance(
                            layer, HyperConnectionHybridLayer
                        ):
                            layer_kwargs["mhc_recompute_manager"] = mhc_manager
                        hidden_states, _ = layer(**layer_kwargs)
                    else:  # MambaLayer, Expert, or MLP
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                        )

                # The attention layer (currently a simplified transformer layer)
                # outputs a tuple of (hidden_states, context). Context is intended
                # for cross-attention, and is not needed in our model.
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

                self._finalize_mhc_recompute_layer(
                    mhc_manager=mhc_manager,
                    hidden_states=hidden_states,
                    is_last_in_recompute_block=mhc_is_last_in_recompute_block[l_no],
                )

        if self.config.enable_hyper_connections and self.post_process:
            hidden_states = HyperConnectionModule.output_contract(
                hidden_states, self.config.num_residual_streams
            )

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        # Ensure that the tensor passed between pipeline parallel stages is
        # viewless. See related notes in TransformerBlock and TransformerLayer
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return hidden_states

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Optional[tuple] = None,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """
        Returns a sharded state dictionary for the current object.

        This function constructs a sharded state dictionary by iterating over the layers
        in the current object, computing the sharded state dictionary for each layer,
        and combining the results into a single dictionary.

        Parameters:
            prefix (str): The prefix to use for the state dictionary keys.
            sharded_offsets (tuple): The sharded offsets to use for the state dictionary.
            metadata (dict): Additional metadata to use when computing the sharded state dictionary.

        Returns:
            dict: The sharded state dictionary for the current object.
        """

        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'

        for local_layer_idx, layer in enumerate(self.layers):

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = (
                f'{layer_prefix}{local_layer_idx}.'  # module list index in HybridStack
            )

            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )

            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module,
                        f'{prefix}{name}.',
                        sharded_offsets,
                        metadata,
                        tp_group=self.tp_group,
                    )
                )

        return sharded_state_dict


# Backward-compatible aliases
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack

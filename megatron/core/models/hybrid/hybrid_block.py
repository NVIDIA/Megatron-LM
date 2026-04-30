# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    LayerPatternItem,
    Symbols as LayerSymbols,
    get_layer_type_physical_count,
    is_layer_group,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
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


class HybridStack(MegatronModule):
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
        layer_type_list: Optional[list[LayerPatternItem]] = None,
        pp_layer_offset: int = 0,
        logical_layer_offset: int = 0,
        is_layer_group_stack: bool = False,
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
        self.logical_layer_offset = logical_layer_offset
        self.is_layer_group_stack = is_layer_group_stack

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
        physical_layer_offset = pp_layer_offset
        for layer_type in self.layer_type_list:
            layer_number = physical_layer_offset + 1
            if is_layer_group(layer_type):
                quant_init_context = nullcontext()
            elif self.config.fp8:
                quant_init_context = get_fp8_context(
                    self.config, physical_layer_offset, is_init=True
                )
            elif self.config.fp4:
                quant_init_context = get_fp4_context(
                    self.config, physical_layer_offset, is_init=True
                )
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                if is_layer_group(layer_type):
                    layer = HybridStack(
                        config=self.config,
                        submodules=submodules,
                        pre_process=True,
                        layer_type_list=list(layer_type),
                        pp_layer_offset=physical_layer_offset,
                        logical_layer_offset=logical_layer_offset + len(self.layers),
                        is_layer_group_stack=True,
                        post_layer_norm=False,
                        post_process=False,
                        device=device,
                        dtype=dtype,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                    )
                elif layer_type == LayerSymbols.MAMBA:
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
            self.layers.append(layer)
            physical_layer_offset += get_layer_type_physical_count(layer_type)

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
            if is_layer_group(layer_type):
                shapes = layer.mamba_state_shapes_per_request()
                if shapes is not None:
                    return shapes
            elif layer_type == LayerSymbols.MAMBA:
                return layer.mamba_state_shapes_per_request()
        return None

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
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

        if inference_context and inference_context.is_static_batching():
            # NOTE(bnorick): match BaseInferenceContext attributes for
            # mamba_ssm.utils.generation.BaseInferenceContext,
            # this hack supports eval
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if sequence_len_offset is not None:
            pass
        elif (
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

        with outer_fp8_context:
            for layer in self.layers:
                # Layers have 1-indexed layer numbers attribute.
                if isinstance(layer, HybridStack):
                    inner_quant_context = nullcontext()
                else:
                    inner_quant_context = get_inner_quant_context(
                        self.config, layer.layer_number - 1
                    )
                with inner_quant_context:
                    if isinstance(layer, HybridStack):
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                    elif isinstance(layer, TransformerLayer):
                        hidden_states, _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
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

        return self._sharded_state_dict(
            prefix=prefix,
            sharded_offsets=sharded_offsets,
            metadata=metadata,
            sharded_layer_prefix=None,
        )

    def _sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Optional[tuple] = None,
        metadata: Optional[dict] = None,
        sharded_layer_prefix: Optional[str] = None,
    ) -> ShardedStateDict:
        sharded_offsets = sharded_offsets or ()
        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'
        if sharded_layer_prefix is None:
            sharded_layer_prefix = layer_prefix

        for local_layer_idx, (layer_type, layer) in enumerate(zip(self.layer_type_list, self.layers)):
            state_dict_prefix = f'{layer_prefix}{local_layer_idx}.'
            logical_layer_idx = (
                self.logical_layer_offset
                if self.is_layer_group_stack
                else self.logical_layer_offset + local_layer_idx
            )

            if is_layer_group(layer_type):
                sharded_state_dict.update(
                    layer._sharded_state_dict(
                        state_dict_prefix,
                        sharded_offsets,
                        metadata,
                        sharded_layer_prefix=sharded_layer_prefix,
                    )
                )
                continue

            sharded_prefix = f'{sharded_layer_prefix}{logical_layer_idx}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )

            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                module_sharded_state_dict = sharded_state_dict_default(
                    module,
                    f'{prefix}{name}.',
                    sharded_offsets,
                    metadata,
                    tp_group=self.tp_group,
                )
                if name == "final_norm":
                    replace_prefix_for_sharding(
                        module_sharded_state_dict,
                        f'{prefix}{name}.',
                        f'{prefix}final_layernorm.',
                    )
                sharded_state_dict.update(module_sharded_state_dict)

        return sharded_state_dict


# Backward-compatible aliases
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack

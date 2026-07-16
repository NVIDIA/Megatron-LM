# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.recompute import checkpointed_forward
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import CudaGraphManager, annotate_first_last_layer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_latent_attention import FusedMLASelfAttention
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
    mla_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp
    mtp_block_spec: Optional[ModuleSpec] = None


@dataclass(frozen=True)
class _HybridExecutionStep:
    """One precomputed single-layer or shortcut-pair execution step."""

    layer_index: int
    shortcut_block: Optional[ShortcutMoEBlock] = None


def _install_standalone_cudagraph_manager(layer, config):
    """Restore the regular graph manager for a layer not consumed by a shortcut pair."""
    if getattr(layer, '_shortcut_graph_output_proj', False) and not hasattr(
        layer, 'cudagraph_manager'
    ):
        layer.cudagraph_manager = CudaGraphManager(config)


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

    _supports_moe_shortcut = True

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
        name: str | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
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

        if getattr(self.config, "mla_down_proj_fusion", False):
            submodules = self._fuse_mla_down_proj(submodules)

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
                        name=(name + f".layers.{i}") if name is not None else None,
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
                        name=(name + f".layers.{i}") if name is not None else None,
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
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.MLA:
                    layer = build_module(
                        submodules.mla_layer,
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
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                elif layer_type == LayerSymbols.GDN:
                    gdn_layer_spec = submodules.gdn_layer
                    if self.config.experimental_attention_variant == "gdn2":
                        # 'G' layers build the GDN2 variant when the gdn2 experimental
                        # attention variant is selected.
                        from megatron.core.ssm.gated_delta_net import GatedDeltaNet2

                        gdn_layer_spec = copy.deepcopy(gdn_layer_spec)
                        gdn_layer_spec.submodules.self_attention.module = GatedDeltaNet2
                    layer = build_module(
                        gdn_layer_spec,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                        name=(name + f".layers.{i}") if name is not None else None,
                    )
                else:
                    raise ValueError("unexpected layer_type")
            self.layers.append(layer)

        if self.config.cuda_graph_impl == "local":
            annotate_first_last_layer(self.layers)

        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

        if self.config.moe_shortcut_connection:
            # Each compute/MoE pair resolves to eager-serial, eager-overlap, or graph-overlap.
            # All modes share the composite computation modules; only graph-overlap registers
            # CUDA-graph managers for them.
            execution_plan = []
            shortcut_route_input_managers = nn.ModuleList()
            shortcut_output_shared_managers = nn.ModuleList()
            i = 0
            while i < len(self.layers):
                next_is_moe = (
                    i + 1 < len(self.layers)
                    and self.layer_type_list[i + 1] == LayerSymbols.MOE
                )
                if not next_is_moe:
                    # Mamba/attention constructors defer their regular graph manager whenever
                    # shortcut mode is globally enabled. Restore it when this particular layer
                    # is not the immediate predecessor of an MoE shortcut pair.
                    _install_standalone_cudagraph_manager(self.layers[i], self.config)
                    execution_plan.append(_HybridExecutionStep(layer_index=i))
                    i += 1
                    continue

                paired_type = self.layer_type_list[i]
                if paired_type not in (LayerSymbols.MAMBA, LayerSymbols.ATTENTION):
                    raise ValueError("Shortcut MoE must be preceded by a Mamba or attention layer")
                paired_layer = self.layers[i]
                moe_layer = self.layers[i + 1]
                enable_cudagraph = (
                    getattr(paired_layer, '_shortcut_graph_output_proj', False)
                    and getattr(moe_layer, '_shortcut_graph_shared_experts', False)
                )
                shortcut_block = ShortcutMoEBlock(
                    paired_layer,
                    moe_layer,
                    is_mamba=paired_type == LayerSymbols.MAMBA,
                    is_mtp_layer=self.is_mtp_layer,
                    enable_cudagraph=enable_cudagraph,
                    overlap_a2a=self.config.moe_shortcut_parallel,
                )
                if shortcut_block.route_input_cudagraph_manager is not None:
                    shortcut_route_input_managers.append(
                        shortcut_block.route_input_cudagraph_manager
                    )
                if shortcut_block.cudagraph_manager is not None:
                    shortcut_output_shared_managers.append(shortcut_block.cudagraph_manager)
                execution_plan.append(
                    _HybridExecutionStep(layer_index=i, shortcut_block=shortcut_block)
                )
                i += 2

            # Keep the plan and composite modules alive without registering duplicate paths to
            # paired/MoE parameters in HybridStack.state_dict().
            object.__setattr__(self, '_shortcut_execution_plan', tuple(execution_plan))
            # Managers contain no model parameters, so registering only them propagates train/eval
            # and first-microbatch state without adding duplicate layer paths to the state dict.
            self.shortcut_route_input_managers = shortcut_route_input_managers
            self.shortcut_output_shared_managers = shortcut_output_shared_managers

    def _fuse_mla_down_proj(self, submodules: HybridStackSubmodules) -> HybridStackSubmodules:
        # Avoid modifying the original object so users don't get surprised about their `submodules`
        # being modified underneath them.
        submodules = copy.deepcopy(submodules)
        mla_spec = submodules.mla_layer
        # We always fuse the input layernorm because Hybrid always uses TransformerEngine.
        mla_spec.submodules.input_layernorm = IdentityOp
        mla_spec.submodules.self_attention.module = FusedMLASelfAttention
        mla_spec.submodules.self_attention.submodules.linear_qkv_down_proj = (
            TELayerNormColumnParallelLinear
        )
        mla_spec.submodules.self_attention.submodules.linear_q_down_proj = None
        mla_spec.submodules.self_attention.submodules.linear_kv_down_proj = None
        mla_spec.submodules.sharded_state_dict_keys_map = {
            "self_attention.linear_q_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_kv_down_proj.layer_norm_": "input_layernorm.",
            "self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm.",
        }
        return submodules

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

        if inference_context and inference_context.is_static_batching():
            # NOTE(bnorick): match BaseInferenceContext attributes for
            # mamba_ssm.utils.generation.BaseInferenceContext,
            # this hack supports eval
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            (self.config.cuda_graph_impl == "local" or self.config.flash_decode)
            and inference_context
            and inference_context.is_static_batching()
            and InferenceMode.is_active()
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
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = checkpointed_forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=None,
                    context_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                    use_inner_quantization_context=(use_inner_fp8_context or use_fp4_context),
                )
            elif not self.config.moe_shortcut_connection:
                for layer in self.layers:
                    # Layers have 1-indexed layer numbers attribute.
                    inner_quant_context = get_inner_quant_context(
                        self.config, layer.layer_number - 1
                    )
                    with inner_quant_context:
                        if isinstance(layer, TransformerLayer):
                            hidden_states, _ = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                rotary_pos_emb=rotary_pos_emb,
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
            else:
                # moe_shortcut_connection=True: process (non-MOE, MOE) layer pairs.
                # Shortcut pairs run through ShortcutMoEBlock. Layers that are not
                # followed by an MoE layer run normally.
                def run_layer(layer, hidden_states):
                    inner_quant_context = get_inner_quant_context(
                        self.config, layer.layer_number - 1
                    )
                    with inner_quant_context:
                        if isinstance(layer, TransformerLayer):
                            hidden_states, _ = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                inference_context=inference_context,
                                rotary_pos_emb=rotary_pos_emb,
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
                    # The attention layer outputs a tuple of (hidden_states, context).
                    # Context is intended for cross-attention and is not needed here.
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    return hidden_states

                for step in self._shortcut_execution_plan:
                    if step.shortcut_block is not None:
                        hidden_states = step.shortcut_block.forward(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                            quant_context_factory=get_inner_quant_context,
                            quant_config=self.config,
                        )
                    else:
                        hidden_states = run_layer(self.layers[step.layer_index], hidden_states)

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

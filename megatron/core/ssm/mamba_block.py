# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import torch
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.ssm.mamba_hybrid_layer_allocation import allocate_layers
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    with get_cuda_rng_tracker().fork():
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_no_reinit", False):
                nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        for name, p in module.named_parameters():
            if name in ["conv1d.weight", "out_proj.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            if name in ["in_proj.weight"]:
                nn.init.normal_(p, mean=0.0, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the
            #   > residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of
            #   > 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM):
            # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization
                    nn.init.normal_(
                        p,
                        mean=0.0,
                        std=initializer_range / math.sqrt(n_residuals_per_layer * n_layer),
                    )


@dataclass
class MambaStackSubmodules:
    """
    A class for the module specs for the MambaStack.
    """

    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp


class MambaStack(MegatronModule):
    """
    Constructor for the MambaStack class.

    Args:
        config (TransformerConfig): the model configuration
        submodules (MambaStackSubmodules): the submodules for the stack
        residual_in_fp32 (bool, optional): whether to do residual connections
            in fp32. Defaults to False.
        pre_process (bool, optional): whether to include an embedding layer.
            Defaults to True.
        hybrid_attention_ratio (float, optional): the target ratio of attention layers to
            total layers. Defaults to 0.0.
        hybrid_mlp_ratio (float, optional): the target ratio of mlp layers to total
            layers. Defaults to 0.0.
        hybrid_override_pattern (str, optional): the hybrid layer pattern to override
             with. Defaults to None.
        post_layer_norm (bool, optional): whether to include a final layer norm.
            Defaults to True.
        post_process (bool, optional): whether to include an output layer.
            Defaults to True.
        device (optional): the device to use. Defaults to None.
        dtype (optional): the data type to use. Defaults to None.
        pg_collection (ProcessGroupCollection): the required model communication
            process groups to use.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaStackSubmodules,
        residual_in_fp32=False,
        pre_process: bool = True,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        hybrid_override_pattern: str = None,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(config=config)
        self.residual_in_fp32 = residual_in_fp32
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process

        assert pg_collection is not None, "pg_collection must be provided for MambaStack"

        self.pp_group = pg_collection.pp

        # Required for pipeline parallel schedules
        self.input_tensor = None

        self.hybrid_attention_ratio = hybrid_attention_ratio
        self.hybrid_mlp_ratio = hybrid_mlp_ratio
        self.hybrid_override_pattern = hybrid_override_pattern

        layer_type_list = allocate_layers(
            self.config.num_layers,
            self.hybrid_attention_ratio,
            self.hybrid_mlp_ratio,
            self.hybrid_override_pattern,
        )

        pp_layer_offset = 0
        if self.pp_group.size() > 1:
            pp_layer_offset, layer_type_list = self._select_layers_for_pipeline_parallel(
                layer_type_list
            )

        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(layer_type_list):
            fp8_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
            with fp8_init_context:
                if layer_type == LayerSymbols.MAMBA:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=self.config,
                        residual_in_fp32=residual_in_fp32,
                        layer_number=i + 1 + pp_layer_offset,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.ATTENTION:
                    # Transformer layers apply their own pp_layer_offset
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.config,
                        layer_number=i + 1,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.MLP:
                    # Transformer layers apply their own pp_layer_offset
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=i + 1,
                        pg_collection=pg_collection,
                    )
                else:
                    assert False, "unexpected layer_type"
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

        self.apply(
            partial(
                _init_weights,
                n_layer=self.config.num_layers,
                initializer_range=self.config.init_method_std,
            )
        )

    def _select_layers_for_pipeline_parallel(self, layer_type_list):
        num_layers_per_pipeline_rank = self.config.num_layers // self.pp_group.size()

        assert self.config.virtual_pipeline_model_parallel_size is None, (
            "The Mamba hybrid model does not currently support "
            "virtual/interleaved pipeline parallelism"
        )

        offset = self.pp_group.rank() * num_layers_per_pipeline_rank
        selected_list = layer_type_list[offset : offset + num_layers_per_pipeline_rank]

        return offset, selected_list

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate inference cache for each layer.

        Args:
            batch_size (int): The batch size to use for inference.
            max_seqlen (int): The maximum sequence length to use
                for inference.
            dtype (optional): The data type to use for allocation.
                Defaults to the data type of the model.
        """
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Forward function of the MambaStack class.

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

        if inference_context:
            assert (
                inference_context.is_static_batching()
            ), "Mamba currently does not support dynamic inference batching."
            # NOTE(bnorick): match BaseInferenceContext attributes for
            # mamba_ssm.utils.generation.BaseInferenceContext,
            # this hack supports eval
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            (
                (
                    self.config.cuda_graph_impl == "local"
                    and "full_iteration" not in self.config.cuda_graph_scope
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
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        with outer_fp8_context:
            for layer in self.layers:
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context
                    else nullcontext()
                )
                with inner_fp8_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            sequence_len_offset=sequence_len_offset,
                        )
                    else:  # MambaLayer
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
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
        output = make_viewless_tensor(
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
                f'{layer_prefix}{local_layer_idx}.'  # module list index in MambaBlock
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
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict

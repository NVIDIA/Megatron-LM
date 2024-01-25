# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.custom_layers.transformer_engine import (
    TENorm,
    get_cpu_offload_context,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor


def get_num_layers_to_build(config: TransformerConfig) -> int:

    num_layers_per_pipeline_rank = (
        config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    )

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_rank

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.

        num_layers_to_build = num_layers_per_pipeline_rank

    return num_layers_to_build


@dataclass
class TransformerBlockSubmodules:
    layer_specs: List[ModuleSpec] = None


def _get_block_submodules(
    config: TransformerConfig, spec: Union[TransformerBlockSubmodules, ModuleSpec],
) -> TransformerBlockSubmodules:

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, TransformerLayer):
            num_layers = get_num_layers_to_build(config)
            return TransformerBlockSubmodules(layer_specs=[spec] * num_layers)
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__(config=config)

        self.submodules = _get_block_submodules(config, spec)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        if get_cpu_offload_context is not None:
            (
                self.offload_context,
                self.group_prefetch_offload_commit_async,
            ) = get_cpu_offload_context(
                self.config.cpu_offloading,
                self.config.cpu_offloading_num_layers,
                self.config.cpu_offloading_activations,
                self.config.cpu_offloading_weights,
            )
            self.config._cpu_offloading_context = (
                self.offload_context if self.config.cpu_offloading else None
            )
        else:
            assert (
                self.config.cpu_offloading == False
            ), "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number,)

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # # TODO: add back standalone_embedding_stage
        # if self.num_layers == 0:
        #     # When a standalone embedding stage is used (e.g.,
        #     # args.standalone_embedding_stage == True), virtual pipeline ranks
        #     # on pipeline rank 0 will have zero transformer layers assigned to
        #     # them. This results in the model's input and output tensors to be
        #     # the same, which will cause failure for certain output tensor
        #     # optimizations (e.g., pipeline output deallocation). To remedy
        #     # this, we assign a 'no-op' layer on these ranks, which will
        #     # disconnect the input tensor from the output tensor.
        #     self.num_layers = 1
        #     self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        # else:
        #     self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states, context = tensor_parallel.checkpoint(
                    custom(l, l + self.config.recompute_num_layers),
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                )

                l += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers_per_pipeline_rank):
                if l < self.config.recompute_num_layers:
                    hidden_states, context = tensor_parallel.checkpoint(
                        custom(l, l + 1),
                        self.config.distribute_saved_activations,
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        packed_seq_params,
                    )
                else:
                    hidden_states, context = custom(l, l + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                        packed_seq_params,
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

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
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True,
        )

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=self.config.fp8_margin,
                interval=self.config.fp8_interval,
                fp8_format=fp8_format,
                amax_compute_algo=self.config.fp8_amax_compute_algo,
                amax_history_len=self.config.fp8_amax_history_len,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context and fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full':
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                )
            else:
                for layer in self.layers:
                    with self.offload_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            inference_params=inference_params,
                            packed_seq_params=packed_seq_params,
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

    def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = ()) -> ShardedStateDict:
        assert not sharded_offsets, "Unexpected sharded offsets"
        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = layer._get_layer_offset()

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock
            sharded_pp_offset = [
                (0, global_layer_offset, num_layers)
            ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                prefix=state_dict_prefix, sharded_offsets=sharded_pp_offset
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, layer_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets)
                )

        return sharded_state_dict

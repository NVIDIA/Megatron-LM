# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import re
from contextlib import nullcontext
from dataclasses import dataclass
import torch
from typing import List

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSpec
from megatron.core.utils import make_viewless_tensor, make_sharded_tensor_for_checkpoint


def get_num_layers_to_build(config) -> int:

    num_layers_per_pipeline_rank = \
        config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()

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
class TransformerBlockSpec:
    layers: List[TransformerLayerSpec] = None


class TransformerBlock(MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerBlockSpec,
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
    ):
        super().__init__(config=config)

        self.spec = spec
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        self._build_layers()

    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(spec, layer_number):
            return TransformerLayer(
                config=self.config,
                spec=spec,
                layer_number=layer_number,
            )

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList([build_layer(spec, i + 1) for i, spec in enumerate(self.spec.layers)])

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
                persist_layer_norm=self.config.persist_layer_norm,
                sequence_parallel=self.config.sequence_parallel,
                zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
                normalization=self.config.normalization,
            )

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask, rotary_pos_emb):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_

            return custom_forward

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states = tensor_parallel.checkpoint(
                    custom(l, l + self.config.recompute_num_layers),
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                )

                l += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers_per_pipeline_rank):
                if l < self.config.recompute_num_layers:
                    hidden_states = tensor_parallel.checkpoint(
                        custom(l, l + 1),
                        self.config.distribute_saved_activations,
                        hidden_states,
                        attention_mask,
                        rotary_pos_emb,
                    )
                else:
                    hidden_states = custom(l, l + 1)(hidden_states, attention_mask, rotary_pos_emb)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
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
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe
            )
        else:
            fp8_context = nullcontext()

        with rng_context and fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full':
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                )
            else:
                retriever_output = None
                for layer in self.layers:
                    hidden_states = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=inference_params,
                        retriever_output=retriever_output,
                    )

                    # First Retro decoder layer returns both hidden_states
                    # and retriever_output. Make retriever_output available
                    # to subsequence Retro layers.
                    if isinstance(hidden_states, tuple):
                        assert len(hidden_states) == 2
                        hidden_states, retriever_output = hidden_states

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        # >>>
        # from lutil import tp
        # print("HIDDEN_STATES : %s." % tp(hidden_states))
        # print("RETRIEVER_OUTPUT : %s." % tp(retriever_output))
        # <<<

        return hidden_states

    def sharded_state_dict(self, prefix=''):

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        for layer in self.layers:
            sharded_state_dict.update(layer.sharded_state_dict(prefix=layer_prefix))

        if self.post_process and self.post_layer_norm:
            tensor = self.state_dict(keep_vars=True)['final_layernorm.weight']
            layer_name = f'{prefix}final_layernorm.weight'
            sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(tensor, layer_name)
            tensor = self.state_dict(keep_vars=True)['final_layernorm.bias']
            layer_name = f'{prefix}final_layernorm.bias'
            sharded_state_dict[layer_name] = make_sharded_tensor_for_checkpoint(tensor, layer_name)

        return sharded_state_dict

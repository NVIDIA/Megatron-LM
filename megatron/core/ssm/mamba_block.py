# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from functools import partial
from typing import Union

from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.ssm.mamba_hybrid_layer_allocation import allocate_layers
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor


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
            if name in ["in_proj.weight", "x_proj.weight", "conv1d.weight", "out_proj.weight"]:
                nn.init.kaiming_uniform(p, a=math.sqrt(5))

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
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
    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp


class MambaStack(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaStackSubmodules,
        mamba_ssm_ngroups: int = 8,
        residual_in_fp32=False,
        pre_process: bool = True,
        hybrid_attention_ratio: float = 0.0,
        hybrid_mlp_ratio: float = 0.0,
        hybrid_override_pattern: str = None,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(config=config)
        self.residual_in_fp32 = residual_in_fp32
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process

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
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            pp_layer_offset, layer_type_list = self._select_layers_for_pipeline_parallel(
                layer_type_list
            )

        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(layer_type_list):
            if layer_type == LayerSymbols.MAMBA:
                layer = build_module(
                    submodules.mamba_layer,
                    config=self.config,
                    mamba_ssm_ngroups=mamba_ssm_ngroups,
                    residual_in_fp32=residual_in_fp32,
                    layer_number=i + 1 + pp_layer_offset,
                )
            elif layer_type == LayerSymbols.ATTENTION:
                # Transformer layers apply their own pp_layer_offset
                layer = build_module(
                    submodules.attention_layer, config=self.config, layer_number=i + 1
                )
            elif layer_type == LayerSymbols.MLP:
                # Transformer layers apply their own pp_layer_offset
                layer = build_module(submodules.mlp_layer, config=self.config, layer_number=i + 1)
            else:
                assert True, "unexpected layer_type"
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
            )
        )

    def _select_layers_for_pipeline_parallel(self, layer_type_list):
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        num_layers_per_pipeline_rank = (
            self.config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
        )

        assert parallel_state.get_virtual_pipeline_model_parallel_world_size() is None, (
            "The Mamba hybrid model does not currently support "
            "virtual/interleaved pipeline parallelism"
        )

        offset = pipeline_rank * num_layers_per_pipeline_rank
        selected_list = layer_type_list[offset : offset + num_layers_per_pipeline_rank]

        return offset, selected_list

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
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
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_params=None,
        rotary_pos_emb: Tensor = None,
    ):
        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        if inference_params:
            # NOTE(bnorick): match InferenceParams attributes for mamba_ssm.utils.generation.InferenceParams,
            # this hack supports eval
            inference_params.max_seqlen = inference_params.max_sequence_length
            inference_params.seqlen_offset = inference_params.sequence_len_offset

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
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

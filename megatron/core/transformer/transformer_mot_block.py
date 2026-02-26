# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Transformer MoT (Mixture of Transformers) Block implementation.

This module implements a Transformer block that supports separate processing
for understanding (und) and generation (gen) tokens, following the MoT architecture.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.transformer_mot_layer import (
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import (
    get_pg_rank,
    make_viewless_tensor,
)
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec

try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        te_checkpoint,
    )

    LayerNormImpl = TENorm
else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm


logger = logging.getLogger(__name__)


# Re-export for convenience
__all__ = [
    'TransformerMoTBlock',
    'TransformerMoTBlockSubmodules',
    'MoTTransformerLayer',
    'MoTTransformerLayerSubmodules',
    'SelfAttentionMoT',
    'SelfAttentionMoTSubmodules',
    'get_mot_layer_spec',
]


@dataclass
class TransformerMoTBlockSubmodules:
    """
    Dataclass for specifying the submodules of a transformer MoT block.

    This class defines the structure for configuring the layers and normalization
    within a transformer MoT block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the transformer block.
        layer_norm (Optional[Union[ModuleSpec, torch.nn.Module]], optional): Specification
            or instance of the layer normalization for understanding tokens.
        layer_norm_gen (Optional[Union[ModuleSpec, torch.nn.Module]], optional): Specification
            or instance of the layer normalization for generation tokens.
    """

    layer_specs: List[ModuleSpec] = None
    layer_norm: Optional[Union[ModuleSpec, torch.nn.Module]] = None
    layer_norm_gen: Optional[Union[ModuleSpec, torch.nn.Module]] = None  # MoT: separate norm for gen


def _get_mot_block_submodules(
    config: TransformerConfig,
    spec: Union["TransformerMoTBlockSubmodules", ModuleSpec],
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerMoTBlockSubmodules:
    """
    Retrieve or construct TransformerMoTBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerMoTBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        TransformerMoTBlockSubmodules: The submodules for the transformer MoT block.
    """

    if isinstance(spec, TransformerMoTBlockSubmodules):
        return spec

    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerMoTBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
            return TransformerMoTBlockSubmodules(
                layer_specs=[spec] * num_layers,
                layer_norm=LayerNormImpl,
                layer_norm_gen=LayerNormImpl,
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerMoTBlock(GraphableMegatronModule, MegatronModule):
    """
    Transformer MoT (Mixture of Transformers) Block.

    This block extends the standard TransformerBlock to support separate processing
    paths for understanding (und) and generation (gen) tokens, following the MoT architecture
    used in models like Bagel.

    Key differences from standard TransformerBlock:
    1. Supports separate token indexes for und/gen tokens
    2. Has separate final layer norms for und/gen tokens
    3. Supports freezing und token gradients (freeze_und)
    4. Different forward signatures for training vs inference
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerMoTBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: Optional[int] = None,
        freeze_und: bool = False,
    ):
        """
        Initialize the TransformerMoTBlock.

        Args:
            config (TransformerConfig): Configuration for the transformer model.
            spec (Union[TransformerMoTBlockSubmodules, ModuleSpec]): Specification for submodules.
            post_layer_norm (bool): Whether to apply layer norm after the last layer.
            pre_process (bool): Whether this is the first stage in pipeline parallel.
            post_process (bool): Whether this is the last stage in pipeline parallel.
            pg_collection (ProcessGroupCollection): Process group collection.
            vp_stage (Optional[int]): Virtual pipeline stage number.
            freeze_und (bool): Whether to freeze understanding token gradients.
        """
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        pp_rank = get_pg_rank(pp_group)

        self.submodules = _get_mot_block_submodules(config, spec, vp_stage, pp_rank)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        self.freeze_und = freeze_und

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        # Setup CPU offloading context
        try:
            from megatron.core.extensions.transformer_engine import get_cpu_offload_context
        except ImportError:
            get_cpu_offload_context = None

        if get_cpu_offload_context is not None:
            (self.offload_context, self.group_prefetch_offload_commit_async) = (
                get_cpu_offload_context(
                    self.config.cpu_offloading,
                    self.config.cpu_offloading_num_layers,
                    self.config.num_layers,
                    self.config.cpu_offloading_activations,
                    self.config.cpu_offloading_weights,
                    self.config.cpu_offloading_double_buffering,
                )
            )
            self.config._cpu_offloading_context = (
                self.offload_context if self.config.cpu_offloading else None
            )
        else:
            assert (
                self.config.cpu_offloading is False
            ), "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

    def _build_layers(self):
        """Build transformer layers and final layer norms."""

        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            # Get appropriate quantization context (FP8 and FP4 are mutually exclusive)
            if layer_config.fp8:
                quantization_context = get_fp8_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            elif layer_config.fp4:
                quantization_context = get_fp4_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            else:
                quantization_context = nullcontext()

            with quantization_context:
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
            return module

        # Build transformer layers
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # Build final layer norms (separate for und and gen tokens in MoT)
        if self.has_final_layernorm_in_this_stage():
            # Final layer norm for understanding tokens
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
            # Final layer norm for generation tokens (MoT specific)
            if self.submodules.layer_norm_gen is not None:
                self.final_layernorm_gen = build_module(
                    self.submodules.layer_norm_gen,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.final_layernorm_gen = None
        else:
            self.final_layernorm = None
            self.final_layernorm_gen = None

    def has_final_layernorm_in_this_stage(self):
        """Check if this vpp stage contains the final layernorm."""
        if self.config.mtp_num_layers is None:
            return self.submodules.layer_norm and self.post_process and self.post_layer_norm
        else:
            has_final_layernorm_in_this_stage = False
            for layer in self.layers:
                if layer.layer_number == self.config.num_layers:
                    has_final_layernorm_in_this_stage = True
                    break
            return (
                self.submodules.layer_norm
                and has_final_layernorm_in_this_stage
                and self.post_layer_norm
            )

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input."""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Union[Tensor, "WrappedTensor"],
        attention_mask: Optional[Tensor],
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
        **kwargs,
    ):
        """
        Perform the forward pass through the transformer MoT block.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h].
            attention_mask (Tensor): Boolean tensor for masking self-attention.
            packed_und_token_indexes (Tensor, optional): Indexes of understanding tokens.
            packed_gen_token_indexes (Tensor, optional): Indexes of generation tokens.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            attention_bias (Tensor): Bias tensor for Q * K.T.
            inference_context (BaseInferenceContext, optional): Parameters for inference.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension.
            mode (str): Processing mode - "und" for understanding, "gen" for generation.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor.
        """
        if self.training:
            return self.forward_train(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            return self.forward_inference(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                mode=mode,
                **kwargs,
            )

    def forward_train(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """
        Forward pass for training with MoT (separate und/gen token processing).

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h].
            attention_mask (Tensor): Attention mask.
            packed_und_token_indexes (Tensor): Indexes of understanding tokens.
            packed_gen_token_indexes (Tensor): Indexes of generation tokens.
            context (Tensor, optional): Context for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Rotary embedding cosine.
            rotary_pos_sin (Tensor, optional): Rotary embedding sine.
            attention_bias (Tensor, optional): Attention bias.
            packed_seq_params (PackedSeqParams, optional): Packed sequence parameters.

        Returns:
            Tensor: Output hidden states.
        """
        # Handle WrappedTensor
        try:
            from megatron.core.transformer.transformer_block import WrappedTensor
            if isinstance(hidden_states, WrappedTensor):
                hidden_states = hidden_states.unwrap()
        except ImportError:
            pass

        if not self.pre_process:
            hidden_states = self.input_tensor

        # Make viewless tensor
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Setup quantization contexts
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass with activation checkpointing
            if self.config.recompute_granularity == 'full':
                hidden_states = self._checkpointed_forward_train(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    packed_und_token_indexes=packed_und_token_indexes,
                    packed_gen_token_indexes=packed_gen_token_indexes,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
                # Standard forward without checkpointing
                for l_no, layer in enumerate(self.layers):
                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            packed_und_token_indexes=packed_und_token_indexes,
                            packed_gen_token_indexes=packed_gen_token_indexes,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )

                    # Freeze und tokens if configured
                    if self.freeze_und and packed_und_token_indexes is not None:
                        hidden_states[packed_und_token_indexes] = (
                            hidden_states[packed_und_token_indexes].detach()
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm with MoT (separate norms for und/gen)
        if self.final_layernorm is not None:
            hidden_states = self._apply_final_layernorm_mot(
                hidden_states, packed_und_token_indexes, packed_gen_token_indexes
            )

        return hidden_states, context

    def _apply_final_layernorm_mot(
        self,
        hidden_states: Tensor,
        packed_und_token_indexes: Optional[Tensor],
        packed_gen_token_indexes: Optional[Tensor],
    ) -> Tensor:
        """
        Apply final layer normalization with MoT (separate norms for und/gen tokens).

        Args:
            hidden_states (Tensor): Input hidden states.
            packed_und_token_indexes (Tensor, optional): Indexes of understanding tokens.
            packed_gen_token_indexes (Tensor, optional): Indexes of generation tokens.

        Returns:
            Tensor: Normalized hidden states.
        """
        if packed_und_token_indexes is not None and packed_gen_token_indexes is not None:
            # MoT: Apply separate layer norms
            output = hidden_states.new_zeros(hidden_states.shape)
            output[packed_und_token_indexes] = self.final_layernorm(
                hidden_states[packed_und_token_indexes]
            )
            # Only apply gen layernorm if there are gen tokens
            if len(packed_gen_token_indexes) > 0 and self.final_layernorm_gen is not None:
                output[packed_gen_token_indexes] = self.final_layernorm_gen(
                    hidden_states[packed_gen_token_indexes]
                )
            hidden_states = output
        else:
            # Standard: Apply same norm to all
            hidden_states = self.final_layernorm(hidden_states)

        # Make viewless tensor
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )
        return hidden_states

    def _checkpointed_forward_train(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
    ) -> Tensor:
        """Forward method with activation checkpointing for training."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                packed_und_token_indexes,
                packed_gen_token_indexes,
                context,
                context_mask,
                rotary_pos_emb,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(
                                self.config, layer.layer_number - 1
                            )
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            packed_und_token_indexes=packed_und_token_indexes,
                            packed_gen_token_indexes=packed_gen_token_indexes,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )

                    # Freeze und tokens if configured
                    if self.freeze_und and packed_und_token_indexes is not None:
                        hidden_states[packed_und_token_indexes] = (
                            hidden_states[packed_und_token_indexes].detach()
                        )

                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the te_checkpoint or tensor_parallel.checkpoint"""
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
                    packed_und_token_indexes,
                    packed_gen_token_indexes,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    packed_und_token_indexes,
                    packed_gen_token_indexes,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'uniform':
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )
                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1

                if layer_idx >= recompute_skip_num_layers and (
                    layer_idx - recompute_skip_num_layers
                ) < self.config.recompute_num_layers:
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states,
                        attention_mask,
                        packed_und_token_indexes,
                        packed_gen_token_indexes,
                        context,
                        context_mask,
                        rotary_pos_emb,
                    )
        else:
            raise ValueError(f"Invalid recompute method: {self.config.recompute_method}")

        return hidden_states

    def forward_inference(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        packed_und_token_indexes: Optional[Tensor] = None,
        packed_gen_token_indexes: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        mode: str = "und",
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for inference with MoT.

        Args:
            hidden_states (Tensor): Input tensor.
            attention_mask (Tensor): Attention mask.
            packed_und_token_indexes (Tensor, optional): Indexes of understanding tokens.
            packed_gen_token_indexes (Tensor, optional): Indexes of generation tokens.
            context (Tensor, optional): Context for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Rotary embedding cosine.
            rotary_pos_sin (Tensor, optional): Rotary embedding sine.
            rotary_pos_cos_sin (Tensor, optional): Combined rotary embedding cosine and sine.
            attention_bias (Tensor, optional): Attention bias.
            inference_context (BaseInferenceContext, optional): Inference context.
            packed_seq_params (PackedSeqParams, optional): Packed sequence parameters.
            sequence_len_offset (Tensor, optional): Sequence length offset.
            mode (str): Processing mode - "und" or "gen".

        Returns:
            Tuple[Tensor, Tensor]: Output hidden states and context.
        """
        # Handle WrappedTensor
        try:
            from megatron.core.transformer.transformer_block import WrappedTensor
            if isinstance(hidden_states, WrappedTensor):
                hidden_states = hidden_states.unwrap()
        except ImportError:
            pass

        if not self.pre_process:
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Setup quantization contexts
        if self.config.fp8:
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config)
                if self.config.fp8_recipe == Fp8Recipe.delayed
                else nullcontext()
            )
        elif self.config.fp4:
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            for l_no, layer in enumerate(self.layers):
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
                        )
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                with self.offload_context, inner_quantization_context:
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        packed_und_token_indexes=packed_und_token_indexes,
                        packed_gen_token_indexes=packed_gen_token_indexes,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        rotary_pos_cos_sin=rotary_pos_cos_sin,
                        attention_bias=attention_bias,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                        sequence_len_offset=sequence_len_offset,
                        mode=mode,
                        **kwargs,
                    )

        # Final layer norm based on mode
        if self.final_layernorm is not None:
            if mode == "gen" and self.final_layernorm_gen is not None:
                hidden_states = self.final_layernorm_gen(hidden_states)
            else:
                hidden_states = self.final_layernorm(hidden_states)

            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        return hidden_states, context

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for distributed checkpointing.

        Args:
            prefix (str): Prefix for state dict keys.
            sharded_offsets (tuple): Offsets for sharding.
            metadata (dict, optional): Additional metadata.

        Returns:
            ShardedStateDict: The sharded state dictionary.
        """
        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'

        for local_layer_idx, layer in enumerate(self.layers):
            global_layer_offset = get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )
            global_layer_idx = local_layer_idx + global_layer_offset
            state_dict_prefix = f'{layer_prefix}{global_layer_idx}.'

            sharded_state_dict.update(
                sharded_state_dict_default(
                    layer,
                    state_dict_prefix,
                    sharded_offsets,
                    metadata,
                )
            )

            # Replace local layer index with global layer index for state dict key
            key_repl = (
                f'{layer_prefix}{local_layer_idx}.',
                f'{layer_prefix}{global_layer_idx}.',
            )
            sharded_state_dict = {
                k.replace(*key_repl): v for k, v in sharded_state_dict.items()
            }

        # Final layer norms
        if self.final_layernorm is not None:
            sharded_state_dict.update(
                sharded_state_dict_default(
                    self.final_layernorm,
                    f'{prefix}final_layernorm.',
                    sharded_offsets,
                    metadata,
                )
            )

        if self.final_layernorm_gen is not None:
            sharded_state_dict.update(
                sharded_state_dict_default(
                    self.final_layernorm_gen,
                    f'{prefix}final_layernorm_gen.',
                    sharded_offsets,
                    metadata,
                )
            )

        return sharded_state_dict

# copied from Bagel, only for alignment
# class Qwen2RMSNorm(torch.nn.Module):
#     def __init__(self, config: TransformerConfig, hidden_size: int, eps: float = 1e-6):
#         """
#         Qwen2RMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = torch.nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

#     def extra_repr(self):
#         return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def get_mot_layer_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = None,
    qk_layernorm: bool = True,
    use_te: bool = True,
    use_flex_attention: bool = False,
) -> ModuleSpec:
    """
    Get a default MoT transformer layer specification.

    This creates a ModuleSpec for MoTTransformerLayer with appropriate submodules
    for both understanding and generation tokens.

    Args:
        config (TransformerConfig): Configuration for the transformer model.
        use_te (bool): Whether to use Transformer Engine modules.

    Returns:
        ModuleSpec: Specification for MoTTransformerLayer.
    """
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    if use_te and HAVE_TE:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TEDotProductAttention,
            TENorm,
            TERowParallelLinear,
        )
        from megatron.core.transformer.flex_attention import FlexAttention

        linear_qkv = TEColumnParallelLinear
        linear_proj = TERowParallelLinear
        layernorm = TENorm
        # layernorm = Qwen2RMSNorm # only for alignment with Bagel
        if use_flex_attention:
            core_attention = FlexAttention
        else:
            core_attention = TEDotProductAttention
    else:
        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        linear_qkv = ColumnParallelLinear
        linear_proj = RowParallelLinear
        layernorm = LayerNormImpl
        if use_flex_attention:
            core_attention = FlexAttention
        else:
            core_attention = DotProductAttention

    # Self-attention submodules with MoT support
    self_attention_submodules = SelfAttentionMoTSubmodules(
        linear_qkv=linear_qkv,
        core_attention=core_attention,
        linear_proj=linear_proj,
        q_layernorm=layernorm if qk_layernorm else None,
        k_layernorm=layernorm if qk_layernorm else None,
        # MoT: Generation token projections
        linear_qkv_gen=linear_qkv,
        linear_proj_gen=linear_proj,
        q_layernorm_gen=layernorm if qk_layernorm else None,
        k_layernorm_gen=layernorm if qk_layernorm else None,
    )

    # MLP submodules
    from megatron.core.transformer.mlp import MLP, MLPSubmodules

    mlp_spec = get_mlp_module_spec(
        use_te=use_te,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=False,
        use_te_op_fuser=False, # TODO: support TE op fuser for MoE
    )

    # MoT transformer layer submodules
    layer_submodules = MoTTransformerLayerSubmodules(
        input_layernorm=layernorm,
        input_layernorm_gen=layernorm,
        self_attention=ModuleSpec(
            module=SelfAttentionMoT,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=self_attention_submodules,
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=layernorm,
        pre_mlp_layernorm_gen=layernorm,
        # mlp=ModuleSpec(module=MLP, submodules=mlp_submodules),
        mlp=mlp_spec,
        # mlp_gen=ModuleSpec(module=MLP, submodules=mlp_submodules),
        mlp_gen=mlp_spec,
        mlp_bda=get_bias_dropout_add,
    )
    return ModuleSpec(module=MoTTransformerLayer, submodules=layer_submodules)


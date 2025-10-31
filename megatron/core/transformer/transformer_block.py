# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)

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

get_cpu_offload_context = None
te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    LayerNormImpl = TENorm

elif HAVE_APEX:
    LayerNormImpl = FusedLayerNorm

else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm


logger = logging.getLogger(__name__)


def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if config.pipeline_model_parallel_layout is not None:
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):

        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert (
                layers_to_distribute % pipeline_stages_left == 0
            ), "With uneven pipelineing the left over layers must be divisible by left over stages"
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert (
            num_layers % config.pipeline_model_parallel_size == 0
        ), "num_layers should be divisible by pipeline_model_parallel_size"
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
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

        assert (
            num_layers_per_pipeline_rank % vp_size == 0
        ), f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert (
                num_layers_to_build >= 0
            ), f"Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, f"Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


@dataclass
class TransformerBlockSubmodules:
    """
    Dataclass for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (Optional[Union[ModuleSpec, torch.nn.Module]], optional): Specification
            or instance of the layer normalization to be applied.
    """

    layer_specs: List[ModuleSpec] = None
    layer_norm: Optional[Union[ModuleSpec, torch.nn.Module]] = None


def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        vp_stage (Optional[int]): Virtual pipeline stage number.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(GraphableMegatronModule, MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        pp_rank = get_pg_rank(pp_group)

        self.submodules = _get_block_submodules(config, spec, vp_stage, pp_rank)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

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
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # 1-based index
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

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # @TODO: add back account_for_embedding_in_pipeline_split (see issue #293)
        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if self.submodules.layer_norm and self.post_process and self.post_layer_norm:
            self.final_layernorm = build_module(
                self.submodules.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states, attention_mask, context, context_mask, rotary_pos_emb
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    # Get appropriate inner quantization context
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(
                                self.config, layer.layer_number - 1
                            )
                        # TODO: check if fp4 is supported in this case
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
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            # TODO: check if fp4 is supported in this case
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                    attention_mask,
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
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                # TODO: check if fp4 is supported in this case
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx, layer_idx + 1)(
                        hidden_states, attention_mask, context, context_mask, rotary_pos_emb
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

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        if not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                kwargs.get('inference_context') is not None
                or kwargs.get('inference_params') is not None
            )
            and 'full_iteration' in self.config.cuda_graph_scope
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
            # dynamic_inference_decode_only is not a real argument to forward, it is only used
            # to differentiate the cuda graph used for decode from the one used for non-decode
            # inference.
            dynamic_inference_decode_only = kwargs['inference_context'].is_decode_only()
            # cudagraphmanager returns a singleton tuple, whereas the
            # normal forward returns a tensor, therefore we need
            # to extract the tensor from the tuple
            return super().__call__(
                *args, dynamic_inference_decode_only=dynamic_inference_decode_only, **kwargs
            )[0]
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
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
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

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
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
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
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
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
                        )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        # TODO: remove multiple non_homogeneous_layers=True assignments
        #  once non_homogeneous_layers=False support is dropped
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if self.config.hetereogenous_dist_checkpoint:
            non_homogeneous_layers = True

        if isinstance(self.config.moe_layer_freq, int):
            if self.config.moe_layer_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.moe_layer_freq, list):
            non_homogeneous_layers = True

        if self.config.heterogeneous_block_specs:
            non_homogeneous_layers = True

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        if singleton_local_shards:
            if metadata is not None and metadata.get('non_homogeneous_layers') is False:
                # non_homogeneous_layers=False was set explicitly - emit an override warning
                logger.warning(
                    'non_homogeneous_layers=False is deprecated.'
                    ' Setting non_homogeneous_layers=True.'
                )
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
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

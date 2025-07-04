# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.cuda_graphs import CudaGraphManager
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    deprecate_inference_params,
    is_te_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)


def get_transformer_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    if not parallel_state.is_inside_encoder():
        pp_decoder_start = parallel_state.get_pipeline_model_parallel_decoder_start()
        if pp_decoder_start is not None:
            pipeline_rank = pipeline_rank - pp_decoder_start

    if config.pipeline_model_parallel_size > 1:

        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        elif (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_vritual_model_chunk_in_middle_pipeline_stage = (
                    middle_num_layers // vp_size
                )

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pipeline_rank == 0:
                    offset = vp_stage * total_virtual_chunks
                else:
                    offset = (
                        vp_stage * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + (pipeline_rank - 1)
                        * (
                            num_layers_per_vritual_model_chunk_in_middle_pipeline_stage
                            // middle_pipeline_stages
                        )
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                middle_pipeline_rank = (
                    pipeline_rank
                    if config.num_layers_in_first_pipeline_stage is None
                    else pipeline_rank - 1
                )

                if pipeline_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (
                    pipeline_rank * num_layers_per_virtual_rank
                )

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage(
                        ignore_virtual=False, vp_stage=vp_stage
                    )
                ):
                    offset -= 1
            else:
                offset = pipeline_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if (
                    config.account_for_embedding_in_pipeline_split
                    and not parallel_state.is_pipeline_first_stage(
                        ignore_virtual=False, vp_stage=vp_stage
                    )
                ):
                    offset -= 1
    else:
        offset = 0
    return offset


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (Union[ModuleSpec, type]): Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm (Union[ModuleSpec, type]): Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


class BaseTransformerLayer(ABC):
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    def __init__(self):
        pass


class TransformerLayer(MegatronModule, BaseTransformerLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        model_comm_pgs: Optional[ModelCommProcessGroups] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)

        # Enable cuda graphs.
        if config.enable_cuda_graph or config.external_cuda_graph:
            assert not (
                config.enable_cuda_graph and config.external_cuda_graph
            ), "Cudagraphs and external cudagraphs cannot be enabled at the same time"
            if config.enable_cuda_graph:
                if not self.training:
                    # Cudagraphs for inference are only enabled with the flash decoding kernel
                    assert (
                        self.config.flash_decode
                    ), "--flash-decode is required to use CUDA graphs during inference"
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

        if model_comm_pgs is None:
            model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups()

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config, vp_stage)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        attention_optional_kwargs["model_comm_pgs"] = model_comm_pgs

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        # [Module 8: MLP block]
        additional_mlp_kwargs = {}
        # import here to avoid circular import
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        # MLP expects tp_group but MoELayer expects model_comm_pgs to be passed in.
        # We can change MLP to accept model_comm_pgs but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if submodules.mlp is not a ModuleSpec,we dont have to handle passing additional kwargs
        if isinstance(submodules.mlp, ModuleSpec):
            if submodules.mlp.module in (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs["model_comm_pgs"] = model_comm_pgs
            elif submodules.mlp.module == MLP:
                assert hasattr(
                    model_comm_pgs, 'tp'
                ), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = model_comm_pgs.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(submodules.mlp)}. Using default kwargs.",
                )
        self.mlp = build_module(submodules.mlp, config=self.config, **additional_mlp_kwargs)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            if "layernorm" in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp):
                    self.recompute_input_layernorm = True
                if not isinstance(self.pre_mlp_layernorm, IdentityOp):
                    self.recompute_pre_mlp_layernorm = True
            if "mlp" in self.config.recompute_modules:

                if not isinstance(self.mlp, MoELayer):
                    self.recompute_mlp = True

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    @staticmethod
    def _get_layer_offset(config: TransformerConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "TransformerLayer._get_layer_offset is deprecated."
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        hidden_states, context = self._forward_attention(*args, **kwargs)
        output = self._forward_mlp(hidden_states, kwargs.get("inference_context", None))
        return output, context

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="self_attention")

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="self_attn_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="self_attn_bda")

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    def _forward_mlp(self, hidden_states, inference_context=None):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
        )

        if self.recompute_mlp:
            if self.config.fp8:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    pre_mlp_layernorm_output,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    self.mlp, False, pre_mlp_layernorm_output
                )
        elif should_chunk_mlp_for_prefill:
            # Chunk input along sequence dimension
            num_chunks = min(self.config.mlp_chunks_for_prefill, pre_mlp_layernorm_output.shape[0])
            chunks = pre_mlp_layernorm_output.chunk(num_chunks, dim=0)

            # Compute outputs for each chunk
            outputs = [self.mlp(chunk) for chunk in chunks]

            # Aggregate chunk outputs
            mlp_output = torch.cat([out for out, _ in outputs], dim=0)
            bias_chunks = [bias for _, bias in outputs if bias is not None]
            bias_output = torch.stack(bias_chunks, dim=0).sum(dim=0) if bias_chunks else None
            mlp_output_with_bias = (mlp_output, bias_output)

        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        nvtx_range_pop(suffix="mlp")

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="mlp_bda")
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )
        nvtx_range_pop(suffix="mlp_bda")

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the transformer layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

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
        static_inputs["attention_mask"] = (
            ~(torch.tril(torch.ones((slen_per_cp, seq_length))).bool())
            .to(torch.cuda.current_device())
            .reshape(1, 1, slen_per_cp, seq_length)
            .tile(micro_batch_size, 1, 1, 1)
        )
        return static_inputs

    def setup_manual_hooks(self, make_hook_func):
        """
        Set CUDA Graph manual hooks for the modules that contain direct parameters and are
        covered by cudagraphs.
        """
        self.cuda_graph_manual_hooks = []

        # Select the modules who contain direct parameters and are covered by cudagraphs.
        # Add these modules to the `cuda_graph_manual_hooks` because their hooks will not
        # be automatically triggered when they go through the CUDA Graph path.
        if self.config.cuda_graph_scope == 'full':
            high_level_modules = [self]
        else:
            assert (
                self.config.cuda_graph_scope == 'attn'
            ), "Invalid cuda_graph_scope ${self.config.cuda_graph_scope}"
            high_level_modules = [
                self.input_layernorm,
                self.self_attention,
                self.pre_cross_attn_layernorm,
                self.cross_attention,
                self.pre_mlp_layernorm,
            ]

        param_modules = []
        for module in high_level_modules:
            for submodule in module.modules():
                if next(submodule.parameters(recurse=False), None) is not None:
                    # Module contains direct parameters.
                    param_modules.append(submodule)
                    continue
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
        hidden_states, context = self._forward_attention(*args, **kwargs)

        if self.config.cuda_graph_scope == "full":
            hidden_states = self._forward_mlp(hidden_states)
        cuda_graph_outputs = [hidden_states]

        if context is not None:
            cuda_graph_outputs.append(context)
        return tuple(cuda_graph_outputs)

    def _cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch
        `self.current_microbatch`. TransformerEngine versions>=1.10
        allow keyword arguments with CUDA graph. However, CUDA graph
        acccepts only Tensor inputs and Tensor outputs. Hence,
        `inference_context` and `packed_seq_params` are excluded from
        input list while output is limited to `hidden_states`.
        """

        def _check_cuda_graph_replay_args(*args, **kwargs):
            """Helper function to get optional tensor arguments for CUDA graph."""

            assert len(args) <= 1, "At most one positional argument `hidden_states` is expected."
            if len(args) == 1:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.pop("hidden_states")
            cudagraph_args = [hidden_states]

            optional_inputs = kwargs.copy()
            optional_inputs['is_first_microbatch'] = self.current_microbatch == 0
            try:
                import transformer_engine.pytorch as te  # pylint: disable=unused-import

                def get_zero_attention_mask(slen_per_tpcp, micro_batch_size):
                    sequence_parallel = self.config.sequence_parallel
                    tensor_model_parallel_size = self.config.tensor_model_parallel_size
                    slen_per_cp = (
                        slen_per_tpcp * tensor_model_parallel_size
                        if sequence_parallel
                        else slen_per_tpcp
                    )
                    slen = slen_per_cp * self.config.context_parallel_size
                    return torch.zeros(
                        (micro_batch_size, 1, slen_per_cp, slen),
                        dtype=torch.bool,
                        device=torch.cuda.current_device(),
                    )

                if not is_te_min_version("1.10.0", check_equality=False):
                    for k, v in kwargs.items():
                        if k == "attention_mask":
                            if v is not None:
                                cudagraph_args.append(v)
                                optional_inputs[k] = None
                            else:
                                cudagraph_args.append(
                                    get_zero_attention_mask(
                                        hidden_states.size(0), hidden_states.size(1)
                                    )
                                )
                        else:
                            assert v is None, "Keyword Arguments not supported with CUDA graph."
                elif optional_inputs['attention_mask'] is None:
                    # The attention_mask can be None when there is no padding to the input sequence.
                    # However, an attention_mask Tensor must be passed into cudagraph for replay, so
                    # we create an equivalent zero Tensor as the attention_mask.
                    optional_inputs["attention_mask"] = get_zero_attention_mask(
                        hidden_states.size(0), hidden_states.size(1)
                    )
            except ImportError:
                raise RuntimeError("CUDAGraph requires TransformerEngine, but not installed")
            return tuple(cudagraph_args), optional_inputs

        cg_index = self.current_microbatch % len(self.cuda_graphs)
        assert ('inference_context' not in kwargs or kwargs['inference_context'] is None) and (
            'packed_seq_params' not in kwargs or kwargs['packed_seq_params'] is None
        ), "CUDA graph accepts only Tensor inputs."
        cudagraph_args, cudagraph_kwargs = _check_cuda_graph_replay_args(*args, **kwargs)

        for hook, hook_args in self.cuda_graph_manual_hooks:
            hook(*hook_args)
        cuda_graph_output = self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)

        if cudagraph_kwargs['context'] is not None:
            context = cuda_graph_output[-1]
            cuda_graph_output = cuda_graph_output[:-1]
        else:
            context = None
        if self.config.cuda_graph_scope == "attn":
            # CUDA Graph only covers the attention layer. Feed-forward
            # layer still goes through the normal pass.
            output = self._forward_mlp(*cuda_graph_output)
        else:
            output = cuda_graph_output[0]
        return output, context

    def __call__(self, *args, **kwargs):
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return self.cudagraph_manager(self, args, kwargs)
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                (
                    kwargs.get('inference_context') is not None
                    and kwargs['inference_context'].is_decode_only()
                )
                or (
                    kwargs.get('inference_params') is not None
                    and kwargs['inference_params'].is_decode_only()
                )
            )
        ):
            assert (
                kwargs.get('attention_mask') is None
            ), f"Attention mask must not be set when using CUDA graphs for decode"
            return self.cudagraph_manager(self, args, kwargs)
        elif self.config.external_cuda_graph and self.training:
            if not self.cuda_graphs:
                # Do CUDA Graphs capture.
                cuda_graph_func = self._cuda_graph_capture
            else:
                # Do CUDA Graphs replay.
                cuda_graph_func = self._cuda_graph_replay
            return cuda_graph_func(*args, **kwargs)
        return super(MegatronModule, self).__call__(*args, **kwargs)

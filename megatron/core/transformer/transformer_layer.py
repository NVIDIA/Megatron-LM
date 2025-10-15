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
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_rank,
    is_te_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)


def get_transformer_layer_offset(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0

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

            middle_pipeline_rank = (
                pp_rank if config.num_layers_in_first_pipeline_stage is None else pp_rank - 1
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

                num_layers_per_virtual_model_chunk_in_middle_pipeline_stage = (
                    middle_num_layers // vp_size
                )

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pp_rank == 0:
                    offset = vp_stage * total_virtual_chunks
                else:
                    offset = (
                        vp_stage * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + middle_pipeline_rank
                        * (
                            num_layers_per_virtual_model_chunk_in_middle_pipeline_stage
                            // middle_pipeline_stages
                        )
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                if pp_rank == 0:
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

            # import here to avoid circular import
            from megatron.core.pipeline_parallel.utils import is_vp_first_stage

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (pp_rank * num_layers_per_virtual_rank)

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not (
                    is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage
                ):
                    offset -= 1
            else:
                offset = pp_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not (
                    is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage
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


class TransformerLayer(GraphableMegatronModule, BaseTransformerLayer):
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
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(
            self.config, vp_stage, get_pg_rank(pg_collection.pp)
        )
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

        attention_optional_kwargs["pg_collection"] = pg_collection

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
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if submodules.mlp is not a ModuleSpec,we dont have to handle passing additional kwargs
        if isinstance(submodules.mlp, ModuleSpec):
            if submodules.mlp.module in (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP):
                additional_mlp_kwargs["pg_collection"] = pg_collection
            elif submodules.mlp.module == MLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for MLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            elif TEFusedMLP is not None and submodules.mlp.module == TEFusedMLP:
                assert hasattr(
                    pg_collection, 'tp'
                ), 'TP process group is required for TEFusedMLP in TransformerLayer'
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
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

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        self.recompute_input_layernorm = False
        self.recompute_pre_mlp_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == 'selective':
            if "layernorm" in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp) and (
                    self.config.cuda_graph_impl == "none"
                    or 'attn' not in self.config.cuda_graph_scope
                ):
                    self.recompute_input_layernorm = True
                    if self.config.fp8:
                        self.self_attention.set_for_recompute_input_layernorm()
                if not isinstance(self.pre_mlp_layernorm, IdentityOp) and (
                    self.config.cuda_graph_impl == "none"
                    or (not self.is_moe_layer and 'mlp' not in self.config.cuda_graph_scope)
                    or (
                        self.is_moe_layer
                        and 'moe' not in self.config.cuda_graph_scope
                        and 'moe_router' not in self.config.cuda_graph_scope
                    )
                ):
                    self.recompute_pre_mlp_layernorm = True
                    if self.config.fp8:
                        if isinstance(self.mlp, MoELayer):
                            self.mlp.set_for_recompute_pre_mlp_layernorm()
                        else:
                            from megatron.core.extensions.transformer_engine import (
                                set_save_original_input,
                            )

                            set_save_original_input(self.mlp.linear_fc1)
            if "mlp" in self.config.recompute_modules:
                if not self.is_moe_layer:
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
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        kwargs.pop("dynamic_inference_decode_only", None)
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
        rotary_pos_cos_sin: Optional[Tensor] = None,
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
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
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
            rotary_pos_cos_sin=rotary_pos_cos_sin,
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
                    self.pg_collection.tp,
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

        # Return right here if we are partially capturing the MoE.
        if (
            self.is_moe_layer
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and is_graph_capturing()
            and 'moe_router' in self.config.cuda_graph_scope
        ):
            return mlp_output_with_bias + [residual]

        return self._forward_post_mlp(mlp_output_with_bias, residual)

    def _forward_post_mlp(self, mlp_output_with_bias, residual):
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
        Get the static inputs for the transformer layer. Besides the hidden_states that is
        generated in GraphableMegatronModule, we also add the attention_mask.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)

        if (
            not isinstance(self.self_attention, IdentityOp)
            and 'attn' in self.config.cuda_graph_scope
        ):
            slen_per_cp = seq_length // self.config.context_parallel_size
            static_inputs["attention_mask"] = (
                ~(torch.tril(torch.ones((slen_per_cp, seq_length))).bool())
                .to(torch.cuda.current_device())
                .reshape(1, 1, slen_per_cp, seq_length)
                .tile(micro_batch_size, 1, 1, 1)
            )
        return static_inputs

    def _get_submodules_under_cudagraphs(self):
        """
        Get the submodules that are covered by cudagraphs.
        """
        submodules = []
        if 'attn' in self.config.cuda_graph_scope:
            submodules += [
                self.input_layernorm,
                self.self_attention,
                self.pre_cross_attn_layernorm,
                self.cross_attention,
            ]
        if (not self.is_moe_layer and 'mlp' in self.config.cuda_graph_scope) or (
            self.is_moe_layer and 'moe' in self.config.cuda_graph_scope
        ):
            submodules += [self.pre_mlp_layernorm, self.mlp]
        elif self.is_moe_layer and 'moe_router' in self.config.cuda_graph_scope:
            submodules += [self.pre_mlp_layernorm, self.mlp.router]
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                submodules += [self.mlp.shared_experts]
        return submodules

    def _te_cuda_graph_capture(self, *args, **kwargs):
        """
        CUDA Graph capture for this layer using TE interface.
        There are some differences from the normal pass:
        1. In some conditions CUDA graph cannot cover the entire layer. The `cuda_graph_scope`
           attribute can be set to control the scope of the CUDA graph.
        2. If context is None, it cannot be returned as output.
        """
        context = None
        if 'attn' in self.config.cuda_graph_scope:
            hidden_states, context = self._forward_attention(*args, **kwargs)
        else:
            if len(args) > 0:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.pop("hidden_states")

        if (not self.is_moe_layer and 'mlp' in self.config.cuda_graph_scope) or (
            self.is_moe_layer
            and (
                'moe' in self.config.cuda_graph_scope
                or 'moe_router' in self.config.cuda_graph_scope
            )
        ):
            hidden_states = self._forward_mlp(hidden_states)
        if not isinstance(hidden_states, list) and not isinstance(hidden_states, tuple):
            cuda_graph_outputs = [hidden_states]
        else:
            cuda_graph_outputs = list(hidden_states)
        if context is not None:
            cuda_graph_outputs.append(context)
        return tuple(cuda_graph_outputs)

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch `self.current_microbatch` using TE
        interface. TransformerEngine versions>=1.10 allow keyword arguments with CUDA graph.
        However, CUDA graph accepts only Tensor inputs.
        Hence, `inference_context` and `packed_seq_params` are excluded from input list.
        """
        context = None
        if 'attn' not in self.config.cuda_graph_scope:
            hidden_states, context = self._forward_attention(*args, **kwargs)
            args = (hidden_states,)
            kwargs = {}

        assert (kwargs.get('inference_context') is None) and (
            kwargs.get('packed_seq_params') is None
        ), (
            "CUDA graph accepts only Tensor inputs. "
            "inference_context and packed_seq_params are excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )

        cuda_graph_output = list(super()._te_cuda_graph_replay(*args, **kwargs))

        if kwargs.get('context') is not None:
            context = cuda_graph_output.pop()

        if (not self.is_moe_layer and 'mlp' in self.config.cuda_graph_scope) or (
            self.is_moe_layer and 'moe' in self.config.cuda_graph_scope
        ):
            # CUDA Graph captures the whole MLP/MoE part. CUDA Graph output is the layer output.
            assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
            output = cuda_graph_output.pop()
        elif self.is_moe_layer and 'moe_router' in self.config.cuda_graph_scope:
            # CUDA Graph partially captures the MoE.
            # The rest of the layer should go to the normal pass.
            shared_expert_output, routing_map, residual = None, None, None
            mlp_residual = cuda_graph_output.pop()
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                # The shared expert output is the fourth element in the CUDA graph output.
                shared_expert_output = cuda_graph_output.pop()

            # Split cudagraph outputs into function outputs and attribute outputs, and
            # process them separately. Function outputs should have three tensors.
            func_output, attr_outputs = cuda_graph_output[:3], cuda_graph_output[3:]
            if 'moe_preprocess' in self.config.cuda_graph_scope:
                hidden_states, probs, residual = func_output
                valid_cudagraph_attrs = self.mlp.token_dispatcher.valid_cudagraph_attrs
                assert len(attr_outputs) == len(
                    valid_cudagraph_attrs
                ), f"attr_outputs: {len(attr_outputs)} != {len(valid_cudagraph_attrs)}"
                for i, attr_name in enumerate(valid_cudagraph_attrs):
                    hier_attr_name = attr_name.split('.')
                    attr = self.mlp.token_dispatcher
                    for name in hier_attr_name[:-1]:
                        attr = getattr(attr, name)
                    setattr(attr, hier_attr_name[-1], attr_outputs[i])
            else:
                hidden_states, probs, routing_map = func_output
                assert not attr_outputs, "cuda_graph_attr_outputs should be empty"

            # Resume the MoELayer forward pass from the end of the CUDA graph scope.
            # The MoE layer will skip redundant computations when we pass in the calculated values
            # through the keyword arguments. See MoELayer.forward docstring for more details.
            mlp_output_with_bias = self.mlp(
                hidden_states,
                probs=probs,
                routing_map=routing_map,
                residual=residual,
                shared_expert_output=shared_expert_output,
            )
            output = self._forward_post_mlp(mlp_output_with_bias, mlp_residual)
        else:
            # CUDA Graph does not capture the MLP/MoE part at all.
            output = self._forward_mlp(*cuda_graph_output)
        return output, context

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        """Helper function to get tensor arguments for TE CUDA graph."""
        cudagraph_args, cudagraph_kwargs = super()._get_te_cuda_graph_replay_args(*args, **kwargs)

        assert (
            len(cudagraph_args) == 1
        ), "Exactly one positional argument `hidden_states` is expected."
        hidden_states = cudagraph_args[0]

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

            if not is_te_min_version("1.10.0"):
                # TE version < 1.10.0 does not support keyword arguments with CUDA graph.
                for k, v in cudagraph_kwargs.items():
                    if k == "attention_mask":
                        if v is not None:
                            cudagraph_args.append(v)
                            cudagraph_kwargs[k] = None
                        else:
                            cudagraph_args.append(
                                get_zero_attention_mask(
                                    hidden_states.size(0), hidden_states.size(1)
                                )
                            )
                    elif k != 'is_first_microbatch':
                        assert v is None, "Keyword Arguments not supported with CUDA graph."
            elif (
                'attention_mask' in cudagraph_kwargs and cudagraph_kwargs['attention_mask'] is None
            ):
                # The attention_mask can be None when there is no padding to the input sequence.
                # However, an attention_mask Tensor must be passed into cudagraph for replay, so
                # we create an equivalent zero Tensor as the attention_mask.
                cudagraph_kwargs["attention_mask"] = get_zero_attention_mask(
                    hidden_states.size(0), hidden_states.size(1)
                )
        except ImportError:
            raise RuntimeError("CUDAGraph requires TransformerEngine, but not installed")
        return tuple(cudagraph_args), cudagraph_kwargs

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return True
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                (kwargs.get('inference_context') is not None)
                or (kwargs.get('inference_params') is not None)
            )
            and 'full_iteration' not in self.config.cuda_graph_scope
        ):
            if kwargs['inference_context'].is_static_batching():
                using_cuda_graph = kwargs['inference_context'].is_decode_only()
            else:
                # it can happen that non-decode steps have a token count greater than the max
                # supported cuda graph token count. In that case this flag will be set to
                # False by initialize_attention, and we should not use cuda graphs.
                using_cuda_graph = kwargs['inference_context'].using_cuda_graph_this_step()
            if using_cuda_graph:
                return True
        return False

    def __call__(self, *args, **kwargs):
        if self._should_call_local_cudagraph(*args, **kwargs):
            # Inference mode.
            if kwargs.get('inference_context') is not None:
                # dynamic_inference_decode_only is not a real argument to forward, it is only used
                # to differentiate the cuda graph used for decode from the one used for non-decode
                # inference.
                kwargs["dynamic_inference_decode_only"] = kwargs[
                    'inference_context'
                ].is_decode_only()
        return super().__call__(*args, **kwargs)

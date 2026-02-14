# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import functools
import logging
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import torch.distributed
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core.transformer.enums import CudaGraphScope, LayerType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_rank,
    is_te_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext

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
        input_layernorm: Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm: Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm: Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: LayerNormBuilder = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: LayerNormBuilder = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: LayerNormBuilder = IdentityOp
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
        is_mtp_layer: bool = False,
    ):
        self.submodules_config = submodules
        super().__init__(config=config, vp_stage=vp_stage)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        # MTP inner layers use their own layer numbering (starting from 1 within each MTP depth),
        # so they should NOT add the decoder layer offset. The router.py handles MTP layer
        # numbering separately by adding config.num_layers to distinguish MTP layers from decoder
        # layers in the aux loss tracker.
        if is_mtp_layer:
            self.layer_number = layer_number
        else:
            self.layer_number = layer_number + get_transformer_layer_offset(
                self.config, vp_stage, get_pg_rank(pg_collection.pp)
            )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.is_mtp_layer = is_mtp_layer

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = submodules.input_layernorm(
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
        self.pre_cross_attn_layernorm = submodules.pre_cross_attn_layernorm(
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
        self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(
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
                # Pass is_mtp_layer flag to MoELayer to distinguish MTP MoE layers.
                if submodules.mlp.module == MoELayer:
                    additional_mlp_kwargs["is_mtp_layer"] = self.is_mtp_layer
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
            assert self.config.recompute_modules is not None
            if "layernorm" in self.config.recompute_modules:
                if not isinstance(self.input_layernorm, IdentityOp):
                    self.recompute_input_layernorm = True
                    if self.config.fp8 or self.config.fp4:
                        self.self_attention.set_for_recompute_input_layernorm()

                def can_recompute_pre_mlp_layernorm_for_cudagraph():
                    if (
                        not self.is_moe_layer
                        or CudaGraphScope.moe_router not in self.config.cuda_graph_scope
                        or self.config.cuda_graph_impl == "local"
                    ):
                        # Not a MoE layer, or not capturing the router part.
                        return True
                    if (
                        self.config.moe_shared_expert_intermediate_size is not None
                        and self.config.moe_shared_expert_overlap
                    ):
                        # If shared expert overlap is used, we cannot make the pre-mlp layernorm
                        # recomputation, because the shared expert takes the layernorm output as
                        # input, and it is outside of the CUDA graph scope.
                        log_single_rank(
                            logger,
                            logging.WARNING,
                            "pre_mlp_layernorm recompute is not supported with moe router "
                            "cudagraph + shared expert overlap. Disabling pre_mlp_layernorm "
                            "recompute.",
                        )
                        return False
                    if CudaGraphScope.moe_preprocess in self.config.cuda_graph_scope and (
                        self.config.moe_token_dispatcher_type == "alltoall"
                        or self.config.moe_latent_size
                    ):
                        # Only when capturing the preprocess part and using alltoall token
                        # dispatcher or latent MoE can we make the pre-mlp layernorm recomputation.
                        # Because in other cases the layernorm output returns directly as one of the
                        # outputs of the cudagraph, which will be allocated a static buffer, thus
                        # not able to be released.
                        return True
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        "pre_mlp_layernorm recompute is only supported with moe router + "
                        "preprocess cudagraph will alltoall token dispatcher or latent MoE. "
                        "Disabling pre_mlp_layernorm recompute.",
                    )
                    return False

                if (
                    not isinstance(self.pre_mlp_layernorm, IdentityOp)
                    and can_recompute_pre_mlp_layernorm_for_cudagraph()
                ):
                    self.recompute_pre_mlp_layernorm = True
                    if self.config.fp8 or self.config.fp4:
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
        self.offload_attn_norm = (
            self.config.fine_grained_activation_offloading
            and "attn_norm" in self.config.offload_modules
            and not isinstance(self.input_layernorm, IdentityOp)
        )
        self.offload_mlp_norm = (
            self.config.fine_grained_activation_offloading
            and "mlp_norm" in self.config.offload_modules
            and not isinstance(self.pre_mlp_layernorm, IdentityOp)
        )

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def create_mcore_cudagraph_manager(self, config):
        """Register the transformer layer for cudagraphs."""

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        # If full scope, just cudagraph the entire layer
        if not self.config.cuda_graph_scope:
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphScope.attn in self.config.cuda_graph_scope
            and self.submodules_config.self_attention != IdentityOp
        ):
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphScope.mlp in self.config.cuda_graph_scope
            and self.submodules_config.mlp != IdentityOp
        ):
            # Cudagraphing MoE layers are supposed handled by MoeTransforerLayer
            assert not self.is_moe_layer
            self.cudagraph_manager = CudaGraphManager(config)

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
        output = self._forward_mlp(
            hidden_states,
            kwargs.get("inference_context", None),
            padding_mask=kwargs.get("padding_mask", None),
        )
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
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
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
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                    apply_module(self.input_layernorm), hidden_states
                )
        else:
            with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
                input_layernorm_output = apply_module(self.input_layernorm)(hidden_states)

        using_fused_tp_inference_kernel = (not self.training) and (
            self.config.inference_fuse_tp_communication
        )

        if using_fused_tp_inference_kernel:
            # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
            # operation in attention's out_proj (linear_proj)
            self._set_proj_residual(residual)

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
        if using_fused_tp_inference_kernel:
            # In inference optimized transformer layer, there is no bias and dropout
            # The remaining residual add is already handled inside the
            # self attention module.
            hidden_states = attention_output_with_bias[0]
        else:
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                    attention_output_with_bias, residual, self.hidden_dropout
                )
        nvtx_range_pop(suffix="self_attn_bda")

        # Delay the offload of the attention norm until after the self_attn_bda has been computed
        # because the residual is needed in the self_attn_bda.
        if self.offload_attn_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="attn_norm", forced_released_tensors=[residual]
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = apply_module(self.pre_cross_attn_layernorm)(hidden_states)

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

    def _forward_pre_mlp_layernorm(self, hidden_states: Tensor):
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            with off_interface(self.offload_mlp_norm, hidden_states, "mlp_norm") as hidden_states:
                pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                    apply_module(self.pre_mlp_layernorm), hidden_states
                )
        else:
            with off_interface(self.offload_mlp_norm, hidden_states, "mlp_norm") as hidden_states:
                pre_mlp_layernorm_output = apply_module(self.pre_mlp_layernorm)(hidden_states)

        return pre_mlp_layernorm_output

    def _forward_mlp(
        self,
        hidden_states: Tensor,
        inference_context: BaseInferenceContext | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor | list[Tensor | None]:
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                Shape [seq_length, batch_size, hidden_size].
            inference_context: Inference context for optimizations.
            padding_mask (Tensor, optional): Padding mask for MoE routing.
                Shape [bsz, seq_length]. True = padding (exclude), False = valid (include).
                Only used for MoE layers to exclude padding tokens from aux loss computations.
                The MoELayer will internally transform this to [seq_length, bsz] format.
        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)

        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
            and not self.config.transformer_impl == "inference_optimized"
        )

        using_fused_tp_inference_kernel = (not self.training) and (
            self.config.inference_fuse_tp_communication
        )

        if self.recompute_mlp:
            if self.config.fp8 or self.config.fp4:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    pre_mlp_layernorm_output,
                    padding_mask=padding_mask,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(
                    functools.partial(self.mlp, padding_mask=padding_mask),
                    False,
                    pre_mlp_layernorm_output,
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
            if using_fused_tp_inference_kernel:
                # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
                # operation in MLP's fc2.
                self._set_fc2_residual(residual)
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)

        nvtx_range_pop(suffix="mlp")

        if (
            self.is_moe_layer
            and self.config.cuda_graph_impl == "transformer_engine"
            and self.training
            and is_graph_capturing()
            and CudaGraphScope.moe_router in self.config.cuda_graph_scope
        ):
            if self.recompute_pre_mlp_layernorm:
                # Register the recompute hooks to all the cudagraph output tensors, because some
                # tensors are in parallel execution paths and they all need pre_mlp_layernorm to be
                # recomputed in backward pass. For example, the router path and the shared expert
                # path. So only register in one path is risky.
                for tensor in mlp_output_with_bias:
                    self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(tensor)
            return list(mlp_output_with_bias) + [residual]
        else:
            return self._forward_post_mlp(mlp_output_with_bias, residual)

    def _forward_post_mlp(
        self, mlp_output_with_bias: tuple[Tensor, Tensor | None], residual: Tensor
    ) -> Tensor:
        """
        Perform operations after the MLP computation.

        Args:
            mlp_output_with_bias (Tensor): Output tensor of the MLP layer with bias.
            residual (Tensor): Residual tensor.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        using_fused_tp_inference_kernel = (not self.training) and (
            self.config.inference_fuse_tp_communication
        )

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        nvtx_range_push(suffix="mlp_bda")
        if using_fused_tp_inference_kernel:
            # In inference optimized transformer layer, there is no bias and dropout
            # The remaining residual add is already handled inside the
            # MLP module.
            hidden_states = mlp_output_with_bias[0]
        else:
            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual, self.hidden_dropout
                )
        nvtx_range_pop(suffix="mlp_bda")
        # Delay the offload of the mlp norm until after the mlp_bda has been computed
        # because the residual is needed in the mlp_bda.
        if self.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )

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

    def configure_fused_tp_inference(
        self,
        skip_qkv_norm_and_all_gather: bool = False,
        fc2_next_layer_norm_weights: Optional[Tensor] = None,
    ):
        """
        Configure settings for fused TP communication in inference mode.

        Args:
            skip_qkv_norm (bool): Whether to skip norm and all-gather for linear_qkv.
            fc2_next_layer_norm_weights (Optional[Tensor]): Next layer's QKV norm weights
                for current layer's MLP FC2.
        """
        self.self_attention.linear_qkv.skip_norm_and_all_gather = skip_qkv_norm_and_all_gather

        # Use current layer's own MLP FC1 norm weights for attention's/mixer's out_proj
        mlp_fc1_weights = self.get_mlp_layer_norm_weights()
        self._set_proj_next_layer_norm_weights(mlp_fc1_weights)

        self.mlp.linear_fc1.skip_norm_and_all_gather = True
        # Use next layer's attention norm weights for current layer's MLP FC2
        self._set_fc2_next_layer_norm_weights(fc2_next_layer_norm_weights)

    def _set_proj_next_layer_norm_weights(self, weights: Tensor):
        """Set next layer norm weights for attention/mixer's linear_proj."""
        self.self_attention.linear_proj._set_next_layer_norm_weights(weights)

    def _set_fc2_next_layer_norm_weights(self, weights: Optional[Tensor]):
        """Set next layer norm weights for MLP FC2."""
        if weights is None:
            # Create dummy tensor for last layer (same shape as fc1 norm weights)
            weights = torch.empty_like(self.get_mlp_layer_norm_weights())
        self.mlp.linear_fc2._set_next_layer_norm_weights(weights)

    def _set_proj_residual(self, residual: Tensor):
        """Set residual for attention's/mixer's out_proj (linear_proj)."""
        self.self_attention.linear_proj._set_residual(residual)

    def _set_fc2_residual(self, residual: Tensor):
        """Set residual for MLP FC2."""
        self.mlp.linear_fc2._set_residual(residual)

    def get_mlp_layer_norm_weights(self) -> Tensor:
        """
        Get the MLP FC1 layer norm weights.

        Returns:
            Tensor: The layer norm weight data.
        """
        return self.mlp.linear_fc1.layer_norm_weight.data

    def get_qkv_layer_norm_weights(self) -> Tensor:
        """
        Get the QKV layer norm weights.

        Returns:
            Tensor: The layer norm weight data.
        """
        return self.self_attention.linear_qkv.layer_norm_weight.data

    def get_layer_static_inputs(self, seq_length, micro_batch_size):
        """
        Get the static inputs for the transformer layer. Besides the hidden_states that is
        generated in GraphableMegatronModule, we also add the attention_mask.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the static inputs for the layer.
        """
        static_inputs = super().get_layer_static_inputs(seq_length, micro_batch_size)

        if not isinstance(self.self_attention, IdentityOp) and (
            not self.config.cuda_graph_scope or CudaGraphScope.attn in self.config.cuda_graph_scope
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
        if not self.config.cuda_graph_scope:
            return super()._get_submodules_under_cudagraphs()

        submodules = []
        if CudaGraphScope.attn in self.config.cuda_graph_scope:
            submodules += [
                self.input_layernorm,
                self.self_attention,
                self.pre_cross_attn_layernorm,
                self.cross_attention,
            ]
        if (not self.is_moe_layer and CudaGraphScope.mlp in self.config.cuda_graph_scope) or (
            self.is_moe_layer and CudaGraphScope.moe in self.config.cuda_graph_scope
        ):
            submodules += [self.pre_mlp_layernorm, self.mlp]
        elif self.is_moe_layer and CudaGraphScope.moe_router in self.config.cuda_graph_scope:
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
        if not self.config.cuda_graph_scope or CudaGraphScope.attn in self.config.cuda_graph_scope:
            hidden_states, context = self._forward_attention(*args, **kwargs)
        else:
            if len(args) > 0:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.pop("hidden_states")

        if (
            not self.config.cuda_graph_scope
            or (not self.is_moe_layer and CudaGraphScope.mlp in self.config.cuda_graph_scope)
            or (
                self.is_moe_layer
                and (
                    CudaGraphScope.moe in self.config.cuda_graph_scope
                    or CudaGraphScope.moe_router in self.config.cuda_graph_scope
                )
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
        if self.config.cuda_graph_scope and CudaGraphScope.attn not in self.config.cuda_graph_scope:
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

        if (
            not self.config.cuda_graph_scope
            or (not self.is_moe_layer and CudaGraphScope.mlp in self.config.cuda_graph_scope)
            or (self.is_moe_layer and CudaGraphScope.moe in self.config.cuda_graph_scope)
        ):
            # CUDA Graph captures the whole MLP/MoE part. CUDA Graph output is the layer output.
            assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
            output = cuda_graph_output.pop()
            assert (
                not self.config.overlap_moe_expert_parallel_comm
            ), "EP overlap must be \
                disabled when CUDA graph captures the whole MLP/MoE part."
        elif self.is_moe_layer and CudaGraphScope.moe_router in self.config.cuda_graph_scope:
            # CUDA Graph partially captures the MoE.
            # The rest of the layer should go to the normal pass.
            shared_expert_output, routing_map = None, None
            # residual is the last element in the CUDA graph output.
            residual = cuda_graph_output.pop()
            if (
                self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            ):
                # The shared expert output is the last second element in the CUDA graph output.
                shared_expert_output = cuda_graph_output.pop()

            if CudaGraphScope.moe_preprocess in self.config.cuda_graph_scope:
                # CUDA graph output is [hidden_states, probs] + attributes outputs.
                (hidden_states, probs), attr_outputs = cuda_graph_output[:2], cuda_graph_output[2:]
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
                # CUDA graph output is [hidden_states, probs, routing_map].
                assert len(cuda_graph_output) == 3, (
                    "CUDA graph output should be [hidden_states, probs, routing_map], "
                    f"but got {len(cuda_graph_output)} elements"
                )
                hidden_states, probs, routing_map = cuda_graph_output

            # Resume the MoELayer forward pass from the end of the CUDA graph scope.
            # The MoE layer will skip redundant computations when we pass in the calculated values
            # through the keyword arguments. See MoELayer.forward docstring for more details.
            nvtx_range_push(suffix="mlp")
            self.mlp.cudagraph_tensor_store.set(
                hidden_states=hidden_states,
                probs=probs,
                routing_map=routing_map,
                shared_expert_output=shared_expert_output,
            )
            # If EP overlap is enabled, remaining of mlp will be called as fine_grained_callables
            # and should be skipped here.
            if self.config.overlap_moe_expert_parallel_comm:
                probs, routing_map = self.mlp.route(hidden_states)
                hidden_states, probs = self.mlp.preprocess(hidden_states, probs, routing_map)
                nvtx_range_pop(suffix="mlp")
                return residual, hidden_states, probs, shared_expert_output
            mlp_output_with_bias = self.mlp(hidden_states)
            self.mlp.cudagraph_tensor_store.clear()
            nvtx_range_pop(suffix="mlp")

            # If we early returned, layernorm recompute hooks were attached to the output buffer
            # of the cudagraph, so disable the recompute hooks inside _forward_post_mlp
            recompute_pre_mlp_layernorm = self.recompute_pre_mlp_layernorm
            self.recompute_pre_mlp_layernorm = False
            output = self._forward_post_mlp(mlp_output_with_bias, residual)
            self.recompute_pre_mlp_layernorm = recompute_pre_mlp_layernorm
        else:
            # If EP overlap is enabled, needs to return same outputs as submodule.attn
            if self.config.overlap_moe_expert_parallel_comm:
                assert len(cuda_graph_output) == 1, "CUDA Graph output should be the layer output."
                residual = cuda_graph_output.pop()
                if not self.is_moe_layer:
                    return residual, None, None, None
                hidden_states = apply_module(self.pre_mlp_layernorm)(residual)
                shared_expert_output = self.mlp.shared_experts_compute(hidden_states)
                probs, routing_map = self.mlp.route(hidden_states)
                hidden_states, probs = self.mlp.preprocess(hidden_states, probs, routing_map)
                return residual, hidden_states, probs, shared_expert_output

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
            and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
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

    def get_layer_norm_weights(self):
        """
        Get the weights of all layernorms (attention and MLP) in the transformer layer.
        Returns:
            List[Tensor]: A list of layernorm weight tensors.
        """
        return


class MoETransformerLayer(TransformerLayer):
    """
    A Transformer layer specialized for Mixture-of-Experts (MoE) architectures.

    Implements specific functionality to support CUDA graph capture for MoE layers.
    Due to the dynamic nature of MoE, capturing the entire layer in a single CUDA graph
    can be challenging. This class supports "partial" CUDA graphs by decomposing the
    MLP forward pass into router, expert-compute, and post-process stages.
    """

    def __init__(self, *args, **kwargs):
        self.is_moe_layer = True
        self.use_partial_cudagraphs = False
        self.moe_layer_recompute = False
        self.token_dispatcher_attrs = {}

        super().__init__(*args, **kwargs)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        if self.use_partial_cudagraphs:
            return False
        if self.config.cuda_graph_impl != "local":
            return False
        return super()._should_call_local_cudagraph(*args, **kwargs)

    def create_mcore_cudagraph_manager(self, config):
        """
        Initializes the CUDA graph manager(s) for the MoE layer.

        Unlike the standard layer which typically uses a single manager, this method
        can configure multiple graph managers if partial CUDA graphs are enabled via
        `cuda_graph_scope`. This allows capturing the static parts of the MoE pass
        while leaving the expert computation to execute eagerly.
        """

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if not self.config.cuda_graph_scope or CudaGraphScope.moe in self.config.cuda_graph_scope:
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphScope.moe_router in self.config.cuda_graph_scope
            or CudaGraphScope.moe_preprocess in self.config.cuda_graph_scope
        ):
            # full MoE layer recompute with partial_cudagraphs. If not partial cudagraphs, MoE
            # layer recompute is handled by the moe_layer.MoELayer class
            self.moe_layer_recompute = (
                self.config.recompute_granularity == 'selective'
                and "moe" in self.config.recompute_modules
                and self.config.cuda_graph_impl == "local"
            )

            self.use_partial_cudagraphs = True
            self.cudagraph_manager_router = CudaGraphManager(
                self.config, self, function_name="_forward_mlp_router"
            )
            self.cudagraph_manager_postprocess = CudaGraphManager(
                self.config, self, function_name="_forward_mlp_postprocess"
            )

    def _forward_mlp_router(self, hidden_states, padding_mask=None):
        """
        Executes the router phase of the MoE block.

        This includes the pre-MLP layernorm and the routing logic.
        This method is isolated so it can be captured by `cudagraph_manager_router`.
        """

        residual = hidden_states
        self.mlp.fwd_execution_map = "route"
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)
        router_outputs = self.mlp(
            pre_mlp_layernorm_output, intermediate_tensors=(), padding_mask=padding_mask
        )

        for attr_name in self.mlp.token_dispatcher.cudagraph_attrs:
            attr = getattr(self.mlp.token_dispatcher, attr_name)
            if torch.is_tensor(attr):
                if attr_name in self.token_dispatcher_attrs:
                    self.token_dispatcher_attrs[attr_name].copy_(attr)
                else:
                    self.token_dispatcher_attrs[attr_name] = attr.detach()

        return residual, *router_outputs

    def _forward_mlp_expert_compute(self, hidden_states, probs):
        """
        Executes the actual computation of the experts.

        This phase takes the routing information and inputs, dispatches them to the
        appropriate experts, and computes the results. In partial graph modes, this
        step runs eagerly between the router and postprocess graph replays.
        """

        for name, attr in self.token_dispatcher_attrs.items():
            setattr(self.mlp.token_dispatcher, name, attr)

        self.mlp.fwd_execution_map = "expert_compute"
        return self.mlp(None, intermediate_tensors=(hidden_states, probs))

    def _forward_mlp_postprocess(self, residual, output, shared_expert_output, mlp_bias):
        """
        Executes the post-processing phase of the MoE block.

        Handles combining the expert outputs, applying biases, re-registering
        activation recomputation hooks if necessary, and performing the final
        Bias-Dropout-Add. This method is isolated so it can be captured by cudagraphs.

        """

        self.mlp.fwd_execution_map = "postprocess"
        output = self.mlp(None, intermediate_tensors=(output, shared_expert_output))
        return self._forward_post_mlp((output, mlp_bias), residual)

    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
        """
        Orchestrates the MLP forward pass, handling partial CUDA graph execution logic.

        If `use_partial_cudagraphs` is True, this method stitches together the
        router, expert_compute, and postprocess calls.
        """

        if inference_context is not None:
            assert not self.use_partial_cudagraphs, (
                "Partial cudagraphs for MoEs were detected during inference!"
                "Please do not use --cuda-graph-scope moe_router moe_preprocess "
                "alongside inference."
            )

        def _forward_mlp_partial_cudagraphs(
            hidden_states, inference_context=None, padding_mask=None
        ):
            residual, hidden_states, probs, shared_expert_output = self._forward_mlp_router(
                hidden_states, padding_mask=padding_mask
            )
            expert_output, mlp_bias = self._forward_mlp_expert_compute(hidden_states, probs)
            return self._forward_mlp_postprocess(
                residual, expert_output, shared_expert_output, mlp_bias
            )

        if self.use_partial_cudagraphs:
            if self.moe_layer_recompute:
                if self.config.fp8 or self.config.fp4:
                    from megatron.core.extensions.transformer_engine import te_checkpoint

                    return te_checkpoint(
                        _forward_mlp_partial_cudagraphs,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                        padding_mask=padding_mask,
                    )
                else:
                    return tensor_parallel.checkpoint(
                        functools.partial(
                            _forward_mlp_partial_cudagraphs, padding_mask=padding_mask
                        ),
                        False,
                        hidden_states,
                    )
            else:
                return _forward_mlp_partial_cudagraphs(hidden_states, padding_mask=padding_mask)
        else:
            return super()._forward_mlp(hidden_states, padding_mask=padding_mask)

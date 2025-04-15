# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, make_viewless_tensor

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]


try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDelayedScaling,
        TENorm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict, word_emb_weight: Tensor, word_emb_weight_key: str
) -> None:
    """tie the embedding of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        word_emb_weight (Tensor): weight of the word embedding.
        word_emb_weight_key (str): key of the word embedding in the sharded state dict.

    Returns: None, acts in-place
    """
    mtp_word_emb_replica_id = (
        1,  # copy of embedding in pre processing stage
        0,
        parallel_state.get_data_parallel_rank(with_context_parallel=True),
    )
    assert word_emb_weight_key in sharded_state_dict
    del sharded_state_dict[word_emb_weight_key]
    sharded_state_dict[word_emb_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=word_emb_weight,
        key=word_emb_weight_key,
        replica_id=mtp_word_emb_replica_id,
        allow_shape_mismatch=True,
    )


def tie_output_layer_state_dict(
    sharded_state_dict: ShardedStateDict, output_layer_weight: Tensor, output_layer_weight_key: str
) -> None:
    """tie the output layer of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        output_layer_weight (Tensor): weight of the output layer.
        output_layer_weight_key (str): key of the output layer in the sharded state dict.

    Returns: None, acts in-place
    """
    mtp_output_layer_replica_id = (
        1,  # copy of output layer in post processing stage
        0,
        parallel_state.get_data_parallel_rank(with_context_parallel=True),
    )
    assert output_layer_weight_key in sharded_state_dict
    del sharded_state_dict[output_layer_weight_key]
    sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=output_layer_weight,
        key=output_layer_weight_key,
        replica_id=mtp_output_layer_replica_id,
        allow_shape_mismatch=True,
    )


def roll_tensor(tensor, shifts=-1, dims=-1):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(0)
    return rolled_tensor, rolled_tensor.sum()


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the mtp loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
            mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=loss.device)
        tracker["values"][layer_number] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    def clean_loss_in_tracker():
        """Clear the mtp losses."""
        tracker = MTPLossLoggingHelper.tracker
        tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    def reduce_loss_in_tracker():
        """Collect and reduce the mtp losses across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        # Reduce mtp losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        mtp_losses = tracker["values"] * loss_scale
        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            name = f"mtp_{i+1} loss"
            loss = mtp_losses[i]
            if total_loss_dict is not None:
                total_loss_dict[name] = loss
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{name}": loss}, iteration)

        MTPLossLoggingHelper.clean_loss_in_tracker()


@dataclass
class MultiTokenPredictionLayerSubmodules:
    """
    Dataclass for specifying the submodules of a MultiTokenPrediction module.

    Args:
        hnorm (Union[ModuleSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        enorm (Union[ModuleSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        transformer_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer block to be applied.
    """

    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    transformer_layer: Union[ModuleSpec, type] = None
    layer_norm: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(
    transformer_layer_spec: ModuleSpec, use_transformer_engine: bool
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    if use_transformer_engine:
        assert HAVE_TE, "transformer_engine should be installed if use_transformer_engine is True"
        layer_norm_impl = TENorm
        column_parallel_linear_impl = TEColumnParallelLinear
    else:
        layer_norm_impl = LNImpl
        column_parallel_linear_impl = ColumnParallelLinear

    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            transformer_layer=transformer_layer_spec,
            layer_norm=layer_norm_impl,
        ),
    )

    return mtp_layer_spec


def get_mtp_layer_offset(config: TransformerConfig) -> int:
    """Get the offset of the MTP layer."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    return 0


def get_mtp_num_layers_to_build(config: TransformerConfig) -> int:
    """Get the number of MTP layers to build."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    if mpu.is_pipeline_last_stage():
        return config.mtp_num_layers if config.mtp_num_layers else 0
    else:
        return 0


class MTPLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            mtp_loss (torch.Tensor): The mtp loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for mtp loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        (mtp_loss,) = ctx.saved_tensors
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = torch.ones_like(mtp_loss) * mtp_loss_backward_scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MTPLossAutoScaler.main_loss_backward_scale = scale


class MultiTokenPredictionLayer(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
    ):
        super().__init__(config=config)
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number

        self_attention_spec = self.submodules.transformer_layer.submodules.self_attention
        attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
        assert attn_mask_type in SUPPORTED_ATTN_MASK, (
            f"Multi-Token Prediction (MTP) is not jet supported with "
            + f"{attn_mask_type} attention mask type."
            + f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
        )

        self.enorm = build_module(
            self.submodules.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
        # of the i-th tocken's hidden states and the (i + K)-th tocken's decoder input,
        # so the input's shape is [s, b, 2*h].
        # The output will be send to the following transformer layer,
        # so the output's shape should be [s, b, h].
        self.eh_proj = build_module(
            self.submodules.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        self.transformer_layer = build_module(self.submodules.transformer_layer, config=self.config)

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        decoder_input: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
    ):
        """
        Perform the forward pass through the MTP layer.

        Args:
            hidden_states (Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            decoder_input (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
                At the (k - 1)-th MTP module, the i-th element of decoder input is
                the embedding of (i + K)-th tocken.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_params (InferenceParams, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, f"multi token prediction + cross attention is not yet supported."
        assert (
            packed_seq_params is None
        ), f"multi token prediction + sequence packing is not yet supported."

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

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

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        with rng_context, fp8_context:
            decoder_input = self.enorm(decoder_input)
            decoder_input = make_viewless_tensor(
                inp=decoder_input, requires_grad=True, keep_graph=True
            )
            hidden_states = self.hnorm(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
            # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
            # and the (i + K)-th tocken's embedding, and combine them with linear projection.
            hidden_states = torch.cat((decoder_input, hidden_states), -1)
            hidden_states, _ = self.eh_proj(hidden_states)
            # For tensor parallel, all gather after linear_fc.
            hidden_states = all_gather_last_dim_from_tensor_parallel_region(hidden_states)
            # For sequence parallel, scatter after linear_fc and before transformer layer.
            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(hidden_states)
            hidden_states, _ = self.transformer_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )

        # Layer norm before shared head layer.
        hidden_states = self.final_layernorm(hidden_states)
        # TENorm produces a "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        return sharded_state_dict


@dataclass
class MultiTokenPredictionBlockSubmodules:
    """
    Dataclass for specifying the submodules of a multi token prediction block.

    This class defines the structure for configuring the layers, allowing for
    flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the multi token prediction block. Each specification typically
            defines a complete multi token prediction layer (e.g., shared embedding,
            projection matrix, transformer block, shared output head).
    """

    layer_specs: List[ModuleSpec] = None


def _get_mtp_block_submodules(
    config: TransformerConfig, spec: Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]
) -> MultiTokenPredictionBlockSubmodules:
    """
    Retrieve or construct MultiTokenPredictionBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[MultiTokenPredictionBlockSubmodules, ModuleSpec]): Specification for the
            multi token prediction block submodules.
            Can be either a MultiTokenPredictionBlockSubmodules instance or a ModuleSpec.

    Returns:
        MultiTokenPredictionBlockSubmodules: The submodules for the multi token prediction block.
    """

    # Transformer block submodules.
    if isinstance(spec, MultiTokenPredictionBlockSubmodules):
        return spec
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, MultiTokenPredictionBlock):
            return spec.submodules
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(MegatronModule):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential modules to predict
    D additional tokens.

    The k-th MTP module consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self, config: TransformerConfig, spec: Union[TransformerBlockSubmodules, ModuleSpec]
    ):
        super().__init__(config=config)
        self.submodules = _get_mtp_block_submodules(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self._build_layers()
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."

    def _build_layers(self):
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number)

        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        labels: Tensor = None,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
        embedding=None,
        output_layer=None,
        output_weight: Optional[torch.Tensor] = None,
        compute_language_model_loss=None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """
        assert (
            labels is not None
        ), f"labels should not be None for calculating multi token prediction loss."
        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = torch.ones_like(labels)

        hidden_states_main_model = hidden_states
        for layer_number in range(len(self.layers)):
            # Calc logits for the current Multi-Token Prediction (MTP) layers.
            input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
            # embedding
            decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
            # norm, linear projection and transformer
            hidden_states = self.layers[layer_number](
                decoder_input=decoder_input,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **(extra_block_kwargs or {}),
            )
            # output
            mtp_logits, _ = output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
            )
            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            labels, _ = roll_tensor(labels, shifts=-1, dims=-1)
            loss_mask, num_tokens = roll_tensor(loss_mask, shifts=-1, dims=-1)
            mtp_loss = compute_language_model_loss(labels, mtp_logits)
            mtp_loss = loss_mask * mtp_loss
            if self.training:
                MTPLossLoggingHelper.save_loss_to_tracker(
                    torch.sum(mtp_loss) / num_tokens,
                    layer_number,
                    self.config.mtp_num_layers,
                    avg_group=parallel_state.get_tensor_and_context_parallel_group(),
                )
            mtp_loss_scale = self.mtp_loss_scaling_factor / self.config.mtp_num_layers
            if self.config.calculate_per_token_loss:
                hidden_states_main_model = MTPLossAutoScaler.apply(
                    hidden_states_main_model, mtp_loss_scale * mtp_loss
                )
            else:
                hidden_states_main_model = MTPLossAutoScaler.apply(
                    hidden_states_main_model, mtp_loss_scale * mtp_loss / num_tokens
                )

        return hidden_states_main_model

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction module.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction module.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        layer_prefix = f'{prefix}layers.'
        for layer in self.layers:
            offset = get_mtp_layer_offset(self.config)
            sharded_prefix = f'{layer_prefix}{layer.layer_number - 1 }.'

            state_dict_prefix = f'{layer_prefix}{layer.layer_number - 1 - offset}.'
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict

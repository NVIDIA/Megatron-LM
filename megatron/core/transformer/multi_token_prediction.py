# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
    make_viewless_tensor,
)

if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]

try:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False



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


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None):
    """Roll the tensor input along the sequence dimension with Context Parallelism (CP) support.

    This function extends the original roll_tensor to support Context Parallelism, which allows
    MTP to work with CP > 1. When CP is enabled, the sequence dimension is split across CP ranks,
    and tensor rolling requires communication between adjacent CP ranks to properly handle the
    boundary conditions.

    For CP=1 (default behavior): Uses standard torch.roll with zero padding
    For CP>1: Splits tensor into chunks, performs rolling within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.

    Args:
        tensor (Tensor): The input tensor to roll.
        shifts (int): The shift of the tensor (typically -1 for MTP).
        dims (int): The dimension to roll (typically -1 for sequence dimension).
        cp_group (ProcessGroup): The context parallelism process group. If None or size=1,
                               falls back to standard rolling behavior.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor)
    """
    # Standard rolling behavior when CP is not enabled (cp_group is None or size=1)
    if cp_group is None or cp_group.size() == 1:
        rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
        rolled_tensor.select(dims, shifts).fill_(0)
        return rolled_tensor, rolled_tensor.sum()

    # CP-enabled rolling: Split tensor into chunks and handle boundary communication
    # This matches the batch splitting logic in get_batch_on_this_cp_rank() function
    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(torch.roll(tensor_list[i], shifts=shifts, dims=dims))

    # Prepare tensors for communication between CP ranks
    # Each CP rank needs to send boundary elements to adjacent ranks
    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(rolled_tensor_list[i].select(dims, shifts).contiguous())
        empty_tensor = torch.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=torch.cuda.current_device(),
        )
        tensor_recv_list.append(empty_tensor)

    # Get the global rank of next and prev process in the cp group
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    local_rank = torch.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    # Start send and recv ops
    ops = []
    if local_rank != 0:
        req_send_first_part = torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank)
        ops.append(req_send_first_part)
        req_recv_second_part = torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank)
        ops.append(req_recv_second_part)
    else:
        # Inserted elements are set to be 0.0.
        tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank)
        ops.append(req_recv_first_part)
        req_send_second_part = torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank)
        ops.append(req_send_second_part)
    else:
        # For the last CP rank, the removed elements of second part go into the first part
        tensor_recv_list[0] = tensor_send_list[1]

    # Wait for all communication operations to complete
    for op in ops:
        op.wait()

    # Splicing: Replace boundary elements with received elements from adjacent ranks
    # This ensures proper sequence continuity across CP boundaries
    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    # Concatenate the processed chunks back into a single tensor
    rolled_tensor = torch.cat(rolled_tensor_list, dim=dims)

    return rolled_tensor, rolled_tensor.sum()


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_metrics_to_tracker(
        loss: torch.Tensor,
        correct: torch.Tensor,
        total: torch.Tensor,
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
        # Initialize the tracker if not exists.
        if "loss_values" not in tracker:
            tracker["loss_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        if "correct_values" not in tracker:
            tracker["correct_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        if "total_values" not in tracker:
            tracker["total_values"] = torch.zeros(num_layers, device=torch.cuda.current_device())

        tracker["loss_values"][layer_number] += loss.detach()
        tracker["correct_values"][layer_number] += correct.detach()
        tracker["total_values"][layer_number] += total.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_metrics_in_tracker():
        """Clear the mtp losses."""
        tracker = MTPLossLoggingHelper.tracker
        tracker["loss_values"].zero_()
        tracker["correct_values"].zero_()
        tracker["total_values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_metrics_in_tracker():
        """Collect and reduce the mtp metrics across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker or "correct_values" not in tracker or "total_values" not in tracker:
            return
        
        for key in ["loss_values"]:
            values = tracker[key]
            # Reduce mtp losses across ranks.
            if tracker.get('reduce_group') is not None:
                torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
            if tracker.get('avg_group') is not None:
                torch.distributed.all_reduce(
                    values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
                )
        for  key in ["correct_values", "total_values"]:
            values = tracker[key]
            if tracker.get('reduce_group') is not None:
                torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
            if tracker.get('avg_group') is not None:
                torch.distributed.all_reduce(
                    values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.SUM
                )

    @staticmethod
    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_metrics_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "loss_values" not in tracker or "correct_values" not in tracker or "total_values" not in tracker:
            return
        mtp_losses = tracker["loss_values"] * loss_scale
        mtp_corrects = tracker["correct_values"]
        mtp_totals = tracker["total_values"]
        
        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            loss_name = f"mtp_{i+1} loss"
            step_acc_name = f"mtp_{i+1}_acceptance_rate"
            cum_acc_name = f"mtp_{i+1}_cumulative_acceptance_rate"


            loss = mtp_losses[i]

            step_rate = (mtp_corrects[i] / max(mtp_totals[i], 1)) * 100.0
            if total_loss_dict is not None:
                total_loss_dict[loss_name] = total_loss_dict.get(loss_name, 0) + loss
                total_loss_dict[f"{step_acc_name}_correct"] = (
                    total_loss_dict.get(f"{step_acc_name}_correct", 0.0) + mtp_corrects[i]
                )
                total_loss_dict[f"{step_acc_name}_total"] = (
                    total_loss_dict.get(f"{step_acc_name}_total", 0.0) + mtp_totals[i]
                )
                cum_correct = total_loss_dict[f"{step_acc_name}_correct"]
                cum_total = total_loss_dict[f"{step_acc_name}_total"]
                cum_rate = (cum_correct / max(cum_total, 1)) * 100.0
            else:
                cum_rate = 0.0

            if writer is not None:
                writer.add_scalar(loss_name, loss, iteration)
                writer.add_scalar(step_acc_name, step_rate, iteration)
                writer.add_scalar(cum_acc_name, cum_rate, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{loss_name}": loss}, iteration)
                wandb_writer.log({f"{step_acc_name}": step_rate}, iteration)
                wandb_writer.log({f"{cum_acc_name}": cum_rate}, iteration)

        MTPLossLoggingHelper.clean_metrics_in_tracker()


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
        mtp_model_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer, mamba, ... block to be applied.
    """

    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    mtp_model_layer: Union[ModuleSpec, type] = None
    layer_norm: Union[ModuleSpec, type] = None


def get_mtp_layer_spec(
    mtp_model_layer_spec: ModuleSpec, use_transformer_engine: bool
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    return get_mtp_layer_spec_for_backend(
        mtp_model_layer_spec,
        backend=TESpecProvider() if use_transformer_engine else LocalSpecProvider(),
    )


def get_mtp_layer_spec_for_backend(
   mtp_model_layer_spec: ModuleSpec, backend: BackendSpecProvider
) -> ModuleSpec:
    """Get the MTP layer spec.

    Returns:
        ModuleSpec: Module specification with modules from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl: type = backend.layer_norm()
    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            mtp_model_layer=mtp_model_layer_spec,
            layer_norm=layer_norm_impl,
        ),
    )
    return mtp_layer_spec


def get_mtp_layer_offset(config: TransformerConfig) -> int:
    """Get the offset of the MTP layer."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    return 0


def get_mtp_num_layers_to_build(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Get the number of MTP layers to build."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    if mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
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
        mtp_hybrid_override_pattern: str = None,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number
        self.vp_stage = vp_stage
        self.mtp_hybrid_override_pattern = mtp_hybrid_override_pattern

        if config.mtp_num_layers is not None:
            if hasattr(self.submodules.mtp_model_layer.submodules, 'attention_layer'):
                self_attention_spec = self.submodules.mtp_model_layer.submodules.attention_layer
                if self_attention_spec.submodules.self_attention is not None:
                    self_attention_spec = self_attention_spec.submodules.self_attention
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
            tp_comm_buffer_name="mtp_eh_proj"
        )
        if self.config.mtp_num_layers is not None:
            if self.mtp_hybrid_override_pattern is not None:
                model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups()
                # We do not need pre and post process stage for MTP layer, given they are handled in the 
                # MultiTokenPredictionLayer itself.
                self.mtp_model_layer = build_module(
                    self.submodules.mtp_model_layer,
                    self.config,
                    pre_process=False, 
                    post_process=False,
                    hybrid_override_pattern=self.mtp_hybrid_override_pattern,
                    dtype=self.config.params_dtype,
                    model_comm_pgs=model_comm_pgs,
                    vp_stage=self.vp_stage
                )
            else:
                # Uses the transformer block spec for MTP layer. This option is only implemented for the 
                # GPT model. In hybrid model, we model transformer block spec for MTP layers with the hybrid
                # override pattern.
                self.mtp_model_layer = build_module(self.submodules.mtp_model_layer, config=self.config, vp_stage=self.vp_stage)

        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.offload_context = nullcontext()

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
            transformer_layer_rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
            transformer_layer_rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in different layers,
        # currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

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
            # For tensor parallel we need to gather the tensor across the model-parallel
            # ranks after the linear projection. This used to call
            # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
            # the gradient in backward pass and was therefore incorrect in this context.
            # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
            hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
            # For sequence parallel, scatter after linear_fc and before transformer layer.
            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(hidden_states)

            # Use a separate context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is True
            # so that the fp8 weight caching can be triggered correctly.
            with transformer_layer_rng_context, transformer_layer_fp8_context:
                if self.config.mtp_num_layers is not None:
                    # MTP hybrid model.
                    if self.mtp_hybrid_override_pattern is not None:
                        # Since pre-process is set to False, we need to set the input tensor manually.
                        self.mtp_model_layer.set_input_tensor(hidden_states)
                        hidden_states = self.mtp_model_layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            inference_params=inference_params,
                            inference_context=context,
                        )
                    else:
                        # Usual transformer layer spec for MTP layer.
                        hidden_states, _ = self.mtp_model_layer(
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
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        vp_stage: Optional[int] = None,
        mtp_hybrid_override_pattern: str = None,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config=config)
        self.mtp_hybrid_override_pattern = mtp_hybrid_override_pattern
        self.submodules = _get_mtp_block_submodules(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.sequence_parallel = config.sequence_parallel
        self.vp_stage = vp_stage
        self._build_layers()
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."

        # Initialize Context Parallelism (CP) support for MTP
        # This enables MTP to work with CP > 1 by providing the CP process group
        # to the roll_tensor function for proper boundary communication
        if model_comm_pgs is None:
            # Use default MPU process groups if not provided
            model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['cp'])
        else:
            # Ensure the provided process groups include CP
            assert hasattr(
                model_comm_pgs, 'cp'
            ), "MultiTokenPredictionBlock model_comm_pgs must have cp process group"
        self.cp_group = model_comm_pgs.cp

    def _build_layers(self):
        def build_layer(layer_spec, layer_number):
            return build_module(
                layer_spec, config=self.config, layer_number=layer_number, vp_stage=self.vp_stage, mtp_hybrid_override_pattern=self.mtp_hybrid_override_pattern
            )

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
            input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
            position_ids, _ = roll_tensor(position_ids, shifts=-1, dims=-1, cp_group=self.cp_group)
            # embedding
            decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    layer_number=layer_number,
                    hidden_states=hidden_states,
                    decoder_input=decoder_input,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    extra_block_kwargs=extra_block_kwargs,
                )
            else:
                custom_forward = self._custom(layer_number)
                hidden_states = custom_forward(
                    hidden_states=hidden_states,
                    decoder_input=decoder_input,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                    extra_block_kwargs=extra_block_kwargs,
                )
            # output
            mtp_logits, _ = output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
            )
            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            labels, _ = roll_tensor(labels, shifts=-1, dims=-1, cp_group=self.cp_group)
            loss_mask, num_tokens = roll_tensor(
                loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
            )
            mtp_loss = compute_language_model_loss(labels, mtp_logits)
            mtp_loss = loss_mask * mtp_loss
            
            # Acceptance rate: compare predictions with labels
            # mtp_logits shape: [s, b, vocab_size] (tensor parallel: vocab is split)
            # labels shape: [b, s], loss_mask shape: [b, s]
            # Get the tensor parallel group for consistent processing
            tp_group = parallel_state.get_tensor_model_parallel_group()
            
            # For tensor parallel, we need to gather the full vocabulary logits
            # This matches exactly how the loss computation works internally
            if tp_group is not None and tp_group.size() > 1:
                # Gather logits from all TP ranks to get full vocabulary
                mtp_logits_full = gather_from_tensor_model_parallel_region(mtp_logits)  # [s, b, full_vocab_size]
                
                # Get predictions from the full vocabulary
                preds = torch.argmax(mtp_logits_full, dim=-1)  # [s, b]
            else:
                # No tensor parallelism, use predictions directly
                preds = torch.argmax(mtp_logits, dim=-1)  # [s, b]
            
            # Transform labels and mask to match the format expected by loss computation: [b, s] => [s, b]
            labels_match = labels.transpose(0, 1).contiguous()  # [s, b]
            mask_match = loss_mask.transpose(0, 1).contiguous()  # [s, b]
            
            # Handle sequence parallelism if enabled
            if self.sequence_parallel:
                # For sequence parallel, we need to scatter the predictions and labels/masks
                # to match the sequence dimension split
                # TODO: maybe remove it.
                sp_group = parallel_state.get_context_parallel_group()
                if sp_group is not None and sp_group.size() > 1:
                    # Scatter predictions along sequence dimension
                    preds = scatter_to_sequence_parallel_region(preds)  # [s/SP, b]
                    # Scatter labels and mask along sequence dimension  
                    labels_match = scatter_to_sequence_parallel_region(labels_match)  # [s/SP, b]
                    mask_match = scatter_to_sequence_parallel_region(mask_match)  # [s/SP, b]
            
            # Now all tensors have matching shapes for comparison
            correct = ((preds == labels_match) & mask_match.bool()).sum().float()
            total = mask_match.sum().float()
            
            if self.training:
                MTPLossLoggingHelper.save_metrics_to_tracker(
                    torch.sum(mtp_loss) / num_tokens,
                    correct,
                    total,
                    layer_number,
                    self.config.mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
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

    def _checkpointed_forward(self, layer_number, *args, **kwargs):
        def checkpoint_handler(forward_func):
            """Determines whether to use the `te_checkpoint` or `tensor_parallel.checkpoint`"""
            if self.config.fp8:
                from megatron.core.extensions.transformer_engine import te_checkpoint

                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    *args,
                    **kwargs,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func, self.config.distribute_saved_activations, *args, *kwargs.values()
                )

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert (
                self.config.recompute_num_layers == 1
            ), "recompute_num_layers must be 1 for MTP recompute"
            hidden_states = checkpoint_handler(self._custom(layer_number))
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
            )
            hidden_states = self._custom(0, len(self.layers))(*args, **kwargs)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def _custom(self, layer_number):
        def custom_forward(
            hidden_states: Tensor,
            decoder_input: Tensor,
            attention_mask: Tensor,
            rotary_pos_emb: Tensor = None,
            rotary_pos_cos: Tensor = None,
            rotary_pos_sin: Tensor = None,
            inference_params: InferenceParams = None,
            packed_seq_params: PackedSeqParams = None,
            sequence_len_offset: Tensor = None,
            extra_block_kwargs: dict = None,
        ) -> Tensor:
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
            return hidden_states

        return custom_forward

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

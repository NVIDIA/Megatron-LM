# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping, replace_prefix_for_sharding
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import (
    get_pg_rank,
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
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def tie_word_embeddings_state_dict(
    sharded_state_dict: ShardedStateDict,
    word_emb_weight: Tensor,
    word_emb_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the embedding of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        word_emb_weight (Tensor): weight of the word embedding.
        word_emb_weight_key (str): key of the word embedding in the sharded state dict.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel group
        dp_cp_group (torch.distributed.ProcessGroup): The dp-cp comm group

    Returns: None, acts in-place
    """
    mtp_word_emb_replica_id = (
        1,  # copy of embedding in pre processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert word_emb_weight_key in sharded_state_dict
    del sharded_state_dict[word_emb_weight_key]
    sharded_state_dict[word_emb_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=word_emb_weight,
        key=word_emb_weight_key,
        replica_id=mtp_word_emb_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def tie_output_layer_state_dict(
    sharded_state_dict: ShardedStateDict,
    output_layer_weight: Tensor,
    output_layer_weight_key: str,
    tp_group: torch.distributed.ProcessGroup,
    dp_cp_group: torch.distributed.ProcessGroup,
) -> None:
    """tie the output layer of the mtp processing stage in a given sharded state dict.

    Args:
        sharded_state_dict (ShardedStateDict): state dict with the weight to tie.
        output_layer_weight (Tensor): weight of the output layer.
        output_layer_weight_key (str): key of the output layer in the sharded state dict.
        tp_group (torch.distributed.ProcessGroup): The tensor parallel group
        dp_cp_group (torch.distributed.ProcessGroup): The dp-cp comm group

    Returns: None, acts in-place
    """
    mtp_output_layer_replica_id = (
        1,  # copy of output layer in post processing stage
        0,
        get_pg_rank(dp_cp_group),
    )
    assert output_layer_weight_key in sharded_state_dict
    del sharded_state_dict[output_layer_weight_key]
    sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
        tensor=output_layer_weight,
        key=output_layer_weight_key,
        replica_id=mtp_output_layer_replica_id,
        allow_shape_mismatch=True,
        tp_group=tp_group,
        dp_cp_group=dp_cp_group,
    )


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    """Roll the tensor input along the sequence dimension with Context Parallelism (CP) support.

    This function extends the original roll_tensor to support Context Parallelism, which allows
    MTP to work with CP > 1. When CP is enabled, the sequence dimension is split across CP ranks,
    and tensor rolling requires communication between adjacent CP ranks to properly handle the
    boundary conditions.

    For CP=1 (default behavior): Uses standard torch.roll with zero padding
    For CP>1: Splits tensor into chunks, performs rolling within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.

    For packed sequences: Respects sequence boundaries when rolling to avoid mixing tokens
    from different sequences.

    Args:
        tensor (Tensor): The input tensor to roll.
        shifts (int): The shift of the tensor (typically -1 for MTP).
        dims (int): The dimension to roll (typically -1 for sequence dimension).
        cp_group (ProcessGroup): The context parallelism process group. If None or size=1,
                               falls back to standard rolling behavior.
        packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
                                            If provided, respects sequence boundaries.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor)
    """
    # Handle packed sequences cases
    if packed_seq_params is not None:
        return _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group)

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


def _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group=None):
    """Roll tensor with packed sequence support.
    This function handles rolling for packed sequences by respecting sequence boundaries
    """

    # Notice: This is a naive implementation to test the correctness,
    # a better solution will only sync the boundary tokens once.
    assert (
        dims == -1 or dims == tensor.dim() - 1
    ), "Packed sequence roll only supports the last dimension."
    assert shifts == -1, "Packed sequence roll only supports a single-token left shift."
    cu_seqlens = packed_seq_params.cu_seqlens_q
    assert cu_seqlens is not None, "Packed sequence parameters must provide cu_seqlens_q."

    rolled_tensor = tensor.clone()

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        # CP disabled: roll each packed sequence independently within its boundaries
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_slice = tensor[..., start_idx:end_idx]
            rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dims)
            # Zero out the last position(s) that would cross sequence boundaries
            rolled_seq[..., shifts:] = 0
            rolled_tensor[..., start_idx:end_idx] = rolled_seq
        return rolled_tensor, rolled_tensor.sum()

    # CP enabled: each rank owns two chunks per sequence (front and mirrored tail).
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    # Iterate over each sequence individually
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]

        # the idx has been multiplied by cp_size, need to divide it by cp_size to get the local idx
        local_start_idx = start_idx // cp_size
        local_end_idx = end_idx // cp_size

        # Skip empty sequences - this can happen when a sequence is very short and
        # after dividing by cp_size, the local slice has zero length
        local_seq_len = local_end_idx - local_start_idx
        if local_seq_len == 0:
            continue

        tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()

        # The following code is very similar as the code in roll_tensor function
        local_chunks = tensor_slice.chunk(2, dim=dims)
        rolled_chunks = [torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks]

        tensor_send_list = []
        tensor_recv_list = []
        for chunk in rolled_chunks:
            # Skip empty chunks that can occur when the sequence slice is very small
            if chunk.size(dims) == 0:
                tensor_send_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                tensor_recv_list.append(
                    torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device)
                )
                continue
            boundary = chunk.select(dims, shifts).contiguous().clone()
            tensor_send_list.append(boundary)
            tensor_recv_list.append(torch.empty_like(boundary))

        ops = []
        if local_rank != 0:
            ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank))
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank))
        else:
            tensor_recv_list[1].zero_()

        if local_rank != cp_size - 1:
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank))
            ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank))
        else:
            tensor_recv_list[0].copy_(tensor_send_list[1])

        for op in ops:
            op.wait()

        index = [slice(None)] * rolled_chunks[0].dim()
        index[dims] = shifts
        for chunk, recv in zip(rolled_chunks, tensor_recv_list):
            # Skip empty chunks
            if chunk.size(dims) == 0:
                continue
            chunk[tuple(index)] = recv

        seq_result = torch.cat(rolled_chunks, dim=dims)

        # update the rolled tensor
        rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

    return rolled_tensor, rolled_tensor.sum()


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
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
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
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
            name = f"mtp_{i + 1} loss"
            loss = mtp_losses[i]
            if total_loss_dict is not None:
                if name in total_loss_dict:
                    total_loss_dict[name] += loss
                else:
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
        hnorm: Specification or instance of the hidden states normalization to be applied.
        enorm: Specification or instance of the embedding normalization to be applied.
        eh_proj (Union[ModuleSpec, type]): Specification or instance of the
            linear projection to be applied.
        mtp_model_layer (Union[ModuleSpec, type]): Specification
            or instance of the transformer or mamba block to be applied.
    """

    enorm: LayerNormBuilder
    hnorm: LayerNormBuilder
    # TODO(nschank): Move this back below transformer_layer once eh_proj and transformer_layer have
    # their defaults removed.
    layer_norm: LayerNormBuilder

    eh_proj: Union[ModuleSpec, type] = None
    mtp_model_layer: Union[ModuleSpec, type] = None


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
    layer_norm_impl = backend.layer_norm()
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


def mtp_on_this_rank(
    config: TransformerConfig, ignore_virtual: Optional[bool] = True, vp_stage: Optional[int] = None
) -> bool:
    """
    Check if there is MTP on the current rank.

    Behavior:
        - If a custom pipeline model parallel layout is provided in the config:
            - If virtual pipeline parallelism is enabled (and `ignore_virtual` is False), checks
              whether any MTP layers are present on this (pp_rank, vp_stage) pair.
            - Otherwise, checks all virtual pipeline ranks of the current pipeline rank. Returns
              True if any virtual sub-rank includes at least one MTP layer.
        - If no custom layout is provided, assumes all MTP layers (if any) are placed on the last
          pipeline stage. The function returns True only on the last pipeline stage.
    """
    mtp_on_this_rank = False
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if config.pipeline_model_parallel_layout is not None:
        # with custom PP layout, we support put MTP layers on any pipeline stage
        layout = config.pipeline_model_parallel_layout.layout
        if (
            not ignore_virtual
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
            num_layers_to_build = layout[pp_rank][vp_stage].count(LayerType.mtp)
            mtp_on_this_rank = num_layers_to_build > 0
        else:
            for vpp_rank in range(len(layout[pp_rank])):
                num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
                if num_layers_to_build > 0:
                    mtp_on_this_rank = True
                    break
    else:
        # without custom PP layout, we only support put all of MTP layers on the last pipeline stage
        if config.mtp_num_layers is not None:
            mtp_on_this_rank = parallel_state.is_pipeline_last_stage(
                ignore_virtual=ignore_virtual, vp_stage=vp_stage
            )
        else:
            mtp_on_this_rank = False
    return mtp_on_this_rank


def get_mtp_ranks(pp_ranks: List[int], config: TransformerConfig) -> List[int]:
    """Get the ranks of the MTP layers."""
    mtp_ranks = set()
    if config.mtp_num_layers is None:
        return []
    if config.pipeline_model_parallel_layout is None:
        return [pp_ranks[-1]]
    layout = config.pipeline_model_parallel_layout.layout
    for pp_rank in range(len(layout)):
        for vpp_rank in range(len(layout[pp_rank])):
            num_layers_to_build = layout[pp_rank][vpp_rank].count(LayerType.mtp)
            if num_layers_to_build:
                mtp_ranks.add(pp_ranks[pp_rank])
    return list(mtp_ranks)


def get_mtp_layer_offset(config: TransformerConfig, vp_stage: Optional[int] = None) -> int:
    """Get the offset of the MTP layer."""
    if config.pipeline_model_parallel_size > 1:
        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.mtp, vp_stage=vp_stage
            )
        else:
            offset = 0
    else:
        offset = 0
    return offset


def get_mtp_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """Get the number of MTP layers to build."""
    if config.pipeline_model_parallel_layout is not None:
        # If we have a custom PP layout, get the number of mtp layers in the layout array.
        num_layers_to_build = config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.mtp, vp_stage=vp_stage
        )
        assert num_layers_to_build == config.mtp_num_layers or num_layers_to_build == 0, (
            f"Currently, we only support put all of MTP layers on the last pipeline stage, "
            f"so the number of MTP layers to build ({num_layers_to_build}) must match "
            f"mtp_num_layers ({config.mtp_num_layers}) or be 0."
        )
    else:
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage):
            num_layers_to_build = config.mtp_num_layers if config.mtp_num_layers else 0
        else:
            num_layers_to_build = 0
    return num_layers_to_build


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


def process_mtp_loss(
    hidden_states: Tensor,
    labels: Tensor,
    loss_mask: Optional[Tensor],
    output_layer: Callable,
    output_weight: Optional[Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable,
    config: TransformerConfig,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> Tensor:
    """Process Multi-Token Prediction (MTP) loss computation.

    This is a standalone function that handles MTP loss computation. It's used on the
    post_process rank to split concatenated hidden states and compute MTP losses.

    Args:
        hidden_states (Tensor): Hidden states tensor (concatenated with MTP outputs).
        labels (Tensor): Ground truth labels.
        loss_mask (Optional[Tensor]): Mask for loss computation. If None, uses all ones.
        output_layer (Callable): Output layer method to compute logits.
        output_weight (Optional[Tensor]): Optional output weight for shared embeddings.
        runtime_gather_output (Optional[bool]): Whether to gather output at runtime.
        is_training (bool): Whether the model is in training mode.
        compute_language_model_loss (Callable): Method to compute language model loss.
        config (TransformerConfig): Model configuration containing mtp_num_layers etc.
        cp_group (Optional[ProcessGroup]): Context parallelism process group.
        packed_seq_params (Optional[PackedSeqParams]): Packed sequence parameters.

    Returns:
        Tensor: Updated hidden states after MTP loss processing (first chunk only).
    """
    hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
    hidden_states = hidden_states_list[0]

    if labels is None:
        return hidden_states

    mtp_labels = labels.clone()

    if loss_mask is None:
        loss_mask = torch.ones_like(mtp_labels)

    for mtp_layer_number in range(config.mtp_num_layers):
        mtp_logits, _ = output_layer(
            hidden_states_list[mtp_layer_number + 1],
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        mtp_labels, _ = roll_tensor(
            mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        loss_mask, num_tokens = roll_tensor(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
        mtp_loss = loss_mask * mtp_loss
        if is_training:
            MTPLossLoggingHelper.save_loss_to_tracker(
                torch.sum(mtp_loss) / num_tokens,
                mtp_layer_number,
                config.mtp_num_layers,
                avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if config.calculate_per_token_loss:
            hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
        else:
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * mtp_loss / num_tokens
            )

    return hidden_states


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

    For more information, refer to DeepSeek-V3 Technical Report
    https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MultiTokenPredictionLayerSubmodules,
        layer_number: int = 1,
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        # For Mamba path - pattern and submodules to build inner layers directly
        mtp_layer_pattern: Optional[str] = None,
        mamba_submodules: Optional["MambaStackSubmodules"] = None,
    ):
        super().__init__(config=config)
        self.sequence_parallel = config.sequence_parallel
        self.submodules = submodules
        self.layer_number = layer_number + get_mtp_layer_offset(self.config, vp_stage)
        self.vp_stage = vp_stage
        self.cp_group = pg_collection.cp
        self.mtp_layer_pattern = mtp_layer_pattern

        # Validate attention mask type if using transformer-based inner layers
        if self.submodules.mtp_model_layer is not None and hasattr(
            self.submodules.mtp_model_layer, 'submodules'
        ):
            if hasattr(self.submodules.mtp_model_layer.submodules, 'attention_layer'):
                self_attention_spec = self.submodules.mtp_model_layer.submodules.attention_layer
                if self_attention_spec.submodules.self_attention is not None:
                    self_attention_spec = self_attention_spec.submodules.self_attention
                    attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
                    assert attn_mask_type in SUPPORTED_ATTN_MASK, (
                        f"Multi-Token Prediction (MTP) is not yet supported with "
                        f"{attn_mask_type} attention mask type. "
                        f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
                    )
            elif hasattr(self.submodules.mtp_model_layer.submodules, 'self_attention'):
                self_attention_spec = self.submodules.mtp_model_layer.submodules.self_attention
                if self_attention_spec is not None:
                    attn_mask_type = self_attention_spec.params.get('attn_mask_type', '')
                    assert attn_mask_type in SUPPORTED_ATTN_MASK, (
                        f"Multi-Token Prediction (MTP) is not yet supported with "
                        f"{attn_mask_type} attention mask type. "
                        f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
                    )

        self.enorm = self.submodules.enorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = self.submodules.hnorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
        # of the i-th token's hidden states and the (i + K)-th token's decoder input,
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
            tp_comm_buffer_name="mtp_eh_proj",
        )

        # Build inner layers: two possible paths
        # 1. Mamba path: use MambaStack for hybrid pattern support
        # 2. GPT path: single TransformerLayer
        if mtp_layer_pattern is not None and mamba_submodules is not None:
            from megatron.core.ssm.mamba_block import MambaStack

            self.mtp_model_layer = MambaStack(
                config=self.config,
                submodules=mamba_submodules,
                hybrid_override_pattern=mtp_layer_pattern,
                pre_process=True,  # Always receives input from eh_proj
                post_layer_norm=False,  # MTP has its own final_layernorm
                post_process=True,  # MTP layer is self-contained
                pg_collection=pg_collection,
                is_mtp_layer=True,
            )
        elif self.config.mtp_num_layers is not None:
            # GPT path: Uses the transformer block spec for MTP layer
            # MTP inner layers use their own layer numbering (self.layer_number = 1, 2, etc.)
            # rather than continuing from decoder layer numbers. This is consistent with the
            # Mamba path and ensures proper aux loss tracking in router.py.
            self.mtp_model_layer = build_module(
                self.submodules.mtp_model_layer,
                config=self.config,
                vp_stage=self.vp_stage,
                layer_number=self.layer_number,
                is_mtp_layer=True,
            )

        self.final_layernorm = self.submodules.layer_norm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.offload_context = nullcontext()

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        embedding: Callable,
        hidden_states: torch.Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            position_ids (torch.Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding module
                from gpt model to compute the decoder input.
            hidden_states (torch.Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            packed_seq_params (PackedSeqParams): Parameters for packed sequence processing.
        """
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids, _ = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        position_ids, _ = roll_tensor(
            position_ids,
            shifts=-1,
            dims=-1,
            cp_group=self.cp_group,
            packed_seq_params=packed_seq_params,
        )
        # embedding
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return input_ids, position_ids, decoder_input, hidden_states

    def _concat_embeddings(self, hidden_states: torch.Tensor, decoder_input: torch.Tensor):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        decoder_input = apply_module(self.enorm)(decoder_input)
        decoder_input = make_viewless_tensor(inp=decoder_input, requires_grad=True, keep_graph=True)
        hidden_states = apply_module(self.hnorm)(hidden_states)
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
        # At the (k - 1)-th MTP module, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
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
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Unlike transformer_block.py which needs to support mixed-precision in
        # different layers,currently MTP only use global fp8 context.
        if self.config.fp8:
            fp8_context = get_fp8_context(self.config)
            transformer_layer_fp8_context = get_fp8_context(self.config)
        else:
            fp8_context = nullcontext()
            transformer_layer_fp8_context = nullcontext()

        # TODO: currently ignoring FP4 in MTP layers because we need more numerical validation
        with rng_context:
            with fp8_context:
                hidden_states = self._concat_embeddings(hidden_states, decoder_input)

            # Use a separate fp8 context for the transformer layer. This is to ensure that when the
            # transformer layer is cudagraphed, the FP8GlobalStateManager.is_first_fp8_module() is
            # True so that the fp8 weight caching can be triggered correctly.
            with transformer_layer_fp8_context:
                if self.mtp_layer_pattern is not None:
                    hidden_states = self.mtp_model_layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_context=inference_params,
                        packed_seq_params=packed_seq_params,
                    )
                else:
                    # GPT path: single TransformerLayer
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

        hidden_states = self._postprocess(hidden_states)

        return hidden_states

    def _postprocess(self, hidden_states: torch.Tensor):
        """
        Postprocesses the output of the transformer layers.
        """

        # Layer norm before shared head layer.
        hidden_states = apply_module(self.final_layernorm)(hidden_states)
        # TENorm produces a "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        return hidden_states

    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        def checkpoint_handler():
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
            outputs = checkpoint_handler()
        elif self.config.recompute_method == 'block':
            # TODO: implement block-based recompute for MTP
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet." " Skipping recompute."
            )
            outputs = forward_func(*args, **kwargs)
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        embedding=None,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            sequence_len_offset (Tensor, optional): Offset for sequence length, if applicable.
            embedding (Callable): The embedding module from gpt model to compute the decoder input.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, "multi token prediction + cross attention is not yet supported."
        input_ids, position_ids, decoder_input, hidden_states = self._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
        )

        if self.config.recompute_granularity == 'full' and self.training:
            hidden_states = self._checkpointed_forward(
                self._proj_and_transformer_layer,
                hidden_states=hidden_states,
                decoder_input=decoder_input,
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
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
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

        return hidden_states, input_ids, position_ids

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

        # Backward compatibility: GPT MTP checkpoints were saved with the submodule
        # named 'transformer_layer'. Remap checkpoint keys so old checkpoints load
        # correctly. Mamba MTP models keep 'mtp_model_layer' as their native format
        # since no older checkpoints exist for them.
        if self.mtp_layer_pattern is None:
            apply_prefix_mapping(
                sharded_state_dict, {f'{prefix}mtp_model_layer.': f'{prefix}transformer_layer.'}
            )

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

    layer_specs: Optional[List[ModuleSpec]] = None


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

    When `mtp_use_repeated_layer=True` in config, instead of creating N separate MTP layers,
    only 1 layer is created and applied mtp_num_layers times.

    For more information, please refer to DeepSeek-V3 Technical Report
    https://arxiv.org/pdf/2412.19437.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        vp_stage: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        # New: For Mamba path with unified pattern syntax
        mtp_layer_pattern: Optional[str] = None,
        mtp_num_depths: int = 0,
        mamba_submodules: Optional["MambaStackSubmodules"] = None,
    ):
        super().__init__(config=config)
        self.submodules = _get_mtp_block_submodules(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.vp_stage = vp_stage
        self.mtp_layer_pattern = mtp_layer_pattern
        self.mtp_num_depths = mtp_num_depths
        self.mamba_submodules = mamba_submodules
        self.mtp_use_repeated_layer = self.config.mtp_use_repeated_layer

        vp_size = config.virtual_pipeline_model_parallel_size
        assert is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size), (
            f"MTP layers must be placed on the last virtual pipeline stage. "
            f"Got vp_stage={vp_stage} with vp_size={vp_size}. "
            f"Placing MTP layers on different VPP stages is not currently supported."
        )

        # Initialize Context Parallelism (CP) support for MTP
        # This enables MTP to work with CP > 1 by providing the CP process group
        # to the roll_tensor function for proper boundary communication
        if pg_collection is None:
            # Use default MPU process groups if not provided
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp'])
        else:
            # Ensure the provided process groups include CP
            assert hasattr(
                pg_collection, 'cp'
            ), "MultiTokenPredictionBlock pg_collection must have cp process group"

        self._build_layers(pg_collection)
        assert len(self.layers) > 0, "MultiTokenPredictionBlock must have at least one layer."
        self.cp_group = pg_collection.cp

    def _build_layers(self, pg_collection):
        # Determine number of depths to build
        if self.mtp_num_depths > 0:
            num_depths = self.mtp_num_depths
        else:
            num_depths = self.config.mtp_num_layers or len(self.submodules.layer_specs)

        def build_layer_legacy(layer_spec, layer_number):
            """Build layer using legacy spec-based approach."""
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                )
            return module

        def build_layer_with_pattern(layer_spec, layer_number, mtp_layer_pattern, mamba_submodules):
            """Build layer using pattern-based approach (new Mamba path)."""
            fp8_init_context = get_fp8_context(self.config, is_init=True)
            with fp8_init_context:
                module = build_module(
                    layer_spec,
                    config=self.config,
                    layer_number=layer_number,
                    vp_stage=self.vp_stage,
                    pg_collection=pg_collection,
                    mtp_layer_pattern=mtp_layer_pattern,
                    mamba_submodules=mamba_submodules,
                )
            return module

        # New Mamba path: use mtp_layer_pattern and mamba_submodules
        if self.mtp_layer_pattern is not None and self.mamba_submodules is not None:
            if self.mtp_use_repeated_layer:
                # Shared/repeated layer: build one layer, use it for all depths
                layer_spec = self.submodules.layer_specs[0]
                shared_layer = build_layer_with_pattern(
                    layer_spec,
                    layer_number=1,
                    mtp_layer_pattern=self.mtp_layer_pattern,
                    mamba_submodules=self.mamba_submodules,
                )
                self.layers = torch.nn.ModuleList([shared_layer])
            else:
                # Non-shared: each depth gets its own layers
                self.layers = torch.nn.ModuleList(
                    [
                        build_layer_with_pattern(
                            self.submodules.layer_specs[
                                min(i, len(self.submodules.layer_specs) - 1)
                            ],
                            layer_number=i + 1,
                            mtp_layer_pattern=self.mtp_layer_pattern,
                            mamba_submodules=self.mamba_submodules,
                        )
                        for i in range(num_depths)
                    ]
                )
        elif self.mtp_use_repeated_layer:
            # Legacy repeated layer mode
            if len(self.submodules.layer_specs) != 1:
                warnings.warn(
                    "Repeated MTP mode expects exactly 1 layer spec, got "
                    f"{len(self.submodules.layer_specs)} instead. "
                    f"The first layer will be applied {self.config.mtp_num_layers} times."
                )
            self.layers = torch.nn.ModuleList(
                [build_layer_legacy(self.submodules.layer_specs[0], layer_number=1)]
            )
        else:
            # Legacy mode: build from layer_specs
            self.layers = torch.nn.ModuleList(
                [
                    build_layer_legacy(layer_spec, i + 1)
                    for i, layer_spec in enumerate(self.submodules.layer_specs)
                ]
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        extra_block_kwargs: Optional[dict] = None,
        embedding=None,
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
        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config, self.vp_stage)
        hidden_states_list = list(torch.chunk(hidden_states, 1 + offset, dim=0))
        hidden_states = hidden_states_list[offset]
        for iteration in range(self.config.mtp_num_layers):
            layer_idx = 0 if self.mtp_use_repeated_layer else iteration
            (hidden_states, input_ids, position_ids) = self.layers[layer_idx](
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=embedding,
                **(extra_block_kwargs or {}),
            )

            # append the output hidden states of the current mtp layer
            # to the hidden_states_list
            hidden_states_list.append(hidden_states)

        # concat the hidden states of all mtp layers
        hidden_states = torch.cat(hidden_states_list, dim=0)
        return hidden_states

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
            offset = get_mtp_layer_offset(self.config, self.vp_stage)
            sharded_prefix = f'{layer_prefix}{layer.layer_number - 1}.'

            state_dict_prefix = f'{layer_prefix}{layer.layer_number - 1 - offset}.'
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict

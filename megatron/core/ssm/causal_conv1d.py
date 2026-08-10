# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Causal convolution over contiguous context-parallel sequence shards."""

import torch

from megatron.core.tensor_parallel.mappings import all_to_all

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _exchange_initial_states(
    x: torch.Tensor, state_len: int, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor | None:
    """Exchange the preceding rank's tail as the local convolution state.

    All ranks participate in a differentiable ring exchange. Rank 0 zeros the
    wrapped tail to preserve the global causal boundary.
    """
    if state_len < 0:
        raise ValueError(f"state_len must be non-negative, got {state_len}")
    if state_len == 0 or cp_group.size() == 1:
        return None
    if x.shape[1] < state_len:
        raise ValueError(
            "Each local sequence shard must contain at least "
            f"{state_len} tokens for causal convolution, got {x.shape[1]}"
        )

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    batch_size, _, channels = x.shape
    split_size = batch_size * state_len

    # Pack only the boundary tokens; x remains a strided sequence shard.
    tail = x[:, -state_len:, :].reshape(split_size, channels)
    input_splits = [0] * cp_size
    output_splits = [0] * cp_size
    input_splits[(cp_rank + 1) % cp_size] = split_size
    output_splits[(cp_rank - 1) % cp_size] = split_size
    previous_tail = all_to_all(
        cp_group, tail, output_split_sizes_=output_splits, input_split_sizes=input_splits
    ).view(batch_size, state_len, channels)

    if cp_rank == 0:
        # Preserve the autograd path while enforcing the global left boundary.
        previous_tail = previous_tail.clone()
        previous_tail.zero_()
    return previous_tail.transpose(1, 2)


def causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Apply causal Conv1d to a contiguous context-parallel shard.

    Args:
        x: Input tensor of shape ``[B, T, D]``.
        weight: Depthwise weights of shape ``[D, W]``.
        bias: Optional channel-wise bias.
        activation: Optional activation passed to ``causal_conv1d_fn``.
        cp_group: Context-parallel process group ordered by sequence shard.

    Returns:
        Output tensor of shape ``[B, T, D]``.

    Raises:
        ImportError: If the optional ``causal-conv1d`` dependency is unavailable.
    """
    if causal_conv1d_fn is None:
        raise ImportError("causal_conv1d_cp requires the optional causal-conv1d dependency")

    initial_states = _exchange_initial_states(
        x=x, state_len=weight.shape[-1] - 1, cp_group=cp_group
    )
    output = causal_conv1d_fn(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        initial_states=initial_states,
        activation=activation,
    )
    return output.transpose(1, 2)

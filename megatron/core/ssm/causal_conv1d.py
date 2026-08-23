# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Causal convolution over contiguous context-parallel sequence shards, and the determinism
guard the SSM mixers apply to the causal_conv1d backward."""

import os

import torch

from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.utils import is_causal_conv1d_min_version

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _use_causal_conv1d_deterministic_mode():
    """Whether causal_conv1d's backward will take its deterministic reduction.

    Mirrors the kernel's own ``use_deterministic_mode()`` (``csrc/causal_conv1d.cpp``): only a
    leading ``'1'`` or ``'0'`` decides, anything else falls through to torch.
    """
    env = os.environ.get('CAUSAL_CONV1D_DETERMINISTIC')
    if env:
        if env[0] == '1':
            return True
        if env[0] == '0':
            return False
    return torch.are_deterministic_algorithms_enabled()


def assert_causal_conv1d_deterministic(deterministic_mode):
    """Refuse a deterministic run whose convolution cannot be bit-reproducible.

    The conv backward combines each weight-gradient element's per-block partials with
    ``atomicAdd``, which fixes no order; 1.6.0+ uses a per-block workspace and an ordered
    reduce instead. Worst on the channel-last layout GDP and Mamba's fused path both feed the
    conv, where an element takes ``batch * ceil(seqlen / 128)`` contributions rather than the
    channels-first ``batch``.

    Call once at construction. Keyed on ``deterministic_mode``, not torch's global flag, which
    unrelated tests set and never restore.
    """
    if not deterministic_mode or causal_conv1d_fn is None:
        return

    assert _use_causal_conv1d_deterministic_mode(), (
        "deterministic_mode requires a deterministic causal_conv1d backward. Enable it with "
        "torch.use_deterministic_algorithms(True) (which --deterministic-mode does) or "
        "CAUSAL_CONV1D_DETERMINISTIC=1."
    )
    # https://github.com/Dao-AILab/causal-conv1d/pull/88 added the deterministic reduction.
    conv1d_min = "1.6.0"
    assert is_causal_conv1d_min_version(conv1d_min), (
        f"causal_conv1d >= {conv1d_min} is required for deterministic_mode: older builds "
        "reduce the conv weight and bias gradients with atomicAdd, which is not "
        "bit-reproducible, and ignore CAUSAL_CONV1D_DETERMINISTIC."
    )


def _exchange_initial_states(
    x: torch.Tensor,
    state_len: int,
    cp_group: torch.distributed.ProcessGroup,
    initial_state_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Exchange the preceding rank's tail as the local convolution state.

    All ranks participate in a differentiable ring exchange. Rank 0 zeros the
    wrapped tail to preserve the global causal boundary. The optional mask
    removes tail tokens outside the first local packed sequence.
    """
    if state_len == 0 or cp_group.size() == 1:
        return None

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

    if initial_state_mask is not None:
        previous_tail = previous_tail.masked_fill(~initial_state_mask.unsqueeze(-1), 0)

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
    global_seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply causal Conv1d to a contiguous context-parallel shard.

    Args:
        x: Input tensor of shape ``[B, T, D]``. THD callers flatten packed tokens
            along ``T`` and use ``B=1``.
        weight: Depthwise weights of shape ``[D, W]``.
        bias: Optional channel-wise bias.
        activation: Optional activation passed to ``causal_conv1d_fn``.
        cp_group: Context-parallel process group ordered by sequence shard.
        global_seq_idx: Global per-token sequence IDs for packed THD input, replicated
            across CP ranks. IDs must be non-negative. Pass ``None`` for non-packed
            input. The convolution state resets at each sequence boundary.

    Returns:
        Output tensor of shape ``[B, T, D]``.

    Raises:
        ImportError: If the optional ``causal-conv1d`` dependency is unavailable.
        ValueError: If ``global_seq_idx`` has an invalid shape, dtype, or device.
    """
    if causal_conv1d_fn is None:
        raise ImportError("causal_conv1d_cp requires the optional causal-conv1d dependency")
    state_len = weight.shape[-1] - 1
    if state_len < 0:
        raise ValueError(f"state_len must be non-negative, got {state_len}")
    if state_len > 0 and cp_group.size() > 1 and x.shape[1] < state_len:
        raise ValueError(
            "Each local sequence shard must contain at least "
            f"{state_len} tokens for causal convolution, got {x.shape[1]}"
        )

    local_seq_idx = None
    initial_state_mask = None
    if global_seq_idx is not None:
        if x.shape[0] != 1:
            raise ValueError(f"THD input must have batch size 1, got {x.shape[0]}")
        expected_shape = (x.shape[0], cp_group.size() * x.shape[1])
        if global_seq_idx.shape != expected_shape:
            actual_shape = global_seq_idx.shape
            raise ValueError(f"global_seq_idx shape must be {expected_shape}, got {actual_shape}")
        if global_seq_idx.dtype != torch.int32:
            raise ValueError(
                f"global_seq_idx must have dtype torch.int32, got {global_seq_idx.dtype}"
            )
        if global_seq_idx.device != x.device:
            raise ValueError(
                f"global_seq_idx must be on device {x.device}, got {global_seq_idx.device}"
            )
        cp_rank = cp_group.rank()
        shard_start = cp_rank * x.shape[1]
        local_seq_idx = global_seq_idx[:, shard_start : shard_start + x.shape[1]]
        if cp_rank > 0 and state_len > 0:
            previous_seq_idx = global_seq_idx[:, shard_start - state_len : shard_start]
            initial_state_mask = previous_seq_idx == local_seq_idx[:, :1]

    initial_states = _exchange_initial_states(
        x=x, state_len=state_len, cp_group=cp_group, initial_state_mask=initial_state_mask
    )
    output = causal_conv1d_fn(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        seq_idx=local_seq_idx,
        initial_states=initial_states,
        activation=activation,
    )
    return output.transpose(1, 2)

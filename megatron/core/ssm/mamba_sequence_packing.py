# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental packed Mamba with a sequence-relative scan chunk grid."""

from functools import lru_cache
from itertools import accumulate

import torch
from causal_conv1d import causal_conv1d_fn
from einops import rearrange

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_mixer import (
    MAMBA_HAS_STATE_DTYPE,
    MambaMixer,
    mamba_chunk_scan_combined,
)


@lru_cache(maxsize=32)
def aligned_indices(
    lengths: tuple[int, ...], chunk_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Cache bounded, non-differentiable layout metadata across model layers."""
    padded = tuple((n + chunk_size - 1) // chunk_size * chunk_size for n in lengths)
    starts = tuple(accumulate(padded, initial=0))
    real = torch.cat(
        [
            torch.arange(start, start + n, device=device)
            for start, n in zip(starts[:-1], lengths, strict=True)
        ]
    )
    sequence_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32),
        torch.tensor(padded, device=device),
        output_size=starts[-1],
    )[None]
    return real, sequence_ids, starts[-1]


def mamba_sequence_relative_scan(
    mixer: MambaMixer,
    projected: torch.Tensor,
    packed_seq_params: PackedSeqParams | None = None,
    *,
    local_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad only the internal recurrence layout, then restore original CP positions.

    Every sequence begins on a scan chunk boundary. The convolution still uses
    sequence IDs to reset its causal context. Tail padding has no path back to
    real outputs; attention and the surrounding model retain the original layout.
    This is a new numerical reference, not an emulation of stock packed Mamba.
    """
    if not mixer.rmsnorm or mixer.norm_before_gate:
        raise ValueError("sequence-relative Mamba requires RMSNorm after gating")
    if projected.ndim != 3 or projected.shape[1] != 1:
        raise ValueError("sequence-relative Mamba requires packed [sequence, 1, projection] input")
    if packed_seq_params is None:
        lengths = (projected.shape[0],)
    else:
        assert packed_seq_params.qkv_format == "thd"
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        lengths = tuple((cu[1:] - cu[:-1]).tolist())
    assert sum(lengths) == projected.shape[0] and min(lengths) > 0
    cp = mixer.cp
    dim = cp.d_inner_local_tpcp
    if local_gate is None:
        gate = projected[..., :dim]
        recurrent = projected[..., dim:]
    else:
        gate = local_gate
        recurrent = projected
    y = scan_mamba_packed_recurrence(mixer, recurrent, lengths)
    y = cp.post_conv_ssm(y, packed_seq_params)
    if local_gate is None:
        gate = cp.post_conv_ssm(gate.contiguous(), packed_seq_params)
    if gate.shape != y.shape:
        raise ValueError("Mamba gate must match the restored local output layout")
    return mixer.norm(y, gate)


def scan_mamba_packed_recurrence(
    mixer: MambaMixer, recurrent: torch.Tensor, lengths: tuple[int, ...]
) -> torch.Tensor:
    """Scan canonical x/B/C/dt sequences with one aligned convolution and scan.

    Returns canonical y without a gate, normalization or CP transform. Each
    complete sequence starts at recurrence state zero and a chunk boundary.
    """
    if recurrent.ndim != 3 or recurrent.shape[1] != 1:
        raise ValueError("Packed recurrence requires [sequence, 1, channels]")
    if not lengths or min(lengths) < 1 or sum(lengths) != recurrent.shape[0]:
        raise ValueError("Packed recurrence lengths must cover its input")
    cp = mixer.cp
    dim, groups = cp.d_inner_local_tpcp, cp.ngroups_local_tpcp
    real, sequence_ids, total = aligned_indices(lengths, mixer.chunk_size, recurrent.device)
    # Fresh storage has no aliases; an in-place copy avoids a second full-sized
    # padding allocation while retaining the differentiable source gather.
    padded = recurrent.new_zeros((total, 1, recurrent.shape[-1])).index_copy_(0, real, recurrent)
    fields = rearrange(padded, "l b d -> b l d").contiguous()
    xbc, dt = fields.split([dim + 2 * groups * mixer.d_state, cp.nheads_local_tpcp], -1)
    xbc = causal_conv1d_fn(
        rearrange(xbc.contiguous(), "b l d -> b d l"),
        rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
        cp.get_conv1d_bias(),
        seq_idx=sequence_ids,
        activation=mixer.activation,
    )
    x, b, c = (
        rearrange(xbc, "b d l -> b l d")
        .contiguous()
        .split([dim, groups * mixer.d_state, groups * mixer.d_state], -1)
    )
    y = mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim).contiguous(),
        dt.contiguous(),
        -torch.exp(cp.get_A_log().float()),
        rearrange(b, "b l (g n) -> b l g n", n=mixer.d_state).contiguous(),
        rearrange(c, "b l (g n) -> b l g n", n=mixer.d_state).contiguous(),
        mixer.chunk_size,
        D=(
            rearrange(cp.get_D().float(), "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else cp.get_D()
        ),
        z=None,
        seq_idx=sequence_ids,
        dt_bias=cp.get_dt_bias().float(),
        dt_softplus=True,
        **({"state_dtype": mixer.mamba_training_ssm_states_dtype} if MAMBA_HAS_STATE_DTYPE else {}),
    )
    y = rearrange(y, "b l h p -> l b (h p)").contiguous().index_select(0, real)
    return y

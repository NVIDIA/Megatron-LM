# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental packed Mamba recurrence with each sequence aligned to a scan chunk."""

from functools import lru_cache
from itertools import accumulate

import torch
from causal_conv1d import causal_conv1d_fn
from einops import rearrange

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

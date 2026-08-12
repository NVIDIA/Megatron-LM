# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Batch-invariant Mamba decode using buffered chunk replay."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from megatron.core.ssm.ops.ssd_bmm import _bmm_chunk_fwd
from megatron.core.ssm.ops.ssd_chunk_scan import _chunk_scan_fwd
from megatron.core.ssm.ops.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from megatron.core.ssm.ops.ssd_state_passing import _state_passing_fwd


@triton.jit
def _masked_update_rows_kernel(
    states_ptr,
    indices_ptr,
    values_ptr,
    state_row_stride,
    value_row_stride,
    ROW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy contiguous rows, skipping entries whose destination index is -1."""
    src_row = tl.program_id(0)
    dst_row = tl.load(indices_ptr + src_row)
    if dst_row < 0:
        return

    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < ROW_SIZE
    values = tl.load(values_ptr + src_row * value_row_stride + offsets, mask=mask)
    tl.store(states_ptr + dst_row * state_row_stride + offsets, values, mask=mask)


def _masked_update_rows(states: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> None:
    """Copy rows into persistent BIK buffers without touching inactive graph lanes."""
    assert states.ndim == values.ndim == 2
    assert states.stride(1) == values.stride(1) == 1
    assert indices.dtype == torch.int32 and indices.numel() == values.shape[0]
    row_size = states.shape[1]
    assert values.shape[1] == row_size

    block_size = min(triton.next_power_of_2(row_size), 1024)
    grid = (values.shape[0], triton.cdiv(row_size, block_size))
    _masked_update_rows_kernel[grid](
        states,
        indices,
        values,
        states.stride(0),
        values.stride(0),
        ROW_SIZE=row_size,
        BLOCK_SIZE=block_size,
    )


def _mamba_chunk_scan_decode_rows(
    x,
    z,
    dt,
    A,
    B,
    C,
    chunk_size,
    chunk_starts,
    slots,
    target_rows,
    chunk_flags,
    initial_states,
    out,
    D=None,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    """Run the training scan pipeline over buffered decode chunks.

    Each kernel computes only the row or boundary consumed by this decode step,
    while preserving the training kernel's arithmetic for that result.
    """
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        None,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        chunk_starts=chunk_starts,
        target_rows=target_rows,
    )
    states = _chunk_state_fwd(
        B,
        x,
        dt,
        dA_cumsum,
        None,
        states_in_fp32=True,
        chunk_flags=chunk_flags,
        chunk_starts=chunk_starts,
    )
    CB = _bmm_chunk_fwd(
        C,
        B,
        chunk_size,
        None,
        output_dtype=torch.float32,
        target_rows=target_rows,
        chunk_starts=chunk_starts,
    )
    # Scan before state passing because both read the incoming live state and
    # state passing overwrites crossing slots with the outgoing boundary state.
    _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        None,
        out,
        slots,
        D=D,
        z=z,
        initial_states=initial_states,
        target_rows=target_rows,
        chunk_starts=chunk_starts,
    )
    _state_passing_fwd(
        states.flatten(-2),
        dA_cumsum,
        None,
        initial_states=initial_states.flatten(-2),
        seq_idx=slots,
        dst_states=initial_states.flatten(-2),
        dst_indices=slots,
        dst_flags=chunk_flags,
    )


@dataclass
class BatchInvariantDecodeBuffers:
    """Per-slot persistent state for the buffered decode scan."""

    x: torch.Tensor  # (max_requests, chunk_size, nheads, headdim)
    z: torch.Tensor  # (max_requests, chunk_size, nheads, headdim)
    dt: torch.Tensor  # (max_requests, chunk_size, nheads)
    B: torch.Tensor  # (max_requests, chunk_size, ngroups, dstate)
    C: torch.Tensor  # (max_requests, chunk_size, ngroups, dstate)
    # Tokens buffered since the slot's last chunk boundary; doubles as the
    # write cursor for the next token.
    num_buffered: torch.Tensor  # (max_requests,) int32
    # Per-entry target-row output, allocated once and sliced per step.
    out: torch.Tensor  # (max_requests, nheads, headdim)
    target_rows: torch.Tensor  # (max_requests,) int32
    chunk_flags: torch.Tensor  # (max_requests,) int32

    @classmethod
    def allocate(
        cls,
        max_requests: int,
        chunk_size: int,
        nheads: int,
        headdim: int,
        ngroups: int,
        dstate: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "BatchInvariantDecodeBuffers":
        """Allocate the per-slot decode buffers."""
        return cls(
            x=torch.zeros(max_requests, chunk_size, nheads, headdim, device=device, dtype=dtype),
            z=torch.zeros(max_requests, chunk_size, nheads, headdim, device=device, dtype=dtype),
            dt=torch.zeros(max_requests, chunk_size, nheads, device=device, dtype=dtype),
            B=torch.zeros(max_requests, chunk_size, ngroups, dstate, device=device, dtype=dtype),
            C=torch.zeros(max_requests, chunk_size, ngroups, dstate, device=device, dtype=dtype),
            num_buffered=torch.zeros(max_requests, device=device, dtype=torch.int32),
            out=torch.empty(max_requests, nheads, headdim, device=device, dtype=dtype),
            target_rows=torch.empty(max_requests, device=device, dtype=torch.int32),
            chunk_flags=torch.empty(max_requests, device=device, dtype=torch.int32),
        )

    def seed(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        cu_seqlens: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> None:
        """Store each prefill's unfinished chunk for decode replay."""
        chunk_size = self.x.shape[1]
        num_seqs = cu_seqlens.numel() - 1

        seq_starts = cu_seqlens[:-1].to(torch.long)
        seq_ends = cu_seqlens[1:].to(torch.long)
        prefill_lens = seq_ends - seq_starts
        # Covers every case: prefill_len < chunk_size gives prefill_len,
        # boundary-aligned gives 0.
        tail_lens = prefill_lens % chunk_size

        # Fill unused rows with a valid token from the same sequence. The row-gated
        # kernel evaluates a full M-block, so finite padding prevents masked NaNs
        # from reaching the target row through tensor-core operations.
        offsets = torch.arange(chunk_size, device=x.device, dtype=torch.long)
        safe_tail_lens = torch.clamp(tail_lens, min=1)
        safe_tail_offsets = torch.minimum(offsets.unsqueeze(0), (safe_tail_lens - 1).unsqueeze(1))
        safe_tail_starts = torch.where(
            tail_lens > 0, seq_ends - tail_lens, torch.clamp(seq_ends - 1, min=0)
        )
        tail_token_idx = (safe_tail_starts.unsqueeze(1) + safe_tail_offsets).clamp(
            max=x.shape[0] - 1
        )

        slots = batch_indices[:num_seqs]
        _masked_update_rows(self.x.flatten(1), slots, x[tail_token_idx].flatten(1))
        _masked_update_rows(self.z.flatten(1), slots, z[tail_token_idx].flatten(1))
        _masked_update_rows(self.dt.flatten(1), slots, dt[tail_token_idx].flatten(1))
        _masked_update_rows(self.B.flatten(1), slots, B[tail_token_idx].flatten(1))
        _masked_update_rows(self.C.flatten(1), slots, C[tail_token_idx].flatten(1))
        _masked_update_rows(
            self.num_buffered.unsqueeze(1), slots, tail_lens.to(torch.int32).unsqueeze(1)
        )


def batch_invariant_decode_buffered_scan(
    buffers: BatchInvariantDecodeBuffers,
    x: torch.Tensor,  # (decode_batch_size, 1, nheads, headdim)
    z: torch.Tensor,  # (decode_batch_size, 1, nheads, headdim)
    dt: torch.Tensor,  # (decode_batch_size, 1, nheads)
    B: torch.Tensor,  # (decode_batch_size, 1, ngroups, dstate)
    C: torch.Tensor,  # (decode_batch_size, 1, ngroups, dstate)
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_indices: torch.Tensor,
    ssm_state: torch.Tensor,
) -> torch.Tensor:
    """Run one decode token with full chunk-scan arithmetic.

    Mutates the replay buffers and commits ``ssm_state`` when a chunk fills.
    """
    decode_batch_size, tokens_per_entry, nheads, headdim = x.shape
    dstate = B.shape[-1]
    chunk_size = buffers.x.shape[1]
    assert tokens_per_entry == 1, (
        "batch-invariant Mamba decode assumes one new token per request "
        "per call (no speculative decoding)."
    )
    assert ssm_state.dtype == torch.float32, (
        "batch-invariant Mamba decode requires an FP32 SSM state cache to preserve "
        "the state-passing carry across chunk boundaries."
    )
    output_capacity = buffers.out.shape[0]
    assert decode_batch_size <= output_capacity, (
        f"decode batch size {decode_batch_size} exceeds the output buffer capacity "
        f"({output_capacity}); increase max_requests."
    )

    out = buffers.out[:decode_batch_size]
    target_rows = buffers.target_rows[:decode_batch_size]
    chunk_flags = buffers.chunk_flags[:decode_batch_size]

    active = batch_indices >= 0
    safe_slots = batch_indices.clamp_min(0)
    write_pos = buffers.num_buffered[safe_slots].to(torch.long)
    buffer_rows = torch.where(active, safe_slots * chunk_size + write_pos, -1).to(torch.int32)

    _masked_update_rows(buffers.x.view(-1, nheads * headdim), buffer_rows, x[:, 0].flatten(1))
    _masked_update_rows(buffers.z.view(-1, nheads * headdim), buffer_rows, z[:, 0].flatten(1))
    _masked_update_rows(buffers.dt.view(-1, nheads), buffer_rows, dt[:, 0])
    _masked_update_rows(
        buffers.B.view(-1, buffers.B.shape[-2] * dstate), buffer_rows, B[:, 0].flatten(1)
    )
    _masked_update_rows(
        buffers.C.view(-1, buffers.C.shape[-2] * dstate), buffer_rows, C[:, 0].flatten(1)
    )

    crossed = active & (write_pos + 1 == chunk_size)
    target_rows.copy_(torch.where(active, write_pos, -1).to(torch.int32))
    chunk_flags.copy_(crossed.to(torch.int32))
    out.zero_()

    # Run the gated pipeline over the buffers and ssm_state in place. State
    # passing writes crossing slots' boundary states straight into
    # ssm_state, so no scatter is needed afterwards.
    _mamba_chunk_scan_decode_rows(
        buffers.x.view(-1, nheads, headdim),
        buffers.z.view(-1, nheads, headdim),
        buffers.dt.view(-1, nheads),
        A,
        buffers.B.view(-1, buffers.B.shape[-2], dstate),
        buffers.C.view(-1, buffers.C.shape[-2], dstate),
        chunk_size,
        chunk_starts=batch_indices * chunk_size,
        slots=batch_indices,
        target_rows=target_rows,
        chunk_flags=chunk_flags,
        initial_states=ssm_state,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    next_write_pos = torch.where(crossed, 0, write_pos + 1).to(torch.int32)
    _masked_update_rows(
        buffers.num_buffered.unsqueeze(1), batch_indices, next_write_pos.unsqueeze(1)
    )

    return out.unsqueeze(1)


class MambaBatchInvariantDecode:
    """Adapter between a MambaMixer and the buffered decode."""

    def __init__(self, mixer):
        # Training applies z inside the chunk scan before RMSNormGated, so
        # decode buffers and replays z through that same kernel path.
        assert mixer.rmsnorm, "batch_invariant_mode requires rmsnorm=True"
        self.mixer = mixer
        self.buffers: BatchInvariantDecodeBuffers | None = None

    def _get_buffers(self, max_requests, x, B) -> BatchInvariantDecodeBuffers:
        if self.buffers is None:
            nheads, headdim = x.shape[-2:]
            ngroups, dstate = B.shape[-2:]
            self.buffers = BatchInvariantDecodeBuffers.allocate(
                max_requests,
                self.mixer.chunk_size,
                nheads,
                headdim,
                ngroups,
                dstate,
                x.device,
                x.dtype,
            )
        return self.buffers

    def seed(self, x, z, dt, B, C, cu_seqlens, batch_indices, max_requests) -> None:
        """Seed replay buffers from the prefill tail."""
        buffers = self._get_buffers(max_requests, x, B)
        buffers.seed(x, z, dt, B, C, cu_seqlens, batch_indices)

    def step(self, x, z, dt, B, C, batch_indices, ssm_state) -> torch.Tensor:
        """Run one decode step using the mixer's flattened layouts."""
        mixer = self.mixer
        batch = x.shape[0]
        x = x.view(batch, 1, -1, mixer.headdim)
        z = z.view(batch, 1, -1, mixer.headdim)
        B = B.view(batch, 1, mixer.ngroups_local_tp, -1)
        C = C.view(batch, 1, mixer.ngroups_local_tp, -1)

        A = -torch.exp(mixer.cp.get_A_log().float())
        D = mixer.cp.get_D()
        if mixer.D_has_hdim:
            D = D.float().view(-1, mixer.headdim)
        dt_bias = mixer.cp.get_dt_bias().float()

        buffers = self._get_buffers(ssm_state.shape[0], x, B)

        y = batch_invariant_decode_buffered_scan(
            buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
        return y.reshape(batch, 1, -1)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Batch-invariant Mamba decode using buffered chunk replay."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from megatron.core.ssm.ops.mamba2.batch_invariant_decode_rows import (
    bmm_rows,
    chunk_scan_rows,
    chunk_state_passing_rows,
    row_kernel_configs,
)
from megatron.core.ssm.ops.mamba2.ssd_bmm import _bmm_chunk_fwd
from megatron.core.ssm.ops.mamba2.ssd_chunk_scan import _chunk_scan_fwd
from megatron.core.ssm.ops.mamba2.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from megatron.core.ssm.ops.mamba2.ssd_state_passing import _state_passing_fwd


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


@triton.jit
def _decode_step_prep_kernel(
    batch_indices_ptr,
    num_buffered_ptr,
    x_ptr,
    dt_ptr,
    B_ptr,
    C_ptr,
    x_buf_ptr,
    dt_buf_ptr,
    B_buf_ptr,
    C_buf_ptr,
    out_ptr,
    target_rows_ptr,
    chunk_flags_ptr,
    chunk_starts_ptr,
    stride_x,
    stride_dt,
    stride_B,
    stride_C,
    X_SIZE: tl.constexpr,
    DT_SIZE: tl.constexpr,
    BC_SIZE: tl.constexpr,
    OUT_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One decode entry's bookkeeping: integer updates and row copies only.

    For entry e with slot s and write position w = num_buffered[s]: copy the current
    token's x / dt / B / C rows into buffer row s * CHUNK_SIZE + w, set target_rows[e] = w,
    chunk_flags[e] = (w + 1 == CHUNK_SIZE), chunk_starts[e] = s * CHUNK_SIZE, zero out[e]
    and advance num_buffered[s] (0 after a crossing). Inactive entries (s < 0) only get
    target row -1, flag 0 and a zero output.
    """
    e = tl.program_id(0)
    slot = tl.load(batch_indices_ptr + e)
    tl.store(chunk_starts_ptr + e, slot * CHUNK_SIZE)
    offs = tl.arange(0, BLOCK_SIZE)
    zeros = tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty)
    for o in range(0, OUT_SIZE, BLOCK_SIZE):
        tl.store(out_ptr + e * OUT_SIZE + o + offs, zeros, mask=o + offs < OUT_SIZE)
    if slot < 0:
        tl.store(target_rows_ptr + e, -1)
        tl.store(chunk_flags_ptr + e, 0)
        return
    write_pos = tl.load(num_buffered_ptr + slot)
    row = (slot * CHUNK_SIZE + write_pos).to(tl.int64)
    for o in range(0, X_SIZE, BLOCK_SIZE):
        mask = o + offs < X_SIZE
        values = tl.load(x_ptr + e * stride_x + o + offs, mask=mask)
        tl.store(x_buf_ptr + row * X_SIZE + o + offs, values, mask=mask)
    for o in range(0, DT_SIZE, BLOCK_SIZE):
        mask = o + offs < DT_SIZE
        values = tl.load(dt_ptr + e * stride_dt + o + offs, mask=mask)
        tl.store(dt_buf_ptr + row * DT_SIZE + o + offs, values, mask=mask)
    for o in range(0, BC_SIZE, BLOCK_SIZE):
        mask = o + offs < BC_SIZE
        values = tl.load(B_ptr + e * stride_B + o + offs, mask=mask)
        tl.store(B_buf_ptr + row * BC_SIZE + o + offs, values, mask=mask)
        values = tl.load(C_ptr + e * stride_C + o + offs, mask=mask)
        tl.store(C_buf_ptr + row * BC_SIZE + o + offs, values, mask=mask)
    crossed = write_pos + 1 == CHUNK_SIZE
    tl.store(target_rows_ptr + e, write_pos)
    tl.store(chunk_flags_ptr + e, crossed.to(tl.int32))
    tl.store(num_buffered_ptr + slot, tl.where(crossed, 0, write_pos + 1))


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
    use_row_kernels=True,
):
    """Run the training scan pipeline over buffered decode chunks.

    Each kernel computes only the row or boundary consumed by this decode step,
    while preserving the training kernel's arithmetic for that result. ``z`` is
    the current token's gate per entry (decode_batch_size, nheads, headdim): the
    gate is row-local and only the target row is computed, so no other row's z
    is needed.

    When the training kernels are pinned to one config each, the row kernels
    (``batch_invariant_decode_rows``) compute the same bits with that config and
    less work; otherwise the training kernels run in target-rows mode.
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
    configs = row_kernel_configs(C, chunk_size) if use_row_kernels else None
    if configs is not None:
        scan_config, bmm_config, state_config = configs
        CB = bmm_rows(C, B, chunk_size, target_rows, chunk_starts, bmm_config)
        # Scan before the boundary update: both read the slot's incoming state.
        chunk_scan_rows(
            CB,
            x,
            z,
            dt,
            dA_cumsum,
            C,
            out,
            slots,
            target_rows,
            chunk_starts,
            initial_states,
            D,
            scan_config,
        )
        chunk_state_passing_rows(
            x, B, dt, dA_cumsum, chunk_starts, chunk_flags, slots, initial_states, state_config
        )
        return

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

    # No z buffer: the gate z * sigmoid(z) is row-local and decode computes only
    # the current (target) row, so the current token's z is all the scan reads.
    x: torch.Tensor  # (max_requests, chunk_size, nheads, headdim)
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
    chunk_starts: torch.Tensor  # (max_requests,) int32

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
            dt=torch.zeros(max_requests, chunk_size, nheads, device=device, dtype=dtype),
            B=torch.zeros(max_requests, chunk_size, ngroups, dstate, device=device, dtype=dtype),
            C=torch.zeros(max_requests, chunk_size, ngroups, dstate, device=device, dtype=dtype),
            num_buffered=torch.zeros(max_requests, device=device, dtype=torch.int32),
            out=torch.empty(max_requests, nheads, headdim, device=device, dtype=dtype),
            target_rows=torch.empty(max_requests, device=device, dtype=torch.int32),
            chunk_flags=torch.empty(max_requests, device=device, dtype=torch.int32),
            chunk_starts=torch.empty(max_requests, device=device, dtype=torch.int32),
        )

    def seed(
        self,
        x: torch.Tensor,
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

    assert batch_indices.dtype == torch.int32
    out = buffers.out[:decode_batch_size]
    target_rows = buffers.target_rows[:decode_batch_size]
    chunk_flags = buffers.chunk_flags[:decode_batch_size]
    chunk_starts = buffers.chunk_starts[:decode_batch_size]

    # Write the current token into the replay buffers and set this step's target
    # rows, chunk flags and chunk starts in one launch. The scan pipeline does not
    # read num_buffered, so its update happens here too.
    x_rows, dt_rows = x[:, 0].flatten(1), dt[:, 0]
    B_rows, C_rows = B[:, 0].flatten(1), C[:, 0].flatten(1)
    assert all(t.stride(-1) == 1 for t in (x_rows, dt_rows, B_rows, C_rows))
    assert all(t.is_contiguous() for t in (out, buffers.x, buffers.dt, buffers.B, buffers.C))
    if decode_batch_size > 0:
        _decode_step_prep_kernel[(decode_batch_size,)](
            batch_indices,
            buffers.num_buffered,
            x_rows,
            dt_rows,
            B_rows,
            C_rows,
            buffers.x,
            buffers.dt,
            buffers.B,
            buffers.C,
            out,
            target_rows,
            chunk_flags,
            chunk_starts,
            x_rows.stride(0),
            dt_rows.stride(0),
            B_rows.stride(0),
            C_rows.stride(0),
            X_SIZE=nheads * headdim,
            DT_SIZE=nheads,
            BC_SIZE=buffers.B.shape[-2] * dstate,
            OUT_SIZE=nheads * headdim,
            CHUNK_SIZE=chunk_size,
            BLOCK_SIZE=1024,
        )

    # Run the gated pipeline over the buffers and ssm_state in place. State
    # passing writes crossing slots' boundary states straight into
    # ssm_state, so no scatter is needed afterwards.
    _mamba_chunk_scan_decode_rows(
        buffers.x.view(-1, nheads, headdim),
        z[:, 0],
        buffers.dt.view(-1, nheads),
        A,
        buffers.B.view(-1, buffers.B.shape[-2], dstate),
        buffers.C.view(-1, buffers.C.shape[-2], dstate),
        chunk_size,
        chunk_starts=chunk_starts,
        slots=batch_indices,
        target_rows=target_rows,
        chunk_flags=chunk_flags,
        initial_states=ssm_state,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    return out.unsqueeze(1)


class MambaBatchInvariantDecode:
    """Adapter between a MambaMixer and the buffered decode."""

    def __init__(self, mixer):
        # Training applies z inside the chunk scan before RMSNormGated, so
        # decode gates the target row through that same kernel path.
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

    def seed(self, x, dt, B, C, cu_seqlens, batch_indices, max_requests) -> None:
        """Seed replay buffers from the prefill tail."""
        buffers = self._get_buffers(max_requests, x, B)
        buffers.seed(x, dt, B, C, cu_seqlens, batch_indices)

    def step(self, x, z, dt, B, C, batch_indices, ssm_state) -> torch.Tensor:
        """Run one decode step using the mixer's flattened layouts."""
        mixer = self.mixer
        batch = x.shape[0]
        x = x.view(batch, 1, -1, mixer.headdim)
        z = z.view(batch, 1, -1, mixer.headdim)
        B = B.view(batch, 1, mixer.ngroups_local_tp, -1)
        C = C.view(batch, 1, mixer.ngroups_local_tp, -1)

        A = -torch.exp(mixer.cp.get_A_log().float())
        D = mixer.get_ssm_D(mixer.cp.get_D())
        dt_bias = mixer.cp.get_dt_bias().float()

        buffers = self._get_buffers(ssm_state.shape[0], x, B)

        y = batch_invariant_decode_buffered_scan(
            buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
        return y.reshape(batch, 1, -1)

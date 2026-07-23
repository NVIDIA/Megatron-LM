# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Batch-invariant Mamba decode using buffered chunk replay."""

from dataclasses import dataclass

import torch

from megatron.core.ssm.ops.ssd_combined import mamba_chunk_scan_decode_rows


@dataclass
class BatchInvariantDecodeBuffers:
    """Per-slot persistent state for the buffered decode scan."""

    x: torch.Tensor  # (max_batch + 1, chunk_size, nheads, headdim)
    dt: torch.Tensor  # (max_batch + 1, chunk_size, nheads)
    B: torch.Tensor  # (max_batch + 1, chunk_size, ngroups, dstate)
    C: torch.Tensor  # (max_batch + 1, chunk_size, ngroups, dstate)
    # Tokens buffered since the slot's last chunk boundary; doubles as the
    # write cursor for the next token.
    num_buffered: torch.Tensor  # (max_batch + 1,) int32
    # Per-entry target-row output, allocated once and sliced per step.
    out: torch.Tensor  # (max_batch + 1, nheads, headdim)

    @classmethod
    def allocate(
        cls,
        max_batch: int,
        chunk_size: int,
        nheads: int,
        headdim: int,
        ngroups: int,
        dstate: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "BatchInvariantDecodeBuffers":
        """Allocate the per-slot decode buffers."""
        # Padding entries use batch index -1. Map them to an extra row so
        # fixed-shape graph code can write without touching a live request.
        rows = max_batch + 1
        return cls(
            x=torch.zeros(rows, chunk_size, nheads, headdim, device=device, dtype=dtype),
            dt=torch.zeros(rows, chunk_size, nheads, device=device, dtype=dtype),
            B=torch.zeros(rows, chunk_size, ngroups, dstate, device=device, dtype=dtype),
            C=torch.zeros(rows, chunk_size, ngroups, dstate, device=device, dtype=dtype),
            num_buffered=torch.zeros(rows, device=device, dtype=torch.int32),
            out=torch.empty(rows, nheads, headdim, device=device, dtype=dtype),
        )

    @property
    def trash_row(self) -> int:
        """Write sink for padding entries (the buffers' extra last row)."""
        return self.num_buffered.shape[0] - 1

    def map_slots(self, batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map padding entries to the trash row."""
        slots = batch_indices.to(torch.long)
        is_active = slots >= 0
        return slots.masked_fill(~is_active, self.trash_row), is_active

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

        # Redirect padding entries (batch_indices < 0) to the trash row so all
        # writes below are unconditional.
        slots, is_active = self.map_slots(batch_indices[:num_seqs])

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

        self.x[slots] = x[tail_token_idx]
        self.dt[slots] = dt[tail_token_idx]
        self.B[slots] = B[tail_token_idx]
        self.C[slots] = C[tail_token_idx]

        # Keep the trash row's count pinned at 0 so its buffer writes stay in
        # bounds.
        self.num_buffered[slots] = torch.where(
            is_active, tail_lens, torch.zeros_like(tail_lens)
        ).to(torch.int32)


def batch_invariant_decode_buffered_scan(
    buffers: BatchInvariantDecodeBuffers,
    x: torch.Tensor,  # (decode_batch_size, 1, nheads, headdim)
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
        f"({output_capacity}); increase max_batch."
    )

    # Redirect padding entries (batch_indices < 0) to the trash row so the
    # buffer writes below are unconditional.
    slots, is_active = buffers.map_slots(batch_indices)
    # ssm_state is engine-owned and has no trash row: clamp for reads. Its
    # only writes happen in-kernel for crossing slots, which never alias.
    state_slots = slots.clamp(max=buffers.trash_row - 1)

    # Write the new token at each slot's cursor; write_pos is also the
    # token's intra-chunk row, the one row the scan must produce.
    write_pos = buffers.num_buffered[slots].to(torch.long)
    buffers.x[slots, write_pos] = x[:, 0]
    buffers.dt[slots, write_pos] = dt[:, 0]
    buffers.B[slots, write_pos] = B[:, 0]
    buffers.C[slots, write_pos] = C[:, 0]

    # A slot crosses its chunk boundary when this token fills the buffer.
    crossed = (write_pos + 1 == chunk_size) & is_active
    out = buffers.out[:decode_batch_size]

    # Run the gated pipeline over the buffers and ssm_state in place. State
    # passing writes crossing slots' boundary states straight into
    # ssm_state, so no scatter is needed afterwards.
    mamba_chunk_scan_decode_rows(
        buffers.x.view(-1, nheads, headdim),
        buffers.dt.view(-1, nheads),
        A,
        buffers.B.view(-1, buffers.B.shape[-2], dstate),
        buffers.C.view(-1, buffers.C.shape[-2], dstate),
        chunk_size,
        chunk_starts=(slots * chunk_size).to(torch.int32),
        slots=state_slots.to(torch.int32),
        target_rows=write_pos.to(torch.int32),
        chunk_flags=crossed.to(torch.int32),
        initial_states=ssm_state,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    # The scan stored each entry's target row at out[i]; padding entries
    # return zeros.
    y = torch.where(is_active.view(-1, 1, 1), out, torch.zeros_like(out)).unsqueeze(1)

    # Crossed slots restart their buffer; the rest advance. Padding entries
    # write 0 to the trash row, keeping its cursor pinned in bounds.
    buffers.num_buffered[slots] = torch.where(
        crossed | ~is_active, torch.zeros_like(write_pos), write_pos + 1
    ).to(torch.int32)

    return y


class MambaBatchInvariantDecode:
    """Adapter between a MambaMixer and the buffered decode."""

    def __init__(self, mixer):
        # The gate is applied outside the scan (RMSNormGated), so the
        # buffers carry no z. Enforced here because the decode path would
        # otherwise silently drop it.
        assert mixer.rmsnorm, "batch_invariant_mode requires rmsnorm=True"
        self.mixer = mixer
        self.buffers: BatchInvariantDecodeBuffers | None = None

    def _get_buffers(self, max_batch, x, B) -> BatchInvariantDecodeBuffers:
        if self.buffers is None:
            nheads, headdim = x.shape[-2:]
            ngroups, dstate = B.shape[-2:]
            self.buffers = BatchInvariantDecodeBuffers.allocate(
                max_batch,
                self.mixer.chunk_size,
                nheads,
                headdim,
                ngroups,
                dstate,
                x.device,
                x.dtype,
            )
        return self.buffers

    def seed(self, x, dt, B, C, cu_seqlens, batch_indices, max_batch) -> None:
        """Seed replay buffers from the prefill tail."""
        buffers = self._get_buffers(max_batch, x, B)
        buffers.seed(x, dt, B, C, cu_seqlens, batch_indices)

    def step(self, x, dt, B, C, batch_indices, ssm_state) -> torch.Tensor:
        """Run one decode step using the mixer's flattened layouts."""
        mixer = self.mixer
        batch = x.shape[0]
        x = x.view(batch, 1, -1, mixer.headdim)
        B = B.view(batch, 1, mixer.ngroups_local_tp, -1)
        C = C.view(batch, 1, mixer.ngroups_local_tp, -1)

        A = -torch.exp(mixer.cp.get_A_log().float())
        D = mixer.cp.get_D()
        if mixer.D_has_hdim:
            D = D.float().view(-1, mixer.headdim)
        dt_bias = mixer.cp.get_dt_bias().float()

        buffers = self._get_buffers(ssm_state.shape[0], x, B)

        y = batch_invariant_decode_buffered_scan(
            buffers, x, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
        return y.reshape(batch, 1, -1)

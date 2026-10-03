"""Context-parallel state passing for Mamba2 SSD.

The production autograd path combines causal Conv1d and the local SSD scan, then
exchanges only FP32 state summaries and scalar block decays. Backward recomputes
the conv output, applies the reverse state-passing CP boundary scan, and writes SSD gradients
directly into the buffers consumed by causal Conv1d backward.

Verified by tests/unit_tests/ssm/ops/test_ssd_state_passing_cp.py.

Note on the ``permute_p2p`` / ``permute_a2a`` load-balancing modes: converting
Megatron's balanced (zigzag) CP layout to a contiguous causal shard overlaps with
``megatron.core.context_parallel_layout``, whose ``CpPartitionMode`` models the
same zigzag/contiguous pair and which GDN already uses via
``convert_module_input_tensors_cp_partition_mode``. The permutation here is kept
self-contained because it is driven from inside the fused Conv+SSD autograd
function rather than at the module entry point, and because it adds a P2P backend
that the shared helper does not have. Consolidating onto the shared helper is
intended follow-up work, not a decision that the shared path is unsuitable.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from einops import rearrange
from mamba_ssm.ops.triton.ssd_combined import (
    _bmm_chunk_bwd,
    _bmm_chunk_fwd,
    _chunk_cumsum_bwd,
    _chunk_cumsum_fwd,
    _chunk_scan_bwd_dC,
    _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_ddAcs_stable,
    _chunk_scan_bwd_dstates,
    _chunk_scan_bwd_dz,
    _chunk_scan_chunk_state_bwd_dx,
    _chunk_scan_fwd,
    _chunk_state_bwd_db,
    _chunk_state_fwd,
    _state_passing_bwd,
    _state_passing_fwd,
    causal_conv1d_bwd_function,
    causal_conv1d_fwd_function,
    ensure_stride,
)
from mamba_ssm.utils.torch import custom_bwd, custom_fwd


def _all_gather_stack(x, group, async_op=False):
    """Gather directly into a stacked contiguous tensor."""
    world = dist.get_world_size(group)
    x = x.contiguous()
    gathered = torch.empty((world, *x.shape), device=x.device, dtype=x.dtype)
    work = dist.all_gather_into_tensor(gathered, x, group=group, async_op=async_op)
    return gathered, work


@triton.jit
def _state_passing_summary_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    dA_cs_ptr,
    summary_ptr,
    # Matrix dimensions
    dim: tl.constexpr,
    nchunks: tl.constexpr,
    nheads: tl.constexpr,
    state_numel: tl.constexpr,
    # Strides
    stride_states_batch: tl.constexpr,
    stride_states_chunk: tl.constexpr,
    stride_states_head: tl.constexpr,
    stride_states_dim: tl.constexpr,
    stride_dA_cs_batch: tl.constexpr,
    stride_dA_cs_head: tl.constexpr,
    stride_dA_cs_chunk: tl.constexpr,
    # Meta-parameters
    BATCH_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    batch_idx = pid_b * BATCH_STRIDE
    states_ptr += batch_idx * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += batch_idx * stride_dA_cs_batch + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    state = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total_decay = 0.0
    for _ in range(nchunks):
        new_state = tl.load(states_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        state = tl.exp(dA_cs) * state + new_state
        total_decay += dA_cs
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk

    state_offset = (batch_idx * nheads + pid_h) * dim
    tl.store(summary_ptr + state_offset + offs_m, state, mask=offs_m < dim)
    decay_offset = state_numel + batch_idx * nheads + pid_h
    tl.store(summary_ptr + decay_offset, tl.exp(total_decay), mask=pid_m == 0)


def _state_passing_summary_fwd(states, dA_chunk_cumsum, *, active_batch=None, batch_stride=1):
    """Pack final states and block decays into one contiguous FP32 payload.

    active_batch and batch_stride select the interleaved front slots used by the
    virtual state-passing CP rank-0 summary optimization.
    """
    assert states.dtype == torch.float32
    assert dA_chunk_cumsum.dtype == torch.float32
    batch, nchunks, nheads, dim = states.shape
    active_batch = batch if active_batch is None else active_batch
    assert active_batch > 0
    assert (active_batch - 1) * batch_stride < batch

    state_numel = batch * nheads * dim
    summary_numel = state_numel + batch * nheads
    if active_batch < batch:
        summary = torch.zeros(summary_numel, device=states.device, dtype=torch.float32)
    else:
        summary = torch.empty(summary_numel, device=states.device, dtype=torch.float32)

    block_size = 256
    grid = (triton.cdiv(dim, block_size), active_batch, nheads)
    _state_passing_summary_fwd_kernel[grid](
        states,
        dA_chunk_cumsum,
        summary,
        dim=dim,
        nchunks=nchunks,
        nheads=nheads,
        state_numel=state_numel,
        stride_states_batch=states.stride(0),
        stride_states_chunk=states.stride(1),
        stride_states_head=states.stride(2),
        stride_states_dim=states.stride(3),
        stride_dA_cs_batch=dA_chunk_cumsum.stride(0),
        stride_dA_cs_head=dA_chunk_cumsum.stride(1),
        stride_dA_cs_chunk=dA_chunk_cumsum.stride(2),
        BATCH_STRIDE=batch_stride,
        BLOCK_SIZE=block_size,
    )
    return summary, state_numel


@triton.jit
def _state_passing_summary_bwd_kernel(
    # Pointers to matrices
    dout_ptr,
    dA_cs_ptr,
    dinitstates_ptr,
    # Matrix dimensions
    dim: tl.constexpr,
    nchunks: tl.constexpr,
    # Strides
    stride_dout_batch: tl.constexpr,
    stride_dout_chunk: tl.constexpr,
    stride_dout_head: tl.constexpr,
    stride_dout_dim: tl.constexpr,
    stride_dA_cs_batch: tl.constexpr,
    stride_dA_cs_head: tl.constexpr,
    stride_dA_cs_chunk: tl.constexpr,
    stride_dinitstates_batch: tl.constexpr,
    stride_dinitstates_head: tl.constexpr,
    stride_dinitstates_dim: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    dout_ptr += (
        pid_b * stride_dout_batch + pid_h * stride_dout_head + (nchunks - 1) * stride_dout_chunk
    )
    dA_cs_ptr += (
        pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + (nchunks - 1) * stride_dA_cs_chunk
    )

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dout_ptrs = dout_ptr + offs_m * stride_dout_dim
    dstates = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for _ in range(nchunks):
        dout = tl.load(dout_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        dstates = tl.exp(dA_cs) * dstates + dout
        dout_ptrs -= stride_dout_chunk
        dA_cs_ptr -= stride_dA_cs_chunk

    dinitstates_ptr += pid_b * stride_dinitstates_batch + pid_h * stride_dinitstates_head
    tl.store(dinitstates_ptr + offs_m * stride_dinitstates_dim, dstates, mask=offs_m < dim)


def _state_passing_summary_bwd(dout, dA_chunk_cumsum):
    """Compute only the gradient entering the first local chunk."""
    assert dout.dtype == torch.float32
    assert dA_chunk_cumsum.dtype == torch.float32
    batch, nchunks, nheads, dim = dout.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)

    dinitstates = torch.empty((batch, nheads, dim), device=dout.device, dtype=torch.float32)
    block_size = 256
    grid = (triton.cdiv(dim, block_size), batch, nheads)
    _state_passing_summary_bwd_kernel[grid](
        dout,
        dA_chunk_cumsum,
        dinitstates,
        dim=dim,
        nchunks=nchunks,
        stride_dout_batch=dout.stride(0),
        stride_dout_chunk=dout.stride(1),
        stride_dout_head=dout.stride(2),
        stride_dout_dim=dout.stride(3),
        stride_dA_cs_batch=dA_chunk_cumsum.stride(0),
        stride_dA_cs_head=dA_chunk_cumsum.stride(1),
        stride_dA_cs_chunk=dA_chunk_cumsum.stride(2),
        stride_dinitstates_batch=dinitstates.stride(0),
        stride_dinitstates_head=dinitstates.stride(1),
        stride_dinitstates_dim=dinitstates.stride(2),
        BLOCK_SIZE=block_size,
    )
    return dinitstates


@triton.jit
def _state_passing_boundary_scan_kernel(
    # Pointers to matrices
    boundary_values_ptr,
    block_decays_ptr,
    boundary_output_ptr,
    # Matrix dimensions
    dim: tl.constexpr,
    world: tl.constexpr,
    rank: tl.constexpr,
    # Strides
    stride_values_rank: tl.constexpr,
    stride_values_batch: tl.constexpr,
    stride_values_head: tl.constexpr,
    stride_decays_rank: tl.constexpr,
    stride_decays_batch: tl.constexpr,
    stride_decays_head: tl.constexpr,
    stride_output_batch: tl.constexpr,
    stride_output_head: tl.constexpr,
    # Meta-parameters
    VIRTUAL_STATE_PASSING_CP: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    state = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    nsegments = 2 * world if VIRTUAL_STATE_PASSING_CP else world
    for step in range(nsegments):
        segment = nsegments - 1 - step if REVERSE else step
        if VIRTUAL_STATE_PASSING_CP:
            if segment == rank:
                tl.store(
                    boundary_output_ptr
                    + (2 * pid_b) * stride_output_batch
                    + pid_h * stride_output_head
                    + offs_m,
                    state,
                    mask=offs_m < dim,
                )
            if segment == nsegments - 1 - rank:
                tl.store(
                    boundary_output_ptr
                    + (2 * pid_b + 1) * stride_output_batch
                    + pid_h * stride_output_head
                    + offs_m,
                    state,
                    mask=offs_m < dim,
                )
            source_rank = segment if segment < world else nsegments - 1 - segment
            source_batch = 2 * pid_b if segment < world else 2 * pid_b + 1
            include = True
        else:
            source_rank = segment
            source_batch = pid_b
            include = source_rank > rank if REVERSE else source_rank < rank

        if include:
            new_state = tl.load(
                boundary_values_ptr
                + source_rank * stride_values_rank
                + source_batch * stride_values_batch
                + pid_h * stride_values_head
                + offs_m,
                mask=offs_m < dim,
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(
                block_decays_ptr
                + source_rank * stride_decays_rank
                + source_batch * stride_decays_batch
                + pid_h * stride_decays_head
            ).to(tl.float32)
            state = scale * state + new_state

    if not VIRTUAL_STATE_PASSING_CP:
        tl.store(
            boundary_output_ptr + pid_b * stride_output_batch + pid_h * stride_output_head + offs_m,
            state,
            mask=offs_m < dim,
        )


def _state_passing_boundary_scan(
    boundary_values, block_decays, rank, state_passing_cp_virtual, reverse
):
    """Scan gathered summaries and return this rank's exclusive causal boundary."""
    assert boundary_values.dtype == torch.float32
    assert block_decays.dtype == torch.float32
    world, batch, nheads = boundary_values.shape[:3]
    state_shape = boundary_values.shape[3:]
    boundary_values_flat = boundary_values.view(world, batch, nheads, -1)
    dim = boundary_values_flat.shape[-1]
    boundary_output = torch.empty(
        batch, nheads, dim, device=boundary_values.device, dtype=torch.float32
    )

    block_size = 256
    if state_passing_cp_virtual:
        assert batch % 2 == 0
        scan_batch = batch // 2
    else:
        scan_batch = batch
    grid = (triton.cdiv(dim, block_size), scan_batch, nheads)
    _state_passing_boundary_scan_kernel[grid](
        boundary_values_flat,
        block_decays,
        boundary_output,
        dim=dim,
        world=world,
        rank=rank,
        stride_values_rank=boundary_values_flat.stride(0),
        stride_values_batch=boundary_values_flat.stride(1),
        stride_values_head=boundary_values_flat.stride(2),
        stride_decays_rank=block_decays.stride(0),
        stride_decays_batch=block_decays.stride(1),
        stride_decays_head=block_decays.stride(2),
        stride_output_batch=boundary_output.stride(0),
        stride_output_head=boundary_output.stride(1),
        VIRTUAL_STATE_PASSING_CP=state_passing_cp_virtual,
        REVERSE=reverse,
        BLOCK_SIZE=block_size,
    )
    return boundary_output.view(batch, nheads, *state_shape)


def _mamba_chunk_scan_combined_state_passing_cp_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_passing_cp_group=None,
    state_passing_cp_virtual=False,
):
    """Mamba combined forward with state-passing CP summary and boundary exchange."""
    assert state_passing_cp_group is not None
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    assert (
        initial_states is None
    ), "external initial_states are not implemented for state-passing CP"
    assert seq_idx is None, "seq_idx is not implemented for state-passing CP"
    assert cu_seqlens is None, "cu_seqlens is not implemented for state-passing CP"
    state_passing_cp_size = dist.get_world_size(state_passing_cp_group)
    rank = dist.get_rank(state_passing_cp_group)
    if state_passing_cp_virtual:
        assert (
            batch % 2 == 0
        ), "virtual state-passing CP expects front/back packed on the batch axis"
    virtual_batch = batch // 2 if state_passing_cp_virtual else None

    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    state_passing_initial_states = None
    state_passing_gathered_decays = None
    if state_passing_cp_size > 1:
        state_dim = headdim * dstate
        if not state_passing_cp_virtual and rank == state_passing_cp_size - 1:
            # Collective shapes are fixed even though no later rank consumes this summary.
            state_numel = batch * nheads * state_dim
            packed_summary = torch.zeros(
                state_numel + batch * nheads, device=x.device, dtype=torch.float32
            )
        else:
            if state_passing_cp_virtual and rank == 0:
                # Rank 0's causal-last segment has no successor.
                active_batch, batch_stride = virtual_batch, 2
            else:
                active_batch, batch_stride = None, 1
            packed_summary, state_numel = _state_passing_summary_fwd(
                rearrange(states, "... p n -> ... (p n)"),
                dA_cumsum[:, :, :, -1],
                active_batch=active_batch,
                batch_stride=batch_stride,
            )

        gathered_summary, gather_work = _all_gather_stack(
            packed_summary, state_passing_cp_group, async_op=True
        )
        CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
        gather_work.wait()
        gathered_states = gathered_summary[:, :state_numel].view(
            state_passing_cp_size, batch, nheads, headdim, dstate
        )
        state_passing_gathered_decays = gathered_summary[:, state_numel:].view(
            state_passing_cp_size, batch, nheads
        )
        state_passing_initial_states = _state_passing_boundary_scan(
            gathered_states,
            state_passing_gathered_decays,
            rank,
            state_passing_cp_virtual=state_passing_cp_virtual,
            reverse=False,
        )
        state_passing_gathered_decays = state_passing_gathered_decays.contiguous()

    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=(
            rearrange(state_passing_initial_states, "... p n -> ... (p n)")
            if state_passing_initial_states is not None
            else None
        ),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=C.dtype,
    )
    states, final_states = [
        rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]
    ]
    if state_passing_cp_size == 1:
        CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    # CB comes from the state-passing branch above when cp_size > 1, and from the
    # branch just above otherwise; the two cases are exhaustive.
    # pylint: disable-next=possibly-used-before-assignment
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)
    return (
        out,
        out_x,
        dt,
        dA_cumsum,
        states,
        final_states,
        state_passing_initial_states,
        state_passing_gathered_decays,
    )


def _mamba_chunk_scan_combined_state_passing_cp_bwd(
    dout,
    x,
    dt,
    A,
    B,
    C,
    out,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dfinal_states=None,
    seq_idx=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    dx=None,
    ddt=None,
    dB=None,
    dC=None,
    dz=None,
    recompute_output=False,
    state_passing_cp_group=None,
    state_passing_initial_states=None,
    state_passing_gathered_decays=None,
    state_passing_cp_virtual=False,
):
    """Mamba combined backward with a state-passing CP reverse-boundary handoff."""
    assert state_passing_cp_group is not None
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)

    assert (
        initial_states is None
    ), "external initial_states are not implemented for state-passing CP"
    assert dfinal_states is None, "dfinal_states are not implemented for state-passing CP"
    assert seq_idx is None, "seq_idx is not implemented for state-passing CP"
    state_passing_cp_size = dist.get_world_size(state_passing_cp_group)
    rank = dist.get_rank(state_passing_cp_group)

    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt_in, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit
    )
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, _ = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=(
            rearrange(state_passing_initial_states, "... p n -> ... (p n)")
            if state_passing_initial_states is not None
            else None
        ),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
    )
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)

    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(
            x,
            z,
            out,
            dout,
            chunk_size=chunk_size,
            has_ddAcs=False,
            D=D,
            dz=dz,
            recompute_output=recompute_output,
        )
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out

    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)

    # Everything in this block is state-passing CP-specific. The local kernels run while the
    # reverse-boundary all-gather is in flight.
    dC_local = dCB = ddA = None
    final_state_grad = None
    if state_passing_cp_size > 1:
        assert (
            state_passing_initial_states is not None and state_passing_gathered_decays is not None
        )
        initial_state_grad = _state_passing_summary_bwd(
            rearrange(dstates, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1]
        )
        initial_state_grad = rearrange(initial_state_grad, "... (p n) -> ... p n", n=dstate)
        gathered_initial_state_grads, gather_work = _all_gather_stack(
            initial_state_grad, state_passing_cp_group, async_op=True
        )

        states_for_scan = states.to(x.dtype)
        dC_local, ddA_cumsum_prev = _chunk_scan_bwd_dC(
            states_for_scan, dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups
        )
        del states_for_scan
        dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups).to(
            CB.dtype
        )
        ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
        gather_work.wait()
        final_state_grad = _state_passing_boundary_scan(
            gathered_initial_state_grads,
            state_passing_gathered_decays,
            rank,
            state_passing_cp_virtual=state_passing_cp_virtual,
            reverse=True,
        )

    state_bwd_outputs = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=(
            rearrange(final_state_grad, "... p n -> ... (p n)")
            if final_state_grad is not None
            else None
        ),
        seq_idx=seq_idx,
        has_initial_states=state_passing_initial_states is not None,
        dstates_dtype=x.dtype,
        # dC already consumed the converted states in the overlapped state-passing CP path.
        states_dtype=x.dtype if dC_local is None else None,
        chunk_size=chunk_size,
    )
    dstates, ddA_chunk_cumsum, _ = state_bwd_outputs[:3]
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)

    if dC_local is None:
        states = rearrange(state_bwd_outputs[3], "... (p n) -> ... p n", n=dstate)
        dC_local, ddA_cumsum_prev = _chunk_scan_bwd_dC(
            states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups
        )
        dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups).to(
            CB.dtype
        )
        ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)

    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(
        x, dt, dA_cumsum, B, CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx
    )
    dB, ddA_next = _chunk_state_bwd_db(
        x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups
    )
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC_local, out=dC_given)
    if z is None:
        dD = dD_from_x

    # ddA_cumsum_prev is set by whichever of the two exhaustive dC_local branches
    # ran, and dD by whichever of the two exhaustive `z` branches ran.
    # pylint: disable-next=possibly-used-before-assignment
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    ddA += ddA_next + ddA_prev
    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(
        ddA,
        ddt,
        dt_in,
        A,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        ddt=ddt_given,
    )

    dinitial_states = None
    # pylint: disable-next=possibly-used-before-assignment
    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz, ddt_bias, dinitial_states)
    return return_vals if not recompute_output else (*return_vals, outz)


# Mixer integration, conv boundary exchange, and state-passing CP layout handling.
def _pack_state_passing_cp_virtual_segments(x: torch.Tensor) -> torch.Tensor:
    """View balanced chunks as interleaved front/back segments on the batch axis."""
    assert x.shape[1] % 2 == 0, "virtual state-passing CP requires two equal local sequence chunks"
    batch = x.shape[0]
    half = x.shape[1] // 2
    return x.reshape(2 * batch, half, *x.shape[2:])


def _unpack_state_passing_cp_virtual_segments(x: torch.Tensor) -> torch.Tensor:
    """View interleaved front/back segments as the balanced local sequence layout."""
    assert x.shape[0] % 2 == 0
    batch = x.shape[0] // 2
    return x.reshape(batch, 2 * x.shape[1], *x.shape[2:])


def _state_passing_cp_chunk_owner_slot(
    chunk_id: int, state_passing_cp_size: int, layout: str
) -> Tuple[int, int]:
    if layout == "contiguous":
        return chunk_id // 2, chunk_id % 2
    if chunk_id < state_passing_cp_size:
        return chunk_id, 0
    return 2 * state_passing_cp_size - 1 - chunk_id, 1


def _state_passing_cp_local_chunk_ids(
    rank: int, state_passing_cp_size: int, layout: str
) -> Tuple[int, int]:
    if layout == "contiguous":
        return 2 * rank, 2 * rank + 1
    return rank, 2 * state_passing_cp_size - 1 - rank


def permute_state_passing_cp_sequence_chunks(
    x: torch.Tensor,
    state_passing_cp_group: torch.distributed.ProcessGroup,
    undo_load_balancing: bool,
    backend: str = "p2p",
) -> torch.Tensor:
    """Exchange sequence chunks between the balanced and contiguous CP layouts.

    ``undo_load_balancing=True`` converts Megatron's balanced front/back layout
    into a contiguous causal shard; ``False`` restores the balanced layout. The
    backward pass applies the inverse permutation.

    This mirrors the zigzag/contiguous conversion in
    ``megatron.core.context_parallel_layout``; see this module's docstring for why
    it is currently implemented separately.
    """
    if dist.get_world_size(state_passing_cp_group) == 1:
        return x
    if backend not in ("p2p", "a2a"):
        raise ValueError(f"Unsupported state-passing CP sequence permutation backend {backend!r}")
    return _StatePassingCPSequenceChunkPermutationFn.apply(
        x, state_passing_cp_group, undo_load_balancing, backend
    )


def undo_state_passing_cp_load_balancing(
    x: torch.Tensor, state_passing_cp_group: torch.distributed.ProcessGroup, backend: str = "p2p"
) -> torch.Tensor:
    """Convert a balanced CP shard into a contiguous causal shard."""
    return permute_state_passing_cp_sequence_chunks(x, state_passing_cp_group, True, backend)


def redo_state_passing_cp_load_balancing(
    x: torch.Tensor, state_passing_cp_group: torch.distributed.ProcessGroup, backend: str = "p2p"
) -> torch.Tensor:
    """Convert a contiguous causal shard back into the balanced CP layout."""
    return permute_state_passing_cp_sequence_chunks(x, state_passing_cp_group, False, backend)


class _StatePassingCPSequenceChunkPermutationFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        state_passing_cp_group: torch.distributed.ProcessGroup,
        undo_load_balancing: bool,
        backend: str,
    ):
        ctx.state_passing_cp_group = state_passing_cp_group
        ctx.undo_load_balancing = undo_load_balancing
        ctx.backend = backend
        return _permute_state_passing_cp_sequence_chunks(
            x, state_passing_cp_group, undo_load_balancing, backend
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = _permute_state_passing_cp_sequence_chunks(
            grad_output, ctx.state_passing_cp_group, not ctx.undo_load_balancing, ctx.backend
        )
        return grad_x, None, None, None


def _permute_state_passing_cp_sequence_chunks(
    x: torch.Tensor,
    state_passing_cp_group: torch.distributed.ProcessGroup,
    undo_load_balancing: bool,
    backend: str = "p2p",
) -> torch.Tensor:
    if backend == "a2a":
        return _permute_state_passing_cp_sequence_chunks_a2a(
            x, state_passing_cp_group, undo_load_balancing
        )
    if backend != "p2p":
        raise ValueError(f"Unsupported state-passing CP sequence permutation backend {backend!r}")
    return _permute_state_passing_cp_sequence_chunks_p2p(
        x, state_passing_cp_group, undo_load_balancing
    )


def _permute_state_passing_cp_sequence_chunks_p2p(
    x: torch.Tensor,
    state_passing_cp_group: torch.distributed.ProcessGroup,
    undo_load_balancing: bool,
) -> torch.Tensor:
    state_passing_cp_size = dist.get_world_size(state_passing_cp_group)
    rank = dist.get_rank(state_passing_cp_group)
    assert (
        x.size(0) % 2 == 0
    ), "State-passing CP load-balanced sequence shard must contain two chunks"
    chunk_len = x.size(0) // 2
    input_layout = "balanced" if undo_load_balancing else "contiguous"
    output_layout = "contiguous" if undo_load_balancing else "balanced"
    input_chunks = _state_passing_cp_local_chunk_ids(rank, state_passing_cp_size, input_layout)
    output_chunks = _state_passing_cp_local_chunk_ids(rank, state_passing_cp_size, output_layout)

    # P2P receive buffers must be non-overlapping and dense. In backward,
    # grad_output can inherit a non-standard stride from the downstream layout.
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    ops = []
    sends = []

    for out_slot, chunk_id in enumerate(output_chunks):
        src_rank, src_slot = _state_passing_cp_chunk_owner_slot(
            chunk_id, state_passing_cp_size, input_layout
        )
        out_slice = out[out_slot * chunk_len : (out_slot + 1) * chunk_len]
        if src_rank == rank:
            src = x[src_slot * chunk_len : (src_slot + 1) * chunk_len]
            out_slice.copy_(src)
        else:
            ops.append(
                dist.P2POp(dist.irecv, out_slice, group=state_passing_cp_group, group_peer=src_rank)
            )

    for in_slot, chunk_id in enumerate(input_chunks):
        dst_rank, _ = _state_passing_cp_chunk_owner_slot(
            chunk_id, state_passing_cp_size, output_layout
        )
        if dst_rank != rank:
            send = x[in_slot * chunk_len : (in_slot + 1) * chunk_len].contiguous()
            sends.append(send)
            ops.append(
                dist.P2POp(dist.isend, send, group=state_passing_cp_group, group_peer=dst_rank)
            )

    for work in dist.batch_isend_irecv(ops):
        work.wait()
    return out


def _permute_state_passing_cp_sequence_chunks_a2a(
    x: torch.Tensor,
    state_passing_cp_group: torch.distributed.ProcessGroup,
    undo_load_balancing: bool,
) -> torch.Tensor:
    state_passing_cp_size = dist.get_world_size(state_passing_cp_group)
    rank = dist.get_rank(state_passing_cp_group)
    assert (
        x.size(0) % 2 == 0
    ), "State-passing CP load-balanced sequence shard must contain two chunks"
    chunk_len = x.size(0) // 2
    input_layout = "balanced" if undo_load_balancing else "contiguous"
    output_layout = "contiguous" if undo_load_balancing else "balanced"
    input_chunks = _state_passing_cp_local_chunk_ids(rank, state_passing_cp_size, input_layout)

    local_destinations = [
        _state_passing_cp_chunk_owner_slot(chunk_id, state_passing_cp_size, output_layout)
        for chunk_id in input_chunks
    ]
    send_slot_order = sorted(range(2), key=lambda slot: local_destinations[slot])
    local_chunks = (x[:chunk_len], x[chunk_len:])
    send_buf = torch.cat([local_chunks[slot] for slot in send_slot_order], dim=0).contiguous()

    input_split_chunks = [0] * state_passing_cp_size
    for dst_rank, _ in local_destinations:
        input_split_chunks[dst_rank] += 1

    output_split_chunks = [0] * state_passing_cp_size
    recv_slots_by_source = [[] for _ in range(state_passing_cp_size)]
    for src_rank in range(state_passing_cp_size):
        src_chunks = _state_passing_cp_local_chunk_ids(
            src_rank, state_passing_cp_size, input_layout
        )
        src_destinations = [
            _state_passing_cp_chunk_owner_slot(chunk_id, state_passing_cp_size, output_layout)
            for chunk_id in src_chunks
        ]
        for src_slot in sorted(range(2), key=lambda slot: src_destinations[slot]):
            dst_rank, dst_slot = src_destinations[src_slot]
            if dst_rank == rank:
                output_split_chunks[src_rank] += 1
                recv_slots_by_source[src_rank].append(dst_slot)

    input_split_sizes = [count * chunk_len for count in input_split_chunks]
    output_split_sizes = [count * chunk_len for count in output_split_chunks]
    recv_buf = torch.empty_like(x, memory_format=torch.contiguous_format)
    dist.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=state_passing_cp_group,
    )

    target_slots: List[Optional[torch.Tensor]] = [None, None]
    offset = 0
    for src_rank in range(state_passing_cp_size):
        for dst_slot in recv_slots_by_source[src_rank]:
            target_slots[dst_slot] = recv_buf[offset : offset + chunk_len]
            offset += chunk_len
    assert all(
        slot is not None for slot in target_slots
    ), "Incomplete state-passing CP sequence A2A reassembly"
    return torch.cat(target_slots, dim=0)


@triton.jit
def _route_virtual_state_passing_cp_conv_boundary_kernel(
    gathered,
    output,
    batch: tl.constexpr,
    channels: tl.constexpr,
    halo: tl.constexpr,
    rank: tl.constexpr,
    world: tl.constexpr,
    stride_g_rank: tl.constexpr,
    stride_g_batch: tl.constexpr,
    stride_g_channel: tl.constexpr,
    stride_g_halo: tl.constexpr,
    stride_o_batch: tl.constexpr,
    stride_o_channel: tl.constexpr,
    stride_o_halo: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    slot = pid_b % 2
    local_batch = pid_b // 2

    if not REVERSE:
        if slot == 0:
            valid = rank > 0
            source_rank = rank - 1
            source_batch = 2 * local_batch
        else:
            valid = True
            source_rank = rank + 1
            source_batch = 2 * local_batch + 1
            if rank == world - 1:
                source_rank = rank
                source_batch = 2 * local_batch
    else:
        if slot == 0:
            valid = True
            source_rank = rank + 1
            source_batch = 2 * local_batch
            if rank == world - 1:
                source_rank = rank
                source_batch = 2 * local_batch + 1
        else:
            valid = rank > 0
            source_rank = rank - 1
            source_batch = 2 * local_batch + 1

    offsets_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offsets_h = tl.arange(0, BLOCK_H)
    mask = (offsets_c[:, None] < channels) & (offsets_h[None, :] < halo)
    values = tl.load(
        gathered
        + source_rank * stride_g_rank
        + source_batch * stride_g_batch
        + offsets_c[:, None] * stride_g_channel
        + offsets_h[None, :] * stride_g_halo,
        mask=mask & valid,
        other=0.0,
    )
    tl.store(
        output
        + pid_b * stride_o_batch
        + offsets_c[:, None] * stride_o_channel
        + offsets_h[None, :] * stride_o_halo,
        values,
        mask=mask,
    )


def _route_virtual_state_passing_cp_conv_boundary(
    gathered: torch.Tensor,
    batch: int,
    rank: int,
    world: int,
    reverse: bool,
    channel_last_output: bool,
) -> torch.Tensor:
    channels, halo = gathered.shape[-2:]
    if channel_last_output:
        output = torch.empty(
            2 * batch, halo, channels, device=gathered.device, dtype=gathered.dtype
        ).transpose(1, 2)
    else:
        output = torch.empty(
            2 * batch, channels, halo, device=gathered.device, dtype=gathered.dtype
        )
    block_c = 128
    block_h = triton.next_power_of_2(halo)
    _route_virtual_state_passing_cp_conv_boundary_kernel[
        (2 * batch, triton.cdiv(channels, block_c))
    ](
        gathered,
        output,
        batch=batch,
        channels=channels,
        halo=halo,
        rank=rank,
        world=world,
        stride_g_rank=gathered.stride(0),
        stride_g_batch=gathered.stride(1),
        stride_g_channel=gathered.stride(2),
        stride_g_halo=gathered.stride(3),
        stride_o_batch=output.stride(0),
        stride_o_channel=output.stride(1),
        stride_o_halo=output.stride(2),
        REVERSE=reverse,
        BLOCK_C=block_c,
        BLOCK_H=block_h,
    )
    return output


def _gather_causal_conv1d_state_passing_cp_boundary(
    boundary: torch.Tensor,
    state_passing_cp_group: torch.distributed.ProcessGroup,
    state_passing_cp_virtual: bool,
    reverse: bool,
    channel_last_output: bool,
) -> torch.Tensor:
    """Gather a predecessor state or successor gradient for causal conv."""
    world = dist.get_world_size(state_passing_cp_group)
    rank = dist.get_rank(state_passing_cp_group)
    batch, channels, halo = boundary.shape
    gathered = boundary.new_empty(world * batch, channels, halo)
    dist.all_gather_into_tensor(gathered, boundary.contiguous(), group=state_passing_cp_group)
    gathered = gathered.view(world, batch, channels, halo)

    if state_passing_cp_virtual:
        assert batch % 2 == 0
        return _route_virtual_state_passing_cp_conv_boundary(
            gathered,
            batch // 2,
            rank,
            world,
            reverse=reverse,
            channel_last_output=channel_last_output,
        )

    if channel_last_output:
        output = torch.empty(
            batch, halo, channels, device=boundary.device, dtype=boundary.dtype
        ).transpose(1, 2)
    else:
        output = torch.empty_like(boundary, memory_format=torch.contiguous_format)
    source_rank = rank + 1 if reverse else rank - 1
    if 0 <= source_rank < world:
        output.copy_(gathered[source_rank])
    else:
        output.zero_()
    return output


def _causal_conv1d_state_passing_cp_fwd(
    x,
    weight,
    bias,
    seq_idx=None,
    initial_states=None,
    final_states_out=None,
    activation=None,
    state_passing_cp_group=None,
    state_passing_cp_virtual=False,
):
    """Run causal Conv1d using a gathered state-passing CP predecessor state."""
    assert causal_conv1d_fwd_function is not None
    assert state_passing_cp_group is not None
    assert seq_idx is None, "seq_idx is not implemented for state-passing CP"
    assert final_states_out is None
    halo = weight.shape[-1] - 1
    assert halo > 0 and x.size(-1) >= halo
    state_passing_conv_initial_states = initial_states
    if state_passing_conv_initial_states is None:
        state_passing_conv_initial_states = _gather_causal_conv1d_state_passing_cp_boundary(
            x[..., -halo:],
            state_passing_cp_group,
            state_passing_cp_virtual=state_passing_cp_virtual,
            reverse=False,
            channel_last_output=True,
        )
    out = causal_conv1d_fwd_function(
        x,
        weight,
        bias,
        seq_idx,
        state_passing_conv_initial_states,
        final_states_out,
        activation in ["silu", "swish"],
    )
    return out, state_passing_conv_initial_states


def _causal_conv1d_state_passing_cp_bwd(
    x,
    weight,
    bias,
    dout,
    seq_idx=None,
    initial_states=None,
    dfinal_states=None,
    dx=None,
    return_dinitial_states=False,
    activation=None,
    state_passing_cp_group=None,
    state_passing_cp_virtual=False,
):
    """Run causal Conv1d backward and route its state-passing CP halo gradient."""
    assert causal_conv1d_bwd_function is not None
    assert state_passing_cp_group is not None
    assert seq_idx is None, "seq_idx is not implemented for state-passing CP"
    assert dfinal_states is None
    assert not return_dinitial_states, "the state-passing boundary gradient is consumed internally"
    if dout.stride(2) != 1 and dout.stride(1) != 1:
        dout = dout.contiguous()
    dx, dweight, dbias, dinitial_states = causal_conv1d_bwd_function(
        x,
        weight,
        bias,
        dout,
        seq_idx,
        initial_states,
        dfinal_states,
        dx,
        True,
        activation in ["silu", "swish"],
    )
    halo = weight.shape[-1] - 1
    grad_tail = _gather_causal_conv1d_state_passing_cp_boundary(
        dinitial_states,
        state_passing_cp_group,
        state_passing_cp_virtual=state_passing_cp_virtual,
        reverse=True,
        channel_last_output=False,
    )
    dx[..., -halo:].add_(grad_tail)
    return dx, dweight, dbias


class MambaSplitConv1dScanCombinedStatePassingCPFn(torch.autograd.Function):
    """State-passing CP counterpart of ``MambaSplitConv1dScanCombinedFn``."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        initial_states=None,
        seq_idx=None,
        dt_limit=(0.0, float("inf")),
        return_final_states=False,
        activation="silu",
        rmsnorm_weight=None,
        rmsnorm_eps=1e-6,
        outproj_weight=None,
        outproj_bias=None,
        headdim=None,
        ngroups=1,
        norm_before_gate=True,
        state_passing_cp_group=None,
        state_passing_cp_virtual=False,
    ):
        assert activation in [None, "silu", "swish"]
        assert state_passing_cp_group is not None
        assert (
            initial_states is None
        ), "external initial_states are not implemented for state-passing CP"
        assert seq_idx is None, "seq_idx packed input is not implemented for state-passing CP"
        assert (
            not return_final_states
        ), "return_final_states is not implemented for state-passing CP"
        assert rmsnorm_weight is None, "fused RMSNorm is not implemented for state-passing CP"
        assert outproj_weight is None and outproj_bias is None
        if D.dim() == 1:
            assert headdim is not None
            (nheads,) = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert d_nonssm == 0, "non-SSM channels are not implemented for state-passing CP"
        assert zxbcdt.shape == (
            batch,
            seqlen,
            2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads,
        )
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(
            zxbcdt, [2 * d_nonssm, dim, dim + 2 * ngroups * dstate, nheads], dim=-1
        )
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv, state_passing_conv_initial_states = _causal_conv1d_state_passing_cp_fwd(
            rearrange(ensure_stride(xBC), "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            seq_idx,
            None,
            None,
            activation,
            state_passing_cp_group,
            state_passing_cp_virtual,
        )
        xBC_conv = rearrange(xBC_conv, "b d s -> b s d")
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None

        (
            out,
            out_x,
            dt_out,
            dA_cumsum,
            states,
            final_states,
            state_passing_initial_states,
            state_passing_gathered_decays,
        ) = _mamba_chunk_scan_combined_state_passing_cp_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            dt_softplus=True,
            dt_limit=dt_limit,
            state_passing_cp_group=state_passing_cp_group,
            state_passing_cp_virtual=state_passing_cp_virtual,
        )
        out = rearrange(out, "b s h p -> b s (h p)")
        rstd = None

        ctx.save_for_backward(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out_x,
            A,
            D,
            dt_bias,
            initial_states,
            seq_idx,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
            state_passing_conv_initial_states,
            state_passing_initial_states,
            state_passing_gathered_decays,
        )
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        ctx.state_passing_cp_group = state_passing_cp_group
        ctx.state_passing_cp_virtual = state_passing_cp_virtual
        return out if not return_final_states else (out, final_states)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        (
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            out,
            A,
            D,
            dt_bias,
            initial_states,
            seq_idx,
            rmsnorm_weight,
            rstd,
            outproj_weight,
            outproj_bias,
            state_passing_conv_initial_states,
            state_passing_initial_states,
            state_passing_gathered_decays,
        ) = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        assert nheads % ctx.ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert d_nonssm == 0
        recompute_output = outproj_weight is not None

        zx0, z, xBC, dt = torch.split(
            zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1
        )
        # Recompute x, B, C.
        xBC_conv, _ = _causal_conv1d_state_passing_cp_fwd(
            rearrange(ensure_stride(xBC), "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            seq_idx,
            state_passing_conv_initial_states,
            None,
            ctx.activation,
            ctx.state_passing_cp_group,
            ctx.state_passing_cp_virtual,
        )
        xBC_conv = rearrange(xBC_conv, "b d s -> b s d")
        x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)

        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz, dxBC_given, ddt_given = torch.split(
            dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1
        )
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)

        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = (
            _mamba_chunk_scan_combined_state_passing_cp_bwd(
                dout,
                x,
                dt,
                A,
                B,
                C,
                out,
                ctx.chunk_size,
                D=D,
                z=z,
                dt_bias=dt_bias,
                initial_states=initial_states,
                dfinal_states=dfinal_states,
                seq_idx=seq_idx,
                dt_softplus=True,
                dt_limit=ctx.dt_limit,
                dx=dx,
                ddt=ddt_given,
                dB=dB,
                dC=dC,
                dz=dz,
                recompute_output=recompute_output,
                state_passing_cp_group=ctx.state_passing_cp_group,
                state_passing_initial_states=state_passing_initial_states,
                state_passing_gathered_decays=state_passing_gathered_decays,
                state_passing_cp_virtual=ctx.state_passing_cp_virtual,
            )
        )

        dxBC_given_update, dweight, dbias = _causal_conv1d_state_passing_cp_bwd(
            rearrange(ensure_stride(xBC), "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
            rearrange(ensure_stride(dxBC), "b s d -> b d s"),
            seq_idx,
            state_passing_conv_initial_states,
            None,
            rearrange(ensure_stride(dxBC_given), "b s d -> b d s"),
            False,
            ctx.activation,
            ctx.state_passing_cp_group,
            ctx.state_passing_cp_virtual,
        )
        dxBC_given_update = rearrange(dxBC_given_update, "b d s -> b s d")
        if dxBC_given.stride() != dxBC_given_update.stride():
            dxBC_given.copy_(dxBC_given_update)
        else:
            dxBC_given = dxBC_given_update

        drmsnorm_weight = None
        doutproj_weight = None
        doutproj_bias = None
        return (
            dzxbcdt,
            dweight,
            dbias,
            ddt_bias,
            dA,
            dD,
            None,
            dinitial_states,
            None,
            None,
            None,
            None,
            drmsnorm_weight,
            None,
            doutproj_weight,
            doutproj_bias,
            None,
            None,
            None,
            None,
            None,
        )


def mamba_split_conv1d_scan_combined_state_passing_cp(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    seq_idx=None,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
    state_passing_cp_group=None,
    state_passing_cp_virtual=False,
):
    """State-passing CP counterpart of ``mamba_split_conv1d_scan_combined``."""
    return MambaSplitConv1dScanCombinedStatePassingCPFn.apply(
        zxbcdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        chunk_size,
        initial_states,
        seq_idx,
        dt_limit,
        return_final_states,
        activation,
        rmsnorm_weight,
        rmsnorm_eps,
        outproj_weight,
        outproj_bias,
        headdim,
        ngroups,
        norm_before_gate,
        state_passing_cp_group,
        state_passing_cp_virtual,
    )


class MambaStatePassingCPAdapter:
    """Adapter between ``MambaMixer`` and the state-passing CP production path."""

    def __init__(self, mixer):
        self.mixer = mixer

    def forward(self, zxBCdt: torch.Tensor) -> torch.Tensor:
        """Run the fused state-passing Conv+SSD path on a projected activation."""
        mixer = self.mixer
        state_passing_cp_load_balancing = mixer.config.mamba_state_passing_cp_load_balancing
        assert state_passing_cp_load_balancing in ("none", "permute_p2p", "permute_a2a", "virtual")
        if state_passing_cp_load_balancing.startswith("permute_"):
            state_passing_cp_permute_backend = state_passing_cp_load_balancing.removeprefix(
                "permute_"
            )
            zxBCdt = undo_state_passing_cp_load_balancing(
                zxBCdt, mixer.cp.cp_group, backend=state_passing_cp_permute_backend
            )
        zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()
        state_passing_cp_virtual = state_passing_cp_load_balancing == "virtual"
        if state_passing_cp_virtual:
            zxBCdt = _pack_state_passing_cp_virtual_segments(zxBCdt)
        A = -torch.exp(mixer.A_log.float())
        D = (
            rearrange(mixer.D.float(), "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else mixer.D
        )

        assert (
            causal_conv1d_fwd_function is not None and causal_conv1d_bwd_function is not None
        ), "Mamba state-passing CP requires causal-conv1d"
        assert mixer.activation in ["silu", "swish"]
        y = mamba_split_conv1d_scan_combined_state_passing_cp(
            zxBCdt,
            rearrange(mixer.conv1d_weight, "d 1 w -> d w"),
            mixer.conv1d_bias,
            mixer.dt_bias.float(),
            A,
            D,
            mixer.chunk_size,
            activation=mixer.activation,
            headdim=None if mixer.D_has_hdim else mixer.headdim,
            ngroups=mixer.ngroups_local_tp,
            norm_before_gate=mixer.norm_before_gate,
            state_passing_cp_group=mixer.cp.cp_group,
            state_passing_cp_virtual=state_passing_cp_virtual,
        )

        if mixer.rmsnorm:
            # Match MambaMixer's fused path: materialize the BF16 gated scan
            # output before applying RMSNorm.
            y = mixer.norm(y)
        if state_passing_cp_virtual:
            y = _unpack_state_passing_cp_virtual_segments(y)
        y = rearrange(y, "b l d -> l b d").contiguous()
        if state_passing_cp_load_balancing.startswith("permute_"):
            y = redo_state_passing_cp_load_balancing(
                y, mixer.cp.cp_group, backend=state_passing_cp_permute_backend
            )
        return y

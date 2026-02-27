# Copyright (c) 2024, Tri Dao, Albert Gu.

import torch
import triton
import triton.language as tl
from mamba_ssm.ops.triton.softplus import softplus


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {"HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"] is not None}
)
@triton.heuristics({"HAS_INT_STATE": lambda args: args["int_state_ptr"] is not None})
@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    int_state_ptr,
    # Matrix dimensions
    batch,
    seq_len,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_seq,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_seq,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_seq,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_seq,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_seq,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_seq,
    stride_out_head,
    stride_out_dim,
    stride_int_batch,
    stride_int_seq,
    stride_int_head,
    stride_int_dim,
    stride_int_dstate,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    HAS_INT_STATE: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    out_ptrs = out_ptr + offs_m * stride_out_dim

    # 1. State Mapping (handles dynamic batching slot allocation)
    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr)
        # Skip padding tokens (e.g. from graph capture or inactive slots)
        if state_batch_idx < 0:
            for s in range(seq_len):
                out_s_ptrs = out_ptrs + s * stride_out_seq
                tl.store(out_s_ptrs, 0.0, mask=offs_m < dim)
            return
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
        if HAS_INT_STATE:
            int_state_ptr += state_batch_idx * stride_int_batch + pid_h * stride_int_head
    else:
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head
        if HAS_INT_STATE:
            int_state_ptr += pid_b * stride_int_batch + pid_h * stride_int_head

    # Base Pointers for Sequence iteration
    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head

    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head

    # Constant offsets (A, D, and bias do not have a sequence dimension)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    if HAS_INT_STATE:
        int_state_ptrs = int_state_ptr + (
            offs_m[:, None] * stride_int_dim + offs_n[None, :] * stride_int_dstate
        )

    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim

    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate

    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim

    # Load initial historical state and constant parameters
    state = tl.load(
        state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0
    ).to(tl.float32)

    if not TIE_HDIM:
        A = tl.load(
            A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0
        ).to(tl.float32)
    else:
        A = tl.load(A_ptr).to(tl.float32)

    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    # ----------------------------------------------------
    # Sequence Loop (Processes Main Token + Speculative Drafts)
    # ----------------------------------------------------
    for s in range(seq_len):
        x_s_ptrs = x_ptrs + s * stride_x_seq
        dt_s_ptrs = dt_ptrs + s * stride_dt_seq
        B_s_ptrs = B_ptrs + s * stride_B_seq
        C_s_ptrs = C_ptrs + s * stride_C_seq
        if HAS_Z:
            z_s_ptrs = z_ptrs + s * stride_z_seq

        x = tl.load(x_s_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        # Calculate dt and dA
        if not TIE_HDIM:
            dt = tl.load(dt_s_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, softplus(dt), dt)
            dA = tl.exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr + s * stride_dt_seq).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, softplus(dt), dt)
            dA = tl.exp(A * dt)

        # Load B and C
        B = tl.load(B_s_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_s_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_s_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        if not TIE_HDIM:
            dB = B[None, :] * dt[:, None]
        else:
            dB = B * dt

        # ----------------------------------------------------
        # The Core State Recurrence (h_t = dA * h_{t-1} + dB * x_t)
        # ----------------------------------------------------
        state = state * dA + dB * x[:, None]

        # ----------------------------------------------------
        # Dump Intermediate Speculative State Snapshot
        # ----------------------------------------------------
        if HAS_INT_STATE:
            int_state_s_ptrs = int_state_ptrs + s * stride_int_seq
            tl.store(
                int_state_s_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
            )

        # Calculate Output
        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)

        out_s_ptrs = out_ptrs + s * stride_out_seq
        tl.store(out_s_ptrs, out, mask=offs_m < dim)

    # After processing all sequence steps, persist the final state back to HBM
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    intermediate_ssm_states=None,
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim), (batch, seqlen, dim), (batch, nheads, dim) or (batch, seqlen, nheads, dim)
        dt: Matches x
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate), (batch, seqlen, dstate), (batch, ngroups, dstate) or
            (batch, seqlen, ngroups, dstate)
        C: Matches B
        D: (dim,) or (nheads, dim)
        z: Matches x
        dt_bias: (dim,) or (nheads, dim)
        intermediate_ssm_states: Optional buffer of shape (batch, seqlen, nheads, dim, dstate)
                                 or (batch, seqlen, dim, dstate)
    Return:
        out: shape matches x
    """
    has_heads = state.dim() > 3
    if not has_heads:
        state = state.unsqueeze(1)

    # Standardize inputs to explicit sequence and head dimensions: (batch, seq_len, nheads, dim)
    is_seq_unsq = False
    if has_heads:
        if x.dim() == 3:  # (batch, nheads, dim) -> (batch, 1, nheads, dim)
            x = x.unsqueeze(1)
            dt = dt.unsqueeze(1)
            B = B.unsqueeze(1)
            C = C.unsqueeze(1)
            if z is not None:
                z = z.unsqueeze(1)
            is_seq_unsq = True
    else:
        if x.dim() == 2:  # (batch, dim) -> (batch, 1, 1, dim)
            x = x.unsqueeze(1).unsqueeze(2)
            dt = dt.unsqueeze(1).unsqueeze(2)
            B = B.unsqueeze(1).unsqueeze(2)
            C = C.unsqueeze(1).unsqueeze(2)
            if z is not None:
                z = z.unsqueeze(1).unsqueeze(2)
            is_seq_unsq = True
        elif x.dim() == 3:  # (batch, seqlen, dim) -> (batch, seqlen, 1, dim)
            x = x.unsqueeze(2)
            dt = dt.unsqueeze(2)
            B = B.unsqueeze(2)
            C = C.unsqueeze(2)
            if z is not None:
                z = z.unsqueeze(2)

    if A.dim() == 2:
        A = A.unsqueeze(0)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)

    # Set up Intermediate State standardization
    if intermediate_ssm_states is not None:
        if not has_heads and intermediate_ssm_states.dim() == 4:
            intermediate_ssm_states = intermediate_ssm_states.unsqueeze(
                2
            )  # (batch, seqlen, 1, dim, dstate)
        int_state_strides = (
            intermediate_ssm_states.stride(0),
            intermediate_ssm_states.stride(1),
            intermediate_ssm_states.stride(2),
            intermediate_ssm_states.stride(3),
            intermediate_ssm_states.stride(4),
        )
    else:
        intermediate_ssm_states = x  # Dummy pointer
        int_state_strides = (0, 0, 0, 0, 0)

    batch, seq_len, nheads, dim = x.shape
    dstate = state.shape[-1]
    ngroups = B.shape[-2]

    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch, nheads)
    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2), z.stride(3)) if z is not None else (0, 0, 0, 0)
    )

    BLOCK_SIZE_M, num_warps = (
        (32, 4)
        if dstate <= 16
        else (
            (16, 4)
            if dstate <= 32
            else ((8, 4) if dstate <= 64 else ((4, 4) if dstate <= 128 else ((4, 8))))
        )
    )

    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and (dt_bias is None or dt_bias.stride(-1) == 0)
    )

    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            intermediate_ssm_states,
            batch,
            seq_len,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else (0, 0),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
            z_strides[0],
            z_strides[1],
            z_strides[2],
            z_strides[3],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            *int_state_strides,
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )

    # Revert dimensions back to match original x format
    if not has_heads:
        out = out.squeeze(2)
    if is_seq_unsq:
        out = out.squeeze(1)

    return out

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cuDNN Kimi Delta Attention and chunkwise context-parallel orchestration."""

from __future__ import annotations

import inspect
import math
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cudnn.linear_attention.frost.common.piece_chain import build_state_chain, run_state_chain
from cudnn.linear_attention.ops import kimi_delta_attention as _kimi_delta_attention
from cudnn.linear_attention.ops import (
    kimi_delta_attention_summary,
    kimi_delta_attention_summary_bwd,
)
from torch.autograd.function import once_differentiable

_MIN_CUDNN_KDA_COMPUTE_CAPABILITY = (10, 0)

# Importing the public operation above registers these explicit custom ops. The
# distributed autograd wrapper calls them directly so forward does not retain a
# nested autograd graph solely to invoke the local cuDNN backward.
_KDA_FWD = torch.ops.cudnn.kimi_delta_attention_fwd
_KDA_BWD = torch.ops.cudnn.kimi_delta_attention_bwd
_STATE_CHAIN_HAS_REVERSE = "reverse" in inspect.signature(build_state_chain).parameters


class _CudnnKdaBackwardResult(NamedTuple):
    """Gradients matching the positional arguments of ``_CudnnKda.forward``."""

    q: Optional[torch.Tensor]
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    g: Optional[torch.Tensor]
    beta: Optional[torch.Tensor]
    cu_seqlens: None
    first_boundary_end: None
    last_boundary_start: None
    a_log: Optional[torch.Tensor]
    dt_bias: Optional[torch.Tensor]
    cp_group: None
    pre_num_ranks: None
    post_num_ranks: None
    scale: None
    use_qk_l2norm_in_kernel: None
    safe_gate: None
    lower_bound: None


def _require_sm100_or_newer(device: torch.device) -> None:
    """Reject devices on which the selected cuDNN KDA path is unsupported."""
    if device.type != "cuda":
        raise RuntimeError(f"cuDNN KDA requires a CUDA device, got {device.type!r}.")
    capability = torch.cuda.get_device_capability(device)
    if capability < _MIN_CUDNN_KDA_COMPUTE_CAPABILITY:
        major, minor = capability
        raise RuntimeError(
            f"cuDNN KDA requires SM100 or newer, got SM{major}{minor}. "
            "Select kda_backend='fla' on this device."
        )


def _state_chain_plan_kwargs(
    *, pieces: int, heads: int, value_dim: int, key_dim: int, reverse: bool, device: int
) -> dict:
    """Return the cuDNN FROST state-chain plan geometry."""
    kwargs = dict(
        heads_out=heads,
        dim_v=value_dim,
        dim_k=key_dim,
        pieces=pieces,
        rows_per_cta=(16 if (value_dim == 256 and key_dim == 128) else value_dim // 8),
        transpose=reverse,
        has_seed=False,
        has_tail=True,
        emit_summary=False,
        device=device,
        opt_level=3,
    )
    if _STATE_CHAIN_HAS_REVERSE:
        kwargs["reverse"] = reverse
    return kwargs


def _run_state_chain(
    h: torch.Tensor, transition: torch.Tensor, *, reverse: bool = False
) -> torch.Tensor:
    """Compose caller-gathered KDA span summaries with cuDNN FROST."""
    if h.ndim != 4 or transition.ndim != 4 or h.shape[0] == 0:
        raise ValueError("FROST state chain expects non-empty [pieces,H,V,K] summaries")
    pieces, heads, value_dim, key_dim = h.shape
    if key_dim not in (64, 128) or value_dim not in (64, 128, 256):
        raise ValueError("FROST state chain requires K in (64, 128) and V in (64, 128, 256)")
    if transition.shape != (pieces, heads, key_dim, key_dim):
        raise ValueError("FROST transition shape does not match the state summaries")
    if h.dtype != torch.float32 or transition.dtype != torch.float32:
        raise ValueError("FROST state chain requires FP32 summaries")
    if not h.is_cuda or transition.device != h.device:
        raise ValueError("FROST state-chain operands must share one CUDA device")
    if not h.is_contiguous() or not transition.is_contiguous():
        raise ValueError("FROST state-chain operands must be contiguous")

    initial_states = torch.empty_like(h, memory_format=torch.contiguous_format)
    final_state = h.new_empty((1, heads, value_dim, key_dim))
    compiled = build_state_chain(
        **_state_chain_plan_kwargs(
            pieces=pieces,
            heads=heads,
            value_dim=value_dim,
            key_dim=key_dim,
            reverse=reverse,
            device=h.device.index,
        )
    )
    run_state_chain(
        compiled,
        1,
        h,
        transition,
        initial_states,
        None,
        final_state,
        None,
        torch.cuda.current_stream(h.device).cuda_stream,
    )
    return final_state[0]


def _all_gather_boundary_summary(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather one equal-shaped boundary summary from every CP rank."""
    if not tensor.is_contiguous():
        raise ValueError("KDA boundary summary must be contiguous before all-gather")
    world_size = dist.get_world_size(group=group)
    gathered = torch.empty(
        (world_size * tensor.shape[0], *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
    )
    dist.all_gather_into_tensor(gathered, tensor, group=group)
    return gathered.view(world_size, *tensor.shape)


def _pack_boundary_summary(h: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
    """Pack H and transition so one collective carries both tensors."""
    if h.ndim != 3 or transition.ndim != 3:
        raise ValueError("KDA boundary summaries must have [H,D,K] layout")
    if h.shape[0] != transition.shape[0] or h.shape[-1] != transition.shape[-1]:
        raise ValueError("KDA H and transition boundary shapes are incompatible")
    return torch.cat((h, transition), dim=-2)


def _unpack_boundary_summaries(
    packed: torch.Tensor, value_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return packed[..., :value_dim, :], packed[..., value_dim:, :]


def _merge_boundary_chain(
    h_all: torch.Tensor,
    transition_all: torch.Tensor,
    *,
    rank: int,
    num_remote_ranks: int,
    reverse: bool,
) -> torch.Tensor:
    """Return the state entering or leaving this rank's split sequence."""
    if num_remote_ranks == 0:
        return torch.zeros_like(h_all[rank])
    if reverse:
        start, stop = rank + 1, rank + num_remote_ranks + 1
    else:
        start, stop = rank - num_remote_ranks, rank
    # Slicing the packed collective returns strided views. These are the two
    # intentional D2D materializations required by the FROST state chain.
    return _run_state_chain(
        h_all[start:stop].contiguous(), transition_all[start:stop].contiguous(), reverse=reverse
    )


def _op_kwargs(
    *,
    use_qk_l2norm_in_kernel: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
    a_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
) -> dict:
    return {
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "use_beta_sigmoid_in_kernel": False,
        "safe_gate": safe_gate,
        "gate_lower_bound": lower_bound if safe_gate else None,
        "gate_domain": "log",
        "a_log": a_log if safe_gate else None,
        "dt_bias": dt_bias if safe_gate else None,
    }


def _flatten_bthd(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten BTHD to THD, copying only when the source is not contiguous."""
    if tensor.is_contiguous():
        return tensor.view(-1, *tensor.shape[2:])
    return tensor.reshape(-1, *tensor.shape[2:]).contiguous()


def _prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
    a_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Flatten BTHD inputs and normalize FLA's gate contract for cuDNN."""
    if any(tensor.ndim != 4 for tensor in (q, k, v, g)) or beta.ndim != 3:
        raise ValueError("cuDNN KDA expects BTHD Q/K/V/G and BTH beta")
    if any(tensor.shape[:3] != q.shape[:3] for tensor in (k, v, g)):
        raise ValueError("cuDNN KDA inputs must share batch, token, and head dimensions")
    if beta.shape != q.shape[:3]:
        raise ValueError("cuDNN KDA beta must match Q's batch, token, and head dimensions")

    if not use_gate_in_kernel:
        a_log = None
        dt_bias = None
    elif not safe_gate:
        if a_log is None:
            raise ValueError("non-safe in-kernel KDA gate requires A_log")
        gate = g.float()
        if dt_bias is not None:
            gate = gate + dt_bias.float().reshape(g.shape[-2:])
        g = -a_log.float().view(1, 1, -1, 1).exp() * F.softplus(gate)
        a_log = None
        dt_bias = None
    else:
        if a_log is None:
            raise ValueError("safe in-kernel KDA gate requires A_log")
        if lower_bound is None or not -5.0 <= lower_bound < 0.0:
            raise ValueError("safe in-kernel KDA gate requires lower_bound in [-5, 0)")
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(g.shape[-2:])

    tensors = [_flatten_bthd(tensor) for tensor in (q, k, v, g)]
    beta_thd = _flatten_bthd(beta.unsqueeze(-1)).squeeze(-1)
    return tensors, beta_thd, a_log, dt_bias


def _boundary_offsets(
    cu_seqlens_cpu: torch.Tensor, local_tokens: int, expected_entries: int
) -> tuple[int, int]:
    """Return first-sequence end and last-sequence start without a CUDA sync."""
    if cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.ndim != 1:
        raise ValueError("cuDNN KDA CP requires one-dimensional CPU sequence boundaries")
    if cu_seqlens_cpu.dtype not in (torch.int32, torch.int64):
        raise ValueError("cuDNN KDA CP sequence boundaries must use int32 or int64")
    if cu_seqlens_cpu.numel() != expected_entries:
        raise ValueError("CPU and CUDA sequence boundaries must have the same shape")
    values = cu_seqlens_cpu.tolist()
    if len(values) < 2 or values[0] != 0 or values[-1] != local_tokens:
        raise ValueError("CPU sequence boundaries must span all local tokens")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("CPU sequence boundaries must be strictly increasing")
    return int(values[1]), int(values[-2])


def _compute_forward_cp_state(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    last_boundary_start: int,
    a_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    pre_num_ranks: int,
    use_qk_l2norm_in_kernel: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Build the recurrent state entering this rank's first local sequence."""
    last_cu = cu_seqlens[-2:]
    if last_boundary_start:
        last_cu = last_cu - last_boundary_start
    h_local, transition_local = kimi_delta_attention_summary(
        k[last_boundary_start:],
        v[last_boundary_start:],
        g[last_boundary_start:],
        beta[last_boundary_start:],
        last_cu,
        plan_name="kda_summary_frost",
        **_op_kwargs(
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
        ),
    )
    value_dim = h_local.shape[-2]
    packed_all = _all_gather_boundary_summary(
        _pack_boundary_summary(h_local[-1], transition_local[-1]), cp_group
    )
    h_all, transition_all = _unpack_boundary_summaries(packed_all, value_dim)
    incoming_boundary = _merge_boundary_chain(
        h_all, transition_all, rank=cp_rank, num_remote_ranks=pre_num_ranks, reverse=False
    )
    incoming = h_local.new_zeros(
        cu_seqlens.numel() - 1, h_local.shape[1], h_local.shape[2], h_local.shape[3]
    )
    incoming[0].copy_(incoming_boundary)
    return incoming


def _compute_backward_cp_state_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    first_boundary_end: int,
    a_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    post_num_ranks: int,
    scale: float,
    use_qk_l2norm_in_kernel: bool,
    safe_gate: bool,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Build the final-state gradient supplied to the local KDA backward."""
    g_local, transition_backward = kimi_delta_attention_summary_bwd(
        q[:first_boundary_end],
        k[:first_boundary_end],
        g[:first_boundary_end],
        beta[:first_boundary_end],
        grad_output[:first_boundary_end],
        cu_seqlens[:2],
        output_transition=True,
        plan_name="kda_summary_frost",
        scale=scale,
        **_op_kwargs(
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
        ),
    )
    # summary_bwd exposes M^T. The reverse FROST merge expects forward M and
    # applies its own transpose while composing.
    transition_first = transition_backward[0].transpose(-2, -1)
    packed_all = _all_gather_boundary_summary(
        _pack_boundary_summary(g_local[0], transition_first), cp_group
    )
    g_all, transition_all = _unpack_boundary_summaries(packed_all, g_local.shape[-2])
    outgoing_boundary = _merge_boundary_chain(
        g_all, transition_all, rank=cp_rank, num_remote_ranks=post_num_ranks, reverse=True
    )
    grad_final_state = g_local.new_zeros(
        cu_seqlens.numel() - 1, g_local.shape[1], g_local.shape[2], g_local.shape[3]
    )
    grad_final_state[-1].copy_(outgoing_boundary)
    return grad_final_state


class _CudnnKda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        first_boundary_end: Optional[int],
        last_boundary_start: Optional[int],
        a_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
        cp_group: Optional[dist.ProcessGroup],
        pre_num_ranks: int,
        post_num_ranks: int,
        scale: float,
        use_qk_l2norm_in_kernel: bool,
        safe_gate: bool,
        lower_bound: Optional[float],
    ) -> torch.Tensor:
        """Run local cuDNN KDA after composing any state entering this CP rank."""
        if q.ndim != 3 or any(tensor.ndim != 3 for tensor in (k, v, g)) or beta.ndim != 2:
            raise ValueError("cuDNN KDA expects compact THD tensors")
        if cu_seqlens.ndim != 1:
            raise ValueError("cuDNN KDA expects one-dimensional sequence boundaries")

        ctx.local_only = cp_group is None
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.a_log_present = a_log is not None
        ctx.dt_bias_present = dt_bias is not None

        if ctx.local_only:
            incoming = None
        else:
            if first_boundary_end is None or last_boundary_start is None:
                raise RuntimeError("cuDNN KDA CP requires local CPU sequence boundaries")
            ctx.cp_group = cp_group
            ctx.cp_rank = dist.get_rank(group=cp_group)
            ctx.post_num_ranks = post_num_ranks
            ctx.first_boundary_end = first_boundary_end
            incoming = _compute_forward_cp_state(
                k,
                v,
                g,
                beta,
                cu_seqlens,
                last_boundary_start=last_boundary_start,
                a_log=a_log,
                dt_bias=dt_bias,
                cp_group=cp_group,
                cp_rank=ctx.cp_rank,
                pre_num_ranks=pre_num_ranks,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )

        ctx.scale = scale
        output = _KDA_FWD(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            ctx.scale,
            initial_state=incoming,
            output_final_state=False,
            plan_name="kda_frost",
            **_op_kwargs(
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
            ),
        )[0]

        ctx.incoming_present = incoming is not None
        empty = q.new_empty(0)
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            incoming if incoming is not None else empty,
            a_log if a_log is not None else empty,
            dt_bias if dt_bias is not None else empty,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        """Run cuDNN KDA backward after composing any state gradient from later CP ranks."""
        q, k, v, g, beta, cu_seqlens, incoming, a_log, dt_bias = ctx.saved_tensors
        incoming = incoming if ctx.incoming_present else None
        a_log = a_log if ctx.a_log_present else None
        dt_bias = dt_bias if ctx.dt_bias_present else None
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        if ctx.local_only:
            grad_final_state = None
        else:
            grad_final_state = _compute_backward_cp_state_grad(
                q,
                k,
                g,
                beta,
                grad_output,
                cu_seqlens,
                first_boundary_end=ctx.first_boundary_end,
                a_log=a_log,
                dt_bias=dt_bias,
                cp_group=ctx.cp_group,
                cp_rank=ctx.cp_rank,
                post_num_ranks=ctx.post_num_ranks,
                scale=ctx.scale,
                use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
            )

        dq, dk, dv, dg, dbeta, _, da_log, ddt_bias = _KDA_BWD(
            grad_output,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            ctx.scale,
            initial_state=incoming,
            d_final_state=grad_final_state,
            state_checkpoints=None,
            checkpoint_every_n_tokens=0,
            plan_name="kda_frost",
            **_op_kwargs(
                use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
                a_log=a_log,
                dt_bias=dt_bias,
            ),
        )
        return _CudnnKdaBackwardResult(
            q=dq,
            k=dk,
            v=dv,
            g=dg,
            beta=dbeta,
            cu_seqlens=None,
            first_boundary_end=None,
            last_boundary_start=None,
            a_log=da_log if ctx.a_log_present else None,
            dt_bias=ddt_bias if ctx.dt_bias_present else None,
            cp_group=None,
            pre_num_ranks=None,
            post_num_ranks=None,
            scale=None,
            use_qk_l2norm_in_kernel=None,
            safe_gate=None,
            lower_bound=None,
        )


def cudnn_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    cp_context=None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run cuDNN KDA with the public FLA ``chunk_kda`` call contract."""
    supported_kwargs = {"A_log", "dt_bias", "return_intermediate_states"}
    unsupported_kwargs = sorted(set(kwargs) - supported_kwargs)
    if unsupported_kwargs:
        raise TypeError(f"cuDNN KDA does not support keyword arguments {unsupported_kwargs}")
    if initial_state is not None or output_final_state:
        raise NotImplementedError("cuDNN KDA training does not expose recurrent state outputs")
    if kwargs.get("return_intermediate_states", False):
        raise NotImplementedError("cuDNN KDA training does not expose intermediate states")

    _require_sm100_or_newer(q.device)
    original_output_shape = v.shape
    batch, tokens = q.shape[:2]
    a_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    if cp_context is None:
        cp_group = None
        pre_num_ranks = post_num_ranks = 0
        first_boundary_end = last_boundary_start = None
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch + 1) * tokens, tokens, dtype=torch.int32, device=q.device
            )
        elif batch != 1:
            raise ValueError("packed cuDNN KDA requires batch size one")
    else:
        if batch != 1:
            raise ValueError("cuDNN KDA context parallelism requires packed BTHD input")
        cp_group = cp_context.group
        cu_seqlens = cp_context.cu_seqlens
        if cp_group is None or cu_seqlens is None:
            raise ValueError("cuDNN KDA context parallelism requires a group and boundaries")
        rank = dist.get_rank(group=cp_group)
        world_size = dist.get_world_size(group=cp_group)
        pre_num_ranks = cp_context.pre_num_ranks
        post_num_ranks = cp_context.post_num_ranks
        if pre_num_ranks is None or not 0 <= pre_num_ranks <= rank:
            raise ValueError(f"invalid pre_num_ranks={pre_num_ranks} for CP rank {rank}")
        if post_num_ranks is None or not 0 <= post_num_ranks < world_size - rank:
            raise ValueError(f"invalid post_num_ranks={post_num_ranks} for CP rank {rank}")
        if cp_context.cu_seqlens_cpu is None:
            raise ValueError(
                "cuDNN KDA context parallelism requires cp_context.cu_seqlens_cpu "
                "to avoid a device-to-host synchronization"
            )
        first_boundary_end, last_boundary_start = _boundary_offsets(
            cp_context.cu_seqlens_cpu, tokens, cu_seqlens.numel()
        )

    tensors, beta_thd, a_log, dt_bias = _prepare_inputs(
        q,
        k,
        v,
        g,
        beta,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    if not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()
    output = _CudnnKda.apply(
        *tensors,
        beta_thd,
        cu_seqlens,
        first_boundary_end,
        last_boundary_start,
        a_log,
        dt_bias,
        cp_group,
        pre_num_ranks,
        post_num_ranks,
        scale,
        use_qk_l2norm_in_kernel,
        safe_gate and use_gate_in_kernel,
        lower_bound,
    )
    return output.reshape(original_output_shape), None

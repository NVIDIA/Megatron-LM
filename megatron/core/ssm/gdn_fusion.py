# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in fusion of GDN convolution, normalization, layouts, and gate preparation."""

import logging
import os

import torch
import triton

from megatron.core.ssm.gdn_fusion_kernels import (
    conv_norm_repeat_fwd,
    fold_norm_silu_bwd,
    projection_gates_bwd,
)

try:
    from fla.modules.conv.triton.kernels import causal_conv1d_bwd_kernel
    from fla.ops.utils import prepare_chunk_indices
except ImportError:
    causal_conv1d_bwd_kernel = None
    prepare_chunk_indices = None

_LOGGED = False
# Unwrap FLA's autotuner to use a fixed launch configuration.
_LINEAR_BWD = causal_conv1d_bwd_kernel
while _LINEAR_BWD is not None and not isinstance(_LINEAR_BWD, triton.runtime.JITFunction):
    _LINEAR_BWD = getattr(_LINEAR_BWD, "fn", None)


def enabled(module: torch.nn.Module, projection: torch.Tensor) -> bool:
    """Return whether the opt-in preparation path supports this projection."""
    requested = os.getenv("MCORE_GDN_FUSION", "0") == "1"
    return (
        requested
        and _LINEAR_BWD is not None
        and not module.config.deterministic_mode
        and module.activation in ("silu", "swish")
        and module.use_qk_l2norm
        and module.cp_size == 1
        and module.key_head_dim == 128
        and module.value_head_dim == 128
        and module.num_value_heads // module.num_key_heads == 4
        and projection.shape[0] == 1
        and projection.shape[-1] == 5152
        and projection.is_cuda
        and projection.stride(-1) == 1
        and projection.dtype == torch.bfloat16
        and module.conv1d.weight.shape[-1] == 4
    )


def forward_impl(
    projection: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    alog: torch.Tensor,
    dtbias: torch.Tensor,
    cu: torch.Tensor | None,
    bt: int = 8,
    warps: int = 2,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Prepare Q/K/V and gates, returning inverse norms for first-order backward."""
    batch, length, _ = projection.shape
    total = batch * length
    q = torch.empty((batch, length, 16, 128), device=projection.device, dtype=projection.dtype)
    k, v = torch.empty_like(q), torch.empty_like(q)
    gate = projection[..., 3072:5120].view(batch, length, 16, 128)
    g = torch.empty((batch, length, 16), device=projection.device, dtype=torch.float32)
    beta = torch.empty(g.shape, device=g.device, dtype=projection.dtype)
    rstd = torch.empty((batch, length, 8), device=projection.device, dtype=torch.float32)
    nseq = cu.numel() - 1 if cu is not None else 0
    conv_norm_repeat_fwd[(triton.cdiv(total, bt), 24)](
        projection,
        weight,
        bias if bias is not None else projection,
        cu if cu is not None else projection,
        q,
        k,
        v,
        rstd,
        TOTAL=total,
        LENGTH=length,
        SX=projection.stride(1),
        HK=4,
        HV=16,
        D=128,
        WIDTH=4,
        EPS=1e-6,
        HAS_BIAS=bias is not None,
        NSEQ=nseq,
        SEARCH=(nseq + 1).bit_length(),
        BT=bt,
        gate=gate,
        g=g,
        beta=beta,
        alog=alog,
        dtbias=dtbias,
        FULL_PROJECTION=True,
        COPY_GATE=False,
        num_warps=warps,
        enable_fp_fusion=True,
    )
    return (q, k, v, gate, g, beta), rstd


def linear_conv_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu: torch.Tensor | None,
    dz: torch.Tensor,
    dx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use FLA's existing linear convolution kernel, writing into the projection gradient."""
    batch, length, channels = x.shape
    bt, bd, width = 128, 32, weight.shape[1]
    # A single validated sequence needs no chunk indices.
    if cu is not None and cu.numel() == 2:
        cu = None
    chunks = prepare_chunk_indices(cu, bt) if cu is not None else None
    nt = len(chunks) if chunks is not None else triton.cdiv(length, bt)
    dw = torch.empty((batch * nt, channels, width), device=x.device, dtype=torch.float32)
    db = (
        torch.empty((batch * nt, channels), device=x.device, dtype=torch.float32)
        if bias is not None
        else None
    )
    _LINEAR_BWD[(triton.cdiv(channels, bd), nt, batch)](
        x=x,
        y=None,
        weight=weight,
        initial_state=None,
        dht=None,
        dy=dz,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu,
        chunk_indices=chunks,
        B=batch,
        T=length,
        stride_x_n=x.stride(0),
        stride_x_t=x.stride(1),
        stride_x_d=x.stride(2),
        stride_dx_n=dx.stride(0),
        stride_dx_t=dx.stride(1),
        stride_dx_d=dx.stride(2),
        stride_dy_n=dz.stride(0),
        stride_dy_t=dz.stride(1),
        stride_dy_d=dz.stride(2),
        D=channels,
        W=width,
        BT=bt,
        BW=triton.next_power_of_2(width),
        BD=bd,
        NB=triton.cdiv(batch * length, 1024),
        ACTIVATION=None,
        HAS_WEIGHT=True,
        HAS_BIAS=bias is not None,
        USE_INITIAL_STATE=False,
        USE_FINAL_STATE=False,
        IS_VARLEN=cu is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dw.sum(0).to(weight.dtype), db.sum(0).to(bias.dtype) if db is not None else None


def backward_impl(
    projection: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    alog: torch.Tensor,
    dtbias: torch.Tensor,
    cu: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    rstd: torch.Tensor,
    grads: tuple[torch.Tensor, ...],
    bt: int = 16,
    warps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Fold repeated-head gradients and pack the projection and parameter gradients."""
    batch, length, channels = projection.shape
    total = batch * length
    dq, dk, dv, dgate, dg, dbeta = (grad.contiguous() for grad in grads)
    nseq = cu.numel() - 1 if cu is not None else 0
    dz = torch.empty((batch, length, 3072), device=projection.device, dtype=torch.float32)
    fold_norm_silu_bwd[(triton.cdiv(total, bt), 24)](
        projection,
        weight,
        bias if bias is not None else projection,
        cu if cu is not None else projection,
        dq,
        dk,
        dv,
        q,
        k,
        rstd,
        dz,
        TOTAL=total,
        LENGTH=length,
        SX=projection.stride(1),
        HK=4,
        HV=16,
        D=128,
        WIDTH=4,
        HAS_BIAS=bias is not None,
        NSEQ=nseq,
        SEARCH=(nseq + 1).bit_length(),
        BT=bt,
        num_warps=warps,
        enable_fp_fusion=True,
    )
    dx = torch.empty(projection.shape, device=projection.device, dtype=projection.dtype)
    dw, db = linear_conv_backward(projection[..., :3072], weight, bias, cu, dz, dx[..., :3072])
    aux_bt = 32
    partial_shape = (triton.cdiv(total, aux_bt), 16)
    da_partial = torch.empty(partial_shape, device=projection.device, dtype=torch.float32)
    ddt_partial = torch.empty_like(da_partial)
    projection_gates_bwd[(partial_shape[0], 16)](
        projection,
        alog,
        dtbias,
        dgate,
        dg,
        dbeta,
        dx,
        da_partial,
        ddt_partial,
        TOTAL=total,
        SX=projection.stride(1),
        HK=4,
        HV=16,
        D=128,
        BT=aux_bt,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Keep the A_log reduction in FP32 until the final parameter gradient.
    da = (da_partial.sum(0) * alog.float().exp()).to(alog.dtype)
    return dx, dw, db, da, ddt_partial.sum(0).to(dtbias.dtype)


class _FusedPreparation(torch.autograd.Function):
    """First-order autograd for fused GDN preparation."""

    @staticmethod
    def forward(ctx, projection, weight, bias, alog, dtbias, cu):
        """Save inputs and normalization outputs for backward."""
        outputs, rstd = forward_impl(projection, weight, bias, alog, dtbias, cu)
        ctx.save_for_backward(
            projection, weight, bias, alog, dtbias, cu, outputs[0], outputs[1], rstd
        )
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grads):
        """Return projection and preparation-parameter gradients."""
        return (*backward_impl(*ctx.saved_tensors, grads), None)


@torch.compiler.disable
def fused_prepare(
    projection: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    alog: torch.Tensor,
    dtbias: torch.Tensor,
    cu: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Run shape-specialized preparation with first-order autograd support."""
    global _LOGGED
    if not _LOGGED:
        logging.getLogger(__name__).warning(
            "GDN_FUSION_ACTIVE shape=%s strides=%s", tuple(projection.shape), projection.stride()
        )
        _LOGGED = True
    return _FusedPreparation.apply(projection, weight, bias, alog, dtbias, cu)

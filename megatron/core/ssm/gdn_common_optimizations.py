# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fixed-launch FLA convolution for shape-specialized GDN execution."""

import logging
import os

import torch
import triton

from megatron.core.ssm.gdn_fusion import _LINEAR_BWD

try:
    from fla.modules.conv.triton.ops import causal_conv1d_fwd
    from fla.ops.utils import prepare_chunk_indices
except ImportError:
    causal_conv1d_fwd = None
    prepare_chunk_indices = None


def enabled(module: torch.nn.Module, projection: torch.Tensor) -> bool:
    """Return whether the fixed-launch convolution supports this projection."""
    return (
        os.getenv("MCORE_GDN_COMMON_OPT", "0") == "1"
        and _LINEAR_BWD is not None
        and causal_conv1d_fwd is not None
        and prepare_chunk_indices is not None
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


def normalized_cu(cu: torch.Tensor | None) -> torch.Tensor | None:
    """Skip chunk metadata for one previously validated complete sequence."""
    return None if cu is not None and cu.numel() == 2 else cu


def conv_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu: torch.Tensor | None,
    activation: str | None,
) -> torch.Tensor:
    """Run the installed FLA forward with its existing layout handling."""
    return causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        activation=activation,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu,
        BT=64,
        layout_fallback=x.stride(1) != x.shape[-1],
    )[0]


def conv_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu: torch.Tensor | None,
    dy: torch.Tensor,
    activation: str = "silu",
    bt: int = 128,
    bd: int = 32,
    warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Launch the installed FLA SiLU backward with explicit, shared tile settings."""
    batch, length, channels = x.shape
    width = weight.shape[1]
    cu = normalized_cu(cu)
    chunks = prepare_chunk_indices(cu, bt) if cu is not None else None
    nt = len(chunks) if chunks is not None else triton.cdiv(length, bt)
    # Forward uses BT64 and cannot reuse the BT128 backward chunk indices.
    preactivation = conv_forward(x, weight, bias, cu, activation=None)
    dx = torch.empty_like(x)
    dw = torch.empty((batch * nt, channels, width), device=x.device, dtype=torch.float32)
    db = (
        torch.empty((batch * nt, channels), device=x.device, dtype=torch.float32)
        if bias is not None
        else None
    )
    _LINEAR_BWD[(triton.cdiv(channels, bd), nt, batch)](
        x=x,
        y=preactivation,
        weight=weight,
        initial_state=None,
        dht=None,
        dy=dy,
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
        stride_dy_n=dy.stride(0),
        stride_dy_t=dy.stride(1),
        stride_dy_d=dy.stride(2),
        D=channels,
        W=width,
        BT=bt,
        BW=triton.next_power_of_2(width),
        BD=bd,
        NB=triton.cdiv(batch * length, 1024),
        ACTIVATION=activation,
        HAS_WEIGHT=True,
        HAS_BIAS=bias is not None,
        USE_INITIAL_STATE=False,
        USE_FINAL_STATE=False,
        IS_VARLEN=cu is not None,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return dx, dw.sum(0).to(weight.dtype), db.sum(0).to(bias.dtype) if db is not None else None


class _TunedCausalConv(torch.autograd.Function):
    """First-order autograd for fixed-launch FLA convolution."""

    @staticmethod
    def forward(ctx, x, weight, bias, cu, activation):
        """Save convolution inputs for backward recomputation."""
        cu = normalized_cu(cu)
        ctx.save_for_backward(x, weight, bias, cu)
        ctx.activation = activation
        return conv_forward(x, weight, bias, cu, activation)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        """Return input and convolution-parameter gradients."""
        return (*conv_backward(*ctx.saved_tensors, dy, ctx.activation), None, None)


_LOGGED = False


@torch.compiler.disable
def tuned_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None]:
    """Run unfused SiLU convolution with a fixed backward launch configuration."""
    global _LOGGED
    assert initial_state is None and not output_final_state
    assert activation in ("silu", "swish")
    if not _LOGGED:
        logging.getLogger(__name__).warning(
            "GDN_COMMON_CONV_ACTIVE backward_bt=128 bd=32 warps=4 shape=%s", tuple(x.shape)
        )
        _LOGGED = True
    return _TunedCausalConv.apply(x, weight, bias, cu_seqlens, activation), None

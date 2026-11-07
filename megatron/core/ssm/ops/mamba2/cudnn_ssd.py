# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mamba-2 training conv + SSD on the cuDNN frontend's CuTe DSL kernels.

``cudnn_mamba_split_conv1d_scan_combined`` stands in for mamba_ssm's
``mamba_split_conv1d_scan_combined`` (without its fused RMSNorm and output projection). The
depthwise causal conv1d + SiLU runs on causal-conv1d's channel-last kernels, as in the Triton
path, and ``torch.ops.cudnn.mamba2_fwd`` / ``mamba2_bwd`` compute the chunked scan with the D
skip and the ``silu(z)`` gate. Only ``zxBCdt`` and the pre-gate output are saved; the conv is
recomputed in backward.

The kernels run on SM100 / SM103 / SM107 with head dim 64 and state dim 128 only.
"""

import functools
from typing import Optional

import torch

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function, causal_conv1d_fwd_function
except ImportError:
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None

CUDNN_MAMBA2_HEAD_DIM = 64
CUDNN_MAMBA2_STATE_DIM = 128


@functools.lru_cache(maxsize=None)
def _cudnn_mamba2_import_error() -> Optional[str]:
    """Register torch.ops.cudnn.mamba2_* on first use; importing cudnn pulls in the CuTe DSL."""
    try:
        import cudnn.linear_attention.ops  # noqa: F401  pylint: disable=unused-import
    except (ImportError, OSError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if not hasattr(torch.ops.cudnn, "mamba2_fwd"):
        return "the installed cudnn-frontend does not register torch.ops.cudnn.mamba2_fwd"
    return None


def cudnn_mamba2_unsupported_reason(headdim: int, d_state: int) -> Optional[str]:
    """Return why the cuDNN Mamba-2 kernels cannot run a layer on this device, or None."""
    if headdim != CUDNN_MAMBA2_HEAD_DIM or d_state != CUDNN_MAMBA2_STATE_DIM:
        return (
            f"the kernels support mamba_head_dim={CUDNN_MAMBA2_HEAD_DIM} and "
            f"mamba_state_dim={CUDNN_MAMBA2_STATE_DIM} only, got {headdim} and {d_state}"
        )
    if causal_conv1d_fwd_function is None:
        return "causal_conv1d.cpp_functions (causal-conv1d) is not installed"
    import_error = _cudnn_mamba2_import_error()
    if import_error is not None:
        return (
            "the cuDNN Mamba-2 ops need nvidia-cudnn-frontend[cutedsl]>=1.32.0 "
            f"({import_error})"
        )
    if not torch.cuda.is_available():
        return "no CUDA device is available"
    major, minor = torch.cuda.get_device_capability()
    sm = 10 * major + minor
    if not (100 <= sm <= 103 or sm == 107):
        return f"the kernels need SM100-SM103 or SM107, but this device is sm_{sm}"
    return None


def _ensure_stride(t: torch.Tensor) -> torch.Tensor:
    """The channel-last conv needs stride(1) of a [b, l, d] tensor to be a multiple of 8."""
    return t if t.stride(1) % 8 == 0 else t.contiguous()


def _conv1d_silu(xBC, weight, bias, seq_idx):
    """Channel-last causal conv1d + SiLU, [b, l, d] -> [b, l, d]."""
    out = causal_conv1d_fwd_function(
        _ensure_stride(xBC).transpose(1, 2), weight, bias, seq_idx, None, None, True
    )
    return out.transpose(1, 2)


def _heads(t: torch.Tensor, headdim: int) -> torch.Tensor:
    """[b, l, h * p] -> [b * l, h, p]."""
    return t.flatten(0, 1).unflatten(-1, (-1, headdim))


def _split_xBC(xBC: torch.Tensor, d_inner: int, ngroups: int, headdim: int):
    """[b, l, conv_dim] -> x [T, h, p], B [T, g, n], C [T, g, n] views."""
    bc_dim = (xBC.shape[-1] - d_inner) // 2
    x, B, C = torch.split(xBC.flatten(0, 1), [d_inner, bc_dim, bc_dim], dim=-1)
    return (
        x.unflatten(-1, (-1, headdim)),
        B.unflatten(-1, (ngroups, -1)),
        C.unflatten(-1, (ngroups, -1)),
    )


class CudnnMambaSplitConv1dScanCombinedFn(torch.autograd.Function):
    """conv1d + SiLU + cuDNN Mamba-2 scan with the fused z gate, over a packed zxBCdt."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        zxBCdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        cu_seqlens,
        seq_idx,
        headdim,
        ngroups,
        batch_invariant,
    ):
        # pylint: disable=missing-function-docstring
        batch, seqlen, _ = zxBCdt.shape
        nheads = A.shape[0]
        d_inner = nheads * headdim
        conv_dim = conv1d_weight.shape[0]
        assert zxBCdt.shape[-1] == d_inner + conv_dim + nheads, (
            f"zxBCdt has {zxBCdt.shape[-1]} channels, expected z, xBC and dt "
            f"({d_inner} + {conv_dim} + {nheads})"
        )
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        z, xBC, dt = torch.split(zxBCdt, [d_inner, conv_dim, nheads], dim=-1)
        x, B, C = _split_xBC(
            _conv1d_silu(xBC, conv1d_weight, conv1d_bias, seq_idx), d_inner, ngroups, headdim
        )
        # cudnn-frontend 1.33 appends composed_initial_state to the outputs.
        y, _, _, y_pre, *_ = torch.ops.cudnn.mamba2_fwd(
            x,
            dt.flatten(0, 1),
            A,
            B,
            C,
            cu_seqlens,
            D=D,
            dt_bias=dt_bias,
            z=_heads(z, headdim),
            output_pre_gate=True,
            batch_invariant=batch_invariant,
        )
        ctx.save_for_backward(
            zxBCdt, conv1d_weight, conv1d_bias, dt_bias, A, D, cu_seqlens, seq_idx, y_pre
        )
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        ctx.batch_invariant = batch_invariant
        return y.view(batch, seqlen, d_inner)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dy):
        # pylint: disable=missing-function-docstring
        zxBCdt, conv1d_weight, conv1d_bias, dt_bias, A, D, cu_seqlens, seq_idx, y_pre = (
            ctx.saved_tensors
        )
        headdim, ngroups = ctx.headdim, ctx.ngroups
        batch, seqlen, _ = zxBCdt.shape
        nheads = A.shape[0]
        d_inner = nheads * headdim
        conv_dim = conv1d_weight.shape[0]
        z, xBC, dt = torch.split(zxBCdt, [d_inner, conv_dim, nheads], dim=-1)
        x, B, C = _split_xBC(
            _conv1d_silu(xBC, conv1d_weight, conv1d_bias, seq_idx), d_inner, ngroups, headdim
        )
        dx, ddt, dA, dB, dC, dD, ddt_bias, dz, _ = torch.ops.cudnn.mamba2_bwd(
            _heads(dy.contiguous(), headdim),
            x,
            dt.flatten(0, 1),
            A,
            B,
            C,
            cu_seqlens,
            D=D,
            dt_bias=dt_bias,
            z=_heads(z, headdim),
            y_pre=y_pre,
            batch_invariant=ctx.batch_invariant,
        )
        del x, B, C

        dzxBCdt = torch.empty_like(zxBCdt)
        dz_out, dxBC_out, ddt_out = torch.split(dzxBCdt, [d_inner, conv_dim, nheads], dim=-1)
        # TODO: mamba2_bwd returns freshly allocated gradients, so dz and ddt are copied into
        # the strided dzxBCdt slices here, where mamba_split_conv1d_scan_combined writes them
        # in place. Pass dz_out / ddt_out to mamba2_bwd once it accepts output buffers.
        dz_out.copy_(dz.view_as(dz_out))
        ddt_out.copy_(ddt.view_as(ddt_out))
        # TODO: the conv backward takes a single [b, l, conv_dim] gradient, so dx, dB and dC
        # are concatenated here. Have mamba2_bwd write them into views of one dxBC buffer
        # once it accepts output buffers.
        dxBC = torch.cat([dx.flatten(1), dB.flatten(1), dC.flatten(1)], dim=-1)
        dxBC = dxBC.view(batch, seqlen, conv_dim)
        del dx, dB, dC, dz, ddt

        dxBC_in, dweight, dbias, *_ = causal_conv1d_bwd_function(
            _ensure_stride(xBC).transpose(1, 2),
            conv1d_weight,
            conv1d_bias,
            dxBC.transpose(1, 2),
            seq_idx,
            None,
            None,
            _ensure_stride(dxBC_out).transpose(1, 2), # the conv kernel writes directly into this buffer and returns it in dxBC_in
            False,
            True,
        )
        dxBC_in = dxBC_in.transpose(1, 2)

        # confirm that the conv kernel wrote directly into dxBC_out, if not, copy the buffer
        if dxBC_in.stride() != dxBC_out.stride():
            dxBC_out.copy_(dxBC_in)
        return dzxBCdt, dweight, dbias, ddt_bias, dA, dD, None, None, None, None, None


def cudnn_mamba_split_conv1d_scan_combined(
    zxBCdt: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    cu_seqlens: torch.Tensor,
    headdim: int,
    ngroups: int,
    seq_idx: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
) -> torch.Tensor:
    """Gated Mamba-2 SSD output ``y * silu(z)`` for a packed ``zxBCdt``.

    Args:
        zxBCdt: ``[b, l, 2 * d_inner + 2 * ngroups * d_state + nheads]``, contiguous.
        conv1d_weight: ``[conv_dim, d_conv]`` depthwise conv weight.
        conv1d_bias: ``[conv_dim]``.
        dt_bias: ``[nheads]``, added to dt before the softplus.
        A: ``[nheads]`` negative decay rates (``-exp(A_log)``).
        D: ``[nheads]`` or ``[nheads, headdim]`` skip.
        cu_seqlens: int32 ``[num_seqs + 1]`` boundaries over the ``b * l`` tokens; the last
            entry must equal ``b * l``.
        headdim: Mamba head dim.
        ngroups: Number of B / C groups.
        seq_idx: Optional int32 ``[1, l]`` per-token sequence index for the conv of packed input.
        batch_invariant: Use cuDNN's batch-invariant plan, whose per-sequence results do not
            depend on the other sequences in the batch.

    Returns:
        ``[b, l, d_inner]`` gated output, before the output norm.
    """
    return CudnnMambaSplitConv1dScanCombinedFn.apply(
        zxBCdt,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A,
        D,
        cu_seqlens,
        seq_idx,
        headdim,
        ngroups,
        batch_invariant,
    )

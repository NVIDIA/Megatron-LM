# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mamba-2 training conv + SSD on the cuDNN frontend's CuTe DSL kernels.

``cudnn_mamba_split_conv1d_scan_combined`` stands in for mamba_ssm's
``mamba_split_conv1d_scan_combined`` (without its fused RMSNorm and output projection). The
depthwise causal conv1d + SiLU runs on causal-conv1d's channel-last kernels, as in the Triton
path, and cuDNN's MAMBA2 / MAMBA2_BWD kernels compute the chunked scan with the D skip and the
``silu(z)`` gate. Only ``zxBCdt`` and the pre-gate output are saved; the conv is recomputed in
backward. The forward calls ``torch.ops.cudnn.mamba2_fwd``; the backward builds the MAMBA2_BWD
node through the cuDNN graph API, so its gradients land in caller-provided views of
``dzxBCdt`` and of the conv's input gradient instead of freshly allocated tensors.

The kernels run on SM100 / SM103 / SM107 with head dim 64 and state dim 128 only.
"""

import functools
import threading
from typing import Optional

import torch

try:
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function, causal_conv1d_fwd_function
except ImportError:
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None

try:
    from causal_conv1d.cpp_functions import _causal_conv1d_fwd_cpp
except ImportError:
    _causal_conv1d_fwd_cpp = None

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


def _conv1d_silu(xBC, weight, bias, seq_idx, out=None):
    """Channel-last causal conv1d + SiLU, [b, l, d] -> [b, l, d], into ``out`` if its layout allows."""
    xBC = _ensure_stride(xBC)
    if out is None or _causal_conv1d_fwd_cpp is None or out.stride(1) % 8 != 0:
        out = causal_conv1d_fwd_function(
            xBC.transpose(1, 2), weight, bias, seq_idx, None, None, True
        )
        return out.transpose(1, 2)
    _causal_conv1d_fwd_cpp(
        xBC.transpose(1, 2), weight, bias, seq_idx, None, out.transpose(1, 2), None, True
    )
    return out


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


# The MAMBA2_BWD gradients this module binds, each with the dtype of the input it differentiates.
_MAMBA2_BWD_GRAD_OF = {
    "dQ": "q",
    "dK": "k",
    "dV": "v",
    "dG": "g",
    "dA": "a",
    "d_dt_bias": "dt_bias",
    "dD": "d",
    "dZ": "z",
}

# cuDNN handles are not thread-safe: one per thread and device, rebound to the launch stream.
_cudnn_handles = threading.local()


def _cudnn_handle(device: torch.device):
    """This thread's cuDNN handle for ``device``, bound to the current stream."""
    import cudnn

    handles = _cudnn_handles.__dict__.setdefault("by_device", {})
    if device.index not in handles:
        with torch.cuda.device(device):
            handles[device.index] = cudnn.create_handle()
    stream = torch.cuda.current_stream(device).cuda_stream
    cudnn.set_stream(handle=handles[device.index], stream=stream)
    return handles[device.index]


@functools.lru_cache(maxsize=None)
def _mamba2_bwd_graph(signature, batch_invariant: bool, device: torch.device):
    """Build a one-node MAMBA2_BWD graph for inputs of the given (name, shape, dtype) signature.

    Tensors are declared by shape only: the kernels take the strides of the buffers bound at
    execute time.
    """
    import cudnn

    data_type = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int32: cudnn.data_type.INT32,
    }
    input_dtypes = {name: data_type[dtype] for name, _, dtype in signature}
    with torch.cuda.device(device):
        graph = cudnn.pygraph()
        tensors = {
            name: graph.tensor(list(shape), data_type=input_dtypes[name], name=name)
            for name, shape, _ in signature
        }
        grads = graph.mamba2_bwd(**tensors, batch_invariant=batch_invariant, name="mamba2_bwd")
        grad_names = ("dQ", "dK", "dV", "dG", "dA", "d_initial_state", "d_dt_bias", "dD", "dZ")
        for name, grad in zip(grad_names, grads):
            if name in _MAMBA2_BWD_GRAD_OF:
                grad.set_output(True).set_data_type(input_dtypes[_MAMBA2_BWD_GRAD_OF[name]])
                tensors[name] = grad
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans()
    return graph, tensors


def _mamba2_bwd(
    dy, x, dt, A, B, C, cu_seqlens, D, dt_bias, z, y_pre, batch_invariant, *, dx, ddt, dB, dC, dz
):
    """``torch.ops.cudnn.mamba2_bwd`` writing dx, ddt, dB, dC and dz into the given buffers.

    The output buffers may be strided views with a unit innermost stride, like the inputs.
    Returns ``(dA, dD, ddt_bias)``.
    """
    inputs = dict(
        q=C,
        k=B,
        v=x,
        g=dt,
        a=A,
        cu_seqlens=cu_seqlens,
        dO=dy,
        dt_bias=dt_bias,
        d=D,
        z=z,
        O_pre=y_pre,
    )
    signature = tuple((name, tuple(t.shape), t.dtype) for name, t in inputs.items())
    graph, tensors = _mamba2_bwd_graph(signature, bool(batch_invariant), x.device)
    dA, dD, ddt_bias = torch.empty_like(A), torch.empty_like(D), torch.empty_like(dt_bias)
    buffers = dict(inputs, dQ=dC, dK=dB, dV=dx, dG=ddt, dA=dA, d_dt_bias=ddt_bias, dD=dD, dZ=dz)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device=x.device)
    graph.execute(
        {tensors[name]: t for name, t in buffers.items()}, workspace, handle=_cudnn_handle(x.device)
    )
    return dA, dD, ddt_bias


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
        dzxBCdt = torch.empty_like(zxBCdt)
        dz, dxBC_out, ddt = torch.split(dzxBCdt, [d_inner, conv_dim, nheads], dim=-1)
        # The conv output is recomputed into dxBC_out, unused until the conv backward writes it,
        # so no separate buffer is live next to the scan backward's workspace.
        x, B, C = _split_xBC(
            _conv1d_silu(xBC, conv1d_weight, conv1d_bias, seq_idx, out=dxBC_out),
            d_inner,
            ngroups,
            headdim,
        )
        # dz and ddt go straight into their dzxBCdt slices, and dx, dB and dC into one dxBC
        # buffer for the conv backward.
        dxBC = zxBCdt.new_empty(batch, seqlen, conv_dim)
        dx, dB, dC = _split_xBC(dxBC, d_inner, ngroups, headdim)
        dA, dD, ddt_bias = _mamba2_bwd(
            _heads(dy.contiguous(), headdim),
            x,
            dt.flatten(0, 1),
            A,
            B,
            C,
            cu_seqlens,
            D,
            dt_bias,
            _heads(z, headdim),
            y_pre,
            ctx.batch_invariant,
            dx=dx,
            ddt=ddt.flatten(0, 1),
            dB=dB,
            dC=dC,
            dz=_heads(dz, headdim),
        )
        del x, B, C, dx, dB, dC

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

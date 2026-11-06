# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kernel and API contracts of the fused wide-residual shortcut seams.

One FP64 reference for the forward and every gradient; schedules, validation, configs,
checkpoint replay scopes, and the MXFP8 handoff to Transformer Engine (real GTP sharding under
``torchrun --nproc-per-node 2 -m pytest <this file> -k gtp``). Poisoned CUDA-graph replay and
the interval replay kernel live in ``tests/unit_tests/determinism/kernels/``.
"""

import gc
import itertools
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import triton
import triton.language as tl
from torch.multiprocessing.reductions import StorageWeakRef

from megatron.core.fusions.wide_residual_shortcut import (
    PendingWideResidualWrites,
    api,
    backward_kernel,
    forward_kernel,
    kernel_utils,
    shortcut_add_wide_residual_write_read_rmsnorm_quant,
    shortcut_add_wide_residual_write_two_reader_rmsnorm_quant,
)
from megatron.core.fusions.wide_residual_shortcut.mxfp8_handoff import (
    Reader,
    install_linear_hooks,
    reader_for,
)
from megatron.core.fusions.wide_residual_shortcut.residual_replay import (
    ResidualSlot,
    current_manager,
    register_write,
    residual_replay_scope,
)
from megatron.core.transformer.streamwise_residual_ops import (
    _streamwise_sigmoid_write_triton,
    streamwise_sigmoid_writeback,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, bytes_equal
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine_torch import DType

    from megatron.core.extensions.transformer_engine import TELinear
except ImportError:  # BF16-only tests do not need Transformer Engine.
    te = None

# Host-only config tests run anywhere; every other test needs a GPU. Native schedules, the
# MXFP8 handoff and the fused backward (so every training round trip) need a qualified GB300.
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA shortcut fusion")
_GB300 = torch.cuda.is_available() and kernel_utils.qualified_device(
    torch.cuda.get_device_properties(0)
)
REQUIRES_FUSED_BACKWARD = pytest.mark.skipif(
    not _GB300, reason="the fused shortcut backward runs only on a qualified GB300"
)
QUALIFIED = pytest.mark.skipif(
    te is None or not _GB300, reason="qualified GB300 schedules and MXFP8 kernels"
)
F, T = False, True
BF16, F32 = torch.bfloat16, torch.float32
FLAGSHIPS = kernel_utils.FLAGSHIPS
# Rows x width of the Nano/Super/Ultra flagships and of expanded admitted shapes.
MX_SHAPES = [*FLAGSHIPS, (2048, 2048), (2048, 16384), (16384, 2048), (6144, 12288)]
# Seams as (readers, updates): one reader, a shortcut add read twice, a single write read twice,
# a shortcut add read once (a fused block into a lone layer).
SEAMS = [(1, 1), (2, 2), (2, 1), (1, 2)]

# ---------------------------------------------------------------- shared inputs and runners


def inputs(
    streams,
    readers,
    hidden=96,
    tokens=1024,
    *,
    retention=True,
    seed=917,
    pad=2,
    updates=None,
    residual_dtype=F32,
    logit_dtype=F32,
):
    """Seeded leaves ``(residual, updates, controllers, gammas)``.

    ``updates`` defaults to one per reader; controllers carry ``pad`` unused trailing logits;
    retention=False drops the retain controller (static identity) after drawing it.
    """
    torch.manual_seed(seed)

    def leaf(*shape, dtype=F32):
        return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)

    residual = leaf(tokens, streams * hidden, dtype=residual_dtype)
    updates = tuple(leaf(tokens, hidden, dtype=BF16) for _ in range(updates or readers))
    controllers = tuple(leaf(streams + pad, dtype=logit_dtype) for _ in range(2 + readers))
    gammas = tuple(leaf(hidden, dtype=BF16) for _ in range(readers))
    if not retention:
        controllers = (controllers[0], None, *controllers[2:])
    return residual, updates, controllers, gammas


def mx_inputs(m, h, readers, updates=None):
    """Three-stream MXFP8 handoff inputs with unpadded controllers."""
    return inputs(3, readers, h, m, seed=319, pad=0, updates=updates)


def run(raw, streams=3, epsilon=1e-5, **options):
    """Call the public seam function matching the reader count (one or two updates)."""
    residual, updates, controllers, gammas = raw
    options.update(num_streams=streams, epsilon=epsilon)
    shared = updates[1] if len(updates) == 2 else None
    pending = PendingWideResidualWrites(residual, updates[0], shared, *controllers[:2])
    if len(gammas) == 1:
        return shortcut_add_wide_residual_write_read_rmsnorm_quant(
            pending, controllers[2], gammas[0], **options
        )
    return shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(
        pending, controllers[2:], gammas, **options
    )


def boundary(raw, consumer=None, *, mxfp8_handoff=True):
    return run(raw, 3, mxfp8_handoff=mxfp8_handoff, mxfp8_consumer=consumer)


def invoke(raw, quantized=False, consumer=None):
    """Run three streams with Megatron's [tokens, batch=1, hidden] layout."""
    residual, updates, controls, gammas = raw
    m, h = updates[0].shape
    raw = residual.view(m, 1, 3 * h), tuple(t.view(m, 1, h) for t in updates), controls, gammas
    return run(raw, 3, mxfp8_handoff=quantized, mxfp8_consumer=consumer)


def fwd(raw, streams=3, **options):
    """Launch the forward kernel directly: (current, saved tape, reader operands)."""
    residual, updates, controllers, gammas = raw
    return forward_kernel.forward(
        residual, updates, *controllers[:2], controllers[2:], gammas, num_streams=streams, **options
    )


def backward_args(raw, streams=3, **options):
    """Forward, then random reader/carry gradients: (current, (saved, previous, dys, carry))."""
    current, saved, _ = fwd(raw, streams, **options)
    dys = tuple(torch.randn_like(raw[1][0]) for _ in raw[3])
    return current, (saved, raw[0], dys, torch.randn_like(current))


def buffers(outputs):
    """Every buffer of BF16 tensors and rowwise MXFP8 operands (data, then scales)."""
    for tensor in outputs:
        if type(tensor) is torch.Tensor:
            yield tensor
        else:
            yield tensor._rowwise_data
            yield tensor._rowwise_scale_inv


def workspace_tensors(value):
    """Owned scratch and outputs, for poisoning and storage-ownership assertions."""
    if hasattr(value, "allocations"):
        yield from value.allocations
    elif is_dataclass(value):
        for field in fields(value):
            yield from workspace_tensors(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from workspace_tensors(item)
    elif type(value) is torch.Tensor:
        yield value
    elif isinstance(value, torch.Tensor):
        yield value._columnwise_data
        yield value._columnwise_scale_inv


def poison(tensors, integer=37):
    for tensor in tensors:
        tensor.fill_(float("nan") if tensor.dtype.is_floating_point else integer)


def poison_bytes(tensors, byte=0xA5):
    for tensor in tensors:
        tensor.view(torch.uint8).fill_(byte)


def all_bytes_equal(actual, expected):
    actual, expected = list(actual), list(expected)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert (a is None and b is None) or bytes_equal(a, b)


def replay_poisoned(graph, scratch, actual, expected, fill=poison):
    """Three times: poison ``scratch``, replay ``graph``, compare ``actual`` byte for byte."""
    scratch, actual = list(scratch), list(actual)
    for _ in range(3):
        fill(scratch)
        graph.replay()
        all_bytes_equal(actual, expected)


def run_with_scratch(backward, *args, **kwargs):
    """Run a backward recording its CUDA allocations and last (non-warmup) kernel launch."""
    allocations, launches = [], []

    def record_allocation(function):
        def allocate(*args, **kwargs):
            tensor = function(*args, **kwargs)
            if tensor.is_cuda:
                allocations.append(tensor)
            return tensor

        return allocate

    original_run = backward_kernel._backward.run

    def record_launch(*args, **kwargs):
        kernel = original_run(*args, **kwargs)
        if not kwargs.get("warmup", False):
            launches.append(kernel)
        return kernel

    with ExitStack() as stack:
        for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
            stack.enter_context(patch.object(owner, name, record_allocation(getattr(owner, name))))
        stack.enter_context(patch.object(backward_kernel._backward, "run", record_launch))
        grads, columns = backward(*args, **kwargs)
    # TE allocates column buffers in C++; include those through returned operands.
    allocations.extend(workspace_tensors((grads, columns)))
    allocations = list({t.untyped_storage().data_ptr(): t for t in allocations}.values())
    return grads, SimpleNamespace(
        grads=grads, columns=columns, allocations=allocations, kernel=launches[-1]
    )


def replay_config(interval=16):
    return SimpleNamespace(
        residual_stream_recompute_num_layers=interval, moe_shortcut_wide_residual_norm_fusion=True
    )


def written(current, saved, operands):
    """Every tensor a forward writes: residual, reader operands and the kernel-written tape."""
    tape = () if saved is None else (saved.coefficients, *saved.rstd)
    if saved is not None and saved.update_count == 2:
        tape += (saved.update,)
    return [current, *buffers(operands), *tape]


def forced_forward(cfg, raw, streams, **options):
    with patch.object(forward_kernel, "choose_forward_config", side_effect=lambda *a, **k: cfg):
        with torch.no_grad():
            return fwd(raw, streams, **options)


# ---------------------------------------------------------------- the FP64 reference


def reference(raw, streams, epsilon=1e-5, coefficients=None):
    """The seam in FP64: ``(current, *readers)`` rounded like the kernel (the residual once to
    its dtype, BF16 readers of unrounded folds) and the FP64 ``(mix, update, current, folds)``.

    ``mix`` is (write, retain, reads...) per stream from the logits (retain None is the static
    identity) or, as FP64 leaves, from a saved tape's ``coefficients`` table.
    """
    residual, updates, controllers, gammas = raw
    h = updates[0].shape[-1]
    if coefficients is None:
        mix = [2 * controllers[0][:streams].double().sigmoid(), None]
        if controllers[1] is not None:
            mix[1] = 1 - 0.1 * (-controllers[1][:streams].double()).sigmoid()
        mix += [read[:streams].double().sigmoid() for read in controllers[2:]]
    else:
        table = coefficients.detach().double().view(8, streams)[: 2 + len(gammas)]
        mix = [row.clone().requires_grad_() for row in table]
    u = updates[0] if len(updates) == 1 else (updates[0].float() + updates[1].float()).bfloat16()
    u, r = u.double(), residual.double().view(-1, streams, h)
    current = (r if mix[1] is None else mix[1][:, None] * r) + mix[0][:, None] * u[:, None]
    xs = [(read[:, None] * current).sum(1) for read in mix[2:]]
    readers = [
        ((x * (x.square().mean(-1, keepdim=True) + epsilon).rsqrt()) * g.double()).bfloat16()
        for x, g in zip(xs, gammas)
    ]
    return (current.flatten(1).to(residual.dtype), *readers), (mix, u, current, xs)


def fp64_backward(raw, streams, dys, carry, *, epsilon=1e-5, tape=None):
    """FP64 gradients in the kernel's order (dR, du per update, write, retain or None, reads,
    gammas) for upstream ``carry``/``dys`` (None is zero), and per controller the sum of its
    absolute contributions, which bounds the error of any FP32 summation order."""
    leaf = lambda t: None if t is None else t.detach().double().requires_grad_()
    r, us, cs, gs = leaf(raw[0]), [*map(leaf, raw[1])], [*map(leaf, raw[2])], [*map(leaf, raw[3])]
    coefficients = None if tape is None else tape.coefficients
    outputs, (mix, u, current, xs) = reference((r, us, cs, gs), streams, epsilon, coefficients)
    upstream = [
        torch.zeros_like(o) if g is None else g.to(o.dtype) for o, g in zip(outputs, (carry, *dys))
    ]
    live = [m for m in mix if m is not None]
    targets = [r, *us, *live, *gs, current, *xs]
    grads = list(torch.autograd.grad(outputs, targets, upstream, retain_graph=True))
    dr, dus = grads.pop(0), [grads.pop(0) for _ in us]
    dmix, dgs = [grads.pop(0) for _ in live], [grads.pop(0) for _ in gs]
    if tape is not None:  # the saved update is the shortcut add's sum
        dus = dus * tape.update_count
    total, dxs = grads[0], grads[1:]
    with torch.no_grad():
        terms = [total * u[:, None], total * r.view_as(total)]
        terms += [current * dx[:, None] for dx in dxs]
    controllers, bounds = [], []
    for j, m in enumerate(mix):
        if m is None:
            controllers.append(None)
            bounds.append(None)
            continue
        if tape is None:  # d(coefficient)/d(logit): each coefficient reads only its own logit
            d, size = (
                torch.autograd.grad(m.sum(), cs[j], retain_graph=True)[0][:streams],
                cs[j].numel(),
            )
        else:
            d = tape.coefficients.double().view(8, streams)[4 + j]
            size = tape.controller_sizes[j]
        controllers.append(torch.zeros(size, dtype=d.dtype, device=d.device))
        controllers[-1][:streams] = dmix.pop(0) * d
        bounds.append(terms[j].abs().sum((0, 2)) * d.abs())
    return [dr, *dus, *controllers, *dgs], bounds


def assert_matches_fp64(actual, expected, bounds, streams, first=(2e-5, 2e-5), fp32=(5e-4, 2e-4)):
    """``assert_grads_close`` tolerances; FP32 controller gradients also stay within 3e-6 of
    their absolute-contribution bound and keep zero padding."""
    assert [a is None for a in actual] == [b is None for b in expected]
    cast = [None if a is None else b.to(a.dtype) for a, b in zip(actual, expected)]
    assert_grads_close(actual, cast, first=first, fp32=fp32)
    start = len(actual) - 2 * len(bounds) + 2
    for a, b, scale in zip(actual[start:], expected[start:], bounds):
        if a is not None and scale is not None:
            assert torch.count_nonzero(a[streams:]) == 0
            if a.dtype == F32:
                error = (a[:streams].double() - b[:streams]).abs()
                assert torch.all(error <= 3e-6 * scale + torch.finfo(F32).tiny)


def folded_reader(saved, previous, reader):
    """FP64 emulation of the forward's explicit FP32 RN/FMA boundaries."""
    m, h = saved.update.shape
    c = saved.coefficients.double().view(8, saved.num_streams)
    values = previous.double().view(m, saved.num_streams, h)
    x = torch.zeros_like(saved.update, dtype=F32)
    q = torch.zeros((), device=previous.device, dtype=F32)
    for stream in range(saved.num_streams):
        beta = c[2 + reader, stream]
        alpha = (beta * c[1, stream]).float().double()
        x = (alpha * values[:, stream] + x.double()).float()
        q = (beta * c[0, stream] + q.double()).float()
    return (q.double() * saved.update.double() + x.double()).float()


def reference_columns(saved, previous, reader=0):
    """TE's columnwise MXFP8 encoding of the exactly emulated FP32 normalized reader."""
    x = folded_reader(saved, previous, reader)
    normalized = (x * saved.rstd[reader][:, None]) * saved.gammas[reader].float()
    return mxfp8_quantizer(rowwise=False)(normalized)


def mxfp8_quantizer(*, rowwise=True, columnwise=True):
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)
    quantizer.optimize_for_gemm = True
    return quantizer


def assert_grads_close(actual, expected, first=(2e-5, 2e-5), fp32=(5e-4, 2e-4)):
    """FP32 residual (``first``), BF16 (1/128) and other (``fp32``; None skips) gradients."""
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        bf16 = getattr(a, "dtype", None) == BF16
        tolerance = (1 / 128, 1 / 128) if bf16 else first if index == 0 else fp32
        if tolerance:
            torch.testing.assert_close(a, b, rtol=tolerance[0], atol=tolerance[1])


def assert_columns_equal(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert bytes_equal(a._columnwise_data, b._columnwise_data)
            assert bytes_equal(a._columnwise_scale_inv, b._columnwise_scale_inv)


# ---------------------------------------------------------------- numerics


def seam(s, n, u, h, m, rd=F32, ld=F32, eps=1e-5, ctrl=T, ret=T, route=None, frozen=F, **kw):
    return pytest.param(s, n, u, h, m, rd, ld, eps, ctrl, ret, route, frozen, **kw)


_SMALL = ((32, 1024), (96, 1024), (128, 2048), (1024, 1024), (3072, 1024), (4128, 1024))
_BF16_SHAPES = ((3, 96, 1024), (1, 128, 2048), (3, 4096, 1024), (6, 3072, 1024), (2, 10240, 1024))
SEAM_CASES = (
    # Every stream count and seam, rotating over small, partial-tile and production widths.
    [seam(s, *SEAMS[i], *_SMALL[(s + i) % 6]) for s in range(1, 7) for i in range(4)]
    # Production widths with a second RMSNorm epsilon.
    + [seam(3, *seam_, h, 1024, eps=1e-3) for h in (3072, 10240) for seam_ in SEAMS]
    # The widest width leaves near-cancelling controller sums to the contribution bound.
    + [seam(6, 2, 1, 14336, 1024, ctrl=F)]
    # BF16 residual streams and/or BF16 logits.
    + [
        seam(s, *SEAMS[(i + j) % 4], h, m, rd, ld)
        for i, (rd, ld) in enumerate([(BF16, BF16), (BF16, F32), (F32, BF16)])
        for j, (s, h, m) in enumerate(_BF16_SHAPES)
    ]
    # Small widths with and without retention; production shapes and production-size
    # dispatch for other stream counts.
    + [seam(2, 2, 1, 288, 1024), seam(3, 1, 2, 96, 1024, ret=F)]
    + [seam(6, 2, 1, 288, 1024, BF16, BF16, ret=F)]
    + [
        seam(3, *SEAMS[(i + j) % 4], h, m, d, d, ret=j == 0, marks=QUALIFIED)
        for i, (m, h) in enumerate(FLAGSHIPS)
        for j, d in enumerate((F32, F32, BF16))
    ]
    + [seam(s, 2, u, 4096, 8192, marks=QUALIFIED) for s, u in [(2, 1), (6, 2)]]
    # One output's gradient (the others pruned) with frozen parameters.
    + [seam(6, 2, 2, 96, 1024, ret=F, route=0, frozen=T)]
)  # fmt: skip


@GPU
@REQUIRES_FUSED_BACKWARD
@pytest.mark.parametrize(
    "streams,readers,updates,hidden,tokens,rd,ld,eps,ctrl,retention,route,frozen", SEAM_CASES
)
def test_seam_matches_fp64_and_replays_bit_exact(
    streams, readers, updates, hidden, tokens, rd, ld, eps, ctrl, retention, route, frozen
):
    """Outputs (in their own dtypes) and every autograd gradient (pruned outputs, frozen
    parameters) match the FP64 formula and replay bit for bit."""
    raw = inputs(
        streams, readers, hidden, tokens, retention=retention, updates=updates,
        residual_dtype=rd, logit_dtype=ld,
    )  # fmt: skip
    for parameter in (raw[2][1], raw[3][-1]) if frozen else ():
        if parameter is not None:
            parameter.requires_grad_(False)
    with torch.no_grad():
        expected, _ = reference(raw, streams, eps)
    keep = range(len(expected)) if route is None else [route]
    dys = [torch.randn_like(t) if i in keep else None for i, t in enumerate(expected)]
    outputs, grads = assert_replays_bit_exact(
        lambda *args: [y for i, y in enumerate(run(args, streams, eps)) if i in keep],
        raw,
        replays=2,
        grad_outputs={f"out[{k}]": dys[i] for k, i in enumerate(keep)},
    )
    # A BF16 residual is rounded once from FP32, FP64 rounds directly: at most one ulp apart.
    exact = (2e-6, 2e-6) if rd == F32 else (1 / 128, 1e-6)
    for k, i in enumerate(keep):
        rtol, atol = exact if i == 0 else (1 / 128, 1 / 128)
        torch.testing.assert_close(outputs[f"out[{k}]"], expected[i], rtol=rtol, atol=atol)
    want, bounds = fp64_backward(raw, streams, dys[1:], dys[0], epsilon=eps)
    # Frozen parameters get no gradient, nor do a pruned reader's controller and norm weight.
    names = ["in[0]"] + [f"in[{g}][{k}]" for g in (1, 2, 3) for k in range(len(raw[g]))]
    pruned = [i for i in range(readers) if dys[1 + i] is None]
    pruned = {f"in[2][{2 + i}]" for i in pruned} | {f"in[3][{i}]" for i in pruned}
    leaves = [raw[0], *raw[1], *raw[2], *raw[3]]
    live = [t is not None and t.requires_grad and n not in pruned for n, t in zip(names, leaves)]
    assert list(grads) == [n for n, on in zip(names, live) if on]
    got = [grads[n] if on else None for n, on in zip(names, live)]
    assert all(g is None or g.dtype == t.dtype for g, t in zip(got, leaves))
    want = [w if on else None for w, on in zip(want, live)]
    # Controller gradients are FP32 reductions with heavy cancellation; any valid summation
    # order is off by up to ~3e-4 against FP64 (the widest widths only meet the bound).
    first = (2e-5, 2e-5) if route is None else (5e-4, 2e-4)
    assert_matches_fp64(got, want, bounds, streams, first, (5e-4, 5e-4) if ctrl else None)


# fmt: off
# (streams, readers, updates, hidden, rows, columns, retention, grid, pruned input)
BACKWARD_CASES = [
    (6, 1, 1, 4096, 1024, F, T, "auto", None),  # six streams: unstaged dy
    (6, 2, 2, 10240, 1024, T, T, "auto", None),
    (2, 1, 1, 4096, 8192, T, T, "auto", "reader0"),  # model-size dynamic row admission
    (5, 2, 2, 4096, 8192, F, F, "auto", "reader1"),
    (3, 2, 2, 96, 1024, F, T, "auto", "carry"),
    # Explicit and automatic grids with a shared or single update; expanded widths and shapes.
    (3, 1, 1, 3072, 128, T, F, 24, None), (3, 2, 2, 3072, 128, F, T, 24, None),
    (3, 2, 1, 10240, 128, T, T, 80, None), (3, 1, 2, 10240, 128, F, T, 80, None),
    (3, 2, 1, 10240, 1024, T, T, "auto", None), (3, 2, 1, 2048, 16384, T, F, "auto", None),
    (3, 1, 2, 2048, 16384, F, T, "auto", None), (3, 1, 2, 16384, 2048, F, T, "auto", None),
    (3, 2, 2, 13312, 2048, F, F, "auto", None), (3, 2, 1, 13312, 2048, T, T, "auto", None),
    # 128 rows: auto and explicit grids (idle participant groups), with and without retention.
    (1, 2, 2, 4096, 128, F, T, 32, None), (2, 2, 2, 4096, 128, F, F, 32, None),
    (2, 1, 1, 1024, 128, F, F, "auto", None), (4, 2, 2, 1024, 128, F, T, 8, None),
    (6, 1, 1, 1024, 128, F, F, 8, None), (6, 2, 2, 1024, 128, F, T, "auto", None),
    (3, 1, 1, 4096, 128, F, F, "auto", None), (3, 1, 1, 3072, 128, F, T, 336, None),
    (3, 2, 2, 10240, 128, F, F, "auto", None),
]
# fmt: on


@GPU
@REQUIRES_FUSED_BACKWARD
@pytest.mark.parametrize(
    "streams,readers,updates,hidden,rows,columns,retention,grid,pruned", BACKWARD_CASES
)
def test_backward_kernel_matches_fp64(
    streams, readers, updates, hidden, rows, columns, retention, grid, pruned
):
    """Every gradient (controllers within their absolute-contribution bound) and reader zero's
    MXFP8 columns match the FP64/TE reference."""
    if columns and te is None:
        pytest.skip("MXFP8 columns need Transformer Engine")
    raw = inputs(streams, readers, hidden, rows, retention=retention, updates=updates)
    with torch.no_grad():
        _, (saved, r, dys, carry) = backward_args(raw, streams, mxfp8_handoff=columns)
        if pruned == "carry":
            carry = None
        elif pruned is not None:
            dys = tuple(None if i == int(pruned[-1]) else dy for i, dy in enumerate(dys))
        actual, operands = backward_kernel.backward(
            saved, r, dys, carry, save_columns=columns, config=grid
        )
    expected, bounds = fp64_backward(raw, streams, dys, carry)
    # dR, one du per update (shared by a shortcut add), write, retain, reads, gammas.
    assert len(actual) == 3 + updates + 2 * readers
    if updates == 2:
        assert actual[1] is actual[2]
    elif readers == 2:  # A single write keeps the caller's update (no BF16 sum is saved).
        assert saved.update.data_ptr() == raw[1][0].data_ptr()
    assert_matches_fp64(actual, expected, bounds, streams, (5e-4, 2e-4) if pruned else (2e-5, 2e-5))
    reference = (reference_columns(saved, r) if columns else None,) + (None,) * (readers - 1)
    assert_columns_equal(operands, reference)


# ---------------------------------------------------------------- schedules


def heuristic_cases(cluster):
    """One representative shape per distinct (non-)cluster inference schedule."""
    if cluster:
        grid = ((3072, 4096, 5120, 7168, 10240, 16384), (1, 3, 6), (1, 4, 17, 32, 64))
    else:
        grid = ((3072, 4096, 6144, 10240, 16384), range(1, 7), (32, 96, 128, 192, 288, 640, 1024))
    seen = {}
    for hidden, streams, tokens, readers, quantized in itertools.product(*grid, (1, 2), (F, T)):
        if quantized and tokens % 32:
            continue
        cfg = forward_kernel.choose_forward_config(
            tokens, hidden, streams, readers, mxfp8=quantized, inference=True
        )
        if (cfg.CLUSTER > 1) == cluster and cfg.LOADS_FIRST:
            shape = (quantized, streams, readers, hidden, tokens if cluster else min(tokens, 128))
            seen.setdefault((cfg, quantized) if cluster else cfg, shape)
    return [
        pytest.param(
            "cluster" if cluster else "loads", key[0] if cluster else key, *shape, T,
            id=f"{'cluster' if cluster else 'tree'}{i}", marks=QUALIFIED if shape[0] else (),
        )
        for i, (key, shape) in enumerate(seen.items())
    ]  # fmt: skip


EXPLICIT_LAYOUTS = [  # (hidden, width, warps, parts, split): masked packets, scalar loads,
    (96, 128, 1, 1, F),  # whole tiles with replicated rows, split readers, and several
    (130, 256, 2, 1, F),  # segments with a masked tail.
    (3072, 4096, 8, 2, F),
    (4096, 2048, 4, 1, T),
    (10240, 4096, 16, 1, F),
    (10240, 2048, 16, 2, T),
]


@GPU
@pytest.mark.parametrize(
    "variant,cfg,quantized,streams,readers,hidden,tokens,inference",
    [
        pytest.param(
            "loads",
            forward_kernel._ForwardConfig(w, NUM_WARPS=k, PARTS=p if inf else 1, SPLIT=s and inf),
            F, 3, n, h, 33, inf, id=f"explicit-{h}-{w}-{n}-{inf}",
        )
        for (h, w, k, p, s), n, inf in itertools.product(EXPLICIT_LAYOUTS, (1, 2), (F, T))
        if not (s and n == 1)
    ]
    + heuristic_cases(F)
    + heuristic_cases(T),
)  # fmt: skip
def test_schedule_variants_match(
    variant, cfg, quantized, streams, readers, hidden, tokens, inference
):
    """LOADS_FIRST only reorders loads, so results and saved state are bit-identical; a
    cluster's slices and partial-norm exchange match the same slices in one CTA per row."""
    raw = inputs(streams, readers, hidden, tokens)
    pair = (cfg._replace(LOADS_FIRST=F), cfg._replace(LOADS_FIRST=T))
    if variant == "cluster":
        pair = (cfg, cfg._replace(CLUSTER=1))
    options = dict(mxfp8_handoff=quantized, inference=inference)
    (current, saved, q), (current_b, saved_b, q_b) = (
        forced_forward(c, raw, streams, **options) for c in pair
    )
    if variant == "loads":
        all_bytes_equal(written(current, saved, q), written(current_b, saved_b, q_b))
        return
    assert bytes_equal(current, current_b)
    for a, b in zip(q, q_b, strict=True):
        if type(a) is torch.Tensor:
            # Only the partial-sum order differs: at most one BF16 ulp.
            torch.testing.assert_close(a.float(), b.float(), rtol=1 / 128, atol=1e-6)
        else:
            assert (a._rowwise_data != b._rowwise_data).float().mean() < 2e-3
            assert (a._rowwise_scale_inv != b._rowwise_scale_inv).float().mean() < 2e-3


# ---------------------------------------------------------------- validation and configs


def offset_view(tensor, offset=1):
    """A copy of ``tensor`` whose base is ``offset`` elements past an allocation start."""
    shifted = tensor.new_empty(tensor.numel() + offset)[offset:].view_as(tensor)
    with torch.no_grad():
        shifted.copy_(tensor)
    return shifted


@QUALIFIED
def test_rejects_invalid_inputs():
    """Inputs, shapes, alignment, grids and counter capacity are checked before any compile or
    launch."""
    r, us, cs, gs = inputs(3, 1, 128)
    for message, raw in [
        ("FP32 or BF16 residuals", (r.half(), us, cs, gs)),
        ("contiguous", (r.t().contiguous().t(), us, cs, gs)),
        ("update", (r, (us[0].float(),), cs, gs)),
        ("controllers require", (r, us, (cs[0][:2], *cs[1:]), gs)),
        ("gamma", (r, us, cs, (gs[0][:-1],))),
    ]:
        with torch.no_grad(), pytest.raises(ValueError, match=message):
            run(raw, 3)
    shapes = [(m, 96, "rows") for m in (0, 1, 31, 65)]
    shapes += [(1024, h, "width|H") for h in (0, 1, 31, 48, 1025, 2052, 16416)]
    for m, h, match in shapes:
        saved = kernel_utils.SavedWrite(
            torch.empty((m, h), device="cuda", dtype=BF16), None, (None,), (None,), (3,) * 3
        )
        with pytest.raises(ValueError, match=match):
            backward_kernel.backward(saved, None, (None,))
    _, args = backward_args(inputs(3, 1, 4096, 128))
    with patch.object(backward_kernel._backward, "run", side_effect=AssertionError("launch")):
        for grid in (None, 0, -32, 31, True):
            with pytest.raises(ValueError, match="grid"):
                backward_kernel.backward(*args, config=grid)
        with pytest.raises(ValueError, match="counter"):
            backward_kernel.backward(*args, config=2**31)

    # Inference accepts offset views except for the FP32 residual's packet loads; training also
    # needs aligned updates and residuals, which the backward reads through TMA descriptors.
    # Element offsets 1, 2 and 4 cover every alignment class of FP32 and BF16 operands.
    for i, (hidden, streams) in enumerate([(3072, 3), (4096, 3), (10240, 3), (128, 1), (288, 6)]):
        readers = 1 + i % 2
        for operand in range(3 + 2 * (i % 2)):
            quantized, offset = (i + operand) % 2 == 0, (1, 2, 4)[(i + operand) % 3]
            residual, updates, controllers, gammas = inputs(streams, readers, hidden, 128)
            values = [residual, *updates, *gammas]

            def execute(args, inference):
                raw = args[0], args[1 : readers + 1], controllers, tuple(args[readers + 1 :])
                options = dict(mxfp8_handoff=quantized, inference=inference)
                return written(*fwd(raw, streams, **options))

            expected = [execute(values, inference) for inference in (F, T)]
            values[operand] = offset_view(values[operand], offset)
            misaligned = values[operand].data_ptr() % 16 != 0
            packet_load = operand == 0 and values[0].dtype == F32
            for inference in (F, T):
                # 128 rows; the backward reads residuals and updates through TMA.
                tma = not inference and hidden % 32 == 0 and operand <= readers
                if misaligned and (packet_load or tma):
                    with pytest.raises(ValueError, match="16-byte aligned"):
                        execute(values, inference)
                else:
                    all_bytes_equal(execute(values, inference), expected[inference])


def test_config_tables():
    """Nano/Super/Ultra and BF16-stream tables apply from 2048 rows with cached backward grids;
    every forward/backward config is valid; training owns one CTA per row."""
    choose = forward_kernel.choose_forward_config
    table = forward_kernel._FORWARD_CONFIGS
    for (m, h), readers, mxfp8 in itertools.product(FLAGSHIPS, (1, 2), (F, T)):
        knobs = backward_kernel.get_backward_knobs(m, h, 3, readers)
        assert knobs.TARGET_CTAS == {3072: 456, 4096: 448, 10240: 440}[h]
        assert knobs.EARLY_CARRY == (h == 4096 and readers == 1)
        for rows in (2048, m, m + 1024, 65536):
            assert choose(rows, h, 3, readers, mxfp8=mxfp8) is table[h, readers, mxfp8]
        assert choose(1024, h, 3, readers, mxfp8=mxfp8) is not table[h, readers, mxfp8]
        assert choose(m, h, 2, readers, mxfp8=mxfp8) is not table[h, readers, mxfp8]
    for (hidden, readers, updates, quantized), cfg in forward_kernel._FORWARD_CONFIGS_BF16.items():
        assert cfg.LOADS_FIRST and cfg.WIDTH >= cfg.VECTOR * 32 * cfg.NUM_WARPS
        assert not cfg.PAIRED_REDUCE or readers == 2
        options = dict(mxfp8=quantized, updates=updates, bf16_residual=True)
        assert choose(2048, hidden, 3, readers, **options) is cfg
        assert choose(512, hidden, 3, readers, inference=True, **options) is cfg
        assert choose(1024, hidden, 3, readers, **options) is not cfg  # short training rows
        assert choose(2048, hidden, 3, readers, mxfp8=quantized, updates=updates) is not cfg
    widths = (1, 31, 32, 48, 288, 1025, 2052, 3072, 3584, 4096, 5632, 9728, 10240, 13312)
    widths += (15872, 16384, 17408, 32768)
    rows = (1, 16, 17, 32, 33, 40, 41, 64, 65, 72, 73, 88, 89, 128, 129, 160, 161, 256, 257)
    rows += (448, 449, 512, 1024, 1025, 2048, 4096, 8192, 16384)
    for h, m, s, (n, u), mxfp8, inference, bf16 in itertools.product(
        widths, rows, range(1, 7), SEAMS, (F, T), (F, T), (F, T)
    ):
        cfg = choose(m, h, s, n, mxfp8=mxfp8, inference=inference, updates=u, bf16_residual=bf16)
        assert cfg.WIDTH >= 32 and cfg.WIDTH & (cfg.WIDTH - 1) == 0
        assert 0 <= cfg.OFFLOAD_SEGMENTS <= (h + cfg.WIDTH - 1) // cfg.WIDTH
        assert cfg.REDUCE_VECTOR in (0, 1, 2, 4, 8, 16, 32) and cfg.NUM_WARPS in (4, 8, 16)
        assert cfg.MAXNREG is None or cfg.MAXNREG * 32 * cfg.NUM_WARPS <= 65536
        assert not (cfg.SPLIT or cfg.PAIRED_REDUCE) or n == 2
        assert cfg.PARTS == 1 or h % (32 * cfg.PARTS) == 0
        tuned = 3072 <= h <= 16384 and inference and m <= 1024
        assert cfg.LOADS_FIRST or not tuned
        if not inference or m > 1024:
            # Training and long prefill: one CTA per row, shared by one and two updates.
            assert (cfg.PARTS, cfg.SPLIT, cfg.CLUSTER) == (1, False, 1)
            if not bf16:
                assert cfg == choose(m, h, s, n, mxfp8=mxfp8, inference=inference)
        elif tuned:
            assert cfg.WIDTH >= 128 * cfg.NUM_WARPS and cfg.PARTS * (1 + cfg.SPLIT) in (1, 2, 4)
            if cfg.CLUSTER > 1:
                assert cfg.CLUSTER <= 8 and cfg.PARTS == 1 and (h // cfg.CLUSTER) % cfg.WIDTH == 0
        else:  # Untuned fallback: replicas by row count.
            replicas = min(8, 128 // triton.next_power_of_2(m)) if m <= 64 else 1
            assert cfg.SPLIT == (replicas > 1 and n == 2)
            assert cfg.PARTS * (1 + cfg.SPLIT) <= replicas
    for h, m, s, n in itertools.product(
        range(32, 16385, 32), (32, 1024, 65536), range(1, 7), (1, 2)
    ):
        grid = backward_kernel.get_backward_knobs(m, h, s, n).TARGET_CTAS
        assert grid > 0 and grid % ((h + 255) // 256) == 0
        assert backward_kernel.get_backward_knobs(m, h, s, n, grid=7).TARGET_CTAS == 7


# ---------------------------------------------------------------- checkpoint replay scopes

# Ops: A/W/R are fused seams with (readers, updates) (1, 1)/(2, 1)/(1, 2); N and M are native
# streamwise writes of an FP32 and a BF16 update; S a native write of a strided update, which the
# native replay cannot take.
_OPS = {"A": (1, 1), "W": (2, 1), "R": (1, 2), "N": (0, 1), "M": (0, 1), "S": (0, 1)}


@GPU
@REQUIRES_FUSED_BACKWARD
@pytest.mark.parametrize(
    "ops,interval,dtype,streams,hidden,retention,pad",
    [
        ("W" * 40, 16, F32, 3, 96, T, 0),  # two full intervals and a partial tail
        ("R" * 40, 16, BF16, 3, 96, T, 0),
        ("MSMN", 8, F32, 3, 96, T, 0),  # an unreplayable write interrupts the interval
    ]
    # A fused, native, fused chain over stream counts, with and without retention.
    + [("AMA", 16, F32, s, h, r, 2) for s, r, h in [(1, F, 96), (3, T, 128), (5, F, 96)]]
    + [("AMA", 16, F32, 6, 288, T, 2)],
)  # fmt: skip
def test_replay_matches_saved_states(ops, interval, dtype, streams, hidden, retention, pad):
    """Replay restores every forward state and reproduces every gradient bit for bit, also in
    a retained second backward on another stream, and then releases its storage."""
    torch.manual_seed(853)
    m, size = 1024, streams + pad
    original = [torch.randn(m, streams * hidden, device="cuda", dtype=dtype)]
    for op in ops:
        readers, updates = _OPS[op]
        logits = dict(device="cuda", dtype=dtype if readers else F32)
        width = 2 * hidden if op == "S" else hidden
        update_dtype = F32 if op == "N" else BF16
        original += [
            torch.randn(m, width, device="cuda", dtype=update_dtype) for _ in range(updates)
        ]
        original += [torch.randn(size, **logits), torch.full((size,), -0.5, **logits)]
        if not retention:
            original[-1] = None
        original += [torch.randn(size, device="cuda", dtype=dtype) for _ in range(readers)]
        original += [torch.randn(hidden, device="cuda", dtype=BF16) for _ in range(readers)]

    def execute(replay):
        leaves = [v if v is None else v.detach().clone().requires_grad_() for v in original]
        values = iter(leaves[1:])
        current, outputs, states, slots = leaves[0], [], [], []
        with residual_replay_scope(replay_config(interval if replay else 1), True) as manager:
            for op in ops:
                if manager is not None:
                    manager.advance(1)
                readers, updates = _OPS[op]
                us = [next(values) for _ in range(updates)]
                controllers = [next(values) for _ in range(2 + readers)]
                gammas = [next(values) for _ in range(readers)]
                if readers:
                    current, *ys = run((current, us, controllers, gammas), streams)
                    outputs += ys
                else:
                    current = streamwise_sigmoid_writeback(
                        current,
                        us[0][:, ::2] if op == "S" else us[0],
                        controllers[0],
                        streams,
                        retention_logits=controllers[1],
                        retention_max_forget=0.1,
                    )
                states.append(current.detach().clone())
                if manager is not None:
                    slots.append(manager.slot(current))
            if manager is not None and "S" in ops:
                # The interval after the unreplayable write is anchored at its output.
                k = ops.index("S")
                assert manager.active.checkpoint.tensor.data_ptr() == slots[k].tensor.data_ptr()
                assert len(manager.active.transitions) == len(ops) - k - 1
                assert not slots[k + 1].resident()
        if slots:
            assert any(not slot.resident() for slot in slots)
            all_bytes_equal([slot.ensure() for slot in reversed(slots)], reversed(states))
        loss = current.float().square().mean() + sum(y.float().square().mean() for y in outputs)
        leaves = [leaf for leaf in leaves if leaf is not None]
        first = torch.autograd.grad(loss, leaves, retain_graph=True, allow_unused=True)
        # A second stream must see the restored allocation and its allocator metadata.
        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            second = torch.autograd.grad(loss, leaves, allow_unused=True)
        torch.cuda.current_stream().wait_stream(other)
        all_bytes_equal(first, second)
        del slots
        gc.collect()
        # Ordinary backward releases the replay storage owners after their final launches.
        assert manager is None or not manager.slots
        return states + list(first)

    all_bytes_equal(execute(False), execute(True))


@GPU
@pytest.mark.parametrize(
    "shape,streams,retention,dtype,alternate,forget",
    [
        ((5, 33), 1, F, torch.float16, F, 0.07),
        ((5, 33), 3, T, BF16, T, 0.07),
        ((2, 3, 128), 3, F, F32, F, 0.07),
        ((2, 3, 128), 6, T, torch.float64, T, 0.07),
        ((2, 2, 3, 33), 6, F, BF16, F, 0.07),
        ((2, 2, 3, 33), 1, T, F32, T, 0.07),
        ((32, 96), 3, F, F32, F, 0.1),  # max_forget is a kernel constexpr
    ],
)
def test_native_replay_intervals(shape, streams, retention, dtype, alternate, forget):
    """Native writes free their predecessor at the successor's registration (also from
    alternating streams); intervals count physical layers, not writes; each scope owns its
    workspaces; replay restores the actual native forward bit for bit (ranks, tails, signed
    zeros, logit casts), also consumed out of order from another stream."""
    owners = []
    for seed in (23, 47):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        update = torch.randn(shape, device="cuda", dtype=BF16, generator=generator)
        current = torch.randn(
            (*shape[:-1], streams * shape[-1]), device="cuda", dtype=F32, generator=generator
        )
        update.view(-1)[0] = 0.0
        current.view(-1)[0] = -0.0
        anchor = current.clone()
        write = torch.linspace(-4, 4, streams + 2, device="cuda", dtype=dtype)
        retain = write.flip(0) if retention else None
        side = [torch.cuda.current_stream(), torch.cuda.Stream()]
        links, expected, identities = [], [], []
        with residual_replay_scope(replay_config(3), True) as manager:
            for physical_layers in (1, 2, 1, 1):
                manager.advance(physical_layers)
                for _ in range(2):  # interval capacity counts physical layers, not writes
                    k = len(links)
                    stream = side[k % 2] if alternate else side[0]
                    stream.wait_stream(side[(k - 1) % 2] if alternate else side[0])
                    with torch.cuda.stream(stream):
                        following = _streamwise_sigmoid_write_triton(
                            current, update, write, streams, retain, forget
                        )
                        links.append(
                            register_write(
                                current, following, update, write, retain, streams, forget
                            )
                        )
                        # Freed unless it anchors an interval (the first, or a sealed last write).
                        assert bool(current.untyped_storage().nbytes()) == (k in (0, 4))
                        expected.append(following.clone())
                    identities.append(following.untyped_storage()._cdata)
                    current = following
        side[0].wait_stream(side[1])
        assert [link.current_slot.resident() for link in links] == [F, F, F, T, F, F, F, T]
        assert links[0].interval is links[3].interval
        assert links[3].interval is not links[4].interval
        assert links[4].interval is links[7].interval
        assert [len(links[i].interval.transitions) for i in (0, 4)] == [4, 4]
        owners.append((links, expected, identities, anchor))
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        for links, expected, identities, anchor in reversed(owners):
            for i in (2, 0, 6, 4, 1, 3, 5, 7):
                actual = links[i].current_slot.ensure()
                assert actual.untyped_storage()._cdata == identities[i]
                assert bytes_equal(actual, expected[i])
            assert links[0].interval.materialized
            assert bytes_equal(links[0].previous_slot.ensure(), anchor)
    torch.cuda.current_stream().wait_stream(consumer)
    backings = [owner[0][0].current_slot.backing.data_ptr() for owner in owners]
    assert backings[0] != backings[1]


@GPU
def test_replay_scope_ownership():
    """Nested scopes isolate ownership and always release their endpoint handle."""
    config = replay_config()
    for raises, nested_training in itertools.product((F, T), (F, T)):
        endpoint = ResidualSlot(torch.ones(1, device="cuda"), anchor=False)
        marker = object()
        endpoint.provider = marker
        with residual_replay_scope(config, True) as outer:
            with torch.no_grad():
                assert current_manager() is None
            try:
                with residual_replay_scope(config, nested_training) as inner:
                    assert current_manager() is inner
                    assert inner is not outer
                    if inner is not None:
                        previous = torch.zeros(1, device="cuda")
                        link = inner.register(
                            previous, previous + 1, native=lambda value: value + 1
                        )
                        endpoint = link.current_slot
                        endpoint.provider = marker
                    if raises:
                        raise ValueError("expected scope interruption")
            except ValueError as error:
                assert str(error) == "expected scope interruption"
            assert current_manager() is outer
            if inner is not None:
                assert inner.last is None
            assert endpoint.anchor == (nested_training and not raises)
            assert endpoint.provider is (None if endpoint.anchor else marker)
        assert current_manager() is None
        assert outer.last is None


# ---------------------------------------------------------------- MXFP8 handoff


@pytest.fixture
def reader_work(monkeypatch):
    """Count column reconstructions at the actual entry points without production counters."""
    active = ContextVar("reader_under_test", default=None)
    initialize, backward = Reader.__init__, api._Boundary.backward
    native_launch = backward_kernel.backward

    def initialize_reader(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.reconstructions = self.fused_calls = 0

    def boundary_backward(ctx, *args, **kwargs):
        token = active.set(ctx.mxfp8_reader)
        try:
            return backward(ctx, *args, **kwargs)
        finally:
            active.reset(token)

    def observe_native(*args, **kwargs):
        reader, callback = active.get(), kwargs.get("on_columns_ready")
        if reader is not None and callback is not None:

            def ready(columns):
                if columns[0] is not None:
                    reader.reconstructions += 1
                    reader.fused_calls += 1
                callback(columns)

            kwargs["on_columns_ready"] = ready
        return native_launch(*args, **kwargs)

    monkeypatch.setattr(Reader, "__init__", initialize_reader)
    monkeypatch.setattr(api._Boundary, "backward", staticmethod(boundary_backward))
    monkeypatch.setattr(backward_kernel, "backward", observe_native)


def mxfp8_test(function):
    """Qualified GB300 MXFP8 test with reconstruction counters on every handoff reader."""
    return QUALIFIED(pytest.mark.usefixtures("reader_work")(function))


def linear(h, *, mcore=False, bias=False):
    if mcore:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=h,
            num_attention_heads=1,
            params_dtype=BF16,
            gradient_accumulation_fusion=True,
        )
        layer = TELinear(
            h,
            128,
            parallel_mode="duplicated",
            config=config,
            init_method=torch.nn.init.normal_,
            bias=bias,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
    else:
        layer = te.Linear(
            h, 128, bias=bias, params_dtype=BF16, device="cuda", fuse_wgrad_accumulation=True
        )
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=F32)
    layer.weight.grad_added_to_main_grad = False
    layer.weight.zero_out_wgrad = True
    return layer


def consume(layer, q):
    with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
        if isinstance(layer, TELinear):
            return layer(q)[0]
        result = layer(q)
        install_linear_hooks(q, result, layer)
        return result


def leaf_grads(raw, *extra):
    return [t.grad for t in (raw[0], *raw[1], *raw[2], *raw[3])] + list(extra)


@triton.jit
def normalized_reader_oracle(R, U, C, RS, G, OUT, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    x = tl.full((BLOCK,), 0, tl.float32)
    q = 0.0
    for stream in tl.static_range(3):
        value = tl.load(R + row * 3 * H + stream * H + cols, cols < H, 0)
        beta = tl.load(C + 6 + stream)
        alpha = beta * tl.load(C + 3 + stream)
        x = tl.fma(alpha, value, x)
        q = tl.fma(beta, tl.load(C + stream), q)
    x = tl.fma(q, tl.load(U + row * H + cols, cols < H, 0).to(tl.float32), x)
    normalized = (x * tl.load(RS + row)) * tl.load(G + cols, cols < H, 0).to(tl.float32)
    tl.store(OUT + row * H + cols, normalized, cols < H)


def oracle_normalized(r, update, coefficients, rstd, gamma):
    m, h = update.shape
    x = torch.empty((m, h), device="cuda", dtype=F32)
    normalized_reader_oracle[(m, triton.cdiv(h, 256))](
        r, update, coefficients, rstd, gamma, x, h, 256, enable_fp_fusion=False
    )
    return x


class _ReferenceQuantization(torch.autograd.Function):
    """Use independent FP32 normalization/TE encoding while preserving the dX edge."""

    @staticmethod
    def forward(ctx, operand, normalized):
        return mxfp8_quantizer()(normalized).to_dtype(BF16)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def reference_operand(operand, raw):
    """Recreate both orientations independently; production emits only columns in backward."""
    r, us, _, gs = raw
    # The boundary saves (residual, update, coefficients, *gammas, *rstd).
    _, _, coefficients, *norms = operand.grad_fn.saved_tensors
    update = us[0] if len(us) == 1 else (us[0].float() + us[1].float()).bfloat16()
    normalized = oracle_normalized(r, update, coefficients, norms[len(gs)], gs[0])
    return _ReferenceQuantization.apply(operand, normalized)


def ordered_reductions(patcher, order, first):
    """Record controller-chunk, then parameter reductions, both after ``first``."""
    reduce = backward_kernel.reduce_jobs

    def recorded(*args, **kwargs):
        assert order == ([first] if len(order) == 1 else [first, "controller_chunks"])
        order.append("controller_chunks" if len(order) == 1 else "parameters")
        return reduce(*args, **kwargs)

    patcher.setattr(backward_kernel, "reduce_jobs", recorded)


@mxfp8_test
@pytest.mark.parametrize("m,h", MX_SHAPES)
@pytest.mark.parametrize("readers", [1, 2])
def test_fp32_bytes_and_te_weight_gradient(monkeypatch, m, h, readers):
    raw = mx_inputs(m, h, readers)
    r, us, cs, gs = raw
    # Use the quantized path's statistics so TE receives identical FP32 inputs;
    # BF16-only schedules may use a different RMSNorm reduction order.
    current, saved, operands = fwd(raw, mxfp8_handoff=True)
    x = oracle_normalized(r, saved.update, saved.coefficients, saved.rstd[0], gs[0])
    qref = mxfp8_quantizer()(x.detach()).to_dtype(BF16).requires_grad_(True)
    layer = linear(h)
    out = boundary(raw, layer)
    q, provider = out[1], reader_for(out[1])
    assert q._columnwise_data is None
    assert torch.equal(q._rowwise_data, qref._rowwise_data)
    assert torch.equal(q._rowwise_scale_inv, qref._rowwise_scale_inv)
    compute_wgrad = Reader.compute_wgrad

    def check_columns(self, current, column):
        assert torch.equal(column._columnwise_data, qref._columnwise_data)
        assert torch.equal(column._columnwise_scale_inv, qref._columnwise_scale_inv)
        return compute_wgrad(self, current, column)

    monkeypatch.setattr(Reader, "compute_wgrad", check_columns)
    if readers == 2:
        assert torch.equal(out[2], operands[1])
    dy = torch.randn(m, 128, device="cuda", dtype=BF16)
    yref = consume(layer, qref)
    yref.backward(dy)
    wg = layer.weight.main_grad.clone()
    layer.weight.main_grad.zero_()
    before = provider.reconstructions
    y = consume(layer, q)
    assert torch.equal(y, yref)
    assert provider.reconstructions == before
    torch.autograd.backward(
        (y, out[0], *out[2:]),
        (dy, torch.zeros_like(out[0]), *(torch.zeros_like(value) for value in out[2:])),
    )
    assert provider.reconstructions == before + 1
    assert torch.equal(layer.weight.main_grad, wg)
    assert r.grad is not None and torch.isfinite(r.grad).all()


@mxfp8_test
@pytest.mark.parametrize("m,h", MX_SHAPES)
@pytest.mark.parametrize("readers", [1, 2])
def test_mcore_fused_backward_matches_independent_columns(m, h, readers):
    records = []
    for exclusive in [False, True]:
        raw = mx_inputs(m, h, readers)
        layer = linear(h, mcore=True, bias=True)
        observed_wgrad = []
        layer.weight.register_post_accumulate_grad_hook(
            lambda weight: observed_wgrad.append(weight.main_grad.clone())
        )
        out = boundary(raw, layer)
        provider = reader_for(out[1])
        # TE's flatten/view path must preserve producer lookup.
        q = out[1].view(1, m, h)
        assert reader_for(q) is provider
        y = consume(layer, q if exclusive else reference_operand(out[1], raw).view(1, m, h))
        outputs = [y, *out[0:1], *out[2:]]
        torch.autograd.backward(outputs, [torch.randn_like(t) for t in outputs])
        assert len(observed_wgrad) == 1
        assert torch.equal(observed_wgrad[0], layer.weight.main_grad)
        grads = leaf_grads(raw, layer.weight.main_grad, layer.bias.grad)
        records.append((y.detach().cpu(), [g.cpu() for g in grads]))
        assert provider.fused_calls == int(exclusive)
        assert not provider.pending
        # Keeping outputs/grad_fn alive must not keep the row allocation alive.
        storage = StorageWeakRef(q._rowwise_data.untyped_storage())
        del q, out, y, outputs, raw, layer, grads
        gc.collect()
        assert storage.expired()
    assert torch.equal(records[0][0], records[1][0])
    # Weight and bias gradients remain exact; the rest may change FP32 accumulation order.
    assert all(torch.equal(a, b) for a, b in zip(records[0][1][-2:], records[1][1][-2:]))
    assert_grads_close(records[0][1][:-2], records[1][1][:-2])


@mxfp8_test
@pytest.mark.parametrize("replay", [False, True])
def test_multiple_live_calls_and_replay(replay):
    raw = mx_inputs(8192, 4096, 1)
    layer = linear(4096, mcore=True)
    with residual_replay_scope(replay_config(), replay):
        first = boundary(raw, layer)
        y0 = consume(layer, first[1])
        second = boundary((first[0], *raw[1:]), layer)
        y1 = consume(layer, second[1])
    providers = [reader_for(first[1]), reader_for(second[1])]
    outputs = (y0, y1, second[0])
    torch.autograd.backward(outputs, tuple(torch.ones_like(t) for t in outputs))
    assert all(p.reconstructions == 1 and p.fused_calls == 1 for p in providers)
    assert torch.isfinite(raw[0].grad).all()


@mxfp8_test
def test_handoff_rejects_invalid_consumers():
    """Training needs an exclusive consumer with deferred wgrad and fused columns."""
    with pytest.raises(ValueError, match="consumer"):
        boundary(mx_inputs(1024, 4096, 1))
    raw = mx_inputs(2048, 2048, 1)
    layer = linear(2048, mcore=True)
    layer.fuse_wgrad_accumulation = False
    out = boundary(raw, layer)
    with pytest.raises(RuntimeError, match="requires deferred weight gradients"):
        consume(layer, out[1])
    out = boundary(raw, linear(2048, mcore=True))
    provider = reader_for(out[1])
    # A pending TE GEMM must never silently launch a standalone regeneration.
    provider.pending.append(None)
    with pytest.raises(RuntimeError, match="require fused MXFP8 columns"):
        provider.compute_wgrad(out[0], None)
    provider.pending.clear()


@mxfp8_test
@pytest.mark.parametrize("native_fp8_weight", [False, True])
def test_saved_weight_keeps_identity_and_metadata(native_fp8_weight):
    # GTP's native FP8 parameter carries prefetch/communication state in Python attributes,
    # which a scope-wide saved-tensor hook would lose when autograd detaches the saved weight.
    raw = mx_inputs(1024, 4096, 1)
    with te.fp8_model_init(enabled=native_fp8_weight, recipe=MXFP8BlockScaling()):
        layer = linear(4096)
    layer.weight.prefetch_identity = object()
    out = boundary(raw, layer)
    y = consume(layer, out[1])
    weights = [t for t in y.grad_fn.saved_tensors if t is layer.weight]
    assert weights
    assert weights[0].prefetch_identity is layer.weight.prefetch_identity
    torch.autograd.backward((y, out[0]), (torch.ones_like(y), torch.ones_like(out[0])))
    assert torch.isfinite(layer.weight.main_grad).all()


@mxfp8_test
@pytest.mark.parametrize("side", ["carry", "consumer"])
def test_other_stream_matches_ordinary_backward(side):
    """A carry loss or the consumer on another stream keeps exact gradients and wgrad
    lifetimes (freed consumer-stream allocations are reused immediately)."""
    results = []
    for exclusive in [False, True]:
        raw = mx_inputs(8192, 4096, 1)
        layer = linear(4096, mcore=True)
        out = boundary(raw, layer)
        operand = out[1] if exclusive else reference_operand(out[1], raw)
        other = torch.cuda.Stream()
        if side == "carry":
            y = consume(layer, operand)
            other.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(other):
                carry_loss = out[0].square().sum()
            torch.cuda.current_stream().wait_stream(other)
            torch.autograd.backward((y, carry_loss), (torch.ones_like(y), None))
        else:
            other.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(other):
                y = consume(layer, operand)
            torch.cuda.current_stream().wait_stream(other)
            torch.autograd.backward((y, out[0]), (torch.ones_like(y), torch.ones_like(out[0])))
            # A deferred GEMM on the boundary stream must not read TE buffers after reuse.
            with torch.cuda.stream(other):
                scratch = [
                    torch.empty(8192, 128, device="cuda").fill_(float("nan")) for _ in range(8)
                ]
        torch.cuda.synchronize()
        results.append([g.clone() for g in leaf_grads(raw, layer.weight.main_grad)])
        provider = reader_for(out[1])
        assert provider.reconstructions == provider.fused_calls == int(exclusive)
    all_bytes_equal(*results)


@mxfp8_test
@pytest.mark.parametrize("m,h", kernel_utils.FLAGSHIPS)
@torch.no_grad()
def test_native_update_rebinding_and_captured_descriptors(m, h):
    """Each captured graph must retain the descriptor for its own saved update."""
    _, (saved, r, dys, carry) = backward_args(mx_inputs(m, h, 1))
    changed = replace(saved, update=-saved.update)
    grid = backward_kernel.get_backward_knobs(m, h, 3, 1).TARGET_CTAS
    options = dict(config=grid, save_columns=(True,))

    def outputs(ws):
        column = ws.columns[0]
        return (*ws.grads, column._columnwise_data, column._columnwise_scale_inv)

    graphs, workspaces, expected = [], [], []
    for state in (saved, changed):
        _, independent = run_with_scratch(backward_kernel.backward, state, r, dys, carry, **options)
        expected.append(tuple(None if x is None else x.clone() for x in outputs(independent)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, workspace = run_with_scratch(
                backward_kernel.backward, state, r, dys, carry, **options
            )
        graphs.append(graph)
        workspaces.append(workspace)
    # Alternating captures must preserve independent outputs and descriptor addresses.
    for index in (0, 1, 1, 0):
        scratch, actual = workspace_tensors(workspaces[index]), outputs(workspaces[index])
        replay_poisoned(graphs[index], scratch, actual, expected[index])
    # The changed saved update must affect a derivative with an unchanged predecessor.
    assert not torch.equal(expected[0][2], expected[1][2])


@mxfp8_test
@pytest.mark.parametrize("native_fp8_weight", [False, True])
def test_checkpointed_bf16_handoff_preserves_weight_readiness(native_fp8_weight):
    """A reentrant non-GTP GDP checkpoint must publish one fully computed weight gradient."""
    from megatron.core.tensor_parallel.random import (
        CheckpointWithoutOutput,
        model_parallel_cuda_manual_seed,
    )

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        raw = mx_inputs(8192, 4096, 2)
        with te.fp8_model_init(enabled=native_fp8_weight, recipe=MXFP8BlockScaling()):
            layer = linear(4096, mcore=True)
        observed = []
        # Use the same AccumulateGrad-node hook that Megatron DDP registers.
        accumulator = layer.weight.expand_as(layer.weight).grad_fn.next_functions[0][0]
        accumulator.register_hook(lambda *unused: observed.append(layer.weight.main_grad.clone()))
        # A checkpointed non-GTP projection cannot defer its weight gradient.
        current, q, router = boundary(raw, None, mxfp8_handoff=False)
        assert reader_for(q) is None and q.dtype == BF16
        checkpoint = CheckpointWithoutOutput(fp8=MXFP8BlockScaling())
        with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
            output = checkpoint.checkpoint(lambda value: consume(layer, value), q)
        loss = output.square().sum()
        checkpoint.discard_output_and_register_recompute(loss)
        torch.autograd.backward(
            (loss, current, router), (None, torch.ones_like(current), torch.ones_like(router))
        )
        assert len(observed) == 1
        assert torch.equal(observed[0], layer.weight.main_grad)
        assert torch.count_nonzero(observed[0]) > 0
    finally:
        Utils.destroy_model_parallel()


@QUALIFIED
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("other_stream", [False, True])
def test_deferred_gtp_weight_gradient_matches_independent_columns(
    monkeypatch, checkpointed, other_stream
):
    """GTP completion can precede controller reduction, including checkpoint replay.

    Needs ``torchrun --nproc-per-node 2`` (real GTP sharding); skips with one rank.
    """
    from transformer_engine.pytorch import distributed_weight

    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        _GTP_PARAMS,
        reset_gtp_state,
    )
    from megatron.core.tensor_parallel.random import (
        CheckpointWithoutOutput,
        model_parallel_cuda_manual_seed,
    )
    from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import _make_gtp_linear

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    group = None
    try:
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            pytest.skip("requires at least two torchrun ranks for real GTP sharding")
        group = torch.distributed.new_group(list(range(world_size)))
        model_parallel_cuda_manual_seed(123)
        records = []
        for deferred in (False, True):
            reset_gtp_state()
            raw = mx_inputs(2048, 2048, 2)
            layer = _make_gtp_linear(2048, 128, group, fuse_wgrad_accumulation=True)
            weight = layer.weight
            assert weight.is_gtp_weight_remat and weight.is_distributed_weight
            weight.main_grad = torch.zeros(weight.shape, device="cuda", dtype=F32)
            weight.grad_added_to_main_grad = False
            weight.zero_out_wgrad = True
            current, q, router = boundary(raw, layer)
            reader = reader_for(q)
            if not deferred:
                q = reference_operand(q, raw)
            stream = torch.cuda.Stream() if other_stream else torch.cuda.current_stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                if checkpointed:
                    checkpoint = CheckpointWithoutOutput(fp8=MXFP8BlockScaling())
                    with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                        output = checkpoint.checkpoint(lambda value: consume(layer, value), q)
                    loss = output.square().sum()
                    checkpoint.discard_output_and_register_recompute(loss)
                else:
                    output = consume(layer, q)
                    loss = output.square().sum()
            torch.cuda.current_stream().wait_stream(stream)

            order = []
            finalize = distributed_weight.finalize_weight_grads

            def finish_weight(saved_weight, grads):
                assert saved_weight.is_gtp_weight_remat
                assert not order
                result = finalize(saved_weight, grads)
                order.append("weight")
                return result

            with monkeypatch.context() as patcher:
                if deferred:
                    patcher.setattr(distributed_weight, "finalize_weight_grads", finish_weight)
                    ordered_reductions(patcher, order, "weight")
                torch.autograd.backward(
                    (loss, current, router),
                    (None, torch.ones_like(current), torch.ones_like(router)),
                )
            torch.cuda.synchronize()
            assert layer.weight is weight
            assert not reader.pending
            assert torch.count_nonzero(weight.main_grad) > 0
            if deferred:
                assert order == ["weight", "controller_chunks", "parameters"]
            grads = [g.cpu() for g in leaf_grads(raw)]
            records.append((output.detach().cpu(), weight.main_grad.cpu(), grads))

        torch.testing.assert_close(records[1][0], records[0][0], rtol=0, atol=0)
        torch.testing.assert_close(records[1][1], records[0][1], rtol=0, atol=0)
        assert_grads_close(records[1][2], records[0][2])
    finally:
        reset_gtp_state()
        _GTP_PARAMS.clear()
        if group is not None:
            torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


@mxfp8_test
@pytest.mark.parametrize("hidden", [32, 96, 256, 2048, 3072, 6144, 10240])
def test_generic_norm_matches_unfused_oracle(hidden):
    torch.manual_seed(947)
    rows = 128
    x = torch.randn(rows, hidden, device="cuda")
    gamma = torch.randn(hidden, device="cuda", dtype=BF16)
    quantizer = mxfp8_quantizer(columnwise=False)
    expected_rs = (x.double().square().mean(-1) + 1.0e-5).rsqrt()
    # One identity-retained stream with zero update and a half-strength reader turns
    # residual=2*x into exactly x, isolating the production norm arithmetic.
    update = torch.zeros_like(x, dtype=BF16)
    coefficients = torch.empty(8, device=x.device)
    controller = torch.zeros(1, device=x.device)
    actual_rs = torch.empty(rows, device=x.device, dtype=F32)
    actual = quantizer.make_empty(x.shape, dtype=BF16, device=x.device)
    actual._rowwise_scale_inv.zero_()
    forward_kernel._forward[(rows,)](
        forward_kernel._WriteArgs(2 * x, update, update, torch.empty_like(x), None),
        read_tensors=forward_kernel._ReadArgs(
            (gamma,), (actual_rs,), (actual._rowwise_data,), (actual._rowwise_scale_inv,)
        ),
        controllers=(controller, None, controller, controller),
        saved_coefficients=coefficients,
        H=hidden,
        STREAMS=1,
        HAS_RETENTION=False,
        CONFIG=forward_kernel._ForwardConfig(WIDTH=triton.next_power_of_2(hidden)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Reduction order may change, but statistics must stay accurate against FP64.
    torch.testing.assert_close(actual_rs.double(), expected_rs, rtol=2e-6, atol=0)
    # Isolate encoding: TE must receive the same FP32 prequantization values.
    expected = quantizer((x * actual_rs[:, None]) * gamma.float())
    assert torch.equal(actual._rowwise_data, expected._rowwise_data)
    assert torch.equal(actual._rowwise_scale_inv, expected._rowwise_scale_inv)

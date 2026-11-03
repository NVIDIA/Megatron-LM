# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Kernel and API contracts of the fused wide-residual shortcut boundaries.

``wide_residual_write_read_rmsnorm_quant`` (one reader) and
``shortcut_add_wide_residual_write_two_reader_rmsnorm_quant`` (two readers of a shortcut add, or of
a single write when the pending add has no shared update): independent FP64
and autograd references, backward schedules and scratch ownership, configs and shapes,
checkpoint replay, inference, and the MXFP8 handoff to Transformer Engine (including real GTP
sharding under ``torchrun --nproc-per-node 2 -m pytest <this file> -k gtp``). Bitwise determinism
and poisoned CUDA-graph replay live in ``tests/unit_tests/determinism/kernels/``.
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
    PendingShortcutAdd,
    api,
    backward_kernel,
    forward_kernel,
    kernel_utils,
    replay_kernel,
    shortcut_add_wide_residual_write_read_rmsnorm_quant,
    shortcut_add_wide_residual_write_two_reader_rmsnorm_quant,
    wide_residual_write_read_rmsnorm_quant,
)
from megatron.core.fusions.wide_residual_shortcut.mxfp8_handoff import (
    Reader,
    eligible,
    install_linear_hooks,
    reader_for,
)
from megatron.core.fusions.wide_residual_shortcut.replay_kernel import replay_interval
from megatron.core.fusions.wide_residual_shortcut.residual_replay import (
    ResidualSlot,
    current_manager,
    register_write,
    residual_replay_scope,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.streamwise_residual_ops import (
    _streamwise_sigmoid_write_triton,
    _StreamwiseSigmoidWriteback,
    streamwise_sigmoid_writeback,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import bytes_equal
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
REQUIRES_FUSED_BACKWARD = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not kernel_utils.qualified_device(torch.cuda.get_device_properties(0)),
    reason="the fused shortcut backward runs only on a qualified GB300",
)
QUALIFIED = pytest.mark.skipif(
    te is None
    or not torch.cuda.is_available()
    or not kernel_utils.qualified_device(torch.cuda.get_device_properties(0)),
    reason="qualified GB300 schedules and MXFP8 kernels",
)
F, T = False, True
# Rows x width of the Nano/Super/Ultra flagships and of expanded admitted shapes.
MX_SHAPES = [
    *kernel_utils.FLAGSHIPS,
    (2048, 2048),
    (2048, 16384),
    (16384, 2048),
    (16384, 16384),
    (3072, 6144),
    (6144, 12288),
    (12288, 15360),
    (8192, 13312),
]

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
    residual_dtype=torch.float32,
    logit_dtype=torch.float32,
):
    """Seeded leaves (residual, updates, controllers, gammas).

    ``updates`` defaults to one per reader (two readers: a shortcut add). Controllers carry
    ``pad`` unused trailing logits; retention=False drops the retain controller (static
    identity) after drawing it. The residual and the controller logits are FP32 or BF16.
    """
    torch.manual_seed(seed)
    bf16 = torch.bfloat16

    def leaf(*shape, dtype=torch.float32):
        return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)

    residual = leaf(tokens, streams * hidden, dtype=residual_dtype)
    updates = tuple(leaf(tokens, hidden, dtype=bf16) for _ in range(updates or readers))
    controllers = tuple(leaf(streams + pad, dtype=logit_dtype) for _ in range(2 + readers))
    gammas = tuple(leaf(hidden, dtype=bf16) for _ in range(readers))
    if not retention:
        controllers = (controllers[0], None, *controllers[2:])
    return residual, updates, controllers, gammas


def mx_inputs(m, h, readers, updates=None):
    """Three-stream MXFP8 handoff inputs with unpadded controllers."""
    return inputs(3, readers, h, m, seed=319, pad=0, updates=updates)


def run(raw, streams=3, epsilon=1e-5, **options):
    """Call the one-reader or two-reader public API (a shortcut add with two updates)."""
    residual, updates, controllers, gammas = raw
    options.update(num_streams=streams, epsilon=epsilon)
    if len(gammas) == 1 and len(updates) == 2:
        return shortcut_add_wide_residual_write_read_rmsnorm_quant(
            PendingShortcutAdd(residual, updates[0], updates[1], *controllers[:2]),
            controllers[2],
            gammas[0],
            **options,
        )
    if len(gammas) == 1:
        return wide_residual_write_read_rmsnorm_quant(
            residual, updates[0], *controllers, gammas[0], **options
        )
    shared = updates[1] if len(updates) == 2 else None
    return shortcut_add_wide_residual_write_two_reader_rmsnorm_quant(
        PendingShortcutAdd(residual, updates[0], shared, *controllers[:2]),
        controllers[2:],
        gammas,
        **options,
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


def native_backward(saved, previous, dys, carry, *, config, save_columns=True):
    """Explicit ordinary-launch schedules; a bool routes every reader's columns."""
    columns = (
        save_columns if isinstance(save_columns, tuple) else (save_columns,) * saved.reader_count
    )
    return backward_kernel.backward(
        saved, previous, dys, carry, config=config, prepare_wgrad=columns
    )


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
    """Observe allocations and launch metadata without production debug objects."""
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
        grads=grads,
        columns=columns,
        allocations=allocations,
        kernel=launches[-1] if launches else None,
    )


def capture_backward_config(m, h, streams, readers, *, columns, retention, grid, updates=None):
    """Inspect real backward scheduling while replacing allocation and GPU launch."""

    class Captured(Exception):
        pass

    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        raise Captured

    saved = SimpleNamespace(
        update=SimpleNamespace(shape=(m, h), device="cuda", dtype=torch.bfloat16),
        coefficients=SimpleNamespace(new_empty=lambda *args, **kwargs: object()),
        reader_count=readers,
        update_count=updates or readers,
        num_streams=streams,
        has_retention=retention,
        controller_sizes=(streams, streams if retention else 0) + (streams,) * readers,
        rstd=(None,) * readers,
        gammas=(None,) * readers,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(backward_kernel, "qualified_device", return_value=True))
        stack.enter_context(patch.object(torch.cuda, "get_device_properties", return_value=None))
        stack.enter_context(
            patch.object(
                replay_kernel,
                "replay_columns",
                lambda *args, readers: tuple(object() if need else None for need in readers),
            )
        )
        stack.enter_context(patch.object(backward_kernel, "_descriptor", lambda *args: None))
        stack.enter_context(patch.object(backward_kernel._backward, "warmup", capture))
        stack.enter_context(patch.dict(backward_kernel._RESIDENCY, clear=True))
        try:
            backward_kernel.backward(
                saved,
                SimpleNamespace(device="cuda", dtype=torch.float32),
                (SimpleNamespace(dtype=torch.bfloat16),) * readers,
                SimpleNamespace(dtype=torch.float32),
                columns=(None,) * readers,
                prepare_wgrad=columns,
                config=grid,
            )
        except Captured:
            return SimpleNamespace(
                **{
                    name: getattr(value, "value", value)
                    for name, value in observed["CONFIG"]._asdict().items()
                },
                COLUMN_ROUTES=(
                    (columns,) + (False,) * (readers - 1) if isinstance(columns, bool) else columns
                ),
            )
    raise AssertionError("backward did not prepare a launch")


def replay_config(interval=16):
    return SimpleNamespace(
        residual_stream_recompute_num_layers=interval, moe_shortcut_wide_residual_norm_fusion=True
    )


# ---------------------------------------------------------------- independent references


def mxfp8_quantizer(*, rowwise=True, columnwise=True):
    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)
    quantizer.optimize_for_gemm = True
    return quantizer


def reference(raw, streams, epsilon=1e-5):
    """Independent FP64 formula with the specified BF16 branch-sum/output boundaries.

    The current residual is rounded to the residual's dtype; readers fold unrounded values.
    """
    residual, updates, controllers, gammas = raw
    hidden = updates[0].shape[-1]
    r = residual.double().view(-1, streams, hidden)
    w = 2 * controllers[0][:streams].double().sigmoid()
    t = (
        torch.ones_like(w)
        if controllers[1] is None
        else 1 - 0.1 * (-controllers[1][:streams].double()).sigmoid()
    )
    u = updates[0] if len(updates) == 1 else (updates[0].float() + updates[1].float()).bfloat16()
    current = t[None, :, None] * r + w[None, :, None] * u.double()[:, None, :]
    outputs = []
    for read, gamma in zip(controllers[2:], gammas):
        x = (read[:streams].double().sigmoid()[None, :, None] * current).sum(1)
        rs = (x.square().mean(-1, keepdim=True) + epsilon).rsqrt()
        outputs.append(((x * rs) * gamma.double()).bfloat16())
    return current.flatten(1).to(residual.dtype), *outputs


def folded_reader(saved, previous, reader):
    """Independent FP64 emulation of the forward's explicit FP32 RN/FMA boundaries."""
    m, h = saved.update.shape
    c = saved.coefficients.double().view(8, saved.num_streams)
    values = previous.double().view(m, saved.num_streams, h)
    x = torch.zeros_like(saved.update, dtype=torch.float32)
    q = torch.zeros((), device=previous.device, dtype=torch.float32)
    for stream in range(saved.num_streams):
        beta = c[2 + reader, stream]
        alpha = (beta * c[1, stream]).float().double()
        x = (alpha * values[:, stream] + x.double()).float()
        q = (beta * c[0, stream] + q.double()).float()
    return (q.double() * saved.update.double() + x.double()).float()


def reader_dx(saved, previous, reader, dy):
    """FP32 folded reader, FP64 normalized value, and FP64 RMSNorm input gradient."""
    x = folded_reader(saved, previous, reader)
    rstd = saved.rstd[reader].double()[:, None]
    z = x.double() * rstd
    weighted = dy.double() * saved.gammas[reader].double()
    return x, z, (weighted - z * (weighted * z).mean(-1, keepdim=True)) * rstd


@torch.no_grad()
def standard_backward(saved, previous, dys, carry, *, prepare_wgrad=None):
    """Independent FP64 derivatives, using exact predecessors and saved statistics."""
    streams, readers = saved.num_streams, saved.reader_count
    m, h = saved.update.shape
    coefficients = saved.coefficients.double().view(8, streams)
    values = previous.double().view(m, streams, h)
    update = saved.update.double()
    total = (
        (torch.zeros_like(previous) if carry is None else carry).double().view_as(values).clone()
    )
    read_grads, gamma_grads, columns = [], [], []
    routes = (False,) * readers if prepare_wgrad is None else prepare_wgrad
    current_math = (
        coefficients[1][None, :, None] * values
        + coefficients[0][None, :, None] * update[:, None, :]
    )
    for reader in range(readers):
        dy = torch.zeros_like(saved.update) if dys[reader] is None else dys[reader]
        x, z, dx = reader_dx(saved, previous, reader, dy)
        total += coefficients[2 + reader][None, :, None] * dx[:, None, :]
        read_grads.append((current_math * dx[:, None, :]).sum((0, 2)) * coefficients[6 + reader])
        gamma_grads.append((dy.double() * z).sum(0).float())
        normalized = (x * saved.rstd[reader][:, None]) * saved.gammas[reader].float()
        columns.append(mxfp8_quantizer(rowwise=False)(normalized) if routes[reader] else None)
    residual_grad = (total * coefficients[1][None, :, None]).flatten(1).to(previous.dtype)
    update_grad = (total * coefficients[0][None, :, None]).sum(1).bfloat16()
    control_values = [
        (total * update[:, None, :]).sum((0, 2)) * coefficients[4],
        (total * values).sum((0, 2)) * coefficients[5],
        *read_grads,
    ]
    control_grads = []
    for value, size in zip(control_values, saved.controller_sizes):
        gradient = None
        if size:
            gradient = torch.zeros(size, device=previous.device, dtype=torch.float32)
            gradient[:streams] = value.float()
        control_grads.append(gradient)
    return (
        (residual_grad, *((update_grad,) * saved.update_count), *control_grads, *gamma_grads),
        tuple(columns),
    )


def assert_grads_close(actual, expected, first=(2e-5, 2e-5), fp32=(5e-4, 2e-4)):
    """FP32 residual (``first``), BF16 (1/128) and other (``fp32``; None skips) gradients."""
    for index, (a, b) in enumerate(zip(actual, expected)):
        bf16 = getattr(a, "dtype", None) == torch.bfloat16
        tolerance = (1 / 128, 1 / 128) if bf16 else first if index == 0 else fp32
        if tolerance:
            torch.testing.assert_close(a, b, rtol=tolerance[0], atol=tolerance[1])


def assert_columns_equal(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert bytes_equal(a._columnwise_data, b._columnwise_data)
            assert bytes_equal(a._columnwise_scale_inv, b._columnwise_scale_inv)


# ---------------------------------------------------------------- forward and autograd numerics

WIDEST = (11264, 13312, 14336, 16384)
FORWARD_CASES = (
    [
        (s, n, u, h, m, 1e-5, T)
        for s in range(1, 7)
        for n, u in SEAMS
        for h, m in ((32, 1024), (96, 1024), (128, 2048), (1024, 1024), (3072, 1024), (4128, 1024))
    ]
    # Production widths and a second RMSNorm epsilon; (3072, 1e-5) is listed above.
    + [
        (3, n, u, h, 1024, e, T)
        for h in (3072, 4096, 10240)
        for n, u in SEAMS
        for e in (1e-5, 1e-3)
        if (h, e) != (3072, 1e-5)
    ]
    # Widest admitted widths leave near-cancelling controller sums to the contribution bound.
    + [(s, n, u, h, 1024, 1e-5, F) for h in WIDEST for n, u in SEAMS for s in (1, 3, 6)]
)


@GPU
@pytest.mark.parametrize("streams,readers,updates,hidden,tokens,epsilon,controllers", FORWARD_CASES)
@REQUIRES_FUSED_BACKWARD
def test_forward_and_complete_backward(
    streams, readers, updates, hidden, tokens, epsilon, controllers
):
    """Outputs and autograd gradients against the FP64 formula."""
    raw = inputs(streams, readers, hidden, tokens, updates=updates)
    originals = (raw[0], *raw[1], *raw[2], *raw[3])
    cloned = tuple(t.detach().clone().requires_grad_(True) for t in originals)
    n, u = readers, updates
    ref_raw = (cloned[0], cloned[1 : 1 + u], cloned[1 + u : 3 + u + n], cloned[3 + u + n :])
    actual, expected = run(raw, streams, epsilon), reference(ref_raw, streams, epsilon)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)
    for a, b in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
    dys = tuple(torch.randn_like(t) for t in actual)
    ga = torch.autograd.grad(actual, originals, dys)
    ge = torch.autograd.grad(expected, cloned, dys)
    # Controller gradients are FP32 reductions over tokens x hidden products with heavy
    # cancellation; any valid summation order is off by up to ~3e-4 against the FP64
    # reference (production reaches 2.9e-4 for other dy draws).
    assert_grads_close(ga, ge, fp32=(5e-4, 5e-4) if controllers else None)
    for c in raw[2] if controllers else ():
        index = next(i for i, p in enumerate(originals) if p is c)
        assert torch.count_nonzero(ga[index][streams:]) == 0


BF16 = torch.bfloat16
BF16_STREAMS = [(BF16, BF16), (BF16, torch.float32), (torch.float32, BF16)]


@GPU
@pytest.mark.parametrize("residual_dtype,logit_dtype", BF16_STREAMS)
@pytest.mark.parametrize("readers,updates", SEAMS)
@pytest.mark.parametrize(
    "streams,hidden,tokens",
    [(3, 96, 1024), (1, 128, 2048), (3, 4096, 1024), (6, 3072, 1024), (2, 10240, 1024)],
)
@REQUIRES_FUSED_BACKWARD
def test_bf16_residual_and_logits(
    residual_dtype, logit_dtype, readers, updates, streams, hidden, tokens
):
    """BF16 residual streams and controller logits: FP32 arithmetic, outputs in their own
    dtypes, and every autograd gradient (in its leaf's dtype) against the FP64 formula."""
    raw = inputs(
        streams,
        readers,
        hidden,
        tokens,
        updates=updates,
        residual_dtype=residual_dtype,
        logit_dtype=logit_dtype,
    )
    originals = (raw[0], *raw[1], *raw[2], *raw[3])
    cloned = tuple(t.detach().clone().requires_grad_(True) for t in originals)
    n, u = readers, updates
    ref_raw = (cloned[0], cloned[1 : 1 + u], cloned[1 + u : 3 + u + n], cloned[3 + u + n :])
    actual, expected = run(raw, streams), reference(ref_raw, streams)
    assert actual[0].dtype == residual_dtype
    # The kernel rounds its FP32 residual once; FP64 rounds directly (at most one ulp apart).
    exact = (2e-6, 2e-6) if residual_dtype == torch.float32 else (1 / 128, 1e-6)
    torch.testing.assert_close(actual[0], expected[0], rtol=exact[0], atol=exact[1])
    for a, b in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
    dys = tuple(torch.randn_like(t) for t in actual)
    ga = torch.autograd.grad(actual, originals, dys)
    ge = torch.autograd.grad(expected, cloned, dys)
    assert [g.dtype for g in ga] == [t.dtype for t in originals]
    assert_grads_close(ga, ge, fp32=(5e-4, 5e-4))


@pytest.mark.parametrize("key", sorted(forward_kernel._FORWARD_CONFIGS_BF16))
def test_bf16_stream_configs(key):
    """Every BF16-stream table entry is valid, selected for its shape, and matches FP64."""
    hidden, readers, updates, quantized = key
    cfg = forward_kernel._FORWARD_CONFIGS_BF16[key]
    assert cfg.LOADS_FIRST and cfg.WIDTH >= cfg.VECTOR * 32 * cfg.NUM_WARPS
    assert not cfg.PAIRED_REDUCE or readers == 2
    choose = forward_kernel.choose_forward_config
    options = dict(mxfp8=quantized, updates=updates, bf16_residual=True)
    assert choose(2048, hidden, 3, readers, **options) is cfg
    assert choose(512, hidden, 3, readers, inference=True, **options) is cfg
    assert choose(1024, hidden, 3, readers, **options) is not cfg  # short training rows
    assert choose(2048, hidden, 3, readers, mxfp8=quantized, updates=updates) is not cfg
    if not torch.cuda.is_available():
        return
    bf16 = torch.bfloat16
    raw = inputs(3, readers, hidden, 2048, updates=updates, residual_dtype=bf16, logit_dtype=bf16)
    with torch.no_grad():
        current, _, outputs = fwd(raw, mxfp8_handoff=quantized)
        expected = reference(raw, 3)
    torch.testing.assert_close(current, expected[0], rtol=1 / 128, atol=1e-6)
    for index, (a, b) in enumerate(zip(outputs, expected[1:])):
        if type(a) is torch.Tensor:
            torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
        else:
            relative = (a.dequantize().float() - b.float()).norm() / b.float().norm()
            assert relative < 0.04


CONTRIBUTION_CASES = [
    (s, n, h, t, native)
    for s, n, h in (
        [(s, 2, 4096) for s in (1, 2, 3, 4, 5)]
        + [(s, n, 1024) for s in (2, 3, 4, 5, 6) for n in (1, 2)]
        + [(3, 1, 4096)]
        + [(3, n, h) for h in (3072, 10240) for n in (1, 2)]
    )
    for t in (F, T)
    for native in (F, T)
] + [(s, n, h, T, F) for h in WIDEST for n in (1, 2) for s in (1, 3, 6)]
CONTRIBUTION_CASES = [pytest.param(*c, marks=QUALIFIED) if c[-1] else c for c in CONTRIBUTION_CASES]


@GPU
@pytest.mark.parametrize("streams,readers,hidden,retention,native", CONTRIBUTION_CASES)
@torch.no_grad()
@REQUIRES_FUSED_BACKWARD
def test_controller_contribution_bounds(streams, readers, hidden, retention, native):
    """FP64 controller error stays within the original absolute-contribution bound.

    Column routes only add a separate replay launch before the backward kernel, which never
    reads them, so they cannot change controller gradients. Column bytes are checked against
    TE by test_expanded_native_backward.
    """
    raw = inputs(streams, readers, hidden, 128, retention=retention)
    current, (saved, r, dys, carry) = backward_args(raw, streams)
    grid = (
        (336 if (streams, readers, hidden) == (3, 1, 3072) else hidden // 128) if native else "auto"
    )
    grads, _ = native_backward(saved, r, dys, carry, config=grid, save_columns=False)
    c = saved.coefficients.double().view(8, streams)
    values = current.double().view(128, streams, hidden)
    total = carry.double().view_as(values)
    terms = []
    for reader in range(readers):
        # Reproduce only the reader's FP32 FMA boundary; all derivative math and
        # reductions are independent FP64 formulas using saved statistics.
        dx = reader_dx(saved, r, reader, dys[reader])[2]
        total += c[2 + reader][None, :, None] * dx[:, None, :]
        terms.append(values * dx[:, None, :] * c[6 + reader][None, :, None])
    terms = [
        total * saved.update.double()[:, None, :] * c[4][None, :, None],
        total * r.double().view_as(values) * c[5][None, :, None] if retention else None,
        *terms,
    ]
    for actual, contribution in zip(grads[1 + readers : 3 + 2 * readers], terms):
        if contribution is None:
            assert actual is None
        else:
            scale = contribution.abs().sum((0, 2))
            error = (actual[:streams].double() - contribution.sum((0, 2))).abs()
            assert torch.all(error <= 3e-6 * scale + torch.finfo(torch.float32).tiny)
            assert torch.count_nonzero(actual[streams:]) == 0


@GPU
@pytest.mark.parametrize("streams", [2, 3, 6])
@pytest.mark.parametrize("route", [0, 1, 2])
@pytest.mark.parametrize("retention", [False, True])
@REQUIRES_FUSED_BACKWARD
def test_unused_outputs_and_frozen_parameters(streams, route, retention):
    raw = inputs(streams, 2, retention=retention)
    for frozen in (raw[2][1], raw[3][1]):
        if frozen is not None:
            frozen.requires_grad_(False)
    result = run(raw, streams)
    result[route].float().sum().backward()
    # Frozen parameters and pruned reader/residual branches are both supported.
    result = run(raw, streams)
    sum(t.float().sum() for t in result).backward()
    assert raw[0].grad is not None
    assert (raw[2][1] is None or raw[2][1].grad is None) and raw[3][1].grad is None
    for t in (raw[0], *raw[1], *raw[2], *raw[3]):
        if t is not None and t.grad is not None:
            assert torch.isfinite(t.grad).all()


@GPU
@pytest.mark.parametrize(
    "invalid,message",
    [
        ("residual_dtype", "FP32 or BF16 residuals"),
        ("residual_layout", "contiguous"),
        ("update", "update"),
        ("controller", "controllers require"),
        ("gamma", "gamma"),
    ],
)
def test_boundary_rejects_invalid_inputs_before_backward(invalid, message):
    """The forward validates every input before a kernel (or the trusted backward) uses it.

    No-grad, so the check runs on any GPU (training is GB300-only).
    """
    residual, updates, controllers, gammas = inputs(3, 1, 128)
    if invalid == "residual_dtype":
        residual = residual.half()  # FP32 and BF16 streams are supported; FP16 is not
    elif invalid == "residual_layout":
        residual = residual.t().contiguous().t()
    elif invalid == "update":
        updates = (updates[0].float(),)
    elif invalid == "controller":
        controllers = (controllers[0][:2], *controllers[1:])
    else:
        gammas = (gammas[0][:-1],)
    with torch.no_grad(), pytest.raises(ValueError, match=message):
        run((residual, updates, controllers, gammas), 3)


# ---------------------------------------------------------------- native backward schedules


@GPU
@pytest.mark.parametrize("hidden", [4096, 10240])
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("columns", [False, True])
@REQUIRES_FUSED_BACKWARD
def test_six_stream_native_routing(hidden, readers, columns):
    """Six streams use the same complete backward and column ownership contract."""
    _, args = backward_args(inputs(6, readers, hidden, 1024), 6)
    expected = standard_backward(*args)
    actual = backward_kernel.backward(*args, save_columns=columns)
    assert_grads_close(actual[0], expected[0])
    assert (actual[1][0] is not None) == columns
    assert all(q is None for q in actual[1][1:])
    with patch.object(
        backward_kernel._backward, "warmup", side_effect=AssertionError("unexpected compile")
    ):
        for grid in (None, 0, -32, 31, True):
            with pytest.raises(ValueError, match="grid"):
                backward_kernel.backward(*args, config=grid)


@GPU
@REQUIRES_FUSED_BACKWARD
def test_native_counter_bounds_precede_launch():
    """Counter overflow remains rejected even for direct test-only schedule injection."""
    _, args = backward_args(inputs(3, 1, 4096, 128))
    with patch.object(
        backward_kernel._backward, "warmup", side_effect=AssertionError("unexpected launch")
    ):
        with pytest.raises(ValueError, match="counter"):
            backward_kernel.backward(*args, config=2**31)


@GPU
@pytest.mark.parametrize("streams", [2, 3, 5, 6])
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("complete", [False, True])
@torch.no_grad()
@REQUIRES_FUSED_BACKWARD
def test_model_backward_routes(streams, readers, columns, complete):
    """Automatic routes preserve reconstruction, column ownership and shared gradients."""
    raw = inputs(streams, readers, 4096, 8192)
    current, saved, _ = fwd(raw, streams)
    dys = tuple(torch.randn_like(t) if complete else None for t in raw[1])
    grads, operands = backward_kernel.backward(
        saved, raw[0], dys, torch.randn_like(current), save_columns=columns
    )
    assert (operands[0] is not None) == columns
    assert all(q is None for q in operands[1:])
    if readers == 2:
        assert grads[1] is grads[2]
    for grad in grads:
        assert grad is not None and torch.isfinite(grad).all()


@GPU
@pytest.mark.parametrize("missing", ["carry", "reader0", "reader1"])
@REQUIRES_FUSED_BACKWARD
def test_backward_materializes_pruned_gradients(missing):
    _, (saved, residual, dys, carry) = backward_args(inputs(3, 2, 96, 1024))
    dys = list(dys)
    if missing == "carry":
        carry = None
    else:
        dys[int(missing[-1])] = None
    actual, _ = backward_kernel.backward(saved, residual, tuple(dys), carry)
    expected, _ = standard_backward(saved, residual, tuple(dys), carry)
    assert_grads_close(actual, expected, first=(5e-4, 2e-4))


@GPU
@pytest.mark.parametrize("native", [False, pytest.param(True, marks=QUALIFIED)])
@pytest.mark.parametrize("hidden", [3072, 4096, 10240])
@torch.no_grad()
@REQUIRES_FUSED_BACKWARD
def test_shared_update_gradients(native, hidden):
    """Automatic and explicit schedules share updates and match FP64 derivatives.

    Explicit grids with a strict first-column anchor also reproduce every gradient bit.
    """
    _, args = backward_args(inputs(3, 2, hidden, 128))
    shared = backward_kernel.backward(*args, config=hidden // 128 if native else "auto")
    expected = standard_backward(*args)
    assert shared[0][1] is shared[0][2]
    # FP64 derivative reductions need not reproduce the GPU accumulation order.
    torch.testing.assert_close(shared[0][1], expected[0][1], rtol=1 / 128, atol=1 / 128)
    if native:
        options = dict(config=hidden // 128, save_columns=(True, False))
        all_bytes_equal(native_backward(*args, **options)[0], native_backward(*args, **options)[0])


@GPU
@pytest.mark.parametrize("native", [False, pytest.param(True, marks=QUALIFIED)])
@pytest.mark.parametrize("hidden", [3072, 4096, 10240])
@REQUIRES_FUSED_BACKWARD
def test_single_write_two_reader_gradients(native, hidden):
    """A single write read twice returns one update gradient and saves the update itself."""
    raw = inputs(3, 2, hidden, 1024, updates=1)
    with torch.no_grad():
        _, args = backward_args(raw)
    saved = args[0]
    # The tape keeps the caller's update (no BF16 sum is materialized).
    assert saved.update_count == 1 and saved.update.data_ptr() == raw[1][0].data_ptr()
    with torch.no_grad():
        grads, _ = backward_kernel.backward(*args, config=hidden // 128 if native else "auto")
        expected, _ = standard_backward(*args)
    # dR, du, write, retain, two reads, two gammas.
    assert len(grads) == len(expected) == 8
    assert_grads_close(grads, expected)
    outputs = run(raw)
    gradients = torch.autograd.grad(
        outputs, (raw[0], raw[1][0], *raw[2], *raw[3]), [torch.ones_like(t) for t in outputs]
    )
    assert all(g is not None for g in gradients)
    # Without a shortcut add the inference API also takes a single update.
    with torch.no_grad():
        inference = run(raw)
    assert torch.equal(inference[0], outputs[0])


@QUALIFIED
@pytest.mark.parametrize(
    "shape", MX_SHAPES[3:10] + [(2048, h) for h in WIDEST[:3]] + [(1024, 4128)]
)
@pytest.mark.parametrize("readers,updates", SEAMS)
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("retention", [False, True])
def test_expanded_native_backward(shape, readers, updates, columns, retention):
    """Native backward and its MXFP8 columns match the FP64/TE reference on expanded shapes."""
    m, h = shape
    raw = inputs(3, readers, h, m, retention=retention, updates=updates)
    _, args = backward_args(raw, mxfp8_handoff=columns)
    expected = standard_backward(*args, prepare_wgrad=(columns,) + (False,) * (readers - 1))
    actual = backward_kernel.backward(*args, save_columns=columns)
    assert_grads_close(actual[0], expected[0])
    assert_columns_equal(actual[1], expected[1])


@QUALIFIED
@pytest.mark.parametrize("capacity", [399, 39])
def test_ordinary_admission_capacity_and_replay(monkeypatch, capacity):
    """Only a complete row group must fit; the full ordinary grid need not fit."""
    _, args = backward_args(inputs(3, 2, 10240, 8192))
    options = dict(config=400, prepare_wgrad=(True, False))
    expected = backward_kernel.backward(*args, **options) if capacity >= 40 else None
    original_run = backward_kernel._backward.run
    launches, epochs, queries = [], [], []

    def capture(*args, **kwargs):
        kernel = original_run(*args, **kwargs)
        assert kernel.metadata.launch_cooperative_grid is False
        if not kwargs.get("warmup", False):
            launches.append(kernel)
            epochs.append(kwargs["scheduler"])
        return kernel

    monkeypatch.setattr(backward_kernel._backward, "run", capture)
    # Drop cached residency checks so this capacity is queried (once per launch key).
    monkeypatch.setattr(backward_kernel, "_RESIDENCY", {})
    monkeypatch.setattr(
        backward_kernel.triton.runtime.driver.active.utils,
        "cuOccupancyMaxActiveClusters",
        lambda *args: queries.append(args) or capacity,
    )
    if capacity < 40:
        with pytest.raises(ValueError, match="required residency 40 exceeds capacity: 39"):
            backward_kernel.backward(*args, **options)
        assert not launches
        return
    reference = tuple(workspace_tensors(expected))
    all_bytes_equal(workspace_tensors(backward_kernel.backward(*args, **options)), reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = tuple(workspace_tensors(backward_kernel.backward(*args, **options)))
    fill = lambda scratch: ([t.fill_(1234567) for t in epochs], poison_bytes(scratch))
    replay_poisoned(graph, replayed, replayed, reference, fill)
    assert len(launches) == 2 and len(queries) == 1


# ---------------------------------------------------------------- configs and shapes (host only)


@pytest.mark.parametrize("shape", kernel_utils.FLAGSHIPS)
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("mxfp8", [False, True])
def test_production_configs(shape, readers, mxfp8, monkeypatch):
    """Nano/Super/Ultra rows from 2048 use the tuned forward entry and cached backward grid."""
    cfg = capture_backward_config(*shape, 3, readers, columns=mxfp8, retention=True, grid="auto")
    assert cfg.TARGET_CTAS == {3072: 456, 4096: 448, 10240: 440}[shape[1]]
    choose = forward_kernel.choose_forward_config
    assert (
        choose(*shape, 3, readers, mxfp8=mxfp8)
        == forward_kernel._FORWARD_CONFIGS[shape[1], readers, mxfp8]
    )
    # A learned profile may equal a preset; the table covers 3-stream rows >= 2048 at its widths.
    marker = object()
    monkeypatch.setitem(forward_kernel._FORWARD_CONFIGS, (shape[1], readers, mxfp8), marker)
    for rows in (2048, shape[0], shape[0] + 1024, 65536):
        assert choose(rows, shape[1], 3, readers, mxfp8=mxfp8) is marker
    assert choose(1024, shape[1], 3, readers, mxfp8=mxfp8) is not marker
    assert choose(*shape, 2, readers, mxfp8=mxfp8) is not marker


def test_training_config_domain():
    """Every admitted training shape and short row gets a deterministic one-CTA-per-row config."""
    shapes = (*kernel_utils.SHAPES, *((32, h) for h in kernel_utils.WIDTHS))
    for shape, readers, mxfp8 in itertools.product(shapes, (1, 2), (False, True)):
        cfg = forward_kernel.choose_forward_config(*shape, 3, readers, mxfp8=mxfp8)
        assert cfg == forward_kernel.choose_forward_config(*shape, 3, readers, mxfp8=mxfp8)
        assert (cfg.PARTS, cfg.SPLIT) == (1, False)
        assert cfg.WIDTH >= 32 and cfg.WIDTH & (cfg.WIDTH - 1) == 0
        assert cfg.PIPELINE_DEPTH >= 0
        assert 0 <= cfg.OFFLOAD_SEGMENTS <= (shape[1] + cfg.WIDTH - 1) // cfg.WIDTH
        assert not cfg.PAIRED_REDUCE or readers == 2
        assert cfg.REDUCE_VECTOR in (0, 1, 2, 4, 8, 16, 32) and cfg.NUM_WARPS in (4, 8, 16)
        assert cfg.MAXNREG is None or cfg.MAXNREG > 0
        backward = capture_backward_config(
            *shape, 3, readers, columns=mxfp8, retention=True, grid="auto"
        )
        assert backward.TARGET_CTAS > 0
        assert backward.TARGET_CTAS % ((shape[1] + 255) // 256) == 0


@pytest.mark.parametrize("streams", range(1, 7))
@pytest.mark.parametrize("inference", [False, True])
def test_general_config_domain(streams, inference):
    """Forward and backward configs are valid for unusual widths, rows and stream counts."""
    widths, rows = (1, 31, 32, 48, 288, 1025, 2052, 17408, 32768), (1, 32, 65, 1024, 4096)
    for h, m, readers in itertools.product(widths, rows, (1, 2)):
        cfg = forward_kernel.choose_forward_config(m, h, streams, readers, inference=inference)
        assert cfg.WIDTH >= 32 and cfg.WIDTH & (cfg.WIDTH - 1) == 0
        assert not cfg.PAIRED_REDUCE or readers == 2
        assert not cfg.SPLIT or (inference and readers == 2)
        assert cfg.PARTS == 1 or h % (32 * cfg.PARTS) == 0
    if inference:
        return
    widths, rows = (32, 96, 128, 288, 1024, 2048, 3072, 10240, 16384), (32, 64, 128, 1024, 4096)
    for h, m, readers, columns in itertools.product(widths, rows, (1, 2), (False, True)):
        options = dict(columns=columns, retention=True, grid="auto")
        cfg = capture_backward_config(m, h, streams, readers, **options)
        assert cfg == capture_backward_config(m, h, streams, readers, **options)
        assert cfg.TARGET_CTAS > 0 and cfg.TARGET_CTAS % ((h + 255) // 256) == 0
        assert cfg.COLUMN_ROUTES == (columns,) + (False,) * (readers - 1)
    # Backward can route only reader 1 through the column path.
    options = dict(columns=(False, True), retention=True, grid="auto")
    assert capture_backward_config(1024, 96, streams, 2, **options).COLUMN_ROUTES == (False, True)


@pytest.mark.parametrize(
    "rows,hidden,match",
    [(m, 96, "rows") for m in (0, 1, 31, 65)]
    + [(1024, h, "width|H") for h in (1, 31, 48, 1025, 2052)],
)
def test_backward_config_rejects_unaligned_shapes(rows, hidden, match):
    with pytest.raises(ValueError, match=match):
        capture_backward_config(rows, hidden, 3, 2, columns=False, retention=True, grid="auto")


@GPU
@pytest.mark.parametrize("m,h", [(0, 2048), (2048, 0)])
def test_invalid_shapes(m, h):
    """Forward and backward reject empty rows or width."""
    with pytest.raises(ValueError, match="positive rows"):
        fwd(inputs(3, 1, h, m))
    update = torch.empty((m, h), device="cuda", dtype=torch.bfloat16)
    saved = kernel_utils.SavedWrite(
        update, torch.empty(24, device="cuda"), (None,), (None,), (3,) * 3
    )
    with pytest.raises(ValueError, match="positive rows"):
        backward_kernel.backward(saved, None, (None,))


def test_inference_config_domain():
    """Inference configs are deterministic and valid; long prefill uses the training config.

    Up to 1024 rows at widths 3072-16384, every heuristic tree leaf is a valid LOADS_FIRST
    config; training, long rows and untuned widths keep the original load order.
    """
    tree_rows = (1, 16, 32, 40, 41, 72, 73, 88, 89, 128, 129, 160, 161, 448, 449, 512, 1024)
    for hidden, streams, (readers, updates), quantized in itertools.product(
        range(3072, 16385, 512), range(1, 7), SEAMS, (False, True)
    ):
        for m in tree_rows:
            cfg = forward_kernel.choose_forward_config(
                m, hidden, streams, readers, mxfp8=quantized, inference=True, updates=updates
            )
            assert cfg.LOADS_FIRST and cfg.WIDTH & (cfg.WIDTH - 1) == 0
            assert cfg.WIDTH >= 128 * cfg.NUM_WARPS
            assert cfg.MAXNREG is None or cfg.MAXNREG * 32 * cfg.NUM_WARPS <= 65536
            assert cfg.PARTS * (1 + cfg.SPLIT) in (1, 2, 4)
            assert not (cfg.SPLIT or cfg.PAIRED_REDUCE) or readers == 2
            if cfg.CLUSTER > 1:
                assert cfg.CLUSTER <= 8 and cfg.PARTS == 1 and hidden % cfg.CLUSTER == 0
                assert (hidden // cfg.CLUSTER) % cfg.WIDTH == 0
            # The update count only refines inference leaves of two readers past decode rows.
            shortcut_add = forward_kernel.choose_forward_config(
                m, hidden, streams, readers, mxfp8=quantized, inference=True
            )
            if readers == 1 or updates == 2 or cfg.CLUSTER > 1:
                assert cfg == shortcut_add
        for m, h, inference in ((32, hidden, False), (1025, hidden, True), (32, 2048, True)):
            cfg = forward_kernel.choose_forward_config(
                m, h, streams, readers, mxfp8=quantized, inference=inference, updates=updates
            )
            assert not cfg.LOADS_FIRST
            # Training and long prefill share the shortcut add's configs.
            assert cfg == forward_kernel.choose_forward_config(
                m, h, streams, readers, mxfp8=quantized, inference=inference
            )
    replicas_by_rows = ((1, 8), (16, 8), (17, 4), (32, 4), (33, 2), (64, 2), (65, 1))
    replicas_by_rows += ((1024, 1), (1025, 1), (8192, 1), (16384, 1))
    for hidden, readers, (tokens, replicas), quantized in itertools.product(
        kernel_utils.WIDTHS, (1, 2), replicas_by_rows, (False, True)
    ):
        args = tokens, hidden, 3, readers
        cfg = forward_kernel.choose_forward_config(*args, mxfp8=quantized, inference=True)
        assert cfg == forward_kernel.choose_forward_config(*args, mxfp8=quantized, inference=True)
        assert not cfg.PAIRED_REDUCE or readers == 2
        assert 0 <= cfg.OFFLOAD_SEGMENTS <= (hidden + cfg.WIDTH - 1) // cfg.WIDTH
        parts, split = (cfg.PARTS, cfg.SPLIT)
        if tokens > 1024:
            # Long prefill shares the training configs: one CTA per row.
            assert cfg == forward_kernel.choose_forward_config(*args, mxfp8=quantized)
            assert (parts, split, cfg.CLUSTER) == (1, False, 1)
            continue
        tuned = 3072 <= hidden <= 16384
        assert cfg.LOADS_FIRST == tuned
        if tuned:
            # The tree sets the replicas.
            replicas = parts * (1 + split)
            assert replicas in (1, 2, 4, 8) and split == (replicas > 1 and readers == 2)
            continue
        assert split == (tokens <= 64 and readers == 2)
        assert parts * (1 + split) == replicas


# ---------------------------------------------------------------- checkpoint replay


def _advance(previous, tape):
    """Emulate RN/FMA forward writes in FP64, independently of the GPU primitive."""
    m, h = tape.update.shape
    c = tape.coefficients.double().view(8, tape.num_streams)
    rows = torch.arange(m, device=previous.device)[:, None]
    result = []
    for stream, value in enumerate(previous.view(m, tape.num_streams, h).unbind(1)):
        retained = (value.double() * c[1, stream]).float().double()
        update = tape.update.double() * c[0, stream]
        fused = (stream == 0) | ((stream == 1) & (rows % 32 < 12))
        result.append(
            torch.where(
                fused, (retained + update).float(), (retained + update.float().double()).float()
            )
        )
    return torch.stack(result, 1).flatten(1)


# 16-step intervals: one partial, one full, and two full plus a partial tail.
@GPU
@pytest.mark.parametrize("depth", [1, 16, 40])
@pytest.mark.parametrize("retain", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("readers", [1, 2])
@REQUIRES_FUSED_BACKWARD
def test_deep_replay_independent_gradients(depth, retain, readers):
    """No inverse reference: retain=0 and 0.3 remain valid forward replay cases."""
    torch.manual_seed(1317)
    m, h, streams = 32, 96, 3
    initial = torch.randn(m, streams * h, device="cuda")
    states, tapes = [initial], []
    for step in range(depth):
        dtype = torch.bfloat16 if step % 2 == 0 else torch.float32
        update = torch.randn(m, h, device="cuda", dtype=dtype)
        if step % 7 == 0:
            update.zero_()
        c = torch.rand(8, streams, device="cuda")
        c[0].mul_(2)
        c[1].fill_(retain)
        tape = kernel_utils.SavedWrite(
            update,
            c.flatten(),
            tuple(torch.randn(h, device="cuda", dtype=torch.bfloat16) for _ in range(readers)),
            tuple(torch.empty(m, device="cuda") for _ in range(readers)),
            (streams,) * (2 + readers),
            num_streams=streams,
        )
        for reader in range(readers):
            x = folded_reader(tape, states[-1], reader)
            tape.rstd[reader].copy_((x.double().square().mean(-1) + 1e-5).rsqrt().float())
        tapes.append(tape)
        states.append(_advance(states[-1], tape))
    carry = torch.randn_like(initial)
    for begin in reversed(range(0, depth, 16)):
        end = min(begin + 16, depth)
        terminal = torch.empty_like(initial)
        predecessors, _ = replay_interval(states[begin], tapes[begin:end], terminal_output=terminal)
        all_bytes_equal((*predecessors, terminal), states[begin : end + 1])
        for offset in reversed(range(end - begin)):
            tape, previous = tapes[begin + offset], predecessors[offset]
            dys = tuple(
                torch.randn(m, h, device="cuda", dtype=torch.bfloat16) for _ in range(readers)
            )
            expected, _ = standard_backward(tape, previous, dys, carry)
            actual, _ = backward_kernel.backward(tape, previous, dys, carry)
            assert_grads_close(actual, expected, first=(5e-4, 2e-4))
            carry = actual[0]


@GPU
@pytest.mark.parametrize("columns", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 3e38, "subnormal"])
def test_replay_exceptional_and_subnormal_recurrence(value, columns):
    """Exceptional values preserve classes; finite values (and IEEE subnormals) keep bits."""
    m, h, streams = 32, 96, 3
    coefficients = torch.ones(8, streams, device="cuda")
    coefficients[0].fill_(0.75)
    coefficients[1].fill_(0.3)
    if value == "subnormal":
        torch.manual_seed(29)
        # Normal FP32 residuals near the subnormal boundary; retention pushes them below it.
        previous = (torch.rand(m, streams * h, device="cuda") + 1) * 2.0**-126
        update = torch.rand(m, h, device="cuda") * 2.0**-130
        update[:, ::2] = 0
    else:
        previous = torch.zeros(m, streams * h, device="cuda")
        previous[0, 0] = previous[31, h + 31] = value
        update = torch.ones(m, h, device="cuda")
        update[16, 64] = value
        coefficients[2].copy_(torch.tensor([0.3, 0.2, -0.1], device="cuda"))
    tape = kernel_utils.SavedWrite(
        update,
        coefficients.flatten(),
        (torch.ones(h, device="cuda", dtype=torch.bfloat16),),
        (torch.ones(m, device="cuda"),),
        (streams,) * 3,
        num_streams=streams,
    )
    expected = [previous]
    for _ in range(3):
        expected.append(_advance(expected[-1], tape))
    if value == "subnormal":
        assert (expected[-1] != 0).any() and (expected[-1].abs() < 2.0**-126).all()
    terminal = torch.empty_like(previous)
    replayed, _ = replay_interval(
        previous, (tape,) * 3, prepare_columns=columns, terminal_output=terminal
    )
    for actual, reference in zip((*replayed, terminal), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
        finite = torch.isfinite(reference)
        assert torch.equal(actual[finite].view(torch.uint8), reference[finite].view(torch.uint8))


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape,streams,retention",
    [
        ((5, 33), 1, False),
        ((5, 33), 3, True),
        ((2, 3, 128), 3, False),
        ((2, 3, 128), 6, True),
        ((2, 2, 3, 33), 6, False),
        ((2, 2, 3, 33), 1, True),
    ],
)
def test_native_forward_replay_exact(shape, streams, retention, dtype):
    """Replay the actual native forward, including ranks, tails and logit casts."""
    generator = torch.Generator(device="cuda").manual_seed(2026)
    update = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    current = torch.randn(
        (*shape[:-1], streams * shape[-1]), device="cuda", dtype=torch.float32, generator=generator
    )
    update.view(-1)[0] = 0.0
    current.view(-1)[0] = -0.0
    write = torch.linspace(-4, 4, streams + 2, device="cuda", dtype=dtype)
    retain = write.flip(0) if retention else None
    links, expected, identities = [], [], []
    with residual_replay_scope(replay_config(4), True) as manager:
        for _ in range(3):
            manager.advance(1)
            following = _streamwise_sigmoid_write_triton(
                current, update, write, streams, retain, 0.07
            )
            links.append(register_write(current, following, update, write, retain, streams, 0.07))
            expected.append(following.clone())
            identities.append(following.untyped_storage()._cdata)
            current = following
    assert [link.current_slot.resident() for link in links] == [False, False, True]
    for link, value, identity in zip(links, expected, identities):
        actual = link.current_slot.ensure()
        assert actual.untyped_storage()._cdata == identity
        assert bytes_equal(actual, value)
    assert links[0].interval.materialized


@GPU
def test_physical_layer_intervals_and_independent_workspaces():
    """Atomic two-layer units and actual writes have separate interval counts."""
    owners = []
    for seed in (23, 47):
        torch.manual_seed(seed)
        current = torch.randn(32, 288, device="cuda")
        update = torch.randn(32, 96, device="cuda", dtype=torch.bfloat16)
        write = torch.randn(3, device="cuda")
        links, expected = [], []
        with residual_replay_scope(replay_config(3), True) as manager:
            for physical_layers in (1, 2, 1, 1):
                manager.advance(physical_layers)
                # Two writes within each physical unit: interval capacity must
                # not be interpreted as a count of write transitions.
                for _ in range(2):
                    following = _streamwise_sigmoid_write_triton(
                        current, update, write, 3, None, 0.1
                    )
                    links.append(register_write(current, following, update, write, None, 3, 0.1))
                    expected.append(following.clone())
                    current = following
        assert links[0].interval is links[3].interval
        assert links[3].interval is not links[4].interval
        assert links[4].interval is links[7].interval
        assert [len(links[i].interval.transitions) for i in (0, 4)] == [4, 4]
        owners.append((links, expected))
    consumer = torch.cuda.Stream()
    results = []
    with torch.cuda.stream(consumer):
        # Consume forward slots out of order and from another stream.
        for links, expected in reversed(owners):
            for i in (2, 0, 6, 4):
                results.append((links[i].current_slot.ensure().clone(), expected[i]))
    torch.cuda.current_stream().wait_stream(consumer)
    all_bytes_equal(*zip(*results))
    backings = [owner[0][0].current_slot.backing.data_ptr() for owner in owners]
    assert backings[0] != backings[1]


@GPU
@pytest.mark.parametrize("prune", [False, True])
@REQUIRES_FUSED_BACKWARD
def test_mixed_fused_native_retained_backward(prune):
    """Replay must match saved predecessors across native boundaries and reuse."""
    torch.manual_seed(313)
    source = torch.randn(1024, 288, device="cuda", dtype=torch.float32)
    # Native bias/dropout writers can accumulate an FP32 update; preserve that
    # exact dtype and its low bits across the BF16 fused-writer boundaries.
    updates = [
        torch.randn(
            1024, 96, device="cuda", dtype=torch.bfloat16 if index % 2 == 0 else torch.float32
        )
        for index in range(4)
    ]
    logits = [torch.randn(3, device="cuda") for _ in range(4)]
    retain = [torch.full((3,), -0.5, device="cuda") for _ in range(4)]
    reads = [torch.randn(3, device="cuda") for _ in range(2)]
    gammas = [torch.randn(96, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

    def execute(recompute):
        residual = source.detach().clone().requires_grad_()
        leaves = [residual] + [
            x.detach().clone().requires_grad_() for x in updates + logits + retain + reads + gammas
        ]
        us, ws, ts, bs, gs = leaves[1:5], leaves[5:9], leaves[9:13], leaves[13:15], leaves[15:17]
        outputs = []
        with residual_replay_scope(replay_config(3 if recompute else 1), True) as manager:
            for index in range(4):
                if manager is not None:
                    manager.advance(1)
                if index % 2 == 0:
                    residual, output = wide_residual_write_read_rmsnorm_quant(
                        residual,
                        us[index],
                        ws[index],
                        ts[index],
                        bs[index // 2],
                        gs[index // 2],
                        num_streams=3,
                        epsilon=2e-5,
                    )
                    outputs.append(output)
                else:
                    residual = _StreamwiseSigmoidWriteback.apply(
                        residual, us[index], ws[index], ts[index], 3, 0.7
                    )
        loss = (
            outputs[0].float().square().sum()
            if prune
            else residual.square().sum() + sum(x.float().square().sum() for x in outputs)
        )
        first = torch.autograd.grad(loss, leaves, retain_graph=True, allow_unused=True)
        # A second stream must see the restored allocation and retain its
        # allocator metadata, not merely a wrapped StorageImpl pointer.
        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            second = torch.autograd.grad(loss, leaves, allow_unused=True)
        torch.cuda.current_stream().wait_stream(other)
        all_bytes_equal(first, second)
        return first

    all_bytes_equal(execute(False), execute(True))


# 16-layer intervals: one partial, one full, and two full plus a partial tail.
@GPU
@pytest.mark.parametrize("depth", [1, 16, 40])
@pytest.mark.parametrize("readers,updates", SEAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@REQUIRES_FUSED_BACKWARD
def test_deep_fused_replay_matches_saved_predecessors(depth, readers, updates, dtype):
    """Physical intervals and short tails preserve forward states and all gradients (FP32 or
    BF16 residual streams and logits)."""
    torch.manual_seed(853)
    original = [
        torch.randn(1024, 96, device="cuda", dtype=dtype),
        *[torch.randn(1024, 32, device="cuda", dtype=torch.bfloat16) for _ in range(updates)],
        torch.randn(3, device="cuda", dtype=dtype),
        torch.full((3,), -0.8, device="cuda", dtype=dtype),
        *[torch.randn(3, device="cuda", dtype=dtype) for _ in range(readers)],
        *[torch.randn(32, device="cuda", dtype=torch.bfloat16) for _ in range(readers)],
    ]

    def execute(recompute):
        leaves = [value.detach().clone().requires_grad_() for value in original]
        current, us = leaves[0], leaves[1 : updates + 1]
        controllers = tuple(leaves[updates + 1 : updates + readers + 3])
        gammas = leaves[updates + readers + 3 :]
        losses, slots, states = [], [], []
        with residual_replay_scope(replay_config(16 if recompute else 1), True) as manager:
            for _ in range(depth):
                if manager is not None:
                    manager.advance(1)
                result = run((current, us, controllers, gammas), 3)
                current = result[0]
                losses.extend(output.float().square().mean() for output in result[1:])
                states.append(current.detach().clone())
                if manager is not None:
                    slots.append(manager.slot(current))
        if slots:
            all_bytes_equal([slot.ensure() for slot in reversed(slots)], reversed(states))
        loss = current.square().mean() + sum(losses)
        return states + list(torch.autograd.grad(loss, leaves))

    all_bytes_equal(execute(False), execute(True))


# ---------------------------------------------------------------- residual replay manager


@GPU
@pytest.mark.parametrize("ordinary_bridge", [False, True])
@REQUIRES_FUSED_BACKWARD
def test_replay_slots_release_after_backward(ordinary_bridge):
    """Ordinary backward releases replay storage owners after their final launches."""
    residual, updates, controllers, gammas = inputs(3, 1, 128, 1024)
    normalized = []
    with residual_replay_scope(replay_config(), True) as manager:
        current = residual
        for _ in range(3):
            current, reader = wide_residual_write_read_rmsnorm_quant(
                current, updates[0], *controllers, gammas[0], num_streams=3
            )
            normalized.append(reader)
            if ordinary_bridge:
                current = streamwise_sigmoid_writeback(
                    current,
                    updates[0],
                    controllers[0],
                    3,
                    retention_logits=controllers[1],
                    retention_max_forget=0.1,
                )
    assert len(manager.slots) > 0
    assert any(not slot.resident() for slot in manager.slots.values())
    loss = current.sum() + sum(reader.float().sum() for reader in normalized)
    loss.backward()
    gc.collect()
    assert not manager.slots


@GPU
@pytest.mark.parametrize("alternate_streams", [False, True])
def test_replay_releases_at_successor_registration(alternate_streams):
    """Release intermediates promptly and replay forward from the starting checkpoint."""
    anchor = torch.zeros((1024, 384), device="cuda", dtype=torch.float32)
    streams = [torch.cuda.current_stream(), torch.cuda.Stream()]
    current, links = anchor, []
    with residual_replay_scope(replay_config(), True) as manager:
        for step in range(4):
            stream = streams[step % 2] if alternate_streams else streams[0]
            stream.wait_stream(streams[(step - 1) % 2] if alternate_streams else streams[0])
            with torch.cuda.stream(stream):
                successor = current + 1
                links.append(manager.register(current, successor, native=lambda value: value + 1))
                assert bool(current.untyped_storage().nbytes()) == (step == 0)
                current = successor
    streams[0].wait_stream(streams[1])
    for step, link in enumerate(links):
        link.previous_slot.ensure()
        assert torch.equal(link.previous_slot.tensor, torch.full_like(anchor, step))
    assert torch.equal(anchor, torch.zeros_like(anchor))


@GPU
@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("nested_training", [False, True])
def test_replay_scope_ownership(raises, nested_training):
    """Nested scopes isolate ownership and always release their endpoint handle."""
    config = replay_config()
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
                    link = inner.register(previous, previous + 1, native=lambda value: value + 1)
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


# ---------------------------------------------------------------- inference


def taped_forward(forward, *args, **kwargs):
    """Materialize the tape with the same schedule to isolate the no-tape contract."""
    choose = forward_kernel.choose_forward_config

    def same_schedule(m, h, streams, readers=1, **options):
        """Return the inference schedule with one CTA per row for the taped launch."""
        cfg = choose(m, h, streams, readers, **(options | {"inference": True}))
        # Split inference readers reduce independently; retain normal tape ownership.
        paired = cfg.PAIRED_REDUCE and not cfg.SPLIT
        return cfg._replace(PARTS=1, SPLIT=False, PAIRED_REDUCE=paired)

    with patch.object(forward_kernel, "choose_forward_config", side_effect=same_schedule):
        return forward(*args, **(kwargs | {"inference": False}))


def forbid_backward_state(monkeypatch):
    """Fail if inference enters autograd or allocates backward state.

    Returns a list that gains one entry per non-warmup ``_forward`` launch.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("inference entered autograd or allocated backward state")

    monkeypatch.setattr(api._Boundary, "apply", forbidden)
    monkeypatch.setattr(api, "current_manager", forbidden)
    monkeypatch.setattr(forward_kernel, "SavedWrite", forbidden)
    launch = forward_kernel._forward.run
    observed = []

    def inspect(*args, **kwargs):
        if not kwargs.get("warmup"):
            write, readers = args[0], kwargs["read_tensors"]
            assert kwargs["saved_coefficients"] is None and write.summed_update is None
            assert all(stat is None for stat in readers.rstd)
            observed.append(True)
        return launch(*args, **kwargs)

    monkeypatch.setattr(forward_kernel._forward, "run", inspect)
    return observed


def forced_forward(cfg, raw, streams, **options):
    """Launch the forward kernel with an explicit config."""
    with patch.object(forward_kernel, "choose_forward_config", side_effect=lambda *a, **k: cfg):
        with torch.no_grad():
            return fwd(raw, streams, **options)


# Tokens cover cluster (<= 64) and tree schedules, the MXFP8 scale-padding pattern of the last
# 128-row tile (tokens % 128 = 32, 64, 96 with one or several tiles, or 0), and the
# no_grad/inference_mode entry (tokens % 64). Decode rows (odd counts and 128/7168/14336
# widths) exercise fallback profiles and masked tiles; 10240 forces the runtime retention path.
INFERENCE_BOUNDARY = [
    (m, h, q, t)
    for m in (32, 64, 96, 160, 320, 992, 1024)
    for h in (3072, 4096, 10240)
    for q in (F, T)
    for t in (F, T)
] + [(m, h, F, T) for m in (1, 16, 17, 33, 65) for h in (128, 3072, 4096, 7168, 10240, 14336)]
INFERENCE_BOUNDARY = [pytest.param(*c, marks=QUALIFIED) if c[2] else c for c in INFERENCE_BOUNDARY]


@GPU
@pytest.mark.parametrize("tokens,hidden,quantized,retention", INFERENCE_BOUNDARY)
@pytest.mark.parametrize("readers,updates", SEAMS)
def test_megatron_inference_boundary(
    monkeypatch, tokens, hidden, quantized, retention, readers, updates
):
    """Megatron inference matches the taped forward bit for bit without backward state."""
    raw = inputs(3, readers, hidden, tokens, retention=retention, updates=updates)
    current, _, operands = taped_forward(fwd, raw, mxfp8_handoff=quantized)
    expected = list(buffers((current, *operands)))
    observed = forbid_backward_state(monkeypatch)
    mode = torch.inference_mode if tokens % 64 == 0 else torch.no_grad
    with mode(), InferenceMode.active():
        actual = invoke(raw, quantized)
        assert all(not t.requires_grad and t.grad_fn is None for t in actual)
        assert all(reader_for(t) is None for t in actual)
        assert actual[0].shape == (tokens, 1, 3 * hidden)
        for a, b in zip(buffers(actual), expected, strict=True):
            assert bytes_equal(a.reshape(-1), b.reshape(-1))
    assert observed == [True]


@QUALIFIED
@pytest.mark.parametrize("tokens", [32, 64, 96, 1024])
@pytest.mark.parametrize("hidden", [3072, 4096, 10240])
@pytest.mark.parametrize("readers,updates", SEAMS)
def test_megatron_inference_te_consumer(monkeypatch, tokens, hidden, readers, updates):
    """A Transformer Engine MXFP8 linear consumes the inference handoff like the taped one."""
    raw = inputs(3, readers, hidden, tokens, updates=updates)
    _, _, operands = taped_forward(fwd, raw, mxfp8_handoff=True)
    layer = linear(hidden, mcore=True).eval()
    observed = forbid_backward_state(monkeypatch)
    with (
        torch.inference_mode(),
        InferenceMode.active(),
        te.fp8_autocast(fp8_recipe=MXFP8BlockScaling()),
    ):
        assert eligible(layer, raw[1][0], 3)
        actual = invoke(raw, consumer=layer)
        expected = consume(layer, operands[0].view(tokens, 1, hidden))
        result = consume(layer, actual[1])
        assert torch.equal(result, expected)
        assert not result.requires_grad and result.grad_fn is None
        assert reader_for(actual[1]) is None
        assert torch.count_nonzero(layer.weight.main_grad) == 0
    assert observed == [True]


def loads_first_cases():
    """Explicit masked/scalar/segmented/split/replicated configs, then one representative
    shape per distinct non-cluster config the inference heuristic returns."""
    cases = []
    for (h, width, warps, parts, split), readers, inference in itertools.product(
        [
            (96, 128, 1, 1, F),  # masked packets
            (130, 256, 2, 1, F),  # scalar loads
            (3072, 4096, 8, 2, F),  # whole tiles, replicated rows
            (4096, 2048, 4, 1, T),  # split readers
            (10240, 4096, 16, 1, F),  # several segments with a masked tail
            (10240, 2048, 16, 2, T),
        ],
        (1, 2),
        (F, T),
    ):
        if split and readers == 1:
            continue  # Split readers need two readers.
        if not inference:
            # Training launches one CTA per row (replicas are an inference schedule).
            parts, split = 1, False
        cfg = forward_kernel._ForwardConfig(width, NUM_WARPS=warps, PARTS=parts, SPLIT=split)
        cases.append(pytest.param(cfg, 3, readers, h, 33, F, inference, id=f"explicit{len(cases)}"))
    seen = {}
    for hidden, streams, readers, quantized, tokens in itertools.product(
        (3072, 4096, 6144, 10240, 16384),
        range(1, 7),
        (1, 2),
        (F, T),
        (32, 96, 128, 192, 288, 640, 1024),
    ):
        cfg = forward_kernel.choose_forward_config(
            tokens, hidden, streams, readers, mxfp8=quantized, inference=True
        )
        if cfg.LOADS_FIRST and cfg.CLUSTER == 1 and cfg not in seen:
            seen[cfg] = (streams, readers, hidden, min(tokens, 128), quantized, T)
    return cases + [
        pytest.param(c, *s, id=f"heuristic{i}", marks=QUALIFIED if s[4] else ())
        for i, (c, s) in enumerate(seen.items())
    ]


@GPU
@pytest.mark.parametrize(
    "base,streams,readers,hidden,tokens,quantized,inference", loads_first_cases()
)
def test_loads_first_parity(base, streams, readers, hidden, tokens, quantized, inference):
    """LOADS_FIRST only reorders loads, so results and saved state are bit-identical."""
    raw = inputs(streams, readers, hidden, tokens)
    options = dict(mxfp8_handoff=quantized, inference=inference)
    (current, saved, q), (current_lf, saved_lf, q_lf) = (
        forced_forward(base._replace(LOADS_FIRST=lf), raw, streams, **options) for lf in (F, T)
    )
    assert bytes_equal(current, current_lf)
    for a, b in zip(buffers(q), buffers(q_lf), strict=True):
        assert bytes_equal(a.reshape(-1), b.reshape(-1))
    if not inference:
        all_bytes_equal(
            (saved.coefficients, saved.update, *saved.rstd),
            (saved_lf.coefficients, saved_lf.update, *saved_lf.rstd),
        )


def cluster_configs():
    """One representative shape per distinct cluster config the inference heuristic returns."""
    seen = {}
    for hidden, streams, readers, quantized, tokens in itertools.product(
        (3072, 4096, 5120, 7168, 10240, 16384), (1, 3, 6), (1, 2), (F, T), (1, 4, 17, 32, 64)
    ):
        if quantized and tokens % 32:
            continue
        cfg = forward_kernel.choose_forward_config(
            tokens, hidden, streams, readers, mxfp8=quantized, inference=True
        )
        if cfg.CLUSTER > 1 and (cfg, quantized) not in seen:
            seen[cfg, quantized] = (streams, readers, hidden, tokens)
    return [
        pytest.param(*key, *shape, id=f"{i}", marks=QUALIFIED if key[1] else ())
        for i, (key, shape) in enumerate(seen.items())
    ]


@GPU
@pytest.mark.parametrize("cfg, quantized, streams, readers, hidden, tokens", cluster_configs())
def test_cluster_rows_match_unclustered(cfg, quantized, streams, readers, hidden, tokens):
    """A cluster's slices and partial-norm exchange match the same slices in one CTA per row."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("clusters need sm_90+")
    raw = inputs(streams, readers, hidden, tokens)
    (current, _, q), (current_one, _, q_one) = (
        forced_forward(c, raw, streams, mxfp8_handoff=quantized, inference=True)
        for c in (cfg, cfg._replace(CLUSTER=1))
    )
    assert bytes_equal(current, current_one)
    for a, b in zip(q, q_one, strict=True):
        if type(a) is torch.Tensor:
            # Only the partial-sum order differs: at most one BF16 ulp.
            torch.testing.assert_close(a.float(), b.float(), rtol=1 / 128, atol=1e-6)
        else:
            assert (a._rowwise_data != b._rowwise_data).float().mean() < 2e-3
            assert (a._rowwise_scale_inv != b._rowwise_scale_inv).float().mean() < 2e-3


# ---------------------------------------------------------------- MXFP8 handoff


@pytest.fixture
def reader_work(monkeypatch):
    """Count reconstruction at the actual entry points without production counters."""
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
            params_dtype=torch.bfloat16,
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
            h,
            128,
            bias=bias,
            params_dtype=torch.bfloat16,
            device="cuda",
            fuse_wgrad_accumulation=True,
        )
    layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
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
    x = torch.empty((m, h), device="cuda", dtype=torch.float32)
    normalized_reader_oracle[(m, triton.cdiv(h, 256))](
        r, update, coefficients, rstd, gamma, x, h, 256, enable_fp_fusion=False
    )
    return x


class _ReferenceQuantization(torch.autograd.Function):
    """Use independent FP32 normalization/TE encoding while preserving the dX edge."""

    @staticmethod
    def forward(ctx, operand, normalized):
        return mxfp8_quantizer()(normalized).to_dtype(torch.bfloat16)

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
    qref = mxfp8_quantizer()(x.detach()).to_dtype(torch.bfloat16).requires_grad_(True)
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
    dy = torch.randn(m, 128, device="cuda", dtype=torch.bfloat16)
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
    # Weight and bias gradients remain exact; tuned projection reductions may change the
    # FP32 accumulation order of the rest.
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
    # GTP's native FP8 parameter carries prefetch/communication state in its
    # Python attributes. A scope-wide saved-tensor hook loses that state when
    # autograd detaches the saved weight, even if its pack/unpack return it as-is.
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
@pytest.mark.parametrize("m,h", MX_SHAPES)
@torch.no_grad()
def test_native_update_rebinding_and_captured_descriptors(m, h):
    """Each captured graph must retain the descriptor for its own saved update."""
    _, (saved, r, dys, carry) = backward_args(mx_inputs(m, h, 1))
    changed = replace(saved, update=-saved.update)
    grid = capture_backward_config(m, h, 3, 1, columns=False, retention=True, grid="auto")
    options = dict(config=grid.TARGET_CTAS, save_columns=(True,))

    def outputs(ws):
        column = ws.columns[0]
        return (*ws.grads, column._columnwise_data, column._columnwise_scale_inv)

    graphs, workspaces, expected = [], [], []
    for state in (saved, changed):
        _, independent = run_with_scratch(native_backward, state, r, dys, carry, **options)
        expected.append(tuple(None if x is None else x.clone() for x in outputs(independent)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, workspace = run_with_scratch(native_backward, state, r, dys, carry, **options)
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
        assert reader_for(q) is None and q.dtype == torch.bfloat16
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
            weight.main_grad = torch.zeros(weight.shape, device="cuda", dtype=torch.float32)
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
@pytest.mark.parametrize(
    "hidden", [32, 64, 96, 128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 10240]
)
def test_generic_norm_matches_unfused_oracle(hidden):
    torch.manual_seed(947)
    rows = 128
    x = torch.randn(rows, hidden, device="cuda")
    gamma = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    quantizer = mxfp8_quantizer(columnwise=False)
    expected_rs = (x.double().square().mean(-1) + 1.0e-5).rsqrt()
    # One identity-retained stream with zero update and a half-strength reader
    # turns residual=2*x into exactly x, isolating the production norm arithmetic.
    update = torch.zeros_like(x, dtype=torch.bfloat16)
    coefficients = torch.empty(8, device=x.device)
    controller = torch.zeros(1, device=x.device)
    actual_rs = torch.empty(rows, device=x.device, dtype=torch.float32)
    actual = quantizer.make_empty(x.shape, dtype=torch.bfloat16, device=x.device)
    actual._rowwise_scale_inv.zero_()
    # Supply output buffers even when checking normalization statistics.
    forward_kernel._forward[(rows,)](
        forward_kernel._WriteArgs(2 * x, update, update, torch.empty_like(x), None),
        read_tensors=forward_kernel._ReadArgs(
            (gamma,), (actual_rs,), (actual._rowwise_data,), (actual._rowwise_scale_inv,)
        ),
        controllers=forward_kernel._ControllerArgs(controller, None, controller, controller),
        saved_coefficients=coefficients,
        GEOMETRY=forward_kernel._Geometry(hidden, 1, False),
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


@GPU
def test_replay_interrupts_at_an_unqualified_native_write():
    """A write the native Triton writer cannot take (here: a non-contiguous update) ends the open
    replay interval instead of failing: autograd saves it, the next write starts a new interval
    anchored at its output, and replay inside that interval stays bit-exact."""
    from megatron.core.transformer.streamwise_residual_ops import streamwise_sigmoid_writeback

    config = SimpleNamespace(
        moe_shortcut_wide_residual_norm_fusion=True, residual_stream_recompute_num_layers=8
    )
    torch.manual_seed(5)
    rows, hidden, streams = 1024, 128, 3
    leaves = (
        torch.randn(rows, streams * hidden, device="cuda", requires_grad=True),
        *(
            torch.randn(rows, 2 * hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(4)
        ),
        torch.randn(4, streams, device="cuda", requires_grad=True),  # write logits
        torch.randn(4, streams, device="cuda", requires_grad=True),  # retention logits
    )

    def chain(replay):
        x, *updates, write, retain = (t.detach().clone().requires_grad_(True) for t in leaves)
        outputs = []
        with residual_replay_scope(config, training=replay) as manager:
            residual = x
            for i, update in enumerate(updates):
                # Write 2 takes a strided (non-contiguous) update view; the others are contiguous.
                update = update[:, ::2] if i == 1 else update[:, :hidden].contiguous()
                if manager is not None:
                    manager.advance(1)
                residual = streamwise_sigmoid_writeback(
                    residual,
                    update,
                    write[i],
                    streams,
                    retention_logits=retain[i],
                    retention_max_forget=0.1,
                )
                outputs.append(residual)
            if replay:
                # The interval opened by write 3 is anchored at the unreplayed write 2's output.
                assert manager.active.checkpoint.tensor.data_ptr() == outputs[1].data_ptr()
                assert len(manager.active.transitions) == 2
                assert outputs[2].untyped_storage().nbytes() == 0  # freed, replayed in backward
        residual.float().square().mean().backward()
        grads = [t.grad for t in (x, *updates, write, retain)]
        return residual.detach(), grads

    expected, expected_grads = chain(replay=False)
    actual, actual_grads = chain(replay=True)
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))

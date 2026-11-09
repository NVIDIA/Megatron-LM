# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for the streamwise residual read/write forward kernels and their inference selection.

Call sites: ``streamwise_sigmoid_read`` and ``streamwise_sigmoid_writeback``
(megatron/core/transformer/streamwise_residual_ops.py). Every layer of a wide-residual
HybridStack reads its branch input from the residual stream and writes the branch output back
through them (``StreamwiseSigmoidWideResidualConnection`` and ``StreamwiseSigmoidResidualReadout``
in megatron/core/transformer/wide_residual.py), also in the inference stack
(``wide_residual_hybrid_inference_stack_spec``).

Contract (from the wide-residual math and the operators' documented semantics):

* Layout. The residual stream ``X [..., K * W]`` holds ``K`` contiguous full-width streams
  ``X_k = X[..., k W : (k + 1) W]``. Each controller is a vector of raw logits padded past ``K``
  (``>= K`` entries); only the first ``K`` are part of the input.
* Read: ``out = sum_k sigmoid(l_k) X_k``, shape ``[..., W]``, accumulated in at least fp32 and
  rounded once to ``output_dtype`` (default: the dtype of ``X``).
* Write: ``Y_k = g_k X_k + 2 sigmoid(w_k) U``, shape ``[..., K * W]`` in the dtype of ``X``, with
  ``g_k = 1`` (identity carry) or ``g_k = 1 - f sigmoid(-r_k)`` (learned retention,
  ``0 < f <= 1``), accumulated in at least fp32 and rounded once. ``U [..., W]`` has the dtype
  of ``X``, or bf16/fp16 when ``X`` is fp32 (the update is promoted, not rounded to bf16 first).
* Inputs are read only; the result is a new tensor with the leading dimensions of ``X``.
* Selection. In an inference forward (``InferenceMode`` active and no autograd, as the engine
  runs under ``no_grad`` with parameters that require grad) the fused kernels serve every batch
  size: the training floor of 256 rows and the cap on gradient partials do not apply. Outside
  it, training and the residual recompute under ``no_grad`` keep both limits, so a recompute
  picks the kernel of its original forward. The fused kernels and the torch fallback must meet
  the same contract; the selection is checked with a spy on the fused launchers.

Reference: fp64 from the definition above. Tolerance class: floating point, per row
(``dim=-1``), bounded by a trusted implementation: the same sums with torch's elementwise ops in
fp32 (not the module's own fallback), rounded once to the output dtype. Logits 0 (read factor
1/2, write factor 1) with dyadic inputs make every sum exact in fp32: exact class.

Structure (read after the contract): the forward kernels tile rows by ``BLOCK_BATCH`` 32 (64 from
4096 rows, when ``W < 512``) and columns by ``BLOCK_WIDTH = min(128, W)`` below 512 and 256 from
512 (``_streamwise_block_config``), unroll the ``K`` streams, and specialize on ``K``, ``W``,
``HAS_RETENTION`` and ``MAX_FORGET``. Sizes: rows 0, 1, 31...33, 63...65, 127...129 and
4095...4097; widths 64 (the smallest the dispatch accepts), 127...129, 255...257, 511...513 and
4096; ``K`` 1, 2, 3, 4 and 16 (the largest fused; 17 falls back); bf16, fp16 and fp32 streams,
bf16 and fp32 logits, bf16/fp16 updates into an fp32 stream, read outputs in another dtype,
2-D and 3-D (``[s, b, K W]``) layouts; padded logits poisoned with NaN past ``K``; inputs in
NaN-guarded buffers (an over-read shows as NaN, a write as a changed guard). Production: ``K``
4 of 4096-wide streams at a 8192-token prefill chunk and a 256-token decode batch, bf16 and an
fp32 stream with bf16 updates. Offsets: rows whose element offsets ``row * K * W`` cross 2**31
(``large``).

Widths between 64 and 128 that are not powers of two (``test_widths_below_the_column_tile``
[96], [127]) give a column tile of ``W`` columns, which Triton cannot build (``tl.arange`` needs
a power of two), although the dispatch accepts every ``W >= 64``: suspected O8b-M1 (packet
doc). They fail on main.
"""

import contextlib
import math
from typing import Optional

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer import streamwise_residual_ops as sw
from tests.unit_tests.inference.oracles import oracle_utils as ou

# The kernels are arch-generic Triton: H100 and GB200.
pytestmark = [ou.requires_sm(ou.HOPPER, ou.BLACKWELL), pytest.mark.launch_on_gb200]

PAD = 128  # controller vectors are padded to 128 entries (wide_residual._MIN_CONTROLLER_NUMEL)
FUSED_MAX_STREAMS = 16
# Column tile 128 below 512 columns, 256 from 512; row tile 32, 64 from 4096 rows below 512.
WIDTHS = [64, 128, 129, 255, 256, 257, 511, 512, 513]
ROWS = ou.boundary_sizes(32, 64, multiples=(2,), production=(4095, 4096, 4097))


# ---------------------------------------------------------------------------------------------
# Reference and trusted implementations (from the contract)
# ---------------------------------------------------------------------------------------------


def _streams(x: torch.Tensor, k: int) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], k, x.shape[-1] // k)


def read_reference(x, logits, k):
    """fp64 ``sum_k sigmoid(l_k) X_k`` (call through ``ou.fp64_reference``)."""
    factors = torch.sigmoid(logits[:k])
    return (_streams(x, k) * factors[:, None]).sum(-2)


def write_reference(x, update, write_logits, k, retention_logits=None, max_forget=0.0):
    """fp64 ``Y_k = g_k X_k + 2 sigmoid(w_k) U``."""
    write = 2 * torch.sigmoid(write_logits[:k])
    carry = torch.ones_like(write)
    if retention_logits is not None:
        carry = 1 - max_forget * torch.sigmoid(-retention_logits[:k])
    y = _streams(x, k) * carry[:, None] + write[:, None] * update.unsqueeze(-2)
    return y.flatten(-2)


def read_trusted(x, logits, k, out_dtype):
    """The same sum with torch's elementwise fp32 ops, rounded once to ``out_dtype``."""
    factors = torch.sigmoid(logits[:k].float())
    streams = _streams(x, k).float()
    total = factors[0] * streams[..., 0, :]
    for s in range(1, k):
        total = total + factors[s] * streams[..., s, :]
    return total.to(out_dtype)


def write_trusted(x, update, write_logits, k, retention_logits=None, max_forget=0.0):
    write = 2 * torch.sigmoid(write_logits[:k].float())
    carry = torch.ones_like(write)
    if retention_logits is not None:
        carry = 1 - max_forget * torch.sigmoid(-retention_logits[:k].float())
    y = _streams(x, k).float() * carry[:, None] + write[:, None] * update.float().unsqueeze(-2)
    return y.flatten(-2).to(x.dtype)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def _leading(rows: int, layout: str):
    """2-D ``[rows, ...]`` or the model's ``[s, b, ...]`` with b = 2 (rows rounded up to even)."""
    if layout == "sbh":
        return (max(1, -(-rows // 2)), 2)
    return (rows,)


def _logits(k: int, dtype: torch.dtype, g, scale: float = 2.0) -> torch.Tensor:
    """A padded controller vector: ``k`` random logits, NaN in the padding (never read)."""
    logits = torch.full((max(PAD, k),), float("nan"), device="cuda", dtype=dtype)
    logits[:k] = (torch.randn(k, device="cuda", generator=g) * scale).to(dtype)
    return logits


def _guarded(t: torch.Tensor) -> ou.GuardedBuffer:
    """``t`` in a NaN-guarded buffer one tile of rows wide (64 rows of the widest row)."""
    row_bytes = t.shape[-1] * t.element_size()
    return ou.guarded_like(t, guard_bytes=64 * row_bytes)


def _rows2d(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def _check(out, ref, trusted, what):
    assert out.dtype == trusted.dtype, f"{what}: dtype {out.dtype}, expected {trusted.dtype}"
    assert out.shape == trusted.shape, f"{what}: shape {tuple(out.shape)}"
    if out.numel() == 0:
        return
    ou.assert_close_to_reference(_rows2d(out), _rows2d(ref), _rows2d(trusted), dim=-1, what=what)


def _mode(inference: bool):
    """An inference forward (``InferenceMode``) or a training recompute (``no_grad`` only)."""
    return InferenceMode.active() if inference else contextlib.nullcontext()


def _run_read(x, logits, k, out_dtype=None, *, inference=True):
    with torch.no_grad(), _mode(inference):
        return sw.streamwise_sigmoid_read(x, logits, k, output_dtype=out_dtype)


def _run_write(x, update, logits, k, retention=None, max_forget=0.0, *, inference=True):
    with torch.no_grad(), _mode(inference):
        return sw.streamwise_sigmoid_writeback(
            x, update, logits, k, retention_logits=retention, retention_max_forget=max_forget
        )


def _read_case(rows, width, k, dtype, out_dtype, logits_dtype, layout, *, inference=True):
    g = ou.make_generator(device="cuda")
    shape = (*_leading(rows, layout), k * width)
    x = _guarded(torch.randn(shape, device="cuda", generator=g).to(dtype))
    logits = _guarded(_logits(k, logits_dtype, g))
    x_before, logits_before = x.tensor.clone(), logits.tensor.clone()
    out = _run_read(x.tensor, logits.tensor, k, out_dtype, inference=inference)
    torch.cuda.synchronize()
    ou.check_guards(x=x, logits=logits)
    ou.assert_exact(x.tensor, x_before, what="X (read-only)")
    ou.assert_exact(logits.tensor, logits_before, what="read logits (read-only)")
    resolved = dtype if out_dtype is None else out_dtype
    ref = ou.fp64_reference(read_reference, x.tensor, logits.tensor[:k], k)
    trusted = read_trusted(x.tensor, logits.tensor, k, resolved)
    _check(out, ref, trusted, f"read {tuple(shape)} K={k} {dtype}->{resolved}")


def _write_case(
    rows,
    width,
    k,
    dtype,
    update_dtype,
    logits_dtype,
    layout,
    max_forget: Optional[float],
    *,
    inference=True,
):
    g = ou.make_generator(device="cuda")
    lead = _leading(rows, layout)
    x = _guarded(torch.randn((*lead, k * width), device="cuda", generator=g).to(dtype))
    update = _guarded(torch.randn((*lead, width), device="cuda", generator=g).to(update_dtype))
    write = _guarded(_logits(k, logits_dtype, g))
    retention = None if max_forget is None else _guarded(_logits(k, logits_dtype, g))
    inputs = dict(x=x, update=update, write_logits=write)
    if retention is not None:
        inputs["retention_logits"] = retention
    before = {name: buf.tensor.clone() for name, buf in inputs.items()}
    out = _run_write(
        x.tensor,
        update.tensor,
        write.tensor,
        k,
        None if retention is None else retention.tensor,
        0.0 if max_forget is None else max_forget,
        inference=inference,
    )
    torch.cuda.synchronize()
    ou.check_guards(**inputs)
    for name, buf in inputs.items():
        ou.assert_exact(buf.tensor, before[name], what=f"{name} (read-only)")
    r = None if retention is None else retention.tensor[:k]
    f = 0.0 if max_forget is None else max_forget
    ref = ou.fp64_reference(write_reference, x.tensor, update.tensor, write.tensor[:k], k, r, f)
    trusted = write_trusted(x.tensor, update.tensor, write.tensor, k, r, f)
    _check(out, ref, trusted, f"write {tuple(x.shape)} K={k} {dtype}<-{update_dtype} f={f}")


# ---------------------------------------------------------------------------------------------
# Value tests (inference forward)
# ---------------------------------------------------------------------------------------------

READ_DTYPES = [
    (torch.bfloat16, None),
    (torch.float16, None),
    (torch.float32, None),
    (torch.bfloat16, torch.float32),  # a promoted read
    (torch.float32, torch.bfloat16),  # the branch input of an fp32 residual stream
]
WRITE_DTYPES = [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.float32, torch.float32),
    (torch.float32, torch.bfloat16),  # fp32 residual stream, bf16 branch output
    (torch.float32, torch.float16),
]


def _dtype_id(pair):
    a, b = pair
    name = lambda d: str(d).replace("torch.", "")  # noqa: E731
    return name(a) if b is None or b == a else f"{name(a)}-{name(b)}"


def _read_params():
    cases = ou.pairwise_cases(
        {
            "rows": ROWS,
            "width": WIDTHS,
            "k": [1, 2, 3, 4, FUSED_MAX_STREAMS],
            "dtypes": READ_DTYPES,
            "logits": [torch.float32, torch.bfloat16],
            "layout": ["2d", "sbh"],
        }
    )
    return [
        pytest.param(
            c,
            id=f"rows={c['rows']}-W={c['width']}-K={c['k']}-{_dtype_id(c['dtypes'])}"
            f"-logits={_dtype_id((c['logits'], None))}-{c['layout']}",
        )
        for c in cases
    ]


def _write_params():
    cases = ou.pairwise_cases(
        {
            "rows": ROWS,
            "width": WIDTHS,
            "k": [1, 2, 3, 4, FUSED_MAX_STREAMS],
            "dtypes": WRITE_DTYPES,
            "retention": [None, 0.5, 1.0],
            "logits": [torch.float32, torch.bfloat16],
            "layout": ["2d", "sbh"],
        }
    )
    return [
        pytest.param(
            c,
            id=f"rows={c['rows']}-W={c['width']}-K={c['k']}-{_dtype_id(c['dtypes'])}"
            f"-retention={c['retention']}-logits={_dtype_id((c['logits'], None))}-{c['layout']}",
        )
        for c in cases
    ]


@pytest.mark.parametrize("case", _read_params())
def test_read_matches_reference(case):
    """Inference reads over the tile boundaries, stream counts, dtypes and layouts."""
    x_dtype, out_dtype = case["dtypes"]
    _read_case(
        case["rows"], case["width"], case["k"], x_dtype, out_dtype, case["logits"], case["layout"]
    )


@pytest.mark.parametrize("case", _write_params())
def test_write_matches_reference(case):
    """Inference writes, identity carry and learned retention, over the same structure."""
    x_dtype, update_dtype = case["dtypes"]
    _write_case(
        case["rows"],
        case["width"],
        case["k"],
        x_dtype,
        update_dtype,
        case["logits"],
        case["layout"],
        case["retention"],
    )


@pytest.mark.parametrize("width", [64, 96, 127, 128])
@pytest.mark.parametrize("op", ["read", "write"])
def test_widths_below_the_column_tile(op, width):
    """Stream widths from the dispatch's minimum (64) to the 128-column tile. The dispatch
    accepts every width from 64, so each must meet the contract, fused or not.

    [96] and [127] fail on main (suspected O8b-M1): the kernels' column tile is ``W`` columns
    below 128, and Triton builds tiles only of power-of-two sizes."""
    if op == "read":
        _read_case(33, width, 3, torch.bfloat16, None, torch.float32, "2d")
    else:
        _write_case(33, width, 3, torch.bfloat16, torch.bfloat16, torch.float32, "2d", 0.5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_unit_controllers_are_exact(dtype):
    """Logits 0 (read factor 1/2, write factor 1, retention 1 - f/2) on dyadic streams: every
    sum is exact in fp32, so each output is the exact result rounded once."""
    rows, width, k = 33, 129, 4
    g = ou.make_generator(device="cpu")
    dyadic = dict(mantissa_bits=4, exponents=(-2, 2), dtype=dtype, generator=g)
    x = ou.dyadic_tensor((rows, k * width), **dyadic).cuda()
    update = ou.dyadic_tensor((rows, width), **dyadic).cuda()
    zeros = torch.zeros(PAD, device="cuda")
    zeros[k:] = float("nan")
    with torch.no_grad(), InferenceMode.active():
        read = sw.streamwise_sigmoid_read(x, zeros, k)
        write = sw.streamwise_sigmoid_writeback(
            x, update, zeros, k, retention_logits=zeros, retention_max_forget=0.5
        )
    ref_read = ou.fp64_reference(read_reference, x, zeros[:k], k)
    ref_write = ou.fp64_reference(write_reference, x, update, zeros[:k], k, zeros[:k], 0.5)
    ou.assert_exact(read, ref_read.to(dtype), what="read with unit controllers")
    ou.assert_exact(write, ref_write.to(dtype), what="write with unit controllers")


@pytest.mark.parametrize("op", ["read", "write"])
def test_saturated_controllers(op):
    """Logits of +-30: sigmoid saturates to 1 and to ~1e-13 (a stream that is read or kept
    with weight ~0, and one written with weight 2), within the same call."""
    rows, width, k = 65, 257, 4
    g = ou.make_generator(device="cuda")
    x = torch.randn(rows, k * width, device="cuda", generator=g).bfloat16()
    logits = torch.full((PAD,), float("nan"), device="cuda")
    logits[:k] = torch.tensor([30.0, -30.0, 0.0, -30.0], device="cuda")
    if op == "read":
        out = _run_read(x, logits, k)
        ref = ou.fp64_reference(read_reference, x, logits[:k], k)
        _check(out, ref, read_trusted(x, logits, k, x.dtype), "saturated read")
    else:
        update = torch.randn(rows, width, device="cuda", generator=g).bfloat16()
        retention = torch.full((PAD,), float("nan"), device="cuda")
        retention[:k] = torch.tensor([-30.0, 30.0, 30.0, 0.0], device="cuda")
        out = _run_write(x, update, logits, k, retention, 1.0)
        ref = ou.fp64_reference(write_reference, x, update, logits[:k], k, retention[:k], 1.0)
        _check(out, ref, write_trusted(x, update, logits, k, retention[:k], 1.0), "saturated")


@pytest.mark.parametrize("op", ["read", "write"])
def test_empty_batch(op):
    """No rows: an empty result of the right shape and dtype, no launch error."""
    k, width = 3, 128
    x = torch.empty(0, k * width, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(PAD, device="cuda")
    if op == "read":
        out = _run_read(x, logits, k)
        assert out.shape == (0, width) and out.dtype == torch.bfloat16
    else:
        update = torch.empty(0, width, device="cuda", dtype=torch.bfloat16)
        out = _run_write(x, update, logits, k)
        assert out.shape == (0, k * width) and out.dtype == torch.bfloat16
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------
# Inference selection
# ---------------------------------------------------------------------------------------------


def _spy(monkeypatch, name, calls):
    """Record the calls of the fused launcher ``name`` in ``calls``."""
    original = getattr(sw, name)

    def spy(*args, **kwargs):
        calls.append((name, tuple(args[0].shape)))
        return original(*args, **kwargs)

    monkeypatch.setattr(sw, name, spy)


SELECTION = [
    # (id, rows, width, k, inference, contiguous, fused expected)
    ("inference-1-row", 1, 128, 3, True, True, True),
    ("inference-255-rows", 255, 128, 3, True, True, True),
    ("recompute-255-rows", 255, 128, 3, False, True, False),
    ("recompute-256-rows", 256, 128, 3, False, True, True),
    ("inference-16-streams", 7, 64, 16, True, True, True),
    ("inference-17-streams", 7, 64, 17, True, True, False),
    ("inference-width-32", 7, 32, 3, True, True, False),
    ("inference-strided", 40, 128, 3, True, False, False),
    # past the training cap on gradient partials (16 x 512 backward tiles): 65536 rows of
    # 4096 give 32768 partials > 16384
    ("inference-past-partial-cap", 65536, 4096, 2, True, True, True),
    ("recompute-past-partial-cap", 65536, 4096, 2, False, True, False),
]


@pytest.mark.parametrize("op", ["read", "write"])
@pytest.mark.parametrize(
    "rows, width, k, inference, contiguous, fused",
    [pytest.param(*c[1:], id=c[0]) for c in SELECTION],
)
def test_inference_selects_the_fused_kernels(
    monkeypatch, op, rows, width, k, inference, contiguous, fused
):
    """An inference forward (``InferenceMode``, ``no_grad``, controllers that require grad as
    model parameters do) takes the fused kernels at every batch size; a recompute under
    ``no_grad`` without ``InferenceMode`` keeps the training limits. Unsupported shapes fall
    back. Every path meets the contract (rows sampled for the large case)."""
    calls = []
    _spy(monkeypatch, "_streamwise_sigmoid_read_triton", calls)
    _spy(monkeypatch, "_streamwise_sigmoid_write_triton", calls)
    g = ou.make_generator(device="cuda")
    x = torch.randn(rows, k * width, device="cuda", generator=g).bfloat16()
    if not contiguous:
        x = torch.randn(rows, 2 * k * width, device="cuda", generator=g).bfloat16()[:, ::2]
        assert not x.is_contiguous()
    logits = _logits(k, torch.float32, g).requires_grad_(True)
    update = torch.randn(rows, width, device="cuda", generator=g).bfloat16()
    if op == "read":
        out = _run_read(x, logits, k, inference=inference)
    else:
        out = _run_write(x, update, logits, k, inference=inference)
    assert (
        bool(calls) == fused
    ), f"{op}: fused kernel {'not ' if fused else ''}expected, launcher calls {calls}"
    sample = torch.unique(torch.cat([torch.arange(0, rows, 61), torch.tensor([rows - 1])]))
    sample = sample.cuda()
    xs, us, outs = x[sample], update[sample], out[sample]
    w = logits.detach()
    if op == "read":
        ref = ou.fp64_reference(read_reference, xs, w[:k], k)
        _check(outs, ref, read_trusted(xs, w, k, xs.dtype), "read")
    else:
        ref = ou.fp64_reference(write_reference, xs, us, w[:k], k)
        _check(outs, ref, write_trusted(xs, us, w, k), "write")


# ---------------------------------------------------------------------------------------------
# Production sizes, int32 offsets, sanitizer
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [256, 8192], ids=["decode-256", "prefill-8192"])
@pytest.mark.parametrize(
    "stream_dtype", [torch.bfloat16, torch.float32], ids=["bf16-stream", "fp32-stream"]
)
def test_production_sizes(rows, stream_dtype):
    """Four 4096-wide streams (a 4-stream wide residual of a 4096-hidden model), at a decode
    batch and a prefill chunk: the read in branch precision (bf16) and the write of a bf16
    branch output with learned retention, as the wide-residual connection calls them."""
    k, width = 4, 4096
    _read_case(rows, width, k, stream_dtype, torch.bfloat16, torch.float32, "sbh")
    _write_case(rows, width, k, stream_dtype, torch.bfloat16, torch.float32, "sbh", 0.5)


def _overflow_rows():
    return ou.int32_overflow_params(2 * 8192)


@pytest.mark.large
@pytest.mark.parametrize("op", ["read", "write"])
@pytest.mark.parametrize("rows", _overflow_rows())
def test_offsets_past_int32(op, rows):
    """Rows whose element offsets ``row * K * W`` reach 2**31 (K = 2 streams of 8192): the
    kernels must address them in 64 bits. Rows around the crossing and the last rows are
    compared with the reference."""
    k, width = 2, 8192
    row_elems = k * width
    ou.require_free_memory(rows * row_elems * 2 * 3 + (4 << 30))
    g = ou.make_generator(device="cuda")
    x = torch.randn(rows, row_elems, device="cuda", generator=g, dtype=torch.bfloat16)
    logits = _logits(k, torch.float32, g)
    cross = ou.INT32_LIMIT // row_elems
    sample = torch.tensor(
        sorted({0, 1, max(0, cross - 1), min(rows - 1, cross), min(rows - 1, cross + 1), rows - 1})
    ).cuda()
    if op == "read":
        out = _run_read(x, logits, k)
        torch.cuda.synchronize()
        xs = x[sample]
        ref = ou.fp64_reference(read_reference, xs, logits[:k], k)
        _check(out[sample], ref, read_trusted(xs, logits, k, xs.dtype), f"read, {rows} rows")
    else:
        update = torch.randn(rows, width, device="cuda", generator=g, dtype=torch.bfloat16)
        out = _run_write(x, update, logits, k)
        torch.cuda.synchronize()
        xs, us = x[sample], update[sample]
        ref = ou.fp64_reference(write_reference, xs, us, logits[:k], k)
        _check(out[sample], ref, write_trusted(xs, us, logits, k), f"write, {rows} rows")


@pytest.mark.sanitizer
def test_small_case_for_sanitizer():
    """An inference read and a write with retention whose rows and columns both end inside a
    tile (33 rows, 129 columns, 3 streams): the compute-sanitizer memcheck case."""
    _read_case(33, 129, 3, torch.bfloat16, None, torch.float32, "2d")
    _write_case(33, 129, 3, torch.float32, torch.bfloat16, torch.float32, "2d", 0.5)


# ---------------------------------------------------------------------------------------------
# CPU checks of the reference itself (no kernel)
# ---------------------------------------------------------------------------------------------


def test_reference_matches_the_definition_stream_by_stream():
    g = torch.Generator().manual_seed(0)
    k, width = 3, 5
    x = torch.randn(2, k * width, generator=g, dtype=torch.float64)
    update = torch.randn(2, width, generator=g, dtype=torch.float64)
    logits = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    retention = torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64)
    read = read_reference(x, logits, k)
    write = write_reference(x, update, logits, k, retention, 0.5)
    expected_read = sum(
        1 / (1 + math.exp(-float(logits[s]))) * x[:, s * width : (s + 1) * width] for s in range(k)
    )
    torch.testing.assert_close(read, expected_read)
    for s in range(k):
        carry = 1 - 0.5 / (1 + math.exp(float(retention[s])))
        factor = 2 / (1 + math.exp(-float(logits[s])))
        expected = carry * x[:, s * width : (s + 1) * width] + factor * update
        torch.testing.assert_close(write[:, s * width : (s + 1) * width], expected)

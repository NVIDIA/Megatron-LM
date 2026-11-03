# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_weighted_silu_mul_bounded_kernel`` / ``weighted_silu_mul_bounded``
(megatron/core/inference/moe/batch_invariant.py).

Contract (from the batch-invariant vLLM-backend call site, ``weighted_silu_mul_bounded(y,
weights, valid_tokens * topk * half_n)``, and training's fused weighted SwiGLU; written before
reading the kernel):

* **Computes**, over the flat token-major layout, for every output element
  ``i < min(bound_elems, rows * N)`` with ``r = i // N`` and ``c = i % N``:
  ``out[r, c] = silu(y[r, c]) * y[r, N + c] * weights[r]`` in fp32 with one bf16 rounding,
  **bitwise equal to training's fused ``weighted_swiglu(y, weights[:, None])``** (the docstring's
  training-parity claim, which batch-invariant generation relies on).
* **Inputs:** ``y`` [rows, 2N] bf16 contiguous; ``weights_flat`` [rows] fp32; ``bound_elems`` a
  1-element device int64 element count (``live_rows * N`` from the caller), read on device;
  ``num_programs`` (default SMs * 8) and ``xblock`` (default 1024) only size the persistent grid
  and must not change any bit.
* **Outputs:** a new [rows, N] tensor in ``y.dtype``; elements at or past the bound are not
  written (the bound is in elements, so it may end inside a row). ``y`` and ``weights`` are not
  modified.
* **Tolerance class:** (1) exact against the live training op (GPU only); (2) floating point
  against fp64: per element the single bf16 rounding of an fp32 evaluation, and the harness
  policy per row; (3) exact across grid sizes.

PR #8566 (d2) adds ``glu_options``, ``squared_relu_gate`` and ``weights_flat=None`` (routing
probabilities applied to the expert input): without weights the result must be bitwise training's
``weighted_swiglu(y, ones)``. Those cases run only where the arguments exist.

Structure: elementwise persistent 1-D grid of ``num_programs`` programs, ``XBLOCK`` elements per
step, striding while ``xoffset < bound``; ``HALF_N`` is a compile-time constant; int64 offsets.
Covered: element counts and bounds around ``XBLOCK`` and the grid stride, a bound inside a row,
a bound past the buffer, both production ``XBLOCK`` values, several grid sizes.
"""

import pytest
import torch

from megatron.core.inference.moe import batch_invariant
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = _common.hardware(ou.HOPPER)

XBLOCKS = (1024, 2048)  # production default 1024; 2048 is the determinism test's variant


def run_weighted(y, weights, bound, programs, xblock=1024, **options):
    """``weighted_silu_mul_bounded`` with a guarded, NaN-poisoned output; inputs unchanged."""
    before = [t.clone() for t in (y, weights) if t is not None]
    with _common.captured_allocations(batch_invariant) as allocs:
        out = batch_invariant.weighted_silu_mul_bounded(
            y,
            weights,
            _common.device_scalar(bound, torch.int64),
            num_programs=programs,
            xblock=xblock,
            **options,
        )
    assert len(allocs.buffers) == 1 and out.data_ptr() == allocs.buffers[0].tensor.data_ptr()
    allocs.check("weighted_silu_mul_bounded output")
    for t, b in zip([t for t in (y, weights) if t is not None], before):
        ou.assert_exact(t, b, what="inputs are read-only")
    return out


def check_elements(out, y, weights, bound, what, **options):
    """Elements below the bound against the contract; the rest still poisoned."""
    rows, n = out.shape
    bound = min(bound, rows * n)
    full, tail = divmod(bound, n)
    _glu.assert_swiglu_values(
        out[:full],
        y[:full],
        None if weights is None else weights[:full],
        what=f"{what}: full rows",
        **options,
    )
    if tail:
        _glu.assert_swiglu_values(
            out[full : full + 1, :tail],
            y[full : full + 1],
            None if weights is None else weights[full : full + 1],
            what=f"{what}: the row the bound ends in",
            cols=tail,
            **options,
        )
    flat = out.reshape(-1)
    ou.assert_poisoned(flat[bound:], what=f"{what}: elements past the bound")


def element_cases():
    """(rows, half_n, bound, programs, xblock): bounds around XBLOCK and the grid stride."""
    cases = []
    for xblock in XBLOCKS:
        for programs in (1, 3):
            stride = programs * xblock
            for half_n, rows in ((128, 2 * stride // 128 + 3), (1856, 5), (7, 3 * xblock // 7)):
                total = rows * half_n
                bounds = {0, 1, xblock - 1, xblock, xblock + 1, stride - 1, stride, stride + 1}
                bounds |= {total - 1, total, total + 5, half_n * (rows // 2)}
                for bound in sorted(b for b in bounds if b >= 0):
                    cases.append((rows, half_n, bound, programs, xblock))
    return cases


ELEMENT_CASES = element_cases()


@pytest.mark.parametrize(
    "rows,half_n,bound,programs,xblock",
    ELEMENT_CASES,
    ids=[f"M{r}-N{n}-bound{b}-P{p}-X{x}" for r, n, b, p, x in ELEMENT_CASES],
)
def test_values_and_bound(rows, half_n, bound, programs, xblock):
    """Element bounds around XBLOCK, the grid stride, row ends and the buffer end."""
    g = ou.make_generator(device=DEVICE)
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    out = run_weighted(y, weights, bound, programs, xblock)
    check_elements(out, y, weights, bound, what=f"bound {bound}")


def test_grid_size_does_not_change_bits():
    """``num_programs`` and ``xblock`` only size the grid: outputs are bitwise equal."""
    g = ou.make_generator(device=DEVICE)
    rows, half_n = 300, 256
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    bound = 211 * half_n + 17
    reference = run_weighted(y, weights, bound, 1, 1024)
    for programs, xblock in ((2, 1024), (7, 2048), (148, 1024), (4096, 2048)):
        out = run_weighted(y, weights, bound, programs, xblock)
        flat, ref = out.reshape(-1)[:bound], reference.reshape(-1)[:bound]
        ou.assert_exact(flat, ref, what=f"{programs} programs, xblock {xblock}")
        ou.assert_poisoned(out.reshape(-1)[bound:], what="elements past the bound")


def test_special_values():
    """Special gate and linear values, times a weight."""
    for i, y in enumerate(_glu.special_value_inputs(DEVICE)):
        rows, half_n = y.shape[0], y.shape[1] // 2
        weights = torch.linspace(0.05, 1.0, rows, device=DEVICE, dtype=torch.float32)
        out = run_weighted(y, weights, rows * half_n, 2, 1024)
        _glu.assert_swiglu_special_values(out, y, weights, what=f"special values, grid {i}")


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass."""
    g = ou.make_generator(device=DEVICE)
    y = _glu.glu_inputs((40, 2 * 100), g, DEVICE)
    weights = torch.rand(40, generator=g, device=DEVICE, dtype=torch.float32)
    out = run_weighted(y, weights, 31 * 100 + 3, 2, 1024)
    check_elements(out, y, weights, 31 * 100 + 3, what="sanitizer case")


@_common.needs_cuda
def test_default_grid():
    """The default grid (SMs * 8 programs) through the production call shape."""
    g = ou.make_generator(device=DEVICE)
    rows, half_n = 2048, 1856
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    out = run_weighted(y, weights, 1500 * half_n, None)
    check_elements(out, y, weights, 1500 * half_n, what="default grid")


@_common.needs_cuda
@pytest.mark.parametrize("half_n", [128, 1856, 2688])
def test_matches_training_bitwise(half_n):
    """Bitwise training's fused ``weighted_swiglu`` on every element below the bound."""
    from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu

    g = ou.make_generator(device=DEVICE)
    rows = 513
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    live = 400
    out = run_weighted(y, weights, live * half_n, None)
    expected = weighted_swiglu(y[:live], weights[:live, None])
    ou.assert_exact(out[:live], expected, what="weighted_silu_mul_bounded vs training")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(2 * 1024))
def test_offsets_past_int32(rows):
    """Input offsets ``row * 2N`` reach 2**31: rows around the crossing and the last rows."""
    half_n = 1024
    ou.require_free_memory(rows * 3 * half_n * 2 + rows * 4 + 2**30)
    y = torch.ones(rows, 2 * half_n, dtype=torch.bfloat16, device=DEVICE)
    weights = torch.full((rows,), 0.5, dtype=torch.float32, device=DEVICE)
    g = ou.make_generator(device=DEVICE)
    probe = [0, 2**31 // (2 * half_n) - 1, 2**31 // (2 * half_n), rows - 2, rows - 1]
    probe = sorted({r for r in probe if 0 <= r < rows})
    y[probe] = _glu.glu_inputs((len(probe), 2 * half_n), g, DEVICE)
    out = batch_invariant.weighted_silu_mul_bounded(
        y, weights, _common.device_scalar(rows * half_n, torch.int64)
    )
    _glu.assert_swiglu_values(out[probe], y[probe], weights[probe], what=f"rows {probe}")


# ---------------------------------------------------------------------------------------------
# PR #8566 (d2): GLU options, gated squared ReLU and probabilities applied on the input
# ---------------------------------------------------------------------------------------------

D2_OPTIONS = {
    "clamp": dict(clamp=7.0),
    "clamp-not-bf16": dict(clamp=7.3),
    "offset": dict(offset=1.0),
    "interleave-32": dict(interleave=32),
    "all": dict(clamp=7.0, offset=1.0, interleave=64),
}


@_glu.requires_glu_options
@pytest.mark.parametrize("name", list(D2_OPTIONS))
@pytest.mark.parametrize("weighted", [True, False], ids=["weights", "no-weights"])
def test_d2_swiglu_options(name, weighted):
    """SwiGLU with training's clamp, linear offset and interleaved layout, with or without
    weights: fp32 compute, one rounding."""
    options = D2_OPTIONS[name]
    g = ou.make_generator(device=DEVICE)
    rows, half_n = 300, 1856
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE, scale=4.0)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    weights = weights if weighted else None
    bound = 211 * half_n
    out = run_weighted(y, weights, bound, 3, 1024, **_glu.glu_kwargs(**options))
    check_elements(out, y, weights, bound, what=name, **options)


@_common.needs_cuda
@_glu.requires_glu_options
@pytest.mark.parametrize("clamp", [None, 7.0, 7.3])
def test_d2_matches_training_bitwise(clamp):
    """Bitwise training's fused weighted SwiGLU (``clamped_weighted_swiglu`` with a clamp),
    with weights and without (training then weights by ones)."""
    from megatron.core.fusions.fused_bias_swiglu import clamped_weighted_swiglu, weighted_swiglu

    g = ou.make_generator(device=DEVICE)
    rows, half_n = 513, 1856
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE, scale=4.0)
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    for w in (weights, None):
        out = run_weighted(y, w, rows * half_n, None, **_glu.glu_kwargs(clamp=clamp))
        w_train = torch.ones_like(weights) if w is None else w
        if clamp is None:
            expected = weighted_swiglu(y, w_train[:, None])
        else:
            expected = clamped_weighted_swiglu(y, w_train[:, None], clamp)
        ou.assert_exact(out, expected, what=f"clamp {clamp}, weights {w is not None}")


@_glu.requires_glu_options
@pytest.mark.parametrize("name", ["none"] + list(D2_OPTIONS))
@pytest.mark.parametrize("weighted", [True, False], ids=["weights", "no-weights"])
def test_d2_gated_squared_relu_matches_training(name, weighted):
    """``squared_relu_gate``: bitwise training's gated squared ReLU (bf16 ops), times the fp32
    weight rounded to bf16 when weights are given."""
    options = D2_OPTIONS.get(name, {})
    g = ou.make_generator(device=DEVICE)
    rows, half_n = 300, 1856
    y = _glu.glu_inputs((rows, 2 * half_n), g, DEVICE, scale=4.0)
    y[7, 3] = float("nan")
    y[9, 11] = float("inf")
    weights = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    weights = weights if weighted else None
    live = 211
    out = run_weighted(
        y, weights, live * half_n, 3, 1024, **_glu.glu_kwargs(squared_relu_gate=True, **options)
    )
    expected = _glu.gated_squared_relu_training(
        y[:live], None if weights is None else weights[:live], **options
    )
    _common.assert_exact_except_nan_payload(out[:live], expected, what=f"gated sq-ReLU {name}")
    ou.assert_poisoned(out[live:], what="rows past the bound")

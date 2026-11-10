# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_swiglu_with_probs_kernel`` / ``swiglu_with_probs``
(megatron/core/inference/moe/batch_invariant.py).

Contract (from the batch-invariant ``mcore_fused_moe`` call site and training's fused weighted
SwiGLU, which batch-invariant inference must reproduce bit for bit; written before reading the
kernel):

* **Computes** for every row ``r < min(n_used, M)`` with ``permutation_map[r] >= 0``:
  ``out[r, c] = silu(x[r, c]) * x[r, N + c] * probs[r]`` (gate first), in fp32 with one bf16
  rounding, and **bitwise equal to training's fused ``weighted_swiglu(x, probs[:, None])``**
  (``torch.compile``, the op batch-invariant training runs).
* **Inputs:** ``x`` [M, 2N] bf16 contiguous; ``permutation_map`` [M] int (-1 = padding);
  ``n_used`` 1-element device int32; ``probs`` [M] fp32 routing probabilities;
  ``zero_padding`` (True on the MXFP8 path, whose quantizer and GEMM read padding rows).
* **Outputs:** a new [M, N] bf16 tensor. Padding rows below ``n_used``: +0.0 with
  ``zero_padding``, untouched otherwise. Rows ``>= n_used``: untouched. ``x`` and ``probs``
  are not modified.
* **Tolerance class:** (1) exact against the live training op (the batch-invariance claim;
  GPU only, since the training op is compiled for the GPU); (2) floating point against fp64:
  per element the single bf16 rounding of an fp32 evaluation, and the harness policy per row
  against torch's fp32 ops rounded once.

PR #8566's GLU options and gated squared ReLU are covered by ``test_glu_options.py`` (on a
branch stacked on that PR).

Structure: ``BLOCK_SIZE = min(next_pow2(N), 1024)``; ``NUM_BLOCKS = min(M, 512)`` persistent
programs over ``min(n_used, M)`` rows; ``ZERO_PADDING`` compile-time variant.
"""

import pytest
import torch

from megatron.core.inference.moe import batch_invariant
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = _common.hardware(ou.HOPPER)

PROGRAMS_CAP = 512  # batch_invariant.py: num_blocks = min(num_rows, 512)

CASES = ou.pairwise_cases(
    {
        "width": [1, 31, 33, 1023, 1024, 1025, 1856, 2688],
        "live": list(_glu.LIVE_COUNTS),
        "padding": list(_glu.PADDING_PATTERNS),
        "rows": [1, PROGRAMS_CAP - 1, PROGRAMS_CAP, PROGRAMS_CAP + 1, 2 * PROGRAMS_CAP + 1],
        "zero_padding": [False, True],
    }
)


def _case_id(case):
    zp = "zero" if case["zero_padding"] else "skip"
    return f"N{case['width']}-M{case['rows']}-n{case['live']}-{case['padding']}-{zp}"


def make_inputs(rows, width, g, padding="none", scale=2.0):
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=scale)
    perm = _glu.permutation_map(rows, padding, torch.int32, g, DEVICE)
    probs = torch.rand(rows, generator=g, device=DEVICE, dtype=torch.float32)
    return x, perm, probs


def run_swiglu_with_probs(x, perm, n_used, probs, zero_padding=False, **options):
    """``swiglu_with_probs`` with a guarded, NaN-poisoned output; inputs must stay unchanged."""
    before = (x.clone(), probs.clone(), perm.clone())
    with _common.captured_allocations(batch_invariant) as allocs:
        out = batch_invariant.swiglu_with_probs(
            x, perm, n_used, probs, zero_padding=zero_padding, **options
        )
    assert len(allocs.buffers) == 1 and out.data_ptr() == allocs.buffers[0].tensor.data_ptr()
    allocs.check("swiglu_with_probs output")
    for t, b, name in zip((x, probs, perm), before, ("x", "probs", "permutation_map")):
        ou.assert_exact(t, b, what=f"{name} is read-only")
    return out


def row_masks(perm, n_used):
    index = torch.arange(perm.numel(), device=perm.device)
    below = index < min(n_used, perm.numel())
    live = below & (perm >= 0)
    return live, below & ~live, ~below


def check_rows(out, x, perm, probs, n_used, zero_padding, what, **options):
    live, padding, beyond = row_masks(perm, n_used)
    _glu.assert_swiglu_values(out[live], x[live], probs[live], what=f"{what}: live rows", **options)
    if zero_padding:
        ou.assert_exact(
            out[padding], torch.zeros_like(out[padding]), what=f"{what}: padding rows are +0"
        )
    else:
        ou.assert_poisoned(out[padding], what=f"{what}: padding rows")
    ou.assert_poisoned(out[beyond], what=f"{what}: rows past n_used")


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_values_and_live_rows(case):
    """Pairwise over width, grid size, live count, padding layout and ZERO_PADDING."""
    g = ou.make_generator(device=DEVICE)
    rows, width = case["rows"], case["width"]
    n_used = _glu.live_count(case["live"], rows, min(rows, PROGRAMS_CAP))
    x, perm, probs = make_inputs(rows, width, g, case["padding"])
    out = run_swiglu_with_probs(x, perm, _common.device_scalar(n_used), probs, case["zero_padding"])
    check_rows(out, x, perm, probs, n_used, case["zero_padding"], what=_case_id(case))


def test_special_values():
    """Special gate and linear values, times a probability."""
    for i, x in enumerate(_glu.special_value_inputs(DEVICE)):
        rows = x.shape[0]
        perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
        probs = torch.linspace(0.05, 1.0, rows, device=DEVICE, dtype=torch.float32)
        out = run_swiglu_with_probs(x, perm, _common.device_scalar(rows), probs, True)
        _glu.assert_swiglu_special_values(out, x, probs, what=f"special values, grid {i}")


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass."""
    g = ou.make_generator(device=DEVICE)
    x, perm, probs = make_inputs(130, 1025, g, "runs")
    out = run_swiglu_with_probs(x, perm, _common.device_scalar(100), probs, zero_padding=True)
    check_rows(out, x, perm, probs, 100, True, what="sanitizer case")


# Rows whose SiLU rounds differently through sigmoid multiplication than through division
# (test_inference_kernels.py::test_batch_invariant_swiglu_matches_training), plus random rows.
BOUNDARY_GATES = (0.032470703125, -1.078125, -5.0625, -12.0625)
BOUNDARY_LINEARS = (-0.0186767578125, 0.150390625, -3.875, 39.5)
BOUNDARY_PROBS = (0.688609778881073, 0.0012366651790216565, 0.2819514572620392, 0.7739971280097961)


def parity_inputs(width, g):
    x, _, probs = make_inputs(PROGRAMS_CAP + 1, width, g)
    for r, (gate, linear, p) in enumerate(zip(BOUNDARY_GATES, BOUNDARY_LINEARS, BOUNDARY_PROBS)):
        x[r, :width] = gate
        x[r, width:] = linear
        probs[r] = p
    return x, probs


@_common.needs_cuda
@pytest.mark.parametrize("width", [128, 1856, 2688])
@pytest.mark.parametrize("zero_padding", [False, True])
def test_matches_training_bitwise(width, zero_padding):
    """Bitwise training's fused ``weighted_swiglu`` on every live row."""
    from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu

    g = ou.make_generator(device=DEVICE)
    x, probs = parity_inputs(width, g)
    rows = x.shape[0]
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    n_used = rows - 5
    out = run_swiglu_with_probs(x, perm, _common.device_scalar(n_used), probs, zero_padding)
    live, _, _ = row_masks(perm, n_used)
    expected = weighted_swiglu(x[live], probs[live][:, None])
    ou.assert_exact(out[live], expected, what="swiglu_with_probs vs training weighted_swiglu")

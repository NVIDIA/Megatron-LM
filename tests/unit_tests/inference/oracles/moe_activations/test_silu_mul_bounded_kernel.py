# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_silu_mul_bounded_kernel`` / ``bounded_silu_mul``
(megatron/core/inference/moe/activations.py).

Contract (from the vLLM-backend call site, ``bounded_silu_mul(intermediate1, valid_tokens *
topk)``, and training's GLU; written before reading the kernel):

* **Computes** SwiGLU on the flat token-major FC1 output (row ``t * topk + k`` holds token t,
  slot k): for every row ``r < min(n_rows, M)``, ``out[r, c] = silu(x[r, c]) * x[r, N + c]``
  (gate first), evaluated in fp32 and rounded once to bf16. There is no permutation map: every
  row below the bound is live.
* **Inputs:** ``x`` [M, 2N] bf16, contiguous (even width asserted); ``n_rows`` a 1-element
  device int32 or int64 tensor, read on device (CUDA-graph safe).
* **Outputs:** a new [M, N] bf16 tensor; rows ``>= n_rows`` are not written. ``x`` is not
  modified. ``n_rows = 0`` writes nothing; ``n_rows > M`` writes the M rows and nothing past.
* **Special values:** as ``_swiglu_kernel`` (IEEE evaluation; subnormal intermediates may be
  flushed to ±0).
* **Tolerance class:** floating point, as ``_swiglu_kernel``: per element the single bf16
  rounding of an fp32 evaluation (``_common.assert_rounded_once``), plus the harness policy per
  row against torch's fp32 SiLU rounded once. Untouched rows: still poisoned.

PR #8566 (d2) adds ``glu_options`` and ``squared_relu_gate``; their cases (``test_d2_*``) run only
where the arguments exist. PR #8871 (t3-r03f5, stacked on d2) fixes the grid at 512 programs and
accepts ``n_rows=None``; the cases here hold for either grid.

Structure: ``BLOCK_N = min(next_pow2(N), 1024)``; ``NUM_BLOCKS = min(M, 512)`` persistent programs
(512 on #8871); int64 row offsets. Covered: N around the tile cap, M and ``n_rows`` around the
grid size, both count dtypes, production widths, special values, offsets past 2**31 (``large``).
"""

import pytest
import torch

from megatron.core.inference.moe import activations
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = _common.hardware(ou.HOPPER)

BLOCK_N_CAP = 1024  # activations.py: BLOCK_N = min(triton.next_power_of_2(N), 1024)
PROGRAMS_CAP = 512  # activations.py: NUM_BLOCKS = min(M, 512)

CASES = ou.pairwise_cases(
    {
        "width": [1, 31, 33, 1023, 1024, 1025, 1856, 2688],
        "live": list(_glu.LIVE_COUNTS),
        "rows": [1, 2, PROGRAMS_CAP - 1, PROGRAMS_CAP, PROGRAMS_CAP + 1, 2 * PROGRAMS_CAP + 1],
        "count_dtype": [torch.int32, torch.int64],
    }
)


def _case_id(case):
    dtype = str(case["count_dtype"]).replace("torch.", "")
    return f"N{case['width']}-M{case['rows']}-n{case['live']}-{dtype}"


def run_bounded_silu_mul(x, n_rows, **options):
    """``bounded_silu_mul`` with a guarded, NaN-poisoned output; checks guards and that ``x`` is
    read-only."""
    before = x.clone()
    with _common.captured_allocations(activations) as allocs:
        out = activations.bounded_silu_mul(x, n_rows, **options)
    assert len(allocs.buffers) == 1 and out.data_ptr() == allocs.buffers[0].tensor.data_ptr()
    allocs.check("bounded_silu_mul output")
    ou.assert_exact(x, before, what="x is read-only")
    return out


def check_rows(out, x, n_rows, what, **options):
    live = min(n_rows, x.shape[0])
    _glu.assert_swiglu_values(out[:live], x[:live], what=f"{what}: live rows", **options)
    ou.assert_poisoned(out[live:], what=f"{what}: rows past n_rows")


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_values_and_live_rows(case):
    """Pairwise over width, grid size, live count and count dtype."""
    g = ou.make_generator(device=DEVICE)
    rows, width = case["rows"], case["width"]
    n_rows = _glu.live_count(case["live"], rows, min(rows, PROGRAMS_CAP))
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    out = run_bounded_silu_mul(x, _common.device_scalar(n_rows, case["count_dtype"]))
    check_rows(out, x, n_rows, what=_case_id(case))


def test_special_values():
    """Zeros, subnormals, saturating sigmoids, overflowing exp and products, infinities, NaN."""
    for i, x in enumerate(_glu.special_value_inputs(DEVICE)):
        out = run_bounded_silu_mul(x, _common.device_scalar(x.shape[0]))
        _glu.assert_swiglu_special_values(out, x, what=f"special values, grid {i}")


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass."""
    g = ou.make_generator(device=DEVICE)
    x = _glu.glu_inputs((130, 2 * 1025), g, DEVICE)
    out = run_bounded_silu_mul(x, _common.device_scalar(100))
    check_rows(out, x, 100, what="sanitizer case")


@_common.needs_cuda
def test_cuda_graph_replay_follows_device_count():
    """Captured once, replayed with other live counts (``valid_tokens * topk`` on device)."""
    g = ou.make_generator(device=DEVICE)
    rows, width = 2 * PROGRAMS_CAP, 256
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    n_rows = _common.device_scalar(rows)
    activations.bounded_silu_mul(x, n_rows)  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = activations.bounded_silu_mul(x, n_rows)
    for count in (1, PROGRAMS_CAP + 1, 0, rows, PROGRAMS_CAP - 1):
        ou.poison_(out)
        n_rows.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        check_rows(out, x, count, what=f"replay with n_rows={count}")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(2 * BLOCK_N_CAP))
def test_offsets_past_int32(rows):
    """Input offsets ``row * 2N`` reach 2**31: rows around the crossing and the last rows."""
    width = BLOCK_N_CAP
    ou.require_free_memory(rows * 3 * width * 2 + 2**30)
    x = torch.ones(rows, 2 * width, dtype=torch.bfloat16, device=DEVICE)
    g = ou.make_generator(device=DEVICE)
    probe = [0, 2**31 // (2 * width) - 1, 2**31 // (2 * width), rows - 2, rows - 1]
    probe = sorted({r for r in probe if 0 <= r < rows})
    x[probe] = _glu.glu_inputs((len(probe), 2 * width), g, DEVICE)
    out = activations.bounded_silu_mul(x, _common.device_scalar(rows))
    _glu.assert_swiglu_values(out[probe], x[probe], what=f"rows {probe}")


# ---------------------------------------------------------------------------------------------
# PR #8566 (d2): GLU options and the gated squared ReLU
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
def test_d2_swiglu_options(name):
    """SwiGLU with training's clamp, linear offset and interleaved FC1 layout."""
    options = D2_OPTIONS[name]
    g = ou.make_generator(device=DEVICE)
    rows, width, n_rows = PROGRAMS_CAP + 1, 1856, 400
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)
    out = run_bounded_silu_mul(x, _common.device_scalar(n_rows), **_glu.glu_kwargs(**options))
    check_rows(out, x, n_rows, what=name, **options)


@_glu.requires_glu_options
@pytest.mark.parametrize("name", ["none"] + list(D2_OPTIONS))
def test_d2_gated_squared_relu_matches_training(name):
    """``squared_relu_gate``: bitwise training's gated squared ReLU, NaN and infinities
    included."""
    options = D2_OPTIONS.get(name, {})
    g = ou.make_generator(device=DEVICE)
    rows, width, n_rows = PROGRAMS_CAP + 1, 1856, 450
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)
    x[7, 3] = float("nan")
    x[8, width + 5] = float("nan")
    x[9, 11] = float("inf")
    x[10, width + 12] = float("-inf")
    out = run_bounded_silu_mul(
        x, _common.device_scalar(n_rows), **_glu.glu_kwargs(squared_relu_gate=True, **options)
    )
    expected = _glu.gated_squared_relu_training(x[:n_rows], **options)
    _common.assert_exact_except_nan_payload(out[:n_rows], expected, what=f"gated sq-ReLU {name}")
    ou.assert_poisoned(out[n_rows:], what="rows past n_rows")

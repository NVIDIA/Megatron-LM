# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_mxfp8_quant_swizzle_rows_kernel`` / ``mxfp8_quantize_rows``
(megatron/core/inference/quantization/mxfp8_quantize.py).

Contract (from the batch-invariant MXFP8 call site, ``_quantize_mxfp8_used_rows(x, offs[-1:])``,
the launcher's documented API and the OCP MX spec; written before reading the kernel):

* **Computes** the MXFP8 encoding of ``x`` [M, K] (``_mx_spec``: E4M3 elements, round-up E8M0
  scales over 32-element blocks, cuBLAS 128 x 4 scale tiles) for the rows ``r < min(n_rows, M)``.
* **Inputs:** ``x`` [M, K] bf16, fp16 or fp32, K a multiple of 32; ``n_rows`` a 1-element device
  int32 or int64 tensor, read on device (CUDA-graph safe).
* **Outputs:** new data [M, K] E4M3 and a new 1-D scale buffer (E8M0 view). Data rows at or past
  ``n_rows`` are not written; their scale bytes, and every padding byte, are 0x00. A count past
  the buffer (``n_rows > M``) quantizes the M rows and writes nothing beyond them. ``x`` is not
  modified; a strided ``x`` is quantized by its values or rejected.
* **Tolerance class:** bit-exact against the spec (codes and scales apart); untouched rows still
  poisoned; zero scale bytes exact.

Open PR #8564 (d8) clamps the count to M and rejects a non-contiguous ``x``. On main,
``test_count_past_the_buffer`` writes past the data and scale buffers and
``test_strided_input_is_quantized_or_rejected`` quantizes the wrong elements: both fail on main
because of the bug #8564 fixes and pass with it.

Structure: persistent grid of ``NUM_PROGRAMS = max(1, min(M, 4096))`` programs striding over
rows; each row is the shared ``_mxfp8_quant_swizzle_row`` (``BLOCK_K = next_pow2(K)``). Covered:
M and ``n_rows`` around the 128-row scale tile and around the 4096-program grid, both count
dtypes, input dtypes and value kinds, CUDA-graph replay with changing counts.

Runs on H100 and on GB200 (``launch_on_gb200``), where production uses it.
"""

import pytest
import torch

from megatron.core.inference.quantization import mxfp8_quantize as mq
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common
from tests.unit_tests.inference.oracles.moe_activations import _mx_spec as mx
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

pytestmark = [pytest.mark.launch_on_gb200, _common.hardware(ou.HOPPER, ou.BLACKWELL)]

PROGRAMS_CAP = 4096  # mxfp8_quantize.py: num_programs = max(1, min(M, max_programs=4096))


def run_rows(x, n_rows, *, guard_bytes=None):
    """``mxfp8_quantize_rows`` with guarded outputs (data NaN-poisoned, scales zeroed as the
    launcher asks); checks the guards and that ``x`` is read-only."""
    before = x.clone()
    with _common.launcher_on_cpu(), _common.captured_allocations(mq, guard_bytes) as allocs:
        data, scale = mq.mxfp8_quantize_rows(x, n_rows)
    assert len(allocs.buffers) == 2
    allocs.check("mxfp8_quantize_rows outputs")
    ou.assert_exact(x, before, what="x is read-only")
    return data, scale


def check(data, scale, x, n, what):
    rows = x.shape[0]
    live = torch.arange(rows) < min(n, rows)
    mx.assert_matches_spec(data, scale, x, rows=live, what=what)
    ou.assert_poisoned(data[min(n, rows) :], what=f"{what}: data rows past n_rows")


CASES = ou.pairwise_cases(
    {
        "rows": [1, 127, 128, 129, 257],
        "live": ["0", "1", "127", "128", "129", "M-1", "M"],
        "width": [32, 160, 1856, 2688],
        "kind": list(mx.VALUE_KINDS),
        "dtype": [torch.bfloat16, torch.float16, torch.float32],
        "count_dtype": [torch.int32, torch.int64],
    }
)


def _count(kind, rows):
    return {"M-1": rows - 1, "M": rows}[kind] if kind in ("M-1", "M") else int(kind)


def _case_id(c):
    return (
        f"M{c['rows']}-n{c['live']}-K{c['width']}-{c['kind']}-{str(c['dtype'])[6:]}"
        f"-{str(c['count_dtype'])[6:]}"
    )


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_live_rows_match_spec(case):
    """Pairwise over M, the live count, K, value kind, input dtype and count dtype."""
    g = ou.make_generator()
    rows = case["rows"]
    n = min(_count(case["live"], rows), rows)
    x = mx.mx_inputs(rows, case["width"], case["dtype"], case["kind"], g).to(DEVICE)
    data, scale = run_rows(x, _common.device_scalar(n, case["count_dtype"]))
    check(data, scale, x, n, _case_id(case))


@pytest.mark.parametrize(
    "rows,n", [(4095, 4095), (4096, 4095), (4096, 4096), (4097, 4096), (4097, 4097), (8193, 8193)]
)
def test_grid_wraps(rows, n):
    """M and the live count around the 4096-program persistent grid."""
    g = ou.make_generator()
    x = mx.mx_inputs(rows, 32, torch.bfloat16, "wide", g).to(DEVICE)
    data, scale = run_rows(x, _common.device_scalar(n))
    check(data, scale, x, n, f"M{rows}-n{n}")


@pytest.mark.parametrize("rows,extra", [(128, 1), (129, 1), (129, 3)])
def test_count_past_the_buffer(rows, extra):
    """``n_rows > M`` quantizes the M rows and writes nothing past the outputs.

    ``x`` is the leading rows of a larger tensor of ordinary values, so a kernel that runs past M
    reads finite rows and writes codes and scales that differ from the guard bytes and from the
    zero tile padding. Fails on main (bug fixed by #8564, d8): the loop runs to ``n_rows``.
    """
    g = ou.make_generator()
    width = 128
    backing = mx.mx_inputs(rows + 8, width, torch.bfloat16, "randn", g).to(DEVICE)
    x = backing[:rows]
    guard = 8 * width + 4096
    data, scale = run_rows(x, _common.device_scalar(rows + extra), guard_bytes=guard)
    check(data, scale, x, rows + extra, f"n_rows = M + {extra}")


def test_strided_input_is_quantized_or_rejected():
    """A strided [M, K] view is quantized by its values, or rejected; never misread.

    Fails on main (bug fixed by #8564, d8, which asserts contiguity).
    """
    g = ou.make_generator()
    base = mx.mx_inputs(256, 128, torch.bfloat16, "wide", g).to(DEVICE)
    x = base.t()
    try:
        data, scale = run_rows(x, _common.device_scalar(100))
    except (AssertionError, ValueError, RuntimeError):
        return
    check(data, scale, x.contiguous(), 100, "strided input")


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass."""
    g = ou.make_generator()
    x = mx.mx_inputs(257, 160, torch.bfloat16, "boundary", g).to(DEVICE)
    data, scale = run_rows(x, _common.device_scalar(200))
    check(data, scale, x, 200, "sanitizer case")


@_common.needs_cuda
def test_cuda_graph_replay_follows_device_count():
    """Captured once, replayed with other counts: rows past each count keep their old bytes,
    which the test poisons and zeroes before each replay."""
    g = ou.make_generator()
    rows, width = 4100, 160
    x = mx.mx_inputs(rows, width, torch.bfloat16, "wide", g).to(DEVICE)
    n = _common.device_scalar(rows)
    mq.mxfp8_quantize_rows(x, n)  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        data, scale = mq.mxfp8_quantize_rows(x, n)
    for count in (128, 4097, 0, rows, 4095):
        ou.poison_(data)
        scale.view(torch.uint8).zero_()
        n.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        check(data, scale, x, count, f"replay with n_rows={count}")

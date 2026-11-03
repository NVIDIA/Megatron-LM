# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_swiglu_kernel`` / ``padded_swiglu`` (megatron/core/inference/moe/activations.py).

Contract (from the call site in ``mcore_fused_moe``, the model math of training's GLU and the
launcher's documented API; written before reading the kernel):

* **Computes** the SwiGLU activation of the expert-grouped FC1 output: for every row
  ``r < min(n_used, M)`` with ``permutation_map[r] >= 0``,
  ``out[r, c] = silu(x[r, c]) * x[r, N + c]`` for ``c < N`` (gate first, the
  ``torch.chunk(x, 2, -1)`` convention of training), evaluated in fp32 and rounded once to bf16.
* **Inputs:** ``x`` [M, 2N] bf16, contiguous (even width asserted); ``permutation_map`` [M]
  integer (int32 in production), -1 marks an alignment-padding row; ``n_used`` a 1-element
  device int32 tensor (``inclusive_expert_offsets[-1:]``), read on device so the launch is
  CUDA-graph safe; ``zero_padding``.
* **Outputs:** a new [M, N] bf16 tensor. ``x`` is not modified.
* **Padding and live rows:** rows ``>= n_used`` are not written. Padding rows below ``n_used``
  are written with +0.0 when ``zero_padding`` is set (MXFP8 quantization and the grouped GEMM
  read them) and are not written otherwise. A count past the buffer (``n_used > M``) processes
  the M rows and nothing beyond. ``n_used = 0`` writes nothing.
* **Special values:** IEEE evaluation of ``g / (1 + exp(-g)) * u``: NaN propagates, ``g = +inf``
  gives ``±inf`` (NaN for ``u = 0``), ``g = -inf`` gives NaN. Where the fp32 evaluation passes
  through a subnormal (subnormal inputs, gates below -87.34 whose sigmoid is subnormal), the
  result may be flushed to ±0.
* **Tolerance class:** floating point. (1) Element by element, the result is the single bf16
  rounding of an fp32 evaluation: the correctly rounded fp64 value, or either neighbour within
  2**-16 (relative) of a rounding midpoint, the fp32 evaluation error bound
  (``_common.assert_rounded_once``); (2) the harness policy per row
  (``ou.assert_close_to_reference(dim=-1)``) against torch's fp32 SiLU rounded once on the same
  device. Padding zeros and untouched rows are exact (bitwise / still poisoned).

Open PR #8566 (d2) adds ``glu_options`` (clamp, linear offset, interleaved FC1 layout) and
``squared_relu_gate`` to this launcher. The ``test_d2_*`` cases state their contract (training's
GLU semantics, ``_glu``) and run only where those arguments exist.

Structure (read from the launcher after writing the contract): ``BLOCK_N = min(next_pow2(N),
1024)`` column tiles; a persistent grid of ``NUM_BLOCKS = min(M, 512)`` programs strides over
rows; ``ZERO_PADDING`` is a compile-time variant (production passes False; the BI and MXFP8
callers of the sibling kernels pass True); int64 row offsets. Covered: both ``ZERO_PADDING``
variants; N around the 1024 tile cap and inside one tile (masked tails), production widths;
M and ``n_used`` around the 512-program grid; padding at program starts and wraps; int32 and
int64 maps; CUDA-graph replay with a changed ``n_used``; offsets past 2**31 (``large``).

Runs on H100 and on GB200 (``launch_on_gb200``): the torch-backend BF16 and MXFP8 SwiGLU experts
run this kernel on both.
"""

import pytest
import torch

from megatron.core.inference.moe import activations
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_activations import _common, _glu
from tests.unit_tests.inference.oracles.moe_activations._common import DEVICE

# H100, and GB200 (``launch_on_gb200``), where the MXFP8 SwiGLU expert path runs this kernel.
pytestmark = [pytest.mark.launch_on_gb200, _common.hardware(ou.HOPPER, ou.BLACKWELL)]

BLOCK_N_CAP = 1024  # activations.py: BLOCK_N = min(triton.next_power_of_2(N), 1024)
PROGRAMS_CAP = 512  # activations.py: NUM_BLOCKS = min(M, 512)
PRODUCTION_FFN = (1856, 2688)  # moe_ffn_hidden_size of the nanov3-style MoE layers in tests

WIDTHS = ou.boundary_sizes(32, BLOCK_N_CAP, multiples=(2,), minimum=1, production=PRODUCTION_FFN)
ROWS = (1, 2, PROGRAMS_CAP - 1, PROGRAMS_CAP, PROGRAMS_CAP + 1, 2 * PROGRAMS_CAP + 1)

CASES = ou.pairwise_cases(
    {
        "width": [w for w in WIDTHS if w not in (63, 64, 65, 2047, 2048)],
        "live": list(_glu.LIVE_COUNTS),
        "padding": list(_glu.PADDING_PATTERNS),
        "rows": list(ROWS),
        "zero_padding": [False, True],
        "perm_dtype": [torch.int32, torch.int64],
    }
)


def _case_id(case):
    dtype = str(case["perm_dtype"]).replace("torch.", "")
    zp = "zero" if case["zero_padding"] else "skip"
    return f"N{case['width']}-M{case['rows']}-n{case['live']}-{case['padding']}-{zp}-{dtype}"


def run_padded_swiglu(x, perm, n_used, zero_padding=False, **options):
    """``padded_swiglu`` with its output buffer guarded and NaN-poisoned (rows it must not write
    stay NaN); checks the guard bands and that ``x`` is read-only."""
    before = x.clone()
    with _common.captured_allocations(activations) as allocs:
        out = activations.padded_swiglu(x, perm, n_used, zero_padding=zero_padding, **options)
    assert len(allocs.buffers) == 1 and out.data_ptr() == allocs.buffers[0].tensor.data_ptr()
    allocs.check("padded_swiglu output")
    ou.assert_exact(x, before, what="x is read-only")
    return out


def check_rows(out, x, perm, n_used, zero_padding, what, **options):
    """Live rows against the contract; padding rows zero or untouched; rows past n_used
    untouched."""
    rows = x.shape[0]
    index = torch.arange(rows, device=out.device)
    below = index < min(n_used, rows)
    live = below & (perm.to(out.device) >= 0)
    padding = below & ~live
    _glu.assert_swiglu_values(out[live], x[live], what=f"{what}: live rows", **options)
    if zero_padding:
        ou.assert_exact(
            out[padding], torch.zeros_like(out[padding]), what=f"{what}: padding rows are +0"
        )
    else:
        ou.assert_poisoned(out[padding], what=f"{what}: padding rows")
    ou.assert_poisoned(out[~below], what=f"{what}: rows past n_used")


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_values_and_live_rows(case):
    """Pairwise over width, grid size, live count, padding layout, ZERO_PADDING and map dtype."""
    g = ou.make_generator(device=DEVICE)
    rows, width = case["rows"], case["width"]
    n_used = _glu.live_count(case["live"], rows, min(rows, PROGRAMS_CAP))
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    perm = _glu.permutation_map(rows, case["padding"], case["perm_dtype"], g, DEVICE)
    out = run_padded_swiglu(x, perm, _common.device_scalar(n_used), case["zero_padding"])
    check_rows(out, x, perm, n_used, case["zero_padding"], what=_case_id(case))


@pytest.mark.parametrize("width", [768, 1856, 2688])
def test_production_widths_all_live(width):
    """A full decode-size buffer of a production width, every row live."""
    g = ou.make_generator(device=DEVICE)
    rows = 1024
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    perm = _glu.permutation_map(rows, "none", torch.int32, g, DEVICE)
    out = run_padded_swiglu(x, perm, _common.device_scalar(rows))
    check_rows(out, x, perm, rows, False, what=f"N{width}")


@pytest.mark.parametrize("zero_padding", [False, True])
def test_special_values(zero_padding):
    """Zeros, subnormals, saturating sigmoids, overflowing exp and products, infinities, NaN."""
    for i, x in enumerate(_glu.special_value_inputs(DEVICE)):
        rows = x.shape[0]
        perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
        out = run_padded_swiglu(x, perm, _common.device_scalar(rows), zero_padding)
        _glu.assert_swiglu_special_values(out, x, what=f"special values, grid {i}")


def test_n_used_zero_writes_nothing():
    """A batch with no live rows leaves the whole output untouched."""
    g = ou.make_generator(device=DEVICE)
    x = _glu.glu_inputs((PROGRAMS_CAP + 1, 2 * 96), g, DEVICE)
    perm = _glu.permutation_map(x.shape[0], "runs", torch.int32, g, DEVICE)
    for zero_padding in (False, True):
        out = run_padded_swiglu(x, perm, _common.device_scalar(0), zero_padding)
        ou.assert_poisoned(out, what=f"output with n_used = 0 (zero_padding={zero_padding})")


@pytest.mark.sanitizer
def test_sanitizer_small():
    """Small representative case for the compute-sanitizer pass: two column tiles, padding runs,
    a live count inside the buffer, zeroed padding."""
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = 130, 1025, 100
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    out = run_padded_swiglu(x, perm, _common.device_scalar(n_used), zero_padding=True)
    check_rows(out, x, perm, n_used, True, what="sanitizer case")


@_common.needs_cuda
@pytest.mark.parametrize("zero_padding", [False, True])
def test_cuda_graph_replay_follows_device_count(zero_padding):
    """Captured once, replayed with other live counts: the kernel reads ``n_used`` on device.

    Before each replay the static output is poisoned; rows at or past the new count must stay
    poisoned and the live rows must meet the contract for the new count.
    """
    g = ou.make_generator(device=DEVICE)
    rows, width = 2 * PROGRAMS_CAP, 256
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    perm = _glu.permutation_map(rows, "program-stride", torch.int32, g, DEVICE)
    n_used = _common.device_scalar(rows)
    activations.padded_swiglu(x, perm, n_used, zero_padding=zero_padding)  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = activations.padded_swiglu(x, perm, n_used, zero_padding=zero_padding)
    for count in (1, PROGRAMS_CAP + 1, 0, rows, PROGRAMS_CAP - 1):
        ou.poison_(out)
        n_used.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        check_rows(out, x, perm, count, zero_padding, what=f"replay with n_used={count}")


@pytest.mark.parametrize(
    "rows", ou.int32_overflow_params(2 * BLOCK_N_CAP) + ou.int32_overflow_params(BLOCK_N_CAP)[-1:]
)
def test_offsets_past_int32(rows):
    """Input offsets ``row * 2N`` (and, for the last size, output offsets ``row * N``) reach
    2**31. Checks the rows around the crossing and the last rows."""
    width = BLOCK_N_CAP
    ou.require_free_memory(rows * 3 * width * 2 + 2**30)
    x = torch.ones(rows, 2 * width, dtype=torch.bfloat16, device=DEVICE)
    g = ou.make_generator(device=DEVICE)
    probe = sorted(
        {r for r in (0, 2**31 // (2 * width) - 1, 2**31 // (2 * width), 2**31 // width - 1)}
        | set(range(rows - 3, rows))
    )
    probe = [r for r in probe if 0 <= r < rows]
    x[probe] = _glu.glu_inputs((len(probe), 2 * width), g, DEVICE)
    perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    out = activations.padded_swiglu(x, perm, _common.device_scalar(rows))
    _glu.assert_swiglu_values(out[probe], x[probe], what=f"rows {probe}")


# ---------------------------------------------------------------------------------------------
# PR #8566 (d2): GLU options and the gated squared ReLU
# ---------------------------------------------------------------------------------------------

D2_OPTIONS = {
    "clamp": dict(clamp=7.0),
    "clamp-not-bf16": dict(clamp=7.3),
    "offset": dict(offset=1.0),
    "clamp-offset": dict(clamp=7.0, offset=1.0),
    "interleave-1": dict(interleave=1),
    "interleave-32": dict(interleave=32),
    "all": dict(clamp=7.0, offset=1.0, interleave=64),
}


@_glu.requires_glu_options
@pytest.mark.parametrize("name", list(D2_OPTIONS))
@pytest.mark.parametrize("zero_padding", [False, True])
def test_d2_swiglu_options(name, zero_padding):
    """SwiGLU with training's clamp, linear offset and interleaved layout (``_glu``): fp32
    compute and one rounding, as for the default options."""
    options = D2_OPTIONS[name]
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = PROGRAMS_CAP + 1, 1856, 400
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)  # some |x| > 7: clamp bites
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    out = run_padded_swiglu(
        x, perm, _common.device_scalar(n_used), zero_padding, **_glu.glu_kwargs(**options)
    )
    check_rows(out, x, perm, n_used, zero_padding, what=name, **options)


@_glu.requires_glu_options
@pytest.mark.parametrize("name", ["clamp", "clamp-not-bf16", "offset", "clamp-offset"])
def test_d2_swiglu_options_special_values(name):
    """The clamp keeps NaN NaN (as torch.clamp does) and bounds infinite inputs."""
    options = D2_OPTIONS[name]
    for i, x in enumerate(_glu.special_value_inputs(DEVICE)):
        rows = x.shape[0]
        perm = torch.arange(rows, dtype=torch.int32, device=DEVICE)
        out = run_padded_swiglu(x, perm, _common.device_scalar(rows), **_glu.glu_kwargs(**options))
        _glu.assert_swiglu_special_values(out, x, what=f"{name}, grid {i}", **options)


@_glu.requires_glu_options
@pytest.mark.parametrize("interleave", [1, 32, 64])
def test_d2_interleaved_layout_is_a_relayout(interleave):
    """The interleaved FC1 layout gives bitwise the same output as the same values laid out
    as ``[gate | linear]`` halves."""
    g = ou.make_generator(device=DEVICE)
    rows, width = 300, 1856
    halves = _glu.glu_inputs((rows, 2 * width), g, DEVICE)
    perm = _glu.permutation_map(rows, "random", torch.int32, g, DEVICE)
    n_used = _common.device_scalar(rows)
    expected = run_padded_swiglu(halves, perm, n_used, zero_padding=True)
    interleaved = _glu.interleave_halves(halves, interleave)
    actual = run_padded_swiglu(
        interleaved, perm, n_used, zero_padding=True, **_glu.glu_kwargs(interleave=interleave)
    )
    ou.assert_exact(actual, expected, what=f"interleave {interleave} vs halves")


@_glu.requires_glu_options
@pytest.mark.parametrize("name", ["none"] + list(D2_OPTIONS))
def test_d2_gated_squared_relu_matches_training(name):
    """``squared_relu_gate``: bitwise training's gated squared ReLU (``relu(gate)^2 * linear``
    with training's per-op bf16 roundings, ``_glu.gated_squared_relu_training``), NaN and
    infinities included."""
    options = D2_OPTIONS.get(name, {})
    g = ou.make_generator(device=DEVICE)
    rows, width, n_used = PROGRAMS_CAP + 1, 1856, 500
    x = _glu.glu_inputs((rows, 2 * width), g, DEVICE, scale=4.0)
    x[7, 3] = float("nan")
    x[8, width + 5] = float("nan")
    x[9, 11] = float("inf")
    x[10, width + 12] = float("-inf")
    perm = _glu.permutation_map(rows, "runs", torch.int32, g, DEVICE)
    out = run_padded_swiglu(
        x,
        perm,
        _common.device_scalar(n_used),
        zero_padding=True,
        **_glu.glu_kwargs(squared_relu_gate=True, **options),
    )
    index = torch.arange(rows, device=out.device)
    live = (index < n_used) & (perm >= 0)
    expected = _glu.gated_squared_relu_training(x[live], **options)
    _common.assert_exact_except_nan_payload(out[live], expected, what=f"gated sq-ReLU {name}")
    padding = (index < n_used) & ~live
    ou.assert_exact(out[padding], torch.zeros_like(out[padding]), what="padding rows are +0")
    ou.assert_poisoned(out[index >= n_used], what="rows past n_used")

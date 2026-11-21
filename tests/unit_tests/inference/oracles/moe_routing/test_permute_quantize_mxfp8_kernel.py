# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_permute_quantize_mxfp8_kernel`` (megatron/core/inference/moe/permute.py),
through its launcher ``permute_and_quantize_mxfp8``, and for the ``mcore_fused_moe`` dispatch
that selects it.

``permute_and_quantize_mxfp8`` runs the count, prefix-sum and init-map kernels (own oracles in
this package), this kernel and, with ``zero_padding``, ``_zero_permutation_padding_kernel`` on
the FP8 rows as bytes. ``mcore_fused_moe`` calls it on MXFP8 experts for the non-batch-invariant
squared-ReLU path (P6 in the audit, the default MXFP8 path) and for every batch-invariant MXFP8
path (P5), unless ``disable_fused_quant_kernels``.

Contract (from the launcher's documented API, the permute contract of ``permute_tokens`` and the
OCP MX specification, ``mxfp8_reference``):

* Routing, as ``permute_tokens``: live pairs ``(t, k)`` with ``t < min(valid, rows)`` and a local
  expert ``e``; offsets are the running sum of the counts rounded up to ``alignment``; expert
  ``e``'s block holds its pairs in unspecified order with ``permutation_map = t`` and
  ``permuted_probs = probs[t, k]``; alignment padding rows have ``permutation_map = -1``; with
  ``return_batch_invariant_inverse_map``, ``inverse[t, e]`` is the row of pair ``(t, e)``.
* Payload: an ``MXFP8Tensor`` (``data`` ``float8_e4m3fn`` ``[output_size, K]``, ``scale``
  ``float8_e8m0fnu`` in the cuBLAS 128 x 4 tile layout of ``[output_size, K/32]``, backend
  ``"triton"``). A claimed row holds the MXFP8 encoding of ``hidden[t]``: per 32-element block
  the round-up E8M0 scale ``X`` (smallest power of two in [2^-127, 2^127] with ``amax / X <=
  448``; 2^-127 for an all-zero block) and E4M3 codes ``RNE(x / X)``.
* Every scale byte that belongs to no claimed row (alignment padding, rows ``>= n_used``, tile
  padding columns) is 0x00. With ``zero_padding`` the data bytes of padding rows are 0x00
  (MXFP8 zero); without it they are unspecified. Data rows ``>= n_used`` are not written.
* ``output_size`` is fixed by the shapes (CUDA graphs): a multiple of 128 rows, at least
  ``rows * min(topk, E_l) + E_l * (alignment - 1)``, the most rows any ``valid`` can use.
* Inputs: ``hidden_states`` ``[rows, K]`` bf16, K a multiple of 32 (asserted), contiguous
  (a strided view must be quantized by value or rejected); read-only. Special values:
  NaN and Inf affect only their own block; a NaN decodes to NaN and an Inf to NaN or
  ``+-448 * X`` (the OCP spec leaves both to the implementation).
* Tolerance class: **exact** for the routing outputs; **bit-exact against the spec** for codes
  and scales (reported apart, ``ou.assert_quantized_bit_exact``).

Contract-first caveat: the audit record of this kernel (family 01), which describes the body,
was read before this contract was written; the reference is the OCP specification and shares
no code or index math with the kernel.

Oracle: the routing half via ``routing_reference.check_permutation_bookkeeping``; the payload
against ``mxfp8_reference.quantize_rows(hidden)`` gathered by ``permutation_map``, with the
scale buffer unswizzled by the layout definition; outputs allocated poisoned (PyTorch's
``fill_uninitialized_memory``: FP8 NaN 0x7F, NaN probabilities, ``INT32_MAX`` map entries).

Structure (read after the contract): constexpr ``topk``, ``E_l``, ``REAL_GROUPS = K / 32``,
``BLOCK_K = next_pow2(K)`` (the whole row in registers), ``BLOCK_GROUPS = BLOCK_K / 32``,
``NUM_BLOCKS = min(rows * topk, 512)`` (programs stride over pairs) and ``HAS_INVERSE``; scale
offsets by 128-row and 4-column tiles; the zero-padding kernel by ``min(output_size, 512)``
programs and ``BLOCK_H = min(next_pow2(K), 1024)``. Inputs therefore cover K in {32, 64, 96,
160, 992, 1024, 1056, 1856, 2688} (powers of two and 1-3 padded scale columns, one and two zero
tiles), ``rows * topk`` at 511/512/513, claimed rows across the 128-row scale tiles, ``topk``
{1, 2, 6, 8}, ``E_l`` {1, 2, 7, 32}, first / second / last slice, ``alignment`` {32, 128, 256},
the P6 and P5 flag pairs and each flag alone, bf16 / fp32 probabilities, ``valid`` {rows,
rows-1, 0, 1} and masked CUDA-graph padding rows, and random, wide-range, round-up-tie and
E4M3-midpoint, and zero / signed-zero values, pairwise; production shapes; CUDA-graph replay
with ``valid`` changing; opt-in (``large``), row offsets ``row * K`` at 2^31 (K 7168).

Hardware: Triton's FP8 conversion needs sm_89 or newer; production runs this kernel on GB200
(the MXFP8 grouped GEMM), so the module runs on H100 and on GB200 (``launch_on_gb200``). It
needs a GPU: the Triton 3.8 interpreter rounds fp32 -> E4M3 differently from compiled Triton
(``cvt.rn.satfinite``), so CPU runs are not evidence for this kernel.

Cases that fail on main are regression tests of their fix PRs:

* ``valid > rows`` (finding F1, the count kernel reads past the map): #8563's
  ``TestComputeLocalTokensPerExpert::test_valid_tokens_past_routing_map_rows``
  (``tests/unit_tests/inference/test_moe_permute.py``).
* Blocks with amax <= 448 * 2^-127 (finding O2-N1, scale byte 0x01 instead of 0x00) and strided
  ``hidden_states`` (read as if contiguous; rejected with the fix): #8956's
  ``TestPermuteAndQuantizeMxfp8::test_tiny_groups_take_the_smallest_scale`` and
  ``test_rejects_strided_input`` (``tests/unit_tests/inference/test_mxfp8_utils.py``).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import mxfp8_reference as mx
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

permute = pytest.importorskip("megatron.core.inference.moe.permute")
pytest.importorskip("triton")

pytestmark = [pytest.mark.launch_on_gb200, ou.requires_sm(ou.HOPPER, ou.BLACKWELL)]

DATA_POISON = 0x7F
"""Byte PyTorch's ``fill_uninitialized_memory`` writes into a new ``float8_e4m3fn`` tensor."""

PATHS = {
    # name: (zero_padding, inverse)
    "p6": (False, False),  # non-batch-invariant squared ReLU (default MXFP8 path)
    "p5": (True, True),  # batch-invariant MXFP8
    "inverse-only": (False, True),
    "zero-padding-only": (True, False),
}


def _inputs(rows, k, topk, num_experts, *, masked, kind, pdtype, dev, seed=0):
    g = ou.make_generator(ou.DEFAULT_SEED + seed)
    routing_map = rr.topk_distinct_ids(
        rows, topk, num_experts, masked_rows=masked, generator=g, dev=dev
    )
    hidden = mx.value_kind_inputs(rows, k, kind, g).to(dev)
    probs = torch.rand(rows, topk, generator=g).to(dtype=pdtype, device=dev)
    return hidden, probs, routing_map


def check_mxfp8_permutation(
    out,
    hidden,
    probs,
    routing_map,
    valid,
    start,
    num_local,
    *,
    alignment,
    zero_padding,
    inverse,
    what,
    stale_map=None,
):
    """Assert every clause of the contract on ``permute_and_quantize_mxfp8``'s outputs."""
    assert len(out) == (5 if inverse else 4), what
    mxt, perm_p, perm_map, offsets = out[:4]
    data, scale = mxt.data, mxt.scale
    rows, k = hidden.shape
    topk = routing_map.shape[1]
    assert data.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu, what
    assert mxt.backend == "triton", what
    size = data.shape[0]
    assert data.shape == (size, k) and size % mx.TILE_ROWS == 0, f"data {data.shape} ({what})"
    worst = rows * min(topk, num_local) + num_local * (alignment - 1)
    assert size >= worst, f"output_size {size} < {worst} rows some valid can use ({what})"
    assert scale.numel() == mx.scale_buffer_bytes(size, k), what

    _, _, n_used = rr.check_permutation_bookkeeping(
        perm_p,
        perm_map,
        offsets,
        out[4] if inverse else None,
        probs,
        routing_map,
        valid,
        start,
        num_local,
        alignment=alignment,
        what=what,
        stale_map=stale_map,
    )

    dev = hidden.device
    codes = data.view(torch.uint8)
    scales, tile_padding = mx.unswizzle(scale, size, k // mx.BLOCK)
    row = torch.arange(size, device=dev)
    used = row < n_used
    claimed = used & (perm_map.to(torch.int64) >= 0)
    tokens = perm_map[claimed].to(torch.int64)
    live = max(0, min(valid, rows))
    ref_codes, ref_scales = mx.quantize_rows(hidden[:live])
    ou.assert_quantized_bit_exact(
        codes[claimed],
        scales[claimed],
        ref_codes[tokens],
        ref_scales[tokens],
        elems_per_scale=mx.BLOCK,
        what=f"claimed rows vs MXFP8 spec of hidden[permutation_map] ({what})",
    )
    ou.assert_exact(
        scales[~claimed],
        torch.zeros_like(scales[~claimed]),
        what=f"scale bytes of padding rows and rows >= n_used ({what})",
    )
    ou.assert_exact(
        tile_padding, torch.zeros_like(tile_padding), what=f"scale tile padding bytes ({what})"
    )
    if zero_padding:
        pad_rows = used & ~claimed
        ou.assert_exact(
            codes[pad_rows], torch.zeros_like(codes[pad_rows]), what=f"zeroed padding ({what})"
        )
    ou.assert_exact(
        codes[n_used:],
        torch.full_like(codes[n_used:], DATA_POISON),
        what=f"data rows >= n_used (must be untouched) ({what})",
    )


def run_and_check(
    rows,
    k,
    topk,
    num_local,
    start,
    num_experts,
    *,
    valid,
    masked=0,
    alignment=128,
    path="p6",
    kind="randn",
    pdtype=torch.float32,
    seed=0,
):
    dev = rr.device()
    zero_padding, inverse = PATHS[path]
    hidden, probs, routing_map = _inputs(
        rows, k, topk, num_experts, masked=masked, kind=kind, pdtype=pdtype, dev=dev, seed=seed
    )
    routing_map, _ = rr.with_canary_tail(routing_map, rr.COUNT_BLOCK, start)
    before = (hidden.clone(), probs.clone(), routing_map.clone())
    with rr.poisoned_outputs():
        out = permute.permute_and_quantize_mxfp8(
            hidden,
            probs,
            routing_map,
            start,
            num_local,
            rr.valid_tensor(valid, dev),
            alignment=alignment,
            return_batch_invariant_inverse_map=inverse,
            zero_padding=zero_padding,
        )
    what = (
        f"rows={rows} K={k} topk={topk} E_l={num_local} start={start} valid={valid} a={alignment}"
        f" {path} {kind} probs={pdtype}"
    )
    check_mxfp8_permutation(
        out,
        hidden,
        probs,
        routing_map,
        valid,
        start,
        num_local,
        alignment=alignment,
        zero_padding=zero_padding,
        inverse=inverse,
        what=what,
    )
    for name, now, then in zip(
        ("hidden_states", "probs", "routing_map"), (hidden, probs, routing_map), before
    ):
        ou.assert_exact(now, then, what=f"{name} is read-only")


_VARIANTS = ou.pairwise_cases(
    {
        "k": [32, 64, 96, 160, 992, 1024, 1056, 1856, 2688],
        "topk": [1, 2, 6, 8],
        "num_local": [1, 2, 7, 32],
        "slice": ["first", "second", "last"],
        "alignment": [32, 128, 256],
        "path": list(PATHS),
        "kind": list(mx.VALUE_KINDS),
        "probs": [torch.float32, torch.bfloat16],
        "valid": ["rows", "rows-1", "0", "1", "masked"],
    }
)


def _variant_id(c):
    return (
        f"K{c['k']}-topk{c['topk']}-E{c['num_local']}-{c['slice']}-a{c['alignment']}"
        f"-{c['path']}-{c['kind']}-{str(c['probs'])[6:]}-v{c['valid']}"
    )


@pytest.mark.parametrize("case", _VARIANTS, ids=[_variant_id(c) for c in _VARIANTS])
def test_variants_pairwise(case):
    """Every compiled variant, dispatch flag and value class, pairwise (module docstring)."""
    rows = 97
    num_local, topk = case["num_local"], case["topk"]
    num_experts = max(4 * num_local, 2 * topk)
    start = {"first": 0, "second": num_local, "last": num_experts - num_local}[case["slice"]]
    masked = rows // 3 if case["valid"] == "masked" else 0
    valid = {"rows": rows, "rows-1": rows - 1, "0": 0, "1": 1, "masked": rows}[case["valid"]]
    run_and_check(
        rows,
        case["k"],
        topk,
        num_local,
        start,
        num_experts,
        valid=valid,
        masked=masked,
        alignment=case["alignment"],
        path=case["path"],
        kind=case["kind"],
        pdtype=case["probs"],
    )


def _grid_cases():
    cases = []
    for topk in (1, 8):
        for rows in [1] + rr.rows_around_pairs(rr.MAX_GRID, topk, multiples=(1,)):
            cases.append(pytest.param(rows, topk, id=f"rows{rows}-topk{topk}[pairs{rows * topk}]"))
    return cases


@pytest.mark.parametrize("rows,topk", _grid_cases())
@pytest.mark.parametrize("valid", ["rows", "rows-1"])
def test_grid_cap(rows, topk, valid):
    """``rows * topk`` around the 512-program grid cap (each program strides over pairs)."""
    v = rows if valid == "rows" else max(rows - 1, 0)
    run_and_check(rows, 64, topk, 8, 8, 32, valid=v, alignment=128, path="p5")


@pytest.mark.parametrize("k", [96, 160, 1856])
@pytest.mark.parametrize("rows", [127, 128, 129, 255, 256, 257, 300])
@pytest.mark.parametrize("num_local", [1, 2])
def test_claimed_rows_cross_scale_tiles(k, rows, num_local):
    """Claimed rows on both sides of the 128-row scale tiles, with 1-3 padded scale columns.

    Top-1 onto ``num_local`` experts: every token is claimed, so with one expert the rows
    written are exactly ``[0, rows)``, and with two the second block starts on a tile boundary
    (alignment 128).
    """
    run_and_check(rows, k, 1, num_local, 0, num_local, valid=rows, alignment=128, path="p5")


@pytest.mark.parametrize("kind", ["all-masked", "none-local", "one-expert-local"])
def test_degenerate_routing(kind):
    """No live pairs (all ``-1`` / nothing local) and a single local expert holding every pair."""
    if kind == "all-masked":
        run_and_check(40, 64, 4, 8, 8, 32, valid=40, masked=40, path="p5")
    elif kind == "none-local":
        run_and_check(40, 64, 4, 8, 32, 32, valid=40, path="p6")
    else:
        run_and_check(600, 64, 1, 1, 0, 1, valid=600, path="p5")


@pytest.mark.parametrize(
    "rows,k,topk,num_experts,ep",
    [
        pytest.param(8192, 2688, 6, 128, 1, id="nanov3-ep1"),
        pytest.param(8192, 7168, 8, 256, 8, id="dsv3-ep8"),
        pytest.param(64 * 256, 7168, 8, 256, 64, id="dsv3-ep64-decode-El4"),
    ],
)
@pytest.mark.parametrize("path", ["p6", "p5"])
def test_production_sizes(rows, k, topk, num_experts, ep, path):
    """Production shapes (``BLOCK_K = 8192`` for K 7168) on both production flag pairs."""
    rr.require_cuda()
    num_local = num_experts // ep
    start = (ep // 2) * num_local
    run_and_check(
        rows,
        k,
        topk,
        num_local,
        start,
        num_experts,
        valid=rows - rows // 9,
        masked=rows // 9,
        path=path,
    )


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    run_and_check(50, 160, 4, 8, 8, 32, valid=45, path="p5", kind="boundary")


@pytest.mark.parametrize("path", ["p6", "p5"])
def test_cuda_graph_replay_follows_valid(path):
    """Captured once; replayed with ``valid`` changing. Rows past the new ``n_used`` keep the
    previous step's map entries and must not be written; data, probabilities and scale bytes
    are re-poisoned before every replay (the captured launcher re-zeroes the scales)."""
    dev = rr.require_cuda()
    rows, k, topk, num_local, start, num_experts = 300, 160, 8, 8, 8, 32
    zero_padding, inverse = PATHS[path]
    hidden, probs, routing_map = _inputs(
        rows, k, topk, num_experts, masked=0, kind="boundary", pdtype=torch.float32, dev=dev
    )
    routing_map, _ = rr.with_canary_tail(routing_map, rr.COUNT_BLOCK, start)
    valid = rr.valid_tensor(rows, dev)
    kwargs = dict(return_batch_invariant_inverse_map=inverse, zero_padding=zero_padding)
    args = (hidden, probs, routing_map, start, num_local, valid)
    permute.permute_and_quantize_mxfp8(*args, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = permute.permute_and_quantize_mxfp8(*args, **kwargs)
    for v in [rows, 1, 0, 64, 65, rows - 1, 7, rows]:
        stale_map = out[2].clone()
        out[0].data.view(torch.uint8).fill_(DATA_POISON)
        out[0].scale.view(torch.uint8).fill_(0xFF)
        out[1].fill_(float("nan"))
        valid.fill_(v)
        graph.replay()
        torch.cuda.synchronize()
        check_mxfp8_permutation(
            out,
            hidden,
            probs,
            routing_map,
            v,
            start,
            num_local,
            alignment=128,
            zero_padding=zero_padding,
            inverse=inverse,
            what=f"replay valid={v} {path}",
            stale_map=stale_map,
        )


def test_special_values():
    """NaN and Inf affect only their own block: other blocks stay bit-exact, a NaN decodes to
    NaN and an Inf to NaN or ``+-448 * X``."""
    dev = rr.device()
    rows, k = 4, 160
    g = ou.make_generator()
    hidden = mx.value_kind_inputs(rows, k, "randn", g)
    special = [(0, 3, float("nan")), (1, 40, float("inf")), (2, 70, float("-inf"))]
    special += [(3, 100, float("nan")), (3, 101, float("inf"))]
    for r, c, v in special:
        hidden[r, c] = v
    hidden = hidden.to(dev)
    probs = torch.ones(rows, 1, device=dev)
    routing_map = torch.zeros(rows, 1, dtype=torch.int64, device=dev)
    mxt, _, perm_map, _ = permute.permute_and_quantize_mxfp8(
        hidden, probs, routing_map, 0, 1, rr.valid_tensor(rows, dev)
    )
    order = perm_map[:rows].to(torch.int64).cpu()
    ou.assert_is_permutation(order, rows, what="claimed tokens")
    codes = torch.empty(rows, k, dtype=torch.uint8)
    codes[order] = mxt.data.view(torch.uint8)[:rows].cpu()
    all_scales, _ = mx.unswizzle(mxt.scale.cpu(), mxt.data.shape[0], k // mx.BLOCK)
    scales = torch.empty(rows, k // mx.BLOCK, dtype=torch.uint8)
    scales[order] = all_scales[:rows]
    finite = hidden.cpu()
    finite = torch.where(torch.isfinite(finite), finite, torch.zeros_like(finite))
    ref_codes, ref_scales = mx.quantize_rows(finite)
    bad_blocks = {(r, c // mx.BLOCK) for r, c, _ in special}
    for r in range(rows):
        for b in range(k // mx.BLOCK):
            if (r, b) in bad_blocks:
                continue
            cols = slice(b * mx.BLOCK, (b + 1) * mx.BLOCK)
            ou.assert_exact(codes[r, cols], ref_codes[r, cols], what=f"codes of block {r},{b}")
            ou.assert_exact(scales[r, b], ref_scales[r, b], what=f"scale of block {r},{b}")
    decoded = mx.dequantize_rows(codes, scales)
    for r, c, v in special:
        got = decoded[r, c].item()
        if v != v:
            assert got != got, f"NaN at ({r}, {c}) decodes to {got}"
        else:
            x_scale = mx.e8m0_decode(scales[r, c // mx.BLOCK]).item()
            saturated = got == (448.0 if v > 0 else -448.0) * x_scale
            assert got != got or saturated, f"{v} at ({r}, {c}) decodes to {got}"


def test_rejects_hidden_size_not_multiple_of_32():
    """K must be a multiple of the 32-element block (asserted before any launch)."""
    dev = rr.device()
    hidden = torch.zeros(4, 48, dtype=torch.bfloat16, device=dev)
    with pytest.raises(AssertionError):
        permute.permute_and_quantize_mxfp8(
            hidden,
            torch.ones(4, 1, device=dev),
            torch.zeros(4, 1, dtype=torch.int64, device=dev),
            0,
            1,
            rr.valid_tensor(4, dev),
        )


@pytest.mark.parametrize("live", ou.int32_overflow_params(7168))
def test_int32_row_offsets(live):
    """Hidden rows read and FP8 rows written whose element offsets ``row * K`` reach 2^31
    (``K = 7168``), and scale tiles past 2^24 rows of bytes. Opt-in ``large``; about 8 GiB.

    One local expert, top-1: every one of the ``live`` tokens is claimed, so the rows written
    are exactly ``[0, live)`` and the highest sits at the size's label (fits / crosses /
    row-starts-past). 1024 masked CUDA-graph padding rows follow, inside ``valid``. The routing
    is checked whole; the payload on the first and last 1024 claimed rows and a strided sample
    (the reference runs on the device).
    """
    dev = rr.require_cuda()
    k, padding = 7168, 1024
    rows = live + padding
    ou.require_free_memory(rows * k * 4 + (4 << 30))
    g = ou.make_generator(device=dev)
    hidden = torch.randn(rows, k, device=dev, generator=g, dtype=torch.bfloat16)
    probs = torch.rand(rows, 1, device=dev, generator=g)
    routing_map = torch.zeros(rows, 1, dtype=torch.int64, device=dev)
    routing_map[live:] = -1
    mxt, perm_p, perm_map, offsets = permute.permute_and_quantize_mxfp8(
        hidden, probs, routing_map, 0, 1, rr.valid_tensor(rows, dev)
    )
    n_used = -(-live // 128) * 128
    ou.assert_exact(offsets, torch.tensor([n_used]), what="inclusive offsets")
    tokens = perm_map[:live].to(torch.int64)
    ou.assert_is_permutation(tokens, live, what="claimed tokens")
    ou.assert_exact(perm_p[:live], probs[tokens, 0], what="claimed probabilities")
    ou.assert_exact(
        perm_map[live:n_used], torch.full((n_used - live,), -1, dtype=torch.int32), what="padding"
    )
    sample = torch.cat(
        [
            torch.arange(0, 1024),
            torch.arange(1024, live - 1024, 4099),
            torch.arange(live - 1024, live),
        ]
    ).to(dev)
    ref_codes, ref_scales = mx.quantize_rows(hidden[tokens[sample]])
    size = mxt.data.shape[0]
    scales, _ = mx.unswizzle(mxt.scale, size, k // mx.BLOCK)
    ou.assert_quantized_bit_exact(
        mxt.data.view(torch.uint8)[sample],
        scales[sample],
        ref_codes,
        ref_scales,
        elems_per_scale=mx.BLOCK,
        what=f"rows {live - 1} and below vs the spec ({live} live rows)",
    )


_DISPATCH = [
    # (activation, batch invariant, disable_fused_quant_kernels) -> fused permute, BI flags
    pytest.param("squared_relu", False, False, True, id="p6-squared-relu"),
    pytest.param("swiglu", False, False, False, id="p3-swiglu"),
    pytest.param("squared_relu", False, True, False, id="p3-fused-disabled"),
    pytest.param("squared_relu", True, False, True, id="p5-bi-squared-relu"),
    pytest.param("swiglu", True, False, True, id="p5-bi-swiglu"),
    pytest.param("swiglu", True, True, False, id="p4-bi-fused-disabled"),
]


class _Stop(Exception):
    """Raised by the permute spies: the dispatch decision is all this test needs."""


@pytest.mark.parametrize("activation,bi,disable,fused", _DISPATCH)
def test_mcore_fused_moe_selects_the_fused_kernel(monkeypatch, activation, bi, disable, fused):
    """``mcore_fused_moe`` on MXFP8 experts (its docstring): batch-invariant MXFP8 and
    non-batch-invariant squared ReLU fuse permutation with quantization unless
    ``disable_fused_quant_kernels``; MXFP8 aligns every expert to 128 rows; batch-invariant
    mode asks for zero padding and the inverse map. The permute launchers are replaced by spies
    that record the call and stop, so no kernel or GEMM runs.
    """
    from megatron.core.inference.moe import fused_moe
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    dev = rr.device()
    calls = []

    def spy(name):
        def record(*args, **kwargs):
            calls.append((name, kwargs))
            raise _Stop

        return record

    monkeypatch.setattr(fused_moe, "permute_and_quantize_mxfp8", spy("fused"))
    monkeypatch.setattr(fused_moe, "permute_tokens", spy("unfused"))
    monkeypatch.setattr(fused_moe.batch_invariant, "enabled", lambda: bi)
    monkeypatch.setattr(fused_moe, "HAVE_SCALED_GMM", True)
    num_local, k, n = 2, 64, 128
    weight = MXFP8Tensor(
        data=torch.zeros(num_local, n, k, dtype=torch.float8_e4m3fn, device=dev),
        scale=torch.zeros(num_local * n * k // 32, dtype=torch.float8_e8m0fnu, device=dev),
        backend="triton",
    )
    rows, topk = 8, 2
    with pytest.raises(_Stop):
        fused_moe.mcore_fused_moe(
            torch.zeros(rows, k, dtype=torch.bfloat16, device=dev),
            torch.ones(rows, topk, device=dev),
            weight,
            weight,
            fused_moe.ActivationType(activation),
            num_local,
            0,
            rr.valid_tensor(rows, dev),
            torch.zeros(rows, topk, dtype=torch.int64, device=dev),
            disable_fused_quant_kernels=disable,
        )
    assert len(calls) == 1, calls
    name, kwargs = calls[0]
    assert name == ("fused" if fused else "unfused"), f"called {name}"
    assert kwargs["alignment"] == mx.TILE_ROWS, kwargs
    assert kwargs["zero_padding"] is bi, kwargs
    assert kwargs["return_batch_invariant_inverse_map"] is bi, kwargs
    if not fused:
        assert kwargs["row_alignment"] == mx.TILE_ROWS, kwargs

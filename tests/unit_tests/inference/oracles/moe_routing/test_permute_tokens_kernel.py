# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_permute_tokens_kernel`` and ``_zero_permutation_padding_kernel``
(megatron/core/inference/moe/permute.py), through their launcher ``permute_tokens``.

``permute_tokens`` runs the count, prefix-sum and init-map kernels (own oracles in this package)
and then these two; it is the torch-backend / MXFP8-unfused MoE permutation (paths P1-P4 of the
audit's dispatch map).

Contract (from the launcher's documented API and the grouped-GEMM consumers):

* Live pairs: ``(t, k)`` with ``t < min(valid, rows)`` and ``e = routing_map[t, k] - start`` in
  ``[0, E_l)``; ``c_e`` of them per local expert. Offsets: ``inclusive`` = running sum of
  ``c_e`` rounded up to ``alignment`` (zero stays zero); ``exclusive = inclusive - padded``;
  ``n_used = inclusive[-1]``.
* Expert ``e``'s block ``[exclusive[e], exclusive[e] + c_e)`` holds exactly its live pairs, one
  row each, **in unspecified order**: ``permutation_map[row] = t``, ``permuted_probs[row] =
  probs[t, k]`` and ``permuted_hidden[row] = hidden[t]`` (bit-exact copies, any dtype).
* Alignment padding ``[exclusive[e] + c_e, inclusive[e])``: ``permutation_map = -1``; with
  ``zero_padding`` the hidden rows are ``+0.0``; without it their contents are unspecified.
* Rows ``>= n_used`` of every output are not written (allocated at a fixed maximum size for CUDA
  graphs, ``output_size`` a multiple of ``row_alignment`` and ``>= n_used``).
* With ``return_batch_invariant_inverse_map``: ``inverse[t, e]`` is the row holding pair
  ``(t, e)``, ``-1`` where token ``t`` has no pair on ``e`` (all of row ``t`` for ``t >= valid``).
* Inputs read-only; ``valid`` an int32 device scalar. Preconditions: each token's ``topk`` ids
  are distinct (the router's top-k); the map, hidden and probs contiguous with equal ``rows``.
* Tolerance class: **exact** (integers and bit-exact copies; order within a block free).

Oracle: per expert block, the multiset of ``(token, prob bits)`` against a Python grouping of the
live pairs, then ``permuted_hidden[row] == hidden[permutation_map[row]]`` bitwise for every
claimed row; offsets against a Python aligned running sum; outputs allocated poisoned (PyTorch's
``fill_uninitialized_memory``: NaN / ``INT32_MAX``) so rows ``>= n_used`` must still hold poison.

Structure (read after the contract): permute: constexpr ``topk``, ``E_l``,
``BLOCK_H = min(next_pow2(H), 1024)``, ``NUM_BLOCKS = min(rows * topk, 512)``, ``HAS_INVERSE``;
the grid strides over pairs by ``NUM_BLOCKS``. Zero-padding: ``BLOCK_H``,
``NUM_BLOCKS = min(output_size, 512)``, loop bounded by ``min(n_used, output_size)``, and the
row pointer dtype (bf16 rows here, FP8 rows as bytes from ``permute_and_quantize_mxfp8``).
Inputs therefore cover ``H`` in {1, 33, 1023, 1024, 1025, 2688, 4096} (one and two tiles),
``rows * topk`` at 511/512/513 (grid cap), ``topk`` {1, 2, 6, 8}, ``E_l`` {1, 2, 7, 32} with
``E_l < topk`` included, first / second / last slice, ``alignment`` {1, 16, 32, 128},
``row_alignment`` {1, 128}, inverse map on/off, zero padding on/off, hidden {bf16, fp32} and
probs {fp32, bf16}, ``valid`` {rows, rows-1, 0, 1} and masked CUDA-graph padding rows, pairwise;
production shapes; CUDA-graph replay with ``valid`` changing (the map keeps the previous step's
entries past the new ``n_used``, the hidden rows are re-poisoned; neither may be written); the
zero-padding kernel launched directly on bf16 and byte rows with stale ``-1`` entries past
``n_used``, ``output_size`` at 511/512/513, between guard bands; opt-in (``large``), hidden
and permuted-row element offsets at 2^31 (``rows * H`` with ``H = 7168``) and inverse-map
offsets ``tok * E_l + lid`` past 2^31.

Known finding F1 (audit family 01): with ``valid > rows`` the count kernel over-reads on main
while this kernel clamps, so offsets and placements disagree.
``test_valid_above_rows_permutes_only_the_map_rows`` **fails on origin/main** and passes with
#8563 (``fix/d3-s6-mtp-moe-metadata``).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_routing import routing_reference as rr

permute = pytest.importorskip("megatron.core.inference.moe.permute")
pytest.importorskip("triton")


def _inputs(rows, hidden_dim, topk, num_experts, *, masked, hdtype, pdtype, dev, seed=0):
    g = ou.make_generator(ou.DEFAULT_SEED + seed)
    routing_map = rr.topk_distinct_ids(
        rows, topk, num_experts, masked_rows=masked, generator=g, dev=dev
    )
    hidden = torch.randn(rows, hidden_dim, generator=g).to(dtype=hdtype, device=dev)
    probs = torch.rand(rows, topk, generator=g).to(dtype=pdtype, device=dev)
    return hidden, probs, routing_map


def check_permutation(
    out,
    hidden,
    probs,
    routing_map,
    valid,
    start,
    num_local,
    *,
    alignment,
    row_alignment,
    zero_padding,
    inverse,
    what,
    stale_map=None,
):
    """Assert every clause of the contract on ``permute_tokens``'s outputs.

    ``stale_map``: the permutation map as it was before the call (a CUDA-graph buffer holding a
    previous step's entries); rows ``>= n_used`` must still hold it. Otherwise they must still
    hold the allocation poison.
    """
    if inverse:
        perm_h, perm_p, perm_map, offsets, inv = out
    else:
        perm_h, perm_p, perm_map, offsets = out
    counts = rr.local_counts(routing_map, valid, start, num_local)
    ref_ex, ref_in = rr.aligned_offsets(counts.tolist(), alignment)
    ou.assert_exact(offsets, torch.tensor(ref_in), what=f"inclusive offsets ({what})")
    n_used = ref_in[-1]
    size = perm_h.shape[0]
    assert perm_p.shape == (size,) and perm_map.shape == (size,), what
    assert perm_map.dtype == torch.int32 and perm_p.dtype == probs.dtype, what
    assert perm_h.dtype == hidden.dtype and perm_h.shape[1] == hidden.shape[1], what
    assert size >= n_used and size % row_alignment == 0, f"output_size {size} ({what})"

    pairs = rr.expected_expert_rows(routing_map, valid, start, num_local)
    probs_cpu, map_cpu = probs.cpu(), perm_map.cpu()
    for e in range(num_local):
        lo, hi, end = ref_ex[e], ref_ex[e] + int(counts[e]), ref_in[e]
        tokens = torch.tensor([t for t, _ in pairs[e]], dtype=torch.int64)
        slots = torch.tensor([k for _, k in pairs[e]], dtype=torch.int64)
        expected = torch.stack([tokens, rr.bits_i64(probs_cpu[tokens, slots])], dim=1)
        actual = torch.stack([map_cpu[lo:hi].to(torch.int64), rr.bits_i64(perm_p[lo:hi].cpu())], 1)
        ou.assert_rows_equal_unordered(
            actual, expected, what=f"expert {e} block [{lo},{hi}) (token, prob) ({what})"
        )
        ou.assert_exact(
            perm_map[hi:end],
            torch.full((end - hi,), -1, dtype=torch.int32),
            what=f"expert {e} alignment padding map ({what})",
        )
        if zero_padding:
            ou.assert_exact(
                perm_h[hi:end],
                torch.zeros((end - hi, hidden.shape[1]), dtype=hidden.dtype),
                what=f"expert {e} zeroed padding rows ({what})",
            )
    real = (perm_map[:n_used] >= 0).nonzero().reshape(-1)
    ou.assert_exact(
        perm_h[real], hidden[perm_map[real].long()], what=f"claimed rows copy hidden ({what})"
    )
    if stale_map is None:
        rr.assert_untouched(perm_map[n_used:], what=f"permutation_map rows >= n_used ({what})")
    else:
        ou.assert_exact(
            perm_map[n_used:], stale_map[n_used:], what=f"stale map rows >= n_used ({what})"
        )
    rr.assert_untouched(perm_h[n_used:], what=f"permuted_hidden rows >= n_used ({what})")
    rr.assert_untouched(perm_p[n_used:], what=f"permuted_probs rows >= n_used ({what})")

    if inverse:
        assert inv.shape == (routing_map.shape[0], num_local) and inv.dtype == torch.int32, what
        expected_inv = torch.full(tuple(inv.shape), -1, dtype=torch.int32)
        for e in range(num_local):
            for row in range(ref_ex[e], ref_ex[e] + int(counts[e])):
                expected_inv[int(map_cpu[row]), e] = row
        ou.assert_exact(inv, expected_inv, what=f"inverse map ({what})")


def run_and_check(
    rows,
    hidden_dim,
    topk,
    num_local,
    start,
    num_experts,
    *,
    valid,
    masked=0,
    alignment=1,
    row_alignment=1,
    zero_padding=False,
    inverse=False,
    hdtype=torch.bfloat16,
    pdtype=torch.float32,
    seed=0,
):
    dev = rr.device()
    hidden, probs, routing_map = _inputs(
        rows,
        hidden_dim,
        topk,
        num_experts,
        masked=masked,
        hdtype=hdtype,
        pdtype=pdtype,
        dev=dev,
        seed=seed,
    )
    routing_map, _ = rr.with_canary_tail(routing_map, rr.COUNT_BLOCK, start)
    before = (hidden.clone(), probs.clone(), routing_map.clone())
    with rr.poisoned_outputs():
        out = permute.permute_tokens(
            hidden,
            probs,
            routing_map,
            start,
            num_local,
            rr.valid_tensor(valid, dev),
            alignment=alignment,
            return_batch_invariant_inverse_map=inverse,
            row_alignment=row_alignment,
            zero_padding=zero_padding,
        )
    what = (
        f"rows={rows} H={hidden_dim} topk={topk} E_l={num_local} start={start} valid={valid}"
        f" a={alignment} ra={row_alignment} zp={zero_padding} inv={inverse}"
    )
    check_permutation(
        out,
        hidden,
        probs,
        routing_map,
        valid,
        start,
        num_local,
        alignment=alignment,
        row_alignment=row_alignment,
        zero_padding=zero_padding,
        inverse=inverse,
        what=what,
    )
    for name, now, then in zip(
        ("hidden", "probs", "routing_map"), (hidden, probs, routing_map), before
    ):
        ou.assert_exact(now, then, what=f"{name} is read-only")


_VARIANTS = ou.pairwise_cases(
    {
        "hidden_dim": [1, 33, 1023, 1024, 1025, 2688, 4096],
        "topk": [1, 2, 6, 8],
        "num_local": [1, 2, 7, 32],
        "slice": ["first", "second", "last"],
        "alignment": [1, 16, 32, 128],
        "row_alignment": [1, 128],
        "inverse": [False, True],
        "zero_padding": [False, True],
        "dtypes": [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
        "valid": ["rows", "rows-1", "0", "1", "masked"],
    }
)


def _variant_id(c):
    h, p = (str(d)[6:] for d in c["dtypes"])
    return (
        f"H{c['hidden_dim']}-topk{c['topk']}-E{c['num_local']}-{c['slice']}-a{c['alignment']}"
        f"-ra{c['row_alignment']}-inv{int(c['inverse'])}-zp{int(c['zero_padding'])}-{h}-{p}"
        f"-v{c['valid']}"
    )


@pytest.mark.parametrize("case", _VARIANTS, ids=[_variant_id(c) for c in _VARIANTS])
def test_variants_pairwise(case):
    """Every compiled variant and dispatch flag, pairwise (see the module docstring)."""
    rows = 97
    num_local, topk = case["num_local"], case["topk"]
    num_experts = max(4 * num_local, 2 * topk)
    start = {"first": 0, "second": num_local, "last": num_experts - num_local}[case["slice"]]
    masked = rows // 3 if case["valid"] == "masked" else 0
    valid = {"rows": rows, "rows-1": rows - 1, "0": 0, "1": 1, "masked": rows}[case["valid"]]
    hdtype, pdtype = case["dtypes"]
    run_and_check(
        rows,
        case["hidden_dim"],
        topk,
        num_local,
        start,
        num_experts,
        valid=valid,
        masked=masked,
        alignment=case["alignment"],
        row_alignment=case["row_alignment"],
        zero_padding=case["zero_padding"],
        inverse=case["inverse"],
        hdtype=hdtype,
        pdtype=pdtype,
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
    run_and_check(rows, 64, topk, 8, 8, 32, valid=v, alignment=16, inverse=True, zero_padding=True)


@pytest.mark.parametrize("kind", ["all-masked", "none-local", "one-expert-local"])
def test_degenerate_routing(kind):
    """No live pairs (all ``-1`` / nothing local) and a single local expert holding every pair."""
    if kind == "all-masked":
        run_and_check(40, 64, 4, 8, 8, 32, valid=40, masked=40, alignment=16, zero_padding=True)
    elif kind == "none-local":
        run_and_check(40, 64, 4, 8, 32, 32, valid=40, alignment=16, inverse=True)
    else:
        run_and_check(
            600,
            64,
            1,
            1,
            0,
            1,
            valid=600,
            alignment=128,
            row_alignment=128,
            zero_padding=True,
            inverse=True,
        )


@pytest.mark.parametrize(
    "rows,hidden_dim,topk,num_experts,ep,alignment,row_alignment",
    [
        pytest.param(8192, 2688, 6, 128, 1, 16, 1, id="nanov3-ep1-torch-bf16"),
        pytest.param(8192, 7168, 8, 256, 8, 128, 128, id="dsv3-ep8-mxfp8-unfused"),
        pytest.param(64 * 256, 7168, 8, 256, 64, 16, 1, id="dsv3-ep64-decode-El4"),
    ],
)
@pytest.mark.parametrize("bi", [False, True], ids=["non-bi", "bi"])
def test_production_sizes(rows, hidden_dim, topk, num_experts, ep, alignment, row_alignment, bi):
    """Production shapes; ``bi`` adds the inverse map and zero padding (paths P2 / P4)."""
    rr.require_cuda()
    num_local = num_experts // ep
    start = (ep // 2) * num_local
    run_and_check(
        rows,
        hidden_dim,
        topk,
        num_local,
        start,
        num_experts,
        valid=rows - rows // 9,
        masked=rows // 9,
        alignment=alignment,
        row_alignment=row_alignment,
        zero_padding=bi,
        inverse=bi,
    )


@pytest.mark.sanitizer
def test_sanitizer_small_case():
    """Small representative case for the compute-sanitizer memcheck pass."""
    run_and_check(50, 96, 4, 8, 8, 32, valid=45, alignment=16, inverse=True, zero_padding=True)


@pytest.mark.parametrize("inverse,zero_padding", [(False, False), (True, True)], ids=["p1", "p4"])
def test_cuda_graph_replay_follows_valid(inverse, zero_padding):
    """Captured once; replayed with ``valid`` changing. Rows past the new ``n_used`` keep a larger
    previous step's map entries and must not be written (hidden rows re-poisoned per replay)."""
    dev = rr.require_cuda()
    rows, hidden_dim, topk, num_local, start, num_experts = 300, 192, 8, 8, 8, 32
    hidden, probs, routing_map = _inputs(
        rows,
        hidden_dim,
        topk,
        num_experts,
        masked=0,
        hdtype=torch.bfloat16,
        pdtype=torch.float32,
        dev=dev,
    )
    routing_map, _ = rr.with_canary_tail(routing_map, rr.COUNT_BLOCK, start)
    valid = rr.valid_tensor(rows, dev)
    kwargs = dict(
        alignment=16, return_batch_invariant_inverse_map=inverse, zero_padding=zero_padding
    )
    permute.permute_tokens(hidden, probs, routing_map, start, num_local, valid, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = permute.permute_tokens(hidden, probs, routing_map, start, num_local, valid, **kwargs)
    for v in [rows, 1, 0, 64, 65, rows - 1, 7, rows]:
        # The map keeps the previous step's entries (stale -1 padding marks included); the hidden
        # rows and probabilities are re-poisoned, so a write past the new n_used shows up.
        stale_map = out[2].clone()
        out[0].fill_(float("nan"))
        out[1].fill_(float("nan"))
        valid.fill_(v)
        graph.replay()
        torch.cuda.synchronize()
        check_permutation(
            out,
            hidden,
            probs,
            routing_map,
            v,
            start,
            num_local,
            alignment=16,
            row_alignment=1,
            zero_padding=zero_padding,
            inverse=inverse,
            what=f"replay valid={v}",
            stale_map=stale_map,
        )


def _zero_padding_cases():
    cases = []
    for dtype in (torch.bfloat16, torch.uint8):
        for size in (5, 511, 512, 513, 1300):
            for n_used in sorted({0, 1, size // 2, size - 1, size}):
                name = "bf16" if dtype == torch.bfloat16 else "mxfp8-bytes"
                cases.append(
                    pytest.param(dtype, size, n_used, id=f"{name}-size{size}-used{n_used}")
                )
    return cases


@pytest.mark.parametrize("dtype,output_size,n_used", _zero_padding_cases())
def test_zero_padding_kernel_leaves_the_unused_suffix(dtype, output_size, n_used):
    """``_zero_permutation_padding_kernel`` launched as ``permute_tokens`` (bf16 rows) and
    ``permute_and_quantize_mxfp8`` (FP8 rows as bytes, ``hidden_dim = K``) launch it.

    Contract (its docstring): zero the rows ``< n_used`` whose map entry is ``-1``, without
    touching the unused suffix. The suffix holds stale ``-1`` entries from a larger earlier step
    (CUDA-graph buffers), which must not be zeroed. Grid ``min(output_size, 512)`` and
    ``BLOCK_H = min(next_pow2(H), 1024)`` as the launchers set them; ``H`` crosses one tile.
    """
    dev = rr.device()
    g = ou.make_generator()
    hidden_dim = 1056
    perm_map = torch.randint(-1, 50, (output_size,), generator=g, dtype=torch.int32)
    perm_map[n_used:] = torch.where(torch.arange(output_size - n_used) % 2 == 0, -1, 7)
    perm_map = perm_map.to(dev)
    buf = ou.guarded_empty((output_size, hidden_dim), dtype, dev, guard_bytes=4 * hidden_dim)
    before = buf.tensor.clone()
    blocks = min(output_size, rr.MAX_GRID)
    permute._zero_permutation_padding_kernel[(blocks,)](
        buf.tensor,
        perm_map,
        rr.valid_tensor(n_used, dev),
        hidden_dim,
        output_size,
        BLOCK_H=min(1 << (hidden_dim - 1).bit_length(), rr.MAX_BLOCK_H),
        NUM_BLOCKS=blocks,
    )
    ou.check_guards(hidden=buf)
    padding = torch.zeros(output_size, dtype=torch.bool, device=dev)
    padding[:n_used] = perm_map[:n_used] < 0
    expected = torch.where(padding[:, None], torch.zeros_like(before), before)
    ou.assert_exact(buf.tensor, expected, what=f"zeroed padding, untouched rest ({dtype})")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(7168))
def test_int32_row_offsets(rows):
    """Hidden and permuted rows whose element offsets ``row * H`` cross 2^31 (``H = 7168``,
    one local expert, top-1, so the permuted buffer has ``rows + 1`` rows). Opt-in ``large``;
    about 20 GiB. Hidden rows are random bit patterns (compared bitwise)."""
    dev = rr.require_cuda()
    hidden_dim = 7168
    ou.require_free_memory(5 * rows * hidden_dim * 2 + (2 << 30))
    g = ou.make_generator(device=dev)
    hidden = torch.randint(
        -(2**15), 2**15, (rows, hidden_dim), dtype=torch.int16, device=dev, generator=g
    )
    hidden = hidden.view(torch.bfloat16)
    probs = torch.rand(rows, 1, device=dev, generator=g)
    routing_map = torch.zeros(rows, 1, dtype=torch.int64, device=dev)
    routing_map[rows // 3 :: 7] = -1
    with rr.poisoned_outputs():
        out = permute.permute_tokens(
            hidden, probs, routing_map, 0, 1, rr.valid_tensor(rows, dev), alignment=16
        )
    check_permutation(
        out,
        hidden,
        probs,
        routing_map,
        rows,
        0,
        1,
        alignment=16,
        row_alignment=1,
        zero_padding=False,
        inverse=False,
        what=f"int32 offsets rows={rows}",
    )


@pytest.mark.large
def test_int32_inverse_map_offsets():
    """Finding O1-N3: the inverse-map store offset ``tok * E_l + lid`` is int32.

    With ``E_l = 128`` and ``2^24 + 1`` rows (top-1, token ``t`` on expert ``t % 128``,
    ``HAS_INVERSE``, alignment 128 as on the batch-invariant path), the last token's offset is
    ``2^31``: on origin/main it wraps and the store lands 8 GiB before the ``[rows, E_l]``
    inverse map, which keeps ``-1`` in that row (or the stray store faults). The check is
    vectorised: offsets, every claimed row on its token's expert, then the whole inverse map
    against the one implied by ``permutation_map``. About 20 GiB; opt-in ``large``.
    """
    dev = rr.require_cuda()
    num_local, alignment, hidden_dim = 128, 128, 16
    rows = 2**31 // num_local + 1
    ou.require_free_memory(2 * rows * num_local * 4 + (4 << 30))
    hidden = torch.zeros(rows, hidden_dim, dtype=torch.bfloat16, device=dev)
    probs = torch.ones(rows, 1, dtype=torch.float32, device=dev)
    routing_map = torch.arange(rows, device=dev).remainder_(num_local).view(rows, 1)
    _, _, perm_map, offsets, inv = permute.permute_tokens(
        hidden,
        probs,
        routing_map,
        0,
        num_local,
        rr.valid_tensor(rows, dev),
        alignment=alignment,
        return_batch_invariant_inverse_map=True,
    )
    counts = torch.full((num_local,), rows // num_local, dtype=torch.int64)
    counts[: rows % num_local] += 1
    _, ref_in = rr.aligned_offsets(counts.tolist(), alignment)
    ou.assert_exact(offsets, torch.tensor(ref_in), what="inclusive offsets")
    used = torch.arange(ref_in[-1], device=dev)
    real = used[perm_map[: ref_in[-1]] >= 0]
    assert real.numel() == rows, f"{real.numel()} claimed rows, expected {rows}"
    expert = torch.searchsorted(offsets.to(torch.int64), real, right=True)
    tokens = perm_map[real].long()
    ou.assert_exact(tokens % num_local, expert, what="each claimed row on its token's expert")
    expected = torch.full_like(inv, -1)
    expected[tokens, expert] = real.to(torch.int32)
    for lo in range(0, rows, 1 << 20):
        ou.assert_exact(
            inv[lo : lo + (1 << 20)],
            expected[lo : lo + (1 << 20)],
            what=f"inverse map rows from {lo}",
        )


@pytest.mark.parametrize("alignment", [1, 16])
def test_valid_above_rows_permutes_only_the_map_rows(alignment):
    """``valid > rows`` (finding F1): offsets and placements cover only the map's rows.

    Fails on origin/main: the count kernel reads the canary tail past the map, so the offsets
    reserve rows the permute kernel (which clamps to ``rows``) never claims. Passes with #8563.
    """
    dev = rr.device()
    rows, topk, num_local, start = 6, 2, 8, 0
    hidden, probs, head = _inputs(
        rows,
        64,
        topk,
        num_local - 1,
        masked=0,
        hdtype=torch.bfloat16,
        pdtype=torch.float32,
        dev=dev,
    )
    routing_map, _ = rr.with_canary_tail(head, 4 + rr.COUNT_BLOCK, num_local - 1)
    with rr.poisoned_outputs():
        out = permute.permute_tokens(
            hidden,
            probs,
            routing_map,
            start,
            num_local,
            rr.valid_tensor(rows + 4, dev),
            alignment=alignment,
        )
    check_permutation(
        out,
        hidden,
        probs,
        routing_map,
        rows + 4,
        start,
        num_local,
        alignment=alignment,
        row_alignment=1,
        zero_padding=False,
        inverse=False,
        what="valid > rows",
    )

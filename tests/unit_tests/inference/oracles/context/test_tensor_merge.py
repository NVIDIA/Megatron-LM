# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_tensor_merge_kernel``
(megatron/core/inference/contexts/attention_context/triton/tensor_ops.py, launcher
``tensor_merge``): the merge of a hybrid model's decode and prefill outputs back into packed
token order.

**Contract** (from the call site, ``SSMDynamicInferenceMixin.ssm_dynamic_inference`` in
megatron/core/ssm/ssm_inference.py, and the launcher's docstring). The mixer runs the decode
rows (``a``, the decode partition) and the prefill rows (``b``) separately and merges them:
``tensor_merge(a, b, pos_on_device, output_tensor=out)`` writes

    out[:pos]            = a[:pos]
    out[pos : pos + k]   = b[:k]        k = min(B, N_out - pos)

bit for bit, where ``pos = pos_on_device[0]`` (the real decode tokens) is read **on device**
when the kernel runs, so a captured graph follows each replay's value; production passes the
context's ``int32 [2]`` pair and only element 0 counts. Rows ``out[pos + k:]`` are untouched;
``a``, ``b`` and ``pos_on_device`` are read-only. Without ``output_tensor`` the merge is in
place on ``a`` (``N_out = N_a``): ``a[pos : pos + k] = b[:k]``, and the rest of ``a`` keeps its
values. All tensors share dtype, device and trailing shape and are contiguous (asserted).
Preconditions, checked only under ``check_bounds`` (production does not pass it):
``0 <= pos <= N_a <= N_out``. In production ``out`` has the token count, ``b`` all ``T`` rows of
the prefill output, so ``b`` is truncated by ``pos`` rows. Tolerance class: copy, **exact**.

**Oracle.** Python slicing on a clone of the poisoned output allocation.

**Inputs from the kernel's structure.** One program per output row (grid ``N_out``), each
copying a row of ``a`` (if ``pid < pos``, unless in place) or of ``b`` in one
``BLOCK_SIZE = next_pow2(row)`` tile; ``ROW_SIZE``, ``BLOCK_SIZE`` and ``IS_INPLACE`` are compiled
in, ``B`` and ``N_out`` are runtime ints (specialized when 1 or a multiple of 16;
``tensor_ops.py:374-392``). Factors, combined pairwise: row sizes 1, 1023/1024/1025,
2047/2048/2049 and the SSM output widths 8192 (Nemotron-H 8B at TP 1) and 16384 (56B);
``N_a`` 1, 15/16/17; ``B`` 1, 15/16/17, 33; ``pos`` 0, 1, mid, ``N_a - 1`` and ``N_a``; ``N_out``
as in production (``max(B, N_a)``), exactly ``pos + B``, longer (untouched tail) and one row
short (truncation); in place and not; bf16, fp16, fp32; 2-D and 3-D; ``pos`` as ``int32 [1]``,
the ``int32 [2]`` pair and ``int64 [1]``; aligned and unaligned base pointers. **CUDA-graph
replay with ``pos`` changing.** Production sizes: 8192 tokens at width 8192 and 4096 at 16384.
int32 limits (``large``): output offsets ``row * ROW_SIZE`` reach 2**31.

**Memory.** Every tensor in a guard-banded allocation (guards of one row); the output starts
NaN-poisoned and its untouched rows must stay poisoned; inputs checked unchanged. One small case
runs under compute-sanitizer.

**Hardware.** H100. An output row that starts past 2**31 elements fails on main: finding O6-M1
(``pid * ROW_SIZE`` and ``tensor_b_idx * ROW_SIZE`` are int32, ``tensor_ops.py:80-81, 93-94``).
#9307 (fix/context-kernels-int64-offsets) fixes it and carries that case as its regression test
(``test_tensor_ops.py::TestInt32RowOffsets::test_tensor_merge``); the ``large`` case here stops
at the last size whose rows all start below 2**31.
"""

import math

import pytest
import torch

from megatron.core.inference.contexts.attention_context.triton.tensor_ops import tensor_merge
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.context import copy_buffers as cb

DEVICE = "cuda"

D_INNER_8B = 8192
"""Nemotron-H 8B SSM output width at TP 1 (128 heads x 64)."""
D_INNER_56B = 16384
"""Nemotron-H 56B SSM output width at TP 1 (256 heads x 64)."""

ROWS = ou.boundary_sizes(1024, minimum=1, production=(D_INNER_8B, D_INNER_56B))
N_A = (1, 15, 16, 17)
N_B = (1, 15, 16, 17, 33)
POS = ("zero", "one", "mid", "last", "all")
N_OUT = ("production", "exact", "long", "short")

CASES = ou.pairwise_cases(
    {
        "row": ROWS,
        "n_out": N_OUT,
        "pos": POS,
        "n_b": N_B,
        "n_a": N_A,
        "in_place": (False, True),
        "dtype": (torch.bfloat16, torch.float16, torch.float32),
        "shape": ("2d", "3d"),
        "pos_buffer": ("int32", "pair", "int64"),
        "layout": ("contiguous", "offset"),
    }
)


def _case_id(c):
    dtype = str(c["dtype"]).replace("torch.", "")
    place = "inplace" if c["in_place"] else c["n_out"]
    return (
        f"row{c['row']}-Na{c['n_a']}-B{c['n_b']}-pos_{c['pos']}-{place}-{dtype}-{c['shape']}"
        f"-{c['pos_buffer']}-{c['layout']}"
    )


def _pos_value(kind, n_a):
    return {"zero": 0, "one": 1, "mid": n_a // 2, "last": n_a - 1, "all": n_a}[kind]


def _n_out(kind, n_a, n_b, pos):
    return {
        "production": max(n_b, n_a),
        "exact": max(pos + n_b, n_a),
        "long": max(pos + n_b, n_a) + 3,
        "short": max(pos + n_b - 1, n_a),
    }[kind]


def _pos_tensor(value, kind, other=-7):
    values, dtype = {
        "int32": ([value], torch.int32),
        "pair": ([value, other], torch.int32),
        "int64": ([value], torch.int64),
    }[kind]
    return torch.tensor(values, dtype=dtype, device=DEVICE)


def _trail(row, shape):
    return {"2d": (row,), "3d": (1, row)}[shape]


def _placed_copy(x, layout, guard):
    p = cb.Placed(x.shape, x.dtype, layout, guard_bytes=guard, device=DEVICE)
    p.tensor.copy_(x)
    return p


def _run_and_check(a, b, n_out, pos, pos_kind, layout, in_place, what):
    """Merge placed copies of ``a`` and ``b`` into a placed poisoned output (or in place)."""
    trail = tuple(a.shape[1:])
    guard = max(math.prod(trail) * a.dtype.itemsize, 4096)
    a_p, b_p = _placed_copy(a, layout, guard), _placed_copy(b, layout, guard)
    b_before = b_p.full.clone()
    pos_on_device = _pos_tensor(pos, pos_kind)
    pos_before = pos_on_device.clone()
    if in_place:
        out_p, a_before = a_p, None
        n_out = a.shape[0]
    else:
        out_p = cb.Placed((n_out, *trail), a.dtype, layout, guard_bytes=guard, device=DEVICE)
        a_before = a_p.full.clone()
    expected = out_p.full.clone()
    expected_view = out_p.view_of(expected)
    expected_view[:pos] = a[:pos]
    k = max(0, min(b.shape[0], n_out - pos))
    expected_view[pos : pos + k] = b[:k]

    tensor_merge(
        a_p.tensor, b_p.tensor, pos_on_device, output_tensor=None if in_place else out_p.tensor
    )
    torch.cuda.synchronize()

    if in_place:
        ou.check_guards(a=a_p.buf, b=b_p.buf)
    else:
        ou.check_guards(a=a_p.buf, b=b_p.buf, out=out_p.buf)
        ou.assert_exact(a_p.full, a_before, what="a is read-only")
    ou.assert_exact(out_p.full, expected, what=f"{what}: out[:{pos}] = a, out[{pos}:{pos + k}] = b")
    ou.assert_exact(b_p.full, b_before, what="b is read-only")
    ou.assert_exact(pos_on_device, pos_before, what="pos is read-only")


SANITIZER_CASE = dict(
    row=1025,
    n_out="short",
    pos="mid",
    n_b=17,
    n_a=15,
    in_place=False,
    dtype=torch.bfloat16,
    shape="3d",
    pos_buffer="pair",
    layout="offset",
)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=_case_id(c)) for c in CASES]
    + [
        pytest.param(
            SANITIZER_CASE, id="sanitizer-" + _case_id(SANITIZER_CASE), marks=pytest.mark.sanitizer
        )
    ],
)
def test_merge_matches_contract(case):
    g = ou.make_generator(device=DEVICE)
    trail = _trail(case["row"], case["shape"])
    n_a, n_b = case["n_a"], case["n_b"]
    pos = _pos_value(case["pos"], n_a)
    n_out = _n_out(case["n_out"], n_a, n_b, pos)
    a = cb.random_bits((n_a, *trail), case["dtype"], g)
    b = cb.random_bits((n_b, *trail), case["dtype"], g)
    _run_and_check(
        a, b, n_out, pos, case["pos_buffer"], case["layout"], case["in_place"], what=_case_id(case)
    )


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "tokens,row,padded_decode,real_decode",
    [(8192, D_INNER_8B, 512, 384), (4096, D_INNER_56B, 256, 255)],
    ids=["8B-8192-tokens", "56B-4096-tokens"],
)
def test_merge_at_production_sizes(tokens, row, padded_decode, real_decode):
    """The production call: ``y_decode`` ``[D, 1, d]`` (the padded decode partition),
    ``y_prefill`` ``[T, 1, d]`` and ``y`` ``[T, 1, d]`` bf16, ``pos`` the real decode tokens."""
    g = ou.make_generator(device=DEVICE)
    a = cb.random_bits((padded_decode, 1, row), torch.bfloat16, g)
    b = cb.random_bits((tokens, 1, row), torch.bfloat16, g)
    _run_and_check(a, b, tokens, real_decode, "pair", "contiguous", False, what=f"{tokens} x {row}")


@ou.requires_sm(ou.HOPPER)
def test_cuda_graph_replay_follows_pos():
    """Captured once, replayed with ``pos`` across the decode partition on fresh data, the
    output re-poisoned before every replay."""
    n_a, tokens, row, dtype = 24, 37, 2049, torch.bfloat16
    g = ou.make_generator(device=DEVICE)
    guard = max(row * dtype.itemsize, 4096)
    a_p = cb.Placed((n_a, 1, row), dtype, guard_bytes=guard, device=DEVICE)
    b_p = cb.Placed((tokens, 1, row), dtype, guard_bytes=guard, device=DEVICE)
    out_p = cb.Placed((tokens, 1, row), dtype, guard_bytes=guard, device=DEVICE)
    a_p.tensor.copy_(cb.random_bits(a_p.shape, dtype, g))
    b_p.tensor.copy_(cb.random_bits(b_p.shape, dtype, g))
    pos_on_device = torch.tensor([3, tokens - 3], dtype=torch.int32, device=DEVICE)

    def merge():
        tensor_merge(a_p.tensor, b_p.tensor, pos_on_device, output_tensor=out_p.tensor)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        merge()  # compile outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        merge()

    for pos in (0, n_a, 1, 5, 16, 17, n_a - 1, 8):
        a = cb.random_bits(a_p.shape, dtype, g)
        b = cb.random_bits(b_p.shape, dtype, g)
        a_p.tensor.copy_(a)
        b_p.tensor.copy_(b)
        ou.poison_(out_p.full)
        pos_on_device.copy_(torch.tensor([pos, tokens - pos], dtype=torch.int32))
        expected = out_p.full.clone()
        expected[:pos] = a[:pos]
        expected[pos:] = b[: tokens - pos]
        graph.replay()
        torch.cuda.synchronize()
        ou.check_guards(a=a_p.buf, b=b_p.buf, out=out_p.buf)
        ou.assert_exact(out_p.full, expected, what=f"replay with pos {pos}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "tokens",
    [p for p in ou.int32_overflow_params(D_INNER_8B) if (p.values[0] - 1) * D_INNER_8B < 2**31],
)
def test_merge_offsets_past_int32(tokens):
    """Output offsets ``row * 8192`` (the 8B SSM output at TP 1) up to 2**31, as in production
    (``out`` and ``b`` with ``T`` rows, ``pos`` = 3 decode tokens). The output sits behind a
    2**31-element poisoned front guard, so a wrapped int32 offset writes into the guard instead of
    faulting. An output row that starts past 2**31 fails on main (finding O6-M1) and is #9307's
    regression case; it is reachable only with ``max_tokens`` of at least 262,145 for this width.
    About 22 GB."""
    row, dtype, pos = D_INNER_8B, torch.bfloat16, 3
    numel = tokens * row
    ou.require_free_memory(5 * numel * dtype.itemsize + 2**30)
    g = ou.make_generator(device=DEVICE)
    a = cb.random_bits((8, 1, row), dtype, g)
    b = cb.random_bits((tokens, 1, row), dtype, g)
    out, out_guard = cb.behind_front_guard((tokens, 1, row), dtype, device=DEVICE)
    ou.poison_(out)
    pos_on_device = torch.tensor([pos, tokens - pos], dtype=torch.int32, device=DEVICE)

    tensor_merge(a, b, pos_on_device, output_tensor=out)
    torch.cuda.synchronize()

    stray = cb.changed(out_guard)
    assert stray == 0, f"{stray} elements written into the 2**31 elements before the output"
    ou.assert_exact(out[:pos], a[:pos], what="out[:pos] = a[:pos]")
    ou.assert_exact(out[pos:], b[: tokens - pos], what=f"{tokens} rows of {row}: out[pos:] = b")

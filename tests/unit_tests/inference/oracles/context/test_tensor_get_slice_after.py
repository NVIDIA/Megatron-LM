# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_tensor_get_slice_after_kernel``
(megatron/core/inference/contexts/attention_context/triton/tensor_ops.py, launcher
``tensor_get_slice_after``): the prefill gather of a hybrid model's mixed batch.

**Contract** (from the call site, ``SSMDynamicInferenceMixin.ssm_dynamic_inference`` in
megatron/core/ssm/ssm_inference.py, and the launcher's docstring). A mixed batch packs the
decode tokens first and the prefill tokens after them; the mixer runs the prefill kernels on the
rows after the decode tokens. ``tensor_get_slice_after(x, out, pos_on_device)`` copies them to the
front of ``out``:

    out[i] = x[pos + i]    for 0 <= i < k = min(N_in - pos, N_out)

bit for bit (NaN payloads, -0.0 and denormals kept), where ``pos = pos_on_device[0]`` is read
**on device** when the kernel runs, so a captured CUDA graph follows the value of each replay.
Production passes the context's ``int32 [2]`` buffer (real decode tokens, real prefill tokens);
only element 0 counts. Rows ``out[k:]`` are untouched (production allocates ``out`` with
``torch.empty_like``), and ``x`` and ``pos_on_device`` are read-only. ``x`` and ``out`` have the
same dtype, device and trailing shape (``[T, 1, d]`` in production; a 1-D tensor has one-element
rows) and are contiguous (asserted). Precondition, checked only under ``check_bounds`` (which
production does not pass): ``0 <= pos <= N_in``. ``pos = N_in`` copies nothing, and so does a
``pos`` past ``N_in`` (the empty slice ``x[pos:]``), without touching memory. ``N_in = 0`` does
nothing. Tolerance class: copy, **exact** (bitwise).

**Oracle.** Python slicing on a clone of the poisoned output allocation
(``expected[:k] = x[pos:pos + k]``).

**Inputs from the kernel's structure.** One program per input row (grid ``N_in``), each copying
its row in one ``BLOCK_SIZE = next_pow2(row)`` tile; ``ROW_SIZE`` and ``BLOCK_SIZE`` are
compiled in, the batch sizes are runtime ints, which Triton specializes when 1 or a multiple of
16 (``tensor_ops.py:307-324``). Factors, combined pairwise:
row sizes 1, 1023/1024/1025, 2047/2048/2049 (masked and full tiles) and the ``in_proj`` widths
18560 (Nemotron-H 8B at TP 1) and 37120 (Nemotron-H 56B at TP 1); ``N_in`` 1, 15/16/17, 33;
``N_out`` equal to ``N_in`` (production), one row short of the copy (truncation), longer
(untouched tail) and 1; ``pos`` 0, 1, mid, ``N_in - 1``, ``N_in`` and ``N_in + 1``; bf16, fp16,
fp32; 2-D and 3-D (and 1-D rows separately); ``pos`` as ``int32 [1]``, the production
``int32 [2]`` pair and ``int64 [1]`` (a separate compilation); 16-byte-aligned and unaligned base
pointers (Triton specializes pointer alignment). **CUDA-graph replay with ``pos`` changing**,
the reason the kernel reads ``pos`` on device: captured once, replayed with ``pos`` across the
whole range on fresh data. Production sizes: 8192 tokens of the 8B ``in_proj`` and 4096 of the
56B's. int32 limits (``large``): element offsets ``row * ROW_SIZE`` reach 2**31.

**Memory.** Input and output sit in guard-banded allocations (guards of one row); the output
starts NaN-poisoned and its untouched rows must stay poisoned; the input allocation, ``pos`` and
the second ``pos`` element are checked unchanged. One small case runs under compute-sanitizer.

**Hardware.** H100. A row that starts past 2**31 elements fails on main: finding O6-M1
(``input_idx * ROW_SIZE`` and ``pid * ROW_SIZE`` are int32, ``tensor_ops.py:46-47``). #9307
(fix/context-kernels-int64-offsets) fixes it and carries that case as its regression test
(``test_tensor_ops.py::TestInt32RowOffsets::test_get_slice_after``); the ``large`` sizes here stop
at the one whose last row crosses 2**31.
"""

import math

import pytest
import torch

from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.context import copy_buffers as cb

DEVICE = "cuda"

IN_PROJ_8B = 18560
"""Nemotron-H 8B ``in_proj`` width at TP 1: 2 * 8192 + 2 * 8 * 128 + 128."""
IN_PROJ_56B = 37120
"""Nemotron-H 56B ``in_proj`` width at TP 1: 2 * 16384 + 2 * 8 * 256 + 256."""

ROWS = ou.boundary_sizes(1024, minimum=1, production=(IN_PROJ_8B, IN_PROJ_56B))
N_IN = (1, 15, 16, 17, 33)
N_OUT = ("same", "short", "long", "one")
POS = ("zero", "one", "mid", "last", "end", "past")
POS_BUFFERS = ("int32", "pair", "int64")

CASES = ou.pairwise_cases(
    {
        "row": ROWS,
        "pos": POS,
        "n_in": N_IN,
        "n_out": N_OUT,
        "dtype": (torch.bfloat16, torch.float16, torch.float32),
        "shape": ("2d", "3d"),
        "pos_buffer": POS_BUFFERS,
        "layout": ("contiguous", "offset"),
    }
)


def _case_id(c):
    dtype = str(c["dtype"]).replace("torch.", "")
    return (
        f"row{c['row']}-Nin{c['n_in']}-{c['n_out']}-pos_{c['pos']}-{dtype}-{c['shape']}"
        f"-{c['pos_buffer']}-{c['layout']}"
    )


def _pos_value(kind, n_in):
    return {
        "zero": 0,
        "one": min(1, n_in),
        "mid": n_in // 2,
        "last": max(n_in - 1, 0),
        "end": n_in,
        "past": n_in + 1,
    }[kind]


def _n_out(kind, n_in, pos):
    return {"same": n_in, "short": max(n_in - pos - 1, 1), "long": n_in + 3, "one": 1}[kind]


def _pos_tensor(value, kind, other=-7):
    """The device ``pos``: ``int32 [1]``, the production ``int32 [2]`` (decode, prefill) pair
    with a second element the kernel must ignore, or ``int64 [1]``."""
    values, dtype = {
        "int32": ([value], torch.int32),
        "pair": ([value, other], torch.int32),
        "int64": ([value], torch.int64),
    }[kind]
    return torch.tensor(values, dtype=dtype, device=DEVICE)


def _trail(row, shape):
    return {"1d": (), "2d": (row,), "3d": (1, row)}[shape]


def _row_guard(row, dtype):
    return max(row * dtype.itemsize, 4096)


def _run_and_check(x, n_out, pos, pos_kind, layout, what):
    """Copy ``x`` into a placed input, slice into a placed poisoned output, check everything."""
    dtype, trail = x.dtype, tuple(x.shape[1:])
    guard = _row_guard(math.prod(trail), dtype)
    src = cb.Placed(x.shape, dtype, layout, guard_bytes=guard, device=DEVICE)
    src.tensor.copy_(x)
    out = cb.Placed((n_out, *trail), dtype, layout, guard_bytes=guard, device=DEVICE)
    src_before = src.full.clone()
    expected = out.full.clone()
    k = max(0, min(x.shape[0] - pos, n_out))
    out.view_of(expected)[:k] = x[pos : pos + k]
    pos_on_device = _pos_tensor(pos, pos_kind)
    pos_before = pos_on_device.clone()

    tensor_get_slice_after(src.tensor, out.tensor, pos_on_device)
    torch.cuda.synchronize()

    ou.check_guards(input=src.buf, output=out.buf)
    ou.assert_exact(out.full, expected, what=f"{what}: out[:{k}] = x[{pos}:], rest untouched")
    ou.assert_exact(src.full, src_before, what="input is read-only")
    ou.assert_exact(pos_on_device, pos_before, what="pos is read-only")


SANITIZER_CASE = dict(
    row=1025,
    pos="mid",
    n_in=17,
    n_out="short",
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
def test_slice_after_matches_contract(case):
    g = ou.make_generator(device=DEVICE)
    n_in = case["n_in"]
    pos = _pos_value(case["pos"], n_in)
    n_out = _n_out(case["n_out"], n_in, pos)
    x = cb.random_bits((n_in, *_trail(case["row"], case["shape"])), case["dtype"], g)
    _run_and_check(x, n_out, pos, case["pos_buffer"], case["layout"], what=_case_id(case))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("n_in,pos", [(1, 0), (1, 1), (16, 3), (17, 16), (17, 0), (33, 17)])
def test_one_dimensional_rows(n_in, pos):
    """A 1-D tensor has rows of one element (``ROW_SIZE = BLOCK_SIZE = 1``)."""
    g = ou.make_generator(device=DEVICE)
    x = cb.random_bits((n_in,), torch.float32, g)
    _run_and_check(x, n_in, pos, "pair", "contiguous", what=f"1-D, N_in {n_in}, pos {pos}")


@ou.requires_sm(ou.HOPPER)
def test_empty_input_is_a_no_op():
    """``N_in = 0``: the launcher does not launch; the output stays poisoned."""
    g = ou.make_generator(device=DEVICE)
    x = cb.random_bits((0, 1, 2049), torch.bfloat16, g)
    _run_and_check(x, 4, 0, "pair", "contiguous", what="N_in = 0")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "tokens,row,decode_tokens",
    [(8192, IN_PROJ_8B, 1024), (4096, IN_PROJ_56B, 333)],
    ids=["8B-in_proj-8192-tokens", "56B-in_proj-4096-tokens"],
)
def test_slice_at_production_sizes(tokens, row, decode_tokens):
    """The production call: ``[T, 1, in_proj]`` bf16 into ``empty_like``, ``pos`` the real
    decode tokens in the context's ``int32 [2]`` pair."""
    g = ou.make_generator(device=DEVICE)
    x = cb.random_bits((tokens, 1, row), torch.bfloat16, g)
    _run_and_check(x, tokens, decode_tokens, "pair", "contiguous", what=f"{tokens} x {row}")


@ou.requires_sm(ou.HOPPER)
def test_cuda_graph_replay_follows_pos():
    """Captured once (as in a non-decode CUDA graph), replayed with ``pos`` across the whole
    range on fresh data each time; the output is re-poisoned before every replay, so rows past
    the copy must stay untouched on every replay too."""
    n_in, row, dtype = 37, 2049, torch.bfloat16
    g = ou.make_generator(device=DEVICE)
    guard = _row_guard(row, dtype)
    src = cb.Placed((n_in, 1, row), dtype, guard_bytes=guard, device=DEVICE)
    out = cb.Placed((n_in, 1, row), dtype, guard_bytes=guard, device=DEVICE)
    src.tensor.copy_(cb.random_bits(src.shape, dtype, g))
    pos_on_device = torch.tensor([5, n_in - 5], dtype=torch.int32, device=DEVICE)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tensor_get_slice_after(src.tensor, out.tensor, pos_on_device)  # compile outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tensor_get_slice_after(src.tensor, out.tensor, pos_on_device)

    for pos in (n_in, 0, 1, 5, 16, 17, n_in - 1, 2, n_in // 2, 32):
        x = cb.random_bits(src.shape, dtype, g)
        src.tensor.copy_(x)
        ou.poison_(out.full)
        pos_on_device.copy_(torch.tensor([pos, n_in - pos], dtype=torch.int32))
        expected = out.full.clone()
        out.view_of(expected)[: n_in - pos] = x[pos:]
        graph.replay()
        torch.cuda.synchronize()
        ou.check_guards(input=src.buf, output=out.buf)
        ou.assert_exact(out.full, expected, what=f"replay with pos {pos}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "tokens",
    [p for p in ou.int32_overflow_params(IN_PROJ_8B) if (p.values[0] - 1) * IN_PROJ_8B < 2**31],
)
def test_slice_offsets_past_int32(tokens):
    """Element offsets ``row * 18560`` (the 8B ``in_proj`` at TP 1) up to and past 2**31, with
    ``pos = 0`` so the last rows are both read and written. Both tensors sit behind a 2**31-element
    poisoned front guard, so a wrapped int32 offset reads poison or writes into the guard instead
    of faulting. A row that starts past 2**31 fails on main (finding O6-M1) and is #9307's
    regression case; it is reachable only with ``max_tokens`` above about 115,700 for this width
    (default 16384).
    About 26 GB: 4.3 GB per tensor and per guard, plus the comparison."""
    row, dtype = IN_PROJ_8B, torch.bfloat16
    numel = tokens * row
    ou.require_free_memory(6 * numel * dtype.itemsize + 2**30)
    x, x_guard = cb.behind_front_guard((tokens, 1, row), dtype, device=DEVICE)
    g = ou.make_generator(device=DEVICE)
    x.copy_(cb.random_bits((tokens, 1, row), dtype, g))
    out, out_guard = cb.behind_front_guard((tokens, 1, row), dtype, device=DEVICE)
    ou.poison_(out)
    pos_on_device = torch.tensor([0, tokens], dtype=torch.int32, device=DEVICE)

    tensor_get_slice_after(x, out, pos_on_device)
    torch.cuda.synchronize()

    stray = cb.changed(out_guard)
    assert stray == 0, f"{stray} elements written into the 2**31 elements before the output"
    assert cb.changed(x_guard) == 0, "the input's front guard was written"
    ou.assert_exact(out, x, what=f"{tokens} rows of {row}: out = x[0:]")

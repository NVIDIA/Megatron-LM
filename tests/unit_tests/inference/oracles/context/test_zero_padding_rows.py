# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_zero_padding_rows_kernel``
(megatron/core/inference/contexts/zero_padding_rows_kernel.py, launcher ``zero_padding_rows``,
called through ``DynamicInferenceContext.zero_padding_tokens``).

**Contract** (from the context's padding semantics). Dynamic inference pads the token
dimension: eagerly up to a rounded size, and on CUDA-graph steps up to the captured size.
``zero_padding_rows(x, real_token_count, first_token_idx)`` zeroes, in place, every row ``r`` of
the token-major activation ``x`` (``[T, ...]``) whose token index ``r + first_token_idx`` is at
or past the real token count, read **on device** from the ``int32 [1]`` tensor at the time the
kernel runs (so a captured graph follows the count of each replay). ``first_token_idx`` is the
sequence-parallel shard's offset (``rank * T``). Zeroed rows are ``+0.0`` whatever they held
(NaN included: the activation is never read); real rows keep their bits. Degenerate: an empty
``x`` is a no-op; a count of 0 zeroes every row, a count at or past the shard's end none. The
launcher runs the kernel for contiguous CUDA tensors and a ``masked_fill_`` otherwise; both
must meet the contract. Tolerance class: copy/store, **exact** (bitwise).

**Oracle.** A torch reference: ``expected[token_idx >= count] = 0`` on a clone, from
``torch.arange`` token indices.

**Inputs from the kernel's structure.** ``BLOCK_M = 8`` rows and ``BLOCK_N = min(1024,
next_pow2(row_numel))`` elements per program, grid ``(cdiv(T, 8), cdiv(row_numel, BLOCK_N))``
(``zero_padding_rows_kernel.py:72-75``): rows T in 1/7/8/9/15/16/17 and a production 2048;
row sizes 1, 1023/1024/1025, 2047/2048/2049 and the production hidden sizes 4096, 7168 and 8192
(several column tiles, which no earlier test ran); counts 0, 1, around every row tile
(7/8/9, 15/16/17), T - 1, T, T + 1 and far past T; shard offsets of rank 0, 1 and 3; bf16,
fp16 and fp32; 2-D ``[T, H]`` and 3-D ``[T, 1, H]`` activations; the fallback for a
non-contiguous view. Factors combined pairwise. **CUDA-graph replay with the count changing**:
captured with the count 0 (as at capture time in the engine), replayed with counts across the
whole range and fresh activations, unsharded and as an SP shard. int32 limits: offsets
``row * row_numel`` past 2**31 elements (``large``).

**Memory.** ``x`` is guard-banded with a whole tile of rows; real rows carry NaNs, infinities
and ``-0.0`` that must survive bit for bit, padding rows start NaN. One small case runs under
compute-sanitizer.

**Hardware.** H100. No open PR changes this kernel.
"""

import pytest
import torch

from megatron.core.inference.contexts.zero_padding_rows_kernel import zero_padding_rows
from tests.unit_tests.inference.oracles import oracle_utils as ou

DEVICE = "cuda"

BLOCK_M = 8
BLOCK_N = 1024

ROWS = ou.boundary_sizes(BLOCK_M, minimum=1, production=(2048,))
ROW_NUMELS = ou.boundary_sizes(BLOCK_N, minimum=1, production=(4096, 7168, 8192))
COUNTS = ("zero", "one", "tile-1", "tile", "tile+1", "rows-1", "rows", "rows+1", "far")
"""Count relative to the shard: ``tile`` is the first multiple of BLOCK_M past its start."""


def _count(kind, rows, first):
    tile = first + BLOCK_M
    return {
        "zero": 0,
        "one": first + 1,
        "tile-1": tile - 1,
        "tile": tile,
        "tile+1": tile + 1,
        "rows-1": first + rows - 1,
        "rows": first + rows,
        "rows+1": first + rows + 1,
        "far": first + 100 * rows,
    }[kind]


CASES = ou.pairwise_cases(
    {
        "row_numel": ROW_NUMELS,
        "rows": ROWS,
        "count": COUNTS,
        "dtype": (torch.bfloat16, torch.float16, torch.float32),
        "sp_rank": (0, 1, 3),
        "shape": ("2d", "3d"),
    }
)


def _case_id(c):
    dtype = str(c["dtype"]).replace("torch.", "")
    return f"T{c['rows']}-H{c['row_numel']}-{c['count']}-{dtype}-rank{c['sp_rank']}-{c['shape']}"


def _activation(rows, row_numel, dtype, shape, g):
    """Random rows with NaN, +-inf and -0.0 sprinkled in (real rows must keep their bits)."""
    x = torch.randn(rows, row_numel, generator=g, dtype=torch.float32)
    specials = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0])
    pick = torch.randint(0, 50, (rows, row_numel), generator=g)
    x = torch.where(pick < 4, specials[pick.clamp(max=3)], x).to(dtype)
    return x if shape == "2d" else x.reshape(rows, 1, row_numel)


def _reference(x, count, first):
    expected = x.clone()
    token = torch.arange(first, first + x.shape[0])
    expected[token >= count] = 0
    return expected


def _guarded(x):
    row_bytes = x[0].numel() * x.dtype.itemsize
    return ou.guarded_like(x.to(DEVICE), guard_bytes=BLOCK_M * max(row_bytes, 1))


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


SANITIZER_CASE = dict(
    row_numel=2049, rows=17, count="tile+1", dtype=torch.bfloat16, sp_rank=1, shape="3d"
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
def test_zero_padding_rows_matches_contract(case):
    g = ou.make_generator()
    rows, first = case["rows"], case["sp_rank"] * case["rows"]
    count = _count(case["count"], rows, first)
    x = _activation(rows, case["row_numel"], case["dtype"], case["shape"], g)
    expected = _reference(x, count, first)
    pad = torch.arange(first, first + rows) >= count
    x[pad] = float("nan")  # padding rows are never read

    buf = _guarded(x)
    real_token_count = torch.tensor([count], dtype=torch.int32, device=DEVICE)
    zero_padding_rows(buf.tensor, real_token_count, first_token_idx=first)
    _sync()
    ou.check_guards(x=buf)
    ou.assert_exact(buf.tensor, expected, what=f"count {count}, first token {first}")
    ou.assert_exact(real_token_count, torch.tensor([count]), what="count is read-only")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("count", [0, 5, 9, 17])
def test_non_contiguous_activation_meets_the_same_contract(count):
    """The launcher's fallback (a strided view is not handed to the kernel)."""
    g = ou.make_generator()
    full = _activation(17, 2 * 1025, torch.bfloat16, "2d", g).to(DEVICE)
    view = full[:, ::2]
    expected_full = full.clone()
    expected_full[:, ::2] = _reference(view.cpu(), count, 0).to(DEVICE)
    zero_padding_rows(view, torch.tensor([count], dtype=torch.int32, device=DEVICE))
    ou.assert_exact(full, expected_full, what="strided view")


@ou.requires_sm(ou.HOPPER)
def test_empty_activation_is_a_no_op():
    x = ou.guarded_empty((0, 4096), torch.bfloat16, DEVICE)
    zero_padding_rows(x.tensor, torch.zeros(1, dtype=torch.int32, device=DEVICE))
    _sync()
    ou.check_guards(x=x)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("sp_rank", [0, 2])
def test_cuda_graph_replay_follows_the_count(sp_rank):
    """Captured once with count 0 (as the engine captures), replayed with counts across the
    shard and around every row tile, each time on a fresh activation."""
    rows, row_numel = 37, 2049
    first = sp_rank * rows
    g = ou.make_generator()
    buf = _guarded(_activation(rows, row_numel, torch.bfloat16, "3d", g))
    count = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        zero_padding_rows(buf.tensor, count, first_token_idx=first)  # compile outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        zero_padding_rows(buf.tensor, count, first_token_idx=first)

    replay_counts = [first + n for n in (rows, 5, 8, 9, rows - 1, 1, 3 * rows, 16, 17, 7)]
    replay_counts[2:2] = [0, max(first - 1, 0)]
    for value in replay_counts:
        x = _activation(rows, row_numel, torch.bfloat16, "3d", g)
        expected = _reference(x, value, first)
        buf.tensor.copy_(x)
        count.fill_(value)
        graph.replay()
        _sync()
        ou.check_guards(x=buf)
        ou.assert_exact(buf.tensor, expected, what=f"replay with count {value}, first {first}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("rows", ou.int32_overflow_params(8192))
def test_zero_padding_offsets_past_int32(rows):
    """Element offsets ``row * 8192 + col`` up to and past 2**31 (bf16, 4 GiB)."""
    row_numel = 8192
    ou.require_free_memory(2 * rows * row_numel * 2 + 2**30)
    x = torch.ones((rows, row_numel), dtype=torch.bfloat16, device=DEVICE)
    count = rows - 3
    zero_padding_rows(x, torch.tensor([count], dtype=torch.int32, device=DEVICE))
    _sync()
    nonzero_tail = int(torch.count_nonzero(x[count:]))
    assert nonzero_tail == 0, f"{nonzero_tail} elements of the last 3 rows were not zeroed"
    zeroed_real = int((x[:count] != 1).sum())
    assert zeroed_real == 0, f"{zeroed_real} elements of real rows changed"

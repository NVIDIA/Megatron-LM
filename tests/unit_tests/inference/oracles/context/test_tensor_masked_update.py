# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_tensor_masked_update_kernel_2d`` / ``_3d`` / ``_4d``
(megatron/core/inference/contexts/attention_context/triton/tensor_ops.py, launcher
``tensor_masked_update``): the write-back of a hybrid model's prefill conv and SSM states.

**Contract** (from the call sites, the varlen prefill of ``MambaMixer``, ``GatedDeltaNet`` and
``GatedDeltaProduct``, and the launcher's docstring). After a prefill step each prefill request's
final state must replace its slot in the layer's state cache, and padded request rows (``-1``)
must change nothing: ``tensor_masked_update(states, idx, new_states)`` does, for every ``b``
with ``idx[b] != -1``,

    states[idx[b]] = new_states[b]

bit for bit, over all trailing dimensions; slots no ``idx`` entry names, ``new_states`` and
``idx`` are untouched. ``states [N, ...]`` and ``new_states [B, ...]``: CUDA, the same trailing
shape (asserted), 2, 3 or 4 dimensions (asserted), the same dtype (production; the kernel would
cast silently), **any strides** (the kernel takes every stride). ``idx [B]``: int64 or int32
(``batch_indices`` is int32 for prefill), ``-1`` or a slot in ``[0, N)``, the slots unique
(preconditions, unchecked). Production shapes: conv states ``[N, conv_dim, d_conv]`` (3-D) and
Mamba SSM states ``[N, nheads, headdim, d_state]`` (4-D), each one layer's slice of the
context's cache. ``B = 0`` does nothing. Tolerance class: copy, **exact**.

**Oracle.** Torch indexing on clones (``expected[idx[b]] = new_states[b]``, one row at a time),
which shares none of the kernel's flatten-and-stride arithmetic.

**Inputs from the kernel's structure.** One kernel per rank; grid ``(B, cdiv(row, 1024))``, one
1024-element chunk of the flattened row per program, with the row's coordinates rebuilt from
``SIZE_D*`` and every stride passed at run time (``tensor_ops.py:412-475``). Factors, combined
pairwise: rank 2/3/4; rows 1023/1024/1025 and 2047/2048/2049 elements (one, two and three
chunks), with 3-D and 4-D shapes whose dimensions read through ``.to`` are above 1 (a leading
``D0`` of 1 occurs; see O6-M3 below); fp32, bf16, fp16; int64 and int32 ``idx``; all rows
valid, some ``-1``, all ``-1``, ``-1`` at both ends;
``B`` 1, 5, 17; states and new states each contiguous, one element in (unaligned base),
padded (every other slot, padded trailing dims) or with the last two dims transposed (non-unit
innermost stride), every pair of the two layouts at every rank. Also ``B = 0`` for each rank;
**CUDA-graph replay** with ``idx`` and the new states changing (the call is captured in
non-decode graphs); production sizes (Nemotron-H 8B and 56B conv and SSM states at TP 1); int32
limits (``large``: a slot past 2**31 elements, int64 offsets, expected to pass).

**Memory.** States in a guard-banded allocation (guards of one slot, NaN sentinels): the whole
allocation, gaps of strided layouts included, must equal the expected one. New states and
``idx`` are checked unchanged. One small case runs under compute-sanitizer.

**Hardware.** H100. Finding O6-M3: on main the kernels call ``.to(tl.int64)`` on runtime int
arguments (``stride_new_b``, ``stride_state_b``, ``SIZE_D1``, ``SIZE_D2``), and Triton 3.8
compiles an int argument equal to 1 as a ``constexpr`` without ``.to``, so a unit trailing
dimension or a one-element row fails to compile. Production states never have one (``d_conv``,
``headdim`` and ``d_state`` are above 1). #9306 (fix/masked-update-unit-dims) fixes it and
carries those shapes as its regression test
(``test_tensor_ops.py::test_tensor_masked_update_unit_dimensions``).
"""

import math

import pytest
import torch

from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_masked_update,
)
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.context import copy_buffers as cb

DEVICE = "cuda"

CHUNK = 1024
"""Elements of the flattened row per program (``tensor_ops.py:418``)."""

ROWS = ou.boundary_sizes(CHUNK, degenerate=False)

SHAPES = {
    2: {n: (n,) for n in ROWS},
    3: {
        1023: (33, 31),
        1024: (256, 4),
        1025: (205, 5),
        2047: (89, 23),
        2048: (512, 4),
        2049: (683, 3),
    },
    4: {
        1023: (3, 11, 31),
        1024: (4, 16, 16),
        1025: (5, 5, 41),
        2047: (1, 23, 89),
        2048: (2, 32, 32),
        2049: (1, 3, 683),
    },
}
"""Trailing shapes per rank and row size; every dimension the kernel reads through ``.to`` is
above 1 (O6-M3)."""

PATTERNS = ("all", "mixed", "none", "ends")

LAYOUT_PAIRS = [(s, n) for s in cb.LAYOUTS for n in cb.LAYOUTS]
"""(states, new states) layouts: one factor, so every rank meets every pair (each rank's
kernel addresses the two tensors with their own strides)."""

CASES = [
    dict(c, states_layout=c["layouts"][0], new_layout=c["layouts"][1])
    for c in ou.pairwise_cases(
        {
            "layouts": LAYOUT_PAIRS,
            "ndim": (2, 3, 4),
            "row": ROWS,
            "pattern": PATTERNS,
            "dtype": (torch.float32, torch.bfloat16, torch.float16),
            "idx_dtype": (torch.int64, torch.int32),
            "updates": (1, 5, 17),
        }
    )
]


def _case_id(c):
    dtype = str(c["dtype"]).replace("torch.", "")
    idx = str(c["idx_dtype"]).replace("torch.", "")
    return (
        f"{c['ndim']}d-row{c['row']}-B{c['updates']}-{c['pattern']}-{dtype}-{idx}"
        f"-states_{c['states_layout']}-new_{c['new_layout']}"
    )


def _indices(pattern, updates, slots, dtype, g):
    """``updates`` entries: unique slots of ``[0, slots)`` or ``-1`` by ``pattern``."""
    idx = torch.randperm(slots, generator=g)[:updates]
    if pattern == "none":
        idx[:] = -1
    elif pattern == "mixed":
        skip = torch.rand(updates, generator=g) < 0.5
        skip[0], skip[-1] = False, updates > 1
        idx[skip] = -1
    elif pattern == "ends" and updates > 2:
        idx[0] = idx[-1] = -1
    return idx.to(dtype)


def _guard(shape, dtype):
    return max(math.prod(shape[1:]) * dtype.itemsize, 4096)


def _update_and_check(states, idx, new, what):
    """Run the launcher on placed ``states`` / ``new`` views; compare whole allocations."""
    expected = states.full.clone()
    expected_view = states.view_of(expected)
    for b, slot in enumerate(idx.tolist()):
        if slot != -1:
            expected_view[slot] = new.tensor[b]
    new_before, idx_device = new.full.clone(), idx.to(DEVICE)

    tensor_masked_update(states.tensor, idx_device, new.tensor)
    torch.cuda.synchronize()

    ou.check_guards(states=states.buf, new_states=new.buf)
    ou.assert_exact(states.full, expected, what=what)
    ou.assert_exact(new.full, new_before, what="new states are read-only")
    ou.assert_exact(idx_device, idx, what="idx is read-only")


def _placed_random(shape, dtype, layout, g):
    p = cb.Placed(shape, dtype, layout, guard_bytes=_guard(shape, dtype), device=DEVICE)
    p.full.copy_(cb.random_bits(p.full.shape, dtype, g))
    return p


SANITIZER_CASE = dict(
    row=1025,
    ndim=4,
    states_layout="padded",
    new_layout="transposed",
    pattern="mixed",
    dtype=torch.bfloat16,
    idx_dtype=torch.int32,
    updates=5,
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
def test_masked_update_matches_contract(case):
    g = ou.make_generator(device=DEVICE)
    g_cpu = ou.make_generator()
    trail = SHAPES[case["ndim"]][case["row"]]
    updates, dtype = case["updates"], case["dtype"]
    slots = updates + 3
    states = _placed_random((slots, *trail), dtype, case["states_layout"], g)
    new = _placed_random((updates, *trail), dtype, case["new_layout"], g)
    idx = _indices(case["pattern"], updates, slots, case["idx_dtype"], g_cpu)
    _update_and_check(states, idx, new, what=_case_id(case))


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_no_updates_is_a_no_op(ndim):
    """``B = 0``: a grid with no programs; nothing changes."""
    g = ou.make_generator(device=DEVICE)
    trail = SHAPES[ndim][2049]
    states = _placed_random((4, *trail), torch.float32, "contiguous", g)
    new = _placed_random((0, *trail), torch.float32, "contiguous", g)
    idx = torch.empty(0, dtype=torch.int64)
    _update_and_check(states, idx, new, what=f"{ndim}-D, B = 0")


@ou.requires_sm(ou.HOPPER)
def test_cuda_graph_replay_follows_indices():
    """Captured once (as in a non-decode CUDA graph), replayed with new ``idx`` values (slots
    changing, rows switching to and from ``-1``, all ``-1``) and new states each time."""
    slots, updates, trail, dtype = 12, 6, (4, 16, 33), torch.float32
    g = ou.make_generator(device=DEVICE)
    g_cpu = ou.make_generator()
    states = _placed_random((slots, *trail), dtype, "contiguous", g)
    new = _placed_random((updates, *trail), dtype, "contiguous", g)
    idx = _indices("mixed", updates, slots, torch.int32, g_cpu).to(DEVICE)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tensor_masked_update(states.tensor, idx, new.tensor)  # compile outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tensor_masked_update(states.tensor, idx, new.tensor)

    for pattern in ("all", "mixed", "none", "ends", "mixed"):
        step_idx = _indices(pattern, updates, slots, torch.int32, g_cpu)
        new.tensor.copy_(cb.random_bits(new.shape, dtype, g))
        idx.copy_(step_idx)
        expected = states.full.clone()
        for b, slot in enumerate(step_idx.tolist()):
            if slot != -1:
                expected[slot] = new.tensor[b]
        graph.replay()
        torch.cuda.synchronize()
        ou.check_guards(states=states.buf, new_states=new.buf)
        ou.assert_exact(states.full, expected, what=f"replay, idx {step_idx.tolist()}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "trail,dtype",
    [
        ((10240, 4), torch.bfloat16),
        ((128, 64, 128), torch.float32),
        ((20480, 4), torch.float32),
        ((256, 64, 256), torch.float32),
    ],
    ids=["8B-conv", "8B-ssm", "56B-conv-fp32", "56B-ssm"],
)
def test_masked_update_at_production_sizes(trail, dtype):
    """One layer's conv state ``[slots, conv_dim, d_conv]`` or SSM state ``[slots, nheads,
    headdim, d_state]`` of Nemotron-H 8B / 56B at TP 1, 8 padded prefill rows of which two are
    ``-1``, int32 ``batch_indices`` as in production."""
    g = ou.make_generator(device=DEVICE)
    g_cpu = ou.make_generator()
    slots, updates = 12, 8
    states = _placed_random((slots, *trail), dtype, "contiguous", g)
    new = _placed_random((updates, *trail), dtype, "contiguous", g)
    idx = torch.randperm(slots, generator=g_cpu)[:updates].to(torch.int32)
    idx[-2:] = -1
    _update_and_check(states, idx, new, what=f"production {trail}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("slots", ou.int32_overflow_params(2**20))
def test_masked_update_offsets_past_int32(slots):
    """SSM states of 2**20 elements (Nemotron-H 8B, 128 x 64 x 128 fp32) with the updated slot
    ending at or starting past 2**31 elements; the kernel's offsets are int64, so this should
    pass. 8.6 GB."""
    trail, dtype = (128, 64, 128), torch.float32
    row = math.prod(trail)
    ou.require_free_memory(slots * row * dtype.itemsize + 2**31)
    g = ou.make_generator(device=DEVICE)
    states = torch.zeros((slots, *trail), dtype=dtype, device=DEVICE)
    new = cb.random_bits((2, *trail), dtype, g)
    idx = torch.tensor([slots - 1, 0], dtype=torch.int32, device=DEVICE)

    tensor_masked_update(states, idx, new)
    torch.cuda.synchronize()

    ou.assert_exact(states[slots - 1], new[0], what=f"slot {slots - 1}")
    ou.assert_exact(states[0], new[1], what="slot 0")
    written = int(torch.count_nonzero(states[1 : slots - 1]))
    assert written == 0, f"{written} elements of slots 1..{slots - 2} were written"

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_fill_expert_block_ids_kernel``
(``megatron/core/inference/moe/vllm_fused_moe.py:394``), launched directly.

Contract
--------
Written from the block alignment's contract (the grouped GEMM reads ``expert_ids[b]`` as the
expert of padded block ``b``, ``_fused_moe_kernel``) and the launcher
(``_moe_align_block_size_cuda_graphable``), before reading the kernel body.

* **Computes**, for each local expert ``e`` (one program per expert): ``expert_ids[b] = e`` for
  every block ``b`` in ``[exclusive[e] / BLOCK_SIZE_M, inclusive[e] / BLOCK_SIZE_M)``.
* **Inputs**: ``exclusive_offsets`` and ``inclusive_offsets``, int32 ``[E_local]``, multiples of
  ``BLOCK_SIZE_M`` (the aligned prefix sums of the per-expert pair counts: an empty expert has
  ``inclusive == exclusive``); ``expert_ids``, int32, long enough for the last segment.
* **Outputs**: only those blocks are written. Every other entry keeps its value (the init
  kernel's ``-1``), so blocks past the last segment stay ``-1`` and the GEMM skips them. The
  offsets are read-only.
* **Degenerate**: empty experts (first, last, every one), a single expert, one-block segments.
* **Tolerance class**: exact (integer table).

Variants and boundaries (from the kernel and its launcher, after the contract)
--------------------------------------------------------------------------------
* ``BLOCK_SIZE_M`` constexpr: 16, 32, 64 and 128, every value ``_get_default_config`` picks.
* The store loop covers ``BLOCK = 128`` blocks per iteration (the launcher's constant, ``:481``):
  segments of 0, 1, 2, 127, 128, 129, 255, 256, 257 and 385 blocks run 0 to 4 iterations, with a
  partial last one.
* Grid ``E_local``: 1, 2, 7, 16, 128 and 256 programs.
* Layouts: ``packed`` is the launcher's (each segment starts where the previous one ends);
  ``gaps`` leaves poisoned blocks before and between the segments, so a write one block past a
  segment cannot hide behind the next expert's write, and a write before one shows too.
* int32: block indices are below ``max_blocks``, an int32 table length; no offset reaches 2**31
  (no ``large`` case).

Why a direct test
-----------------
The entry oracle (``test_vllm_fused_moe.py``) sees this table only through the MoE output. A
block written one past its segment is overwritten by the next expert (a race on GPU, never a
wrong value once the race is lost) or lies past ``num_tokens_post_padded``, where the GEMM never
looks, so such a fill is invisible there; segments of 128 or more blocks occur there in one
grid-stride case only, and never at BLOCK_SIZE_M 32 to 128.

Hardware: H100 (integer stores; nothing architecture-specific).
"""

import dataclasses
import importlib
from typing import List, Optional

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.moe_vllm import align_reference as ar

vfm = importlib.import_module("megatron.core.inference.moe.vllm_fused_moe")

DEVICE = torch.device("cuda")

FILL_BLOCK = 128  # blocks per store-loop iteration: the launcher's BLOCK (vllm_fused_moe.py:481)
BLOCK_MS = (16, 32, 64, 128)  # every BLOCK_SIZE_M of _get_default_config
LOOP_SEGMENTS = ou.boundary_sizes(FILL_BLOCK, multiples=(2,), production=(385,))  # 0 ... 385


@dataclasses.dataclass(frozen=True)
class Case:
    """Per-expert segment lengths in blocks, laid out packed or with poisoned gaps."""

    blocks: tuple
    block_m: int = 16
    layout: str = "packed"  # "packed" | "gaps"
    tail: int = 3  # poisoned blocks past the last segment

    def id(self) -> str:
        if len(self.blocks) <= 6:
            lens = "-".join(str(b) for b in self.blocks)
        else:
            lens = f"{len(self.blocks)}experts-{sum(self.blocks)}blocks"
        return f"bm{self.block_m}-{self.layout}-tail{self.tail}-[{lens}]"


def segments(case: Case, g: torch.Generator):
    """Block ranges [lo, hi) per expert; gaps of 0 to 2 blocks in the ``gaps`` layout."""
    if case.layout == "gaps":
        gaps = torch.randint(1, 3, (len(case.blocks),), generator=g, device=DEVICE).tolist()
    else:
        gaps = [0] * len(case.blocks)
    lows, highs, position = [], [], 0
    for gap, n in zip(gaps, case.blocks):
        position += gap
        lows.append(position)
        position += n
        highs.append(position)
    return lows, highs, position + case.tail


def run_fill(case: Case, seed: int = 0) -> None:
    """Launch the kernel as the launcher does and check every entry of the table."""
    g = ou.make_generator(ou.DEFAULT_SEED + seed, device=DEVICE)
    lows, highs, length = segments(case, g)
    bm = case.block_m
    exclusive = ou.guarded_like(torch.tensor(lows, dtype=torch.int32, device=DEVICE) * bm)
    inclusive = ou.guarded_like(torch.tensor(highs, dtype=torch.int32, device=DEVICE) * bm)
    # Guards of at least one store-loop iteration on each side.
    expert_ids = ou.guarded_empty((length,), torch.int32, DEVICE, guard_bytes=4 * FILL_BLOCK * 4)
    before = (exclusive.tensor.clone(), inclusive.tensor.clone())

    vfm._fill_expert_block_ids_kernel[(len(case.blocks),)](
        expert_ids.tensor, exclusive.tensor, inclusive.tensor, BLOCK_SIZE_M=bm, BLOCK=FILL_BLOCK
    )

    expected = ar.expected_expert_ids(
        [lo * bm for lo in lows], [hi * bm for hi in highs], bm, length, ar.INT32_MIN
    )
    what = case.id()
    ou.assert_exact(expert_ids.tensor, expected, what=f"{what}: expert_ids")
    ou.check_guards(expert_ids=expert_ids, exclusive=exclusive, inclusive=inclusive)
    ou.assert_exact(exclusive.tensor, before[0], what=f"{what}: exclusive offsets are read-only")
    ou.assert_exact(inclusive.tensor, before[1], what=f"{what}: inclusive offsets are read-only")


# Segment lengths around the 128-block store loop, cycled over the experts.
CYCLE = (0, 1, 2, 127, 128, 129, 255, 256, 257)


def cycled(num_local: int, offset: int) -> tuple:
    return tuple(CYCLE[(offset + i) % len(CYCLE)] for i in range(num_local))


PAIRWISE = [
    Case(blocks=cycled(c["num_local"], c["offset"]), block_m=c["block_m"], layout=c["layout"])
    for c in ou.pairwise_cases(
        {
            "num_local": [1, 2, 7, 16, 128],
            "block_m": list(BLOCK_MS),
            "layout": ["packed", "gaps"],
            "offset": [0, 3, 5, 7],
        }
    )
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id()) for c in PAIRWISE])
def test_fills_each_segment_and_nothing_else(case):
    """Every block of each segment holds its expert; every other entry keeps its poison."""
    run_fill(case)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("layout", ["packed", "gaps"])
@pytest.mark.parametrize("block_m", BLOCK_MS)
@pytest.mark.parametrize("blocks", LOOP_SEGMENTS)
def test_segment_lengths_around_the_store_loop(blocks, block_m, layout):
    """One long segment between an empty first expert and a one-block last expert.

    0 to 385 blocks: B-1/B/B+1 and 2B-1/2B/2B+1 of the 128-block store loop.
    """
    run_fill(Case(blocks=(0, blocks, 1), block_m=block_m, layout=layout, tail=FILL_BLOCK + 1))


DEGENERATE = {
    "one-expert-empty": Case(blocks=(0,)),
    "one-expert-one-block": Case(blocks=(1,), tail=0),
    "all-empty": Case(blocks=(0,) * 16, block_m=64),
    "ends-empty": Case(blocks=(0, 3, 1, 0), block_m=128, layout="gaps"),
    "no-tail": Case(blocks=(2, 129), block_m=32, tail=0),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(DEGENERATE))
def test_degenerate(name):
    """Empty experts, a single expert, a last segment that ends at the table's end."""
    run_fill(DEGENERATE[name])


def production_blocks(
    num_local: int, pairs: int, block_m: int, g: torch.Generator, hot: Optional[int] = None
) -> tuple:
    """Blocks per expert for ``pairs`` local pairs spread at random (optionally one hot expert)."""
    weights = torch.rand(num_local, generator=g, device=DEVICE)
    if hot is not None:
        weights[hot] = weights.sum()
    draws = torch.multinomial(weights, pairs, replacement=True, generator=g)
    counts = torch.bincount(draws, minlength=num_local)
    return tuple(-(-int(c) // block_m) for c in counts.tolist())


PRODUCTION = {
    # Nemotron 3 Nano at EP 1: 128 experts, top-6, a 8192-token prefill (BLOCK_SIZE_M 128).
    "nemotron-nano-ep1-prefill": (128, 8192 * 6, 128, None),
    # DeepSeek-V3 at EP 1: 256 experts, top-8, 8192 tokens, one hot expert.
    "deepseek-v3-ep1-prefill-hot": (256, 8192 * 8, 128, 7),
    # Qwen3-30B-A3B at EP 8 decode: 64 tokens per rank gathered over 8 ranks, top-8 over 128
    # experts, so about 512 pairs on this rank's 16 experts (BLOCK_SIZE_M 64).
    "qwen3-30b-a3b-ep8-decode": (16, 512, 64, None),
}


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_sizes(name):
    """Deployed expert counts and batch sizes."""
    num_local, pairs, block_m, hot = PRODUCTION[name]
    g = ou.make_generator(device=DEVICE)
    run_fill(Case(blocks=production_blocks(num_local, pairs, block_m, g, hot), block_m=block_m))


@pytest.mark.sanitizer
@ou.requires_sm(ou.HOPPER)
def test_sanitizer():
    """A small case for compute-sanitizer memcheck (also run normally)."""
    run_fill(Case(blocks=(0, 129, 1, 0, 2), block_m=32, layout="gaps", tail=0))


def all_cases() -> List[Case]:
    """Every case of this module (for scratch checks outside pytest)."""
    cases = list(PAIRWISE) + list(DEGENERATE.values())
    cases += [
        Case(blocks=(0, b, 1), block_m=bm, layout=lay, tail=FILL_BLOCK + 1)
        for b in LOOP_SEGMENTS
        for bm in BLOCK_MS
        for lay in ("packed", "gaps")
    ]
    return cases

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of the GTP prefetch plan: the planner (``_plan_prefetches``) and the plan's cursor
(``_PrefetchPlan``).

The plan is a pure function of the recorded consumes, so its invariants are checked by replaying
it on a model of the gather buffers.
"""

import random

import pytest

from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    _plan_prefetches,
    _PrefetchPlan,
)


def _recorded(order, buffers=None, pinned=()):
    """Recorded consumes of weight names; each gathers into ``buffers[name]`` (default: its own).

    ``pinned`` weights must not be prefetched (the embedding).
    """
    buffers = buffers or {}
    return [(name, buffers.get(name, name), name not in pinned) for name in order]


def _replay(consumes, prefetches):
    """Replay ``prefetches`` on the buffers; returns the per-use outcome and fails on any hazard.

    At each consume the weight must be in its buffer: put there by an earlier gather, or by an
    on-demand gather right now. The gathers the consume issues land after its read (its GEMM is
    queued first); none may overwrite the buffer it reads, nor one a use before the target still
    expects to read, and none may be redundant.
    """
    content, outcome = {}, []
    targets = _issue_points(prefetches)
    for p, (weight, buf, _) in enumerate(consumes):
        if content.get(buf) == weight:
            outcome.append("prefetched" if p in targets else "resident")
        else:
            outcome.append("on_demand")
            content[buf] = weight
        for target in prefetches.get(p, ()):
            assert target > p, "a gather is issued at or after the consume it serves"
            t_weight, t_buf, prefetchable = consumes[target]
            assert prefetchable, f"pinned weight {t_weight} was prefetched"
            assert t_buf != buf, f"use {target}'s gather overwrites use {p}'s read"
            for q in range(p + 1, target):
                assert (
                    t_buf != consumes[q][1]
                ), f"use {target}'s gather, issued at {p}, evicts use {q} ({consumes[q][0]})"
            assert content.get(t_buf) != t_weight, "redundant gather"
            content[t_buf] = t_weight
    return outcome


def _issue_points(prefetches):
    """``{use: consume that issues its gather}``."""
    return {t: p for p, targets in prefetches.items() for t in targets}


def test_distinct_weights_prefetch_one_ahead():
    """Single consumes in distinct buffers: exactly the next_w links, plus the cold start."""
    consumes = _recorded("abcd")
    prefetches, cold = _plan_prefetches(consumes)
    assert prefetches == {0: [1], 1: [2], 2: [3]}
    assert cold == 0
    assert _replay(consumes, prefetches) == ["on_demand", "prefetched", "prefetched", "prefetched"]


def test_pinned_head_is_gathered_on_demand():
    consumes = _recorded("Eab", pinned="E")
    prefetches, cold = _plan_prefetches(consumes)
    assert cold is None and prefetches == {0: [1], 1: [2]}


def test_repeated_uses_stay_resident():
    consumes = _recorded("abbb")
    prefetches, cold = _plan_prefetches(consumes)
    assert prefetches == {0: [1]} and cold == 0
    assert _replay(consumes, prefetches)[2:] == ["resident", "resident"]


def test_gather_waits_for_its_buffers_last_reader():
    # d shares a's buffer: its gather follows c's (the previous gather) but must also wait until
    # the consume after a's last read, here b's resident one.
    consumes = _recorded("abcabd", buffers={"d": "a"})
    prefetches, _ = _plan_prefetches(consumes)
    assert _issue_points(prefetches)[5] == 4
    assert _replay(consumes, prefetches)[3:] == ["resident", "resident", "prefetched"]


def test_neighbours_in_one_buffer_gather_on_demand():
    # A shared buffer for adjacent weights cannot be overlapped (pristine parity prevents it).
    consumes = _recorded("ab", buffers={"b": "a"})
    prefetches, _ = _plan_prefetches(consumes)
    assert prefetches == {}
    assert _replay(consumes, prefetches) == ["on_demand", "on_demand"]


def _head_and_mtp(eh_shares_proj):
    """Dense list of the head region: embedding E, trunk t0 t1, then two MTP depths (E, block)
    and the output layer O's three heads. E and O share the vocab buffer.
    """
    block = ["eh", "qkv", "proj", "lat"]
    order = ["E", "t0", "t1"] + ["E"] + block + ["E"] + block + ["O", "O", "O"]
    buffers = {"O": "E"}
    if eh_shares_proj:
        buffers["proj"] = "eh"
    return order, _recorded(order, buffers, pinned="E")


def test_output_layer_gathered_under_the_resident_depth_2_block():
    order, consumes = _head_and_mtp(eh_shares_proj=False)
    prefetches, _ = _plan_prefetches(consumes)
    outcome = _replay(consumes, prefetches)
    e_d2 = order.index("E", 4)
    first_o = order.index("O")
    assert order[e_d2] == "E"
    # Every depth-2 block weight and the later E and O consumes are resident: 2 vocab gathers.
    assert [o for name, o in zip(order, outcome) if name in ("E", "O")] == [
        "on_demand",
        "resident",
        "resident",
        "prefetched",
        "resident",
        "resident",
    ]
    assert outcome[e_d2 + 1 : first_o] == ["resident"] * 4
    # E d2 is the vocab buffer's last reader: O's gather starts right after it.
    assert _issue_points(prefetches)[first_o] == e_d2 + 1


def test_output_layer_waits_for_a_regathered_block_weight():
    """eh_proj and proj share a buffer (the 16-GPU MXFP8 case): both are re-gathered at depth 2,
    and O is issued one gather ahead -- at proj's depth-2 consume.
    """
    order, consumes = _head_and_mtp(eh_shares_proj=True)
    prefetches, _ = _plan_prefetches(consumes)
    outcome = _replay(consumes, prefetches)
    d2 = order.index("E", 4)
    assert [outcome[d2 + 1 + i] for i in range(4)] == [
        "prefetched",
        "resident",
        "prefetched",
        "resident",
    ]
    assert _issue_points(prefetches)[order.index("O")] == d2 + 3  # proj d2


def test_issue_order_is_use_order():
    rng = random.Random(0)
    for _ in range(200):
        names = [f"w{i}" for i in range(rng.randint(2, 7))]
        buffers = {n: f"b{rng.randint(0, 3)}" for n in names}
        consumes = _recorded([rng.choice(names) for _ in range(rng.randint(1, 40))], buffers)
        points = sorted(_issue_points(_plan_prefetches(consumes)[0]).items())
        assert [p for _, p in points] == sorted(p for _, p in points)


@pytest.mark.parametrize("seed", range(30))
def test_random_lists_never_read_a_clobbered_buffer(seed):
    rng = random.Random(seed)
    names = [f"w{i}" for i in range(rng.randint(2, 8))]
    buffers = {n: f"b{rng.randint(0, 3)}" for n in names}
    pinned = {n for n in names if rng.random() < 0.2}
    consumes = _recorded([rng.choice(names) for _ in range(rng.randint(1, 50))], buffers, pinned)
    prefetches, cold = _plan_prefetches(consumes)
    _replay(consumes, prefetches)
    assert cold is None or (cold == 0 and consumes[0][0] not in pinned)


def test_plan_cursor_follows_restarts_and_skips():
    a, b, c = object(), object(), object()
    plan = _PrefetchPlan()
    assert plan.consumes is None  # nothing is recorded before the first fence
    plan.consumes = []
    for w in (a, b, c):
        plan.record(w)
    plan.prefetches = {}
    assert [plan.advance(w) for w in (a, b, c)] == [0, 1, 2]
    # Past the end: the list's first weight restarts it.
    assert plan.advance(a) == 0
    # Out of turn and not the first weight: off the list, the cursor stays.
    assert plan.advance(c) is None and plan.cursor == 1
    assert plan.advance(b) == 1
    # The first weight out of turn restarts the list.
    assert plan.advance(a) == 0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions adapted from DynamicUSP at commit
# 1ff97a4ae0beb5e088fadb9e112265bdd194b24d (src/hot_switch_solver: stage2/solver.py,
# stage2/allocators.py, stage2/evaluator.py, stage2/model.py, api.py and models.py),
# BSD-3-Clause license.
# See ../DYNAMIC_USP_NOTICE.txt for the upstream copyright and license text.

"""DynamicUSP Stage 2: re-pack the samples of one microbatch on its fixed CP groups.

Stage 1 and the count search (``hot_switch``) fix the number of microbatches, the samples of
every microbatch and its CP groups: their rank spans and sizes. Stage 2 keeps all of that.
For one microbatch it builds the assignment of samples to groups again from nothing, at most
twice: once per sample queue (``compute_lpt``, then ``memory_lpt``). A finished construction
is priced exactly, and the better one replaces the Stage 1 assignment only when the slowest
group of the microbatch gets faster by a numeric epsilon. The step time is not part of that
rule: it can rise.

The cost quantities are the Stage 2 hooks of a cost model (see ``models.base.DcpCostModel``):
``stage2_reference_cp_size``, ``stage2_global_constant_ms``, ``stage2_sample_loads``,
``stage2_empty_group_load`` and ``stage2_group_time``. Groups of at most the reference size
are the reference tier; a microbatch of such groups only is left alone.

A cost model can also have a placement guard for the groups above the reference size
(upstream's ring guard): ``stage2_guard_margin``, the margin of a group, and
``stage2_guard_threshold``. The guard acts where upstream's does:

- During a construction a sample joins such a group only if the margin of the group with
  the sample is at least the smaller of its margin without the sample and the threshold: a
  group at or above the threshold stays there, and the margin of a group below it does not
  fall. Of the groups of one size only the one that the token cap selects for the sample
  is asked; when the guard rejects it, no group of that size takes the sample (when the
  capacity predicate of the runtime rejects it, the next group of the size is asked in
  turn).
- A finished construction is dropped when the margin of one of these groups is below the
  threshold.
- The Stage 1 layout is priced whatever its margins are.

A margin that is not a finite number is an error of the cost model, in the Stage 1 layout
as well. A cost model without a threshold has no guard and is never asked for a margin.
As upstream, the threshold is read for every microbatch and converted once with ``float()``.

Differences from upstream (``../ORIGIN.json`` is the provenance record of the port):

- Upstream asks every profile for a guard threshold and a margin; here both are optional.
- The threshold must be an int or a float, and no boolean; upstream takes whatever
  ``float()`` converts, a boolean excepted.
- Upstream keeps the least margin of a priced layout (``guard_min_margin``); no margin is
  recorded here.
- The capacity predicate of the runtime (``fits``: padding and the sequences of a group) is
  applied to every placement, after the token cap and the guard. Where it rejects a
  placement that upstream admits, the next group of the size is tried and the result can
  differ from upstream's; the record of the microbatch then carries ``runtime_capacity``.
- A sample has two sizes: its token count as Stage 1 uses it (the padded length), for the
  token cap and the memory queue, and its logical length, for work, loads, group times and
  guard margins. Upstream has one; the two are equal for a length that the runtime does not
  pad.
- Upstream's balanced index and heap over the groups are replaced by scans over the groups
  of one microbatch, with the same result.
- The groups are taken to cover the ranks (upstream checks it).
- Upstream checks a finished construction once more (no empty group, the token cap, the
  groups of the keep samples). For a layout that holds every sample once, as every layout
  of the planner does, the construction guarantees all of it, and that every group passed
  the capacity predicate, so none of it is checked again.
- Of upstream's outcome codes, those of a run that cannot start (disabled, empty problem,
  profile or features unavailable, unsupported domain) have no counterpart here.
- An exception is contained as in upstream's fallback mode (``repack_layout``); there is no
  raise mode. An error record has no times and no routes: upstream keeps the time of the
  Stage 1 layout, and the constructions finished before the error, when it has them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

# The two sample queues, in the order they are built; the first wins a tie.
COMPUTE_LPT = "compute_lpt"
MEMORY_LPT = "memory_lpt"
# DynamicUSP's default ``keep_threshold``: a sample whose work exceeds this many times the
# work per rank of its microbatch is a keep sample.
DEFAULT_KEEP_THRESHOLD = 8.4

# Outcome of Stage 2 for one microbatch. The four codes after ``ADOPTED`` are DynamicUSP's;
# a re-packed microbatch has no code there.
ADOPTED = "ADOPTED"
NO_PREDICTED_IMPROVEMENT = "NO_PREDICTED_IMPROVEMENT"
NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
PURE_REFERENCE_TOPOLOGY = "PURE_REFERENCE_TOPOLOGY"
PROFILE_EXECUTION_ERROR = "PROFILE_EXECUTION_ERROR"


def time_eps_ms(a: float, b: float) -> float:
    """Numeric tie tolerance of two predicted times in milliseconds."""
    return max(1e-6, 1e-9 * max(abs(a), abs(b), 1.0))


def _finite(value) -> bool:
    """True for a finite number that is not a boolean."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an integer beyond the float range
        return False


def _threshold(value):
    """The guard threshold that a cost model gives, converted once to a float; None stays None."""
    if value is None:
        return None
    number = float(value) if _finite(value) else math.nan
    if not math.isfinite(number):
        raise ValueError("stage2_guard_threshold must be a finite number or None")
    return number


def error_text(error: BaseException, limit: int = 500) -> str:
    """Text of a contained exception for a record, cut to ``limit`` characters; never raises."""
    try:
        text = str(error)
    except Exception:
        text = "<unprintable>"
    return text[:limit]


def _record(code=NO_VALID_CANDIDATE, **more) -> dict:
    """Record of a microbatch that has no finished construction."""
    return dict(code=code, order=None, base_ms=None, selected_ms=None, routes=[], **more)


@dataclass(frozen=True)
class _Microbatch:
    """One microbatch as Stage 2 sees it: its fixed groups and its prepared samples."""

    sizes: tuple  # group sizes in rank order
    tiers: tuple  # the distinct sizes, ascending
    reference: int  # groups of at most this size are the reference tier
    constant: float  # added to the slowest group time
    lengths: Mapping  # logical length of a sample: what the cost model prices
    tokens: Mapping  # token count of a sample: token cap and memory queue
    position: Mapping  # position of a sample in the global batch: the last tie-break
    work: Mapping  # work of a sample
    loads: Mapping  # {sample: {size: load the sample adds to a group of that size}}
    costs: object
    fits: Callable | None
    guard: float | None  # threshold of the placement guard; None: the cost model has none


def _margin(batch, ids, size):
    """Guard margin of a group above the reference size that holds ``ids``: a finite number."""
    margin = batch.costs.stage2_guard_margin(tuple(batch.lengths[i] for i in ids), size)
    if not _finite(margin):
        raise ValueError("the Stage 2 guard margin of a group must be a finite number")
    return margin


def _price(batch, members, finished=False):
    """Slowest group time of a complete layout plus the global constant, in ms.

    A finished construction (``finished``) gives None when the margin of one of its groups
    above the reference size is below the guard threshold. The Stage 1 layout is priced
    without that test.
    """
    times = []
    for ids, size in zip(members, batch.sizes):
        group_ms = batch.costs.stage2_group_time(tuple(batch.lengths[i] for i in ids), size)
        if not _finite(group_ms) or group_ms <= 0.0:
            raise ValueError("the Stage 2 time of a group must be finite and positive")
        if batch.guard is not None and size > batch.reference:
            # As upstream, the margin is taken in every layout and tested only in a finished
            # construction: the absolute test, where a placement met the ratchet.
            margin = _margin(batch, ids, size)
            if finished and float(margin) < batch.guard:
                return None
        times.append(float(group_ms))
    value = max(times) + batch.constant
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("the Stage 2 time of a microbatch must be finite and positive")
    return value


def _construct(batch, queue, keep, cap, noted):
    """One construction: keep samples first, then every other sample in queue order.

    Every sample is placed once. Returns the samples of every group in the order they were
    placed, or None at a dead end: a construction that cannot go on is dropped, never
    repaired.
    """
    sizes, reference = batch.sizes, batch.reference
    groups = range(len(sizes))
    load = []
    for size in sizes:
        initial = batch.costs.stage2_empty_group_load(size)
        if not _finite(initial) or initial < 0.0:
            raise ValueError("the Stage 2 load of an empty group must be finite and nonnegative")
        load.append(float(initial))
    used = [0] * len(sizes)
    members = [[] for _ in sizes]

    def allowed(group, sample):
        """The guard of a placement: the margin with the sample against the ratchet."""
        before = _margin(batch, members[group], sizes[group])
        after = _margin(batch, members[group] + [sample], sizes[group])
        return after >= min(before, batch.guard)

    def candidate(tier, sample, empty_only):
        """The group of one size that takes the sample: least load, then highest first rank."""
        needed = batch.tokens[sample]
        admitted = sorted(
            (
                group
                for group in groups
                if sizes[group] == tier
                and not (empty_only and members[group])
                and (used[group] + needed) * cap[1] <= cap[0] * tier
            ),
            key=lambda group: (load[group], -group),
        )
        for group in admitted:
            # As upstream, a group that the guard rejects ends the size for this sample: the
            # next group of the size is not asked.
            if batch.guard is not None and tier > reference and not allowed(group, sample):
                return None
            if batch.fits is None or batch.fits(members[group] + [sample], tier):
                return group
            # Upstream admits the placement and the capacity predicate of the runtime does
            # not: from here on the construction can differ from upstream's.
            noted["runtime_capacity"] = True
        return None

    def place(sample, tiers, empty_only):
        """Place the sample into the best candidate of ``tiers``; False when there is none."""
        choices = []
        for tier in tiers:
            group = candidate(tier, sample, empty_only)
            if group is None:
                continue
            own = load[group] + batch.loads[sample][tier]
            if not math.isfinite(own):
                raise ValueError("a Stage 2 group load is not finite")
            others = max((load[other] for other in groups if other != group), default=-math.inf)
            # (largest load of the microbatch after the placement, load of the group after
            # it, size, group)
            choices.append((max(others, own), own, tier, group))
        if not choices:
            return False
        low = min(choice[0] for choice in choices)
        tied = [choice for choice in choices if choice[0] <= low + time_eps_ms(choice[0], low)]
        # Within the epsilon: the smaller load of the group itself, then the smaller size.
        own, _, group = min(choice[1:] for choice in tied)
        members[group].append(sample)
        used[group] += batch.tokens[sample]
        load[group] = own
        return True

    # Keep samples, heaviest first: each into an empty group above the reference size.
    wide = [tier for tier in batch.tiers if tier > reference]
    for sample in sorted(keep, key=lambda i: (-batch.work[i], batch.position[i])):
        if not place(sample, wide, True):
            return None
    rest = [sample for sample in queue if sample not in keep]
    for index, sample in enumerate(rest):
        # As many samples left as empty groups: only an empty group takes one.
        empty_only = len(rest) - index == sum(not ids for ids in members)
        if not place(sample, batch.tiers, empty_only):
            return None
    return members


def repack_microbatch(
    groups: Sequence[tuple],
    lengths: Mapping[int, int],
    tokens: Mapping[int, int],
    position: Mapping[int, int],
    costs,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
    fits: Callable[[list, int], bool] | None = None,
):
    """Stage 2 on the Stage 1 layout of one microbatch.

    ``groups`` is ``[(first rank, size, [sample ids])]``, covering the DPxCP ranks; the
    other arguments are the fields of ``_Microbatch``.

    Returns ``(groups, record)``. ``groups`` is None unless the outcome is ``ADOPTED``; then
    it lists the same groups in rank order with their new samples, in the order they were
    placed. ``record`` holds ``code``, the queue ``order`` of an adopted layout,
    ``base_ms`` and ``selected_ms`` (slowest group plus the global constant, before and
    after), ``routes`` (``[queue, ms]`` of every finished construction), and
    ``runtime_capacity`` True when the capacity predicate rejected a placement that the
    token cap and the guard admitted. An error of the cost model is raised, not recorded.
    """
    reference, constant = costs.stage2_reference_cp_size, costs.stage2_global_constant_ms
    if type(reference) is not int or reference <= 0:
        raise ValueError("stage2_reference_cp_size must be a positive integer")
    if not _finite(constant) or constant < 0.0:
        raise ValueError("stage2_global_constant_ms must be finite and nonnegative")
    guard = _threshold(getattr(costs, "stage2_guard_threshold", None))
    groups = sorted(groups, key=lambda group: group[0])
    sizes = tuple(size for _, size, _ in groups)
    stage1 = [list(ids) for _, _, ids in groups]
    record = _record()  # the outcome until a construction is finished
    if not all(stage1):
        # A group without a sample leaves nothing to construct. The runtime forms none.
        return None, record

    # The samples in global-batch order, and what the cost model says about each.
    samples = sorted((i for ids in stage1 for i in ids), key=position.__getitem__)
    tiers = tuple(sorted(set(sizes)))
    work, loads = {}, {}
    for sample in samples:
        work[sample], given = costs.stage2_sample_loads(lengths[sample], tiers)
        loads[sample] = {size: given[size] for size in tiers}
        if not _finite(work[sample]) or work[sample] <= 0.0:
            raise ValueError("the Stage 2 work of a sample must be finite and positive")
        if any(not _finite(load) or load <= 0.0 for load in loads[sample].values()):
            raise ValueError("the Stage 2 loads of a sample must be finite and positive")
    batch = _Microbatch(
        sizes,
        tiers,
        reference,
        float(constant),
        lengths,
        tokens,
        position,
        work,
        loads,
        costs,
        fits,
        guard,
    )
    base = _price(batch, stage1)
    record.update(base_ms=base, selected_ms=base)
    if all(size <= reference for size in sizes):
        record["code"] = PURE_REFERENCE_TOPOLOGY
        return None, record

    # Token cap: the most tokens per rank of a Stage 1 group, kept as the exact ratio
    # (tokens, size) and compared by cross-multiplication.
    cap = None
    for ids, size in zip(stage1, sizes):
        total = sum(tokens[i] for i in ids)
        if cap is None or total * cap[1] > cap[0] * size:
            cap = (total, size)
    # As upstream, the threshold is checked here: after the two returns above.
    if not _finite(keep_threshold) or keep_threshold <= 0.0:
        raise ValueError("keep_threshold must be finite and positive")
    # Keep samples: more work than ``keep_threshold`` times the work per rank. The works are
    # added with the built-in sum in global-batch order, as upstream: from Python 3.12 on it
    # is compensated, so python3.11 can decide differently for a sample exactly on the
    # threshold.
    mean_work = sum(work[i] for i in samples) / sum(sizes)
    if not math.isfinite(mean_work) or mean_work <= 0.0:
        raise ValueError("the mean Stage 2 work of a microbatch must be finite and positive")
    keep = frozenset(i for i in samples if work[i] > keep_threshold * mean_work)
    if len(keep) > sum(size > reference for size in sizes):
        # Every keep sample needs a group above the reference size of its own.
        return None, record

    def tier_loads(sample):
        return [loads[sample][size] for size in tiers]

    # Both queues are descending; an earlier sample of the global batch comes first on a tie.
    compute = sorted(
        samples,
        key=lambda i: (max(tier_loads(i)), min(tier_loads(i)), tokens[i], -position[i]),
        reverse=True,
    )
    memory = sorted(
        samples, key=lambda i: (tokens[i], max(tier_loads(i)), -position[i]), reverse=True
    )
    queues = [(COMPUTE_LPT, compute)]
    if memory != compute:
        queues.append((MEMORY_LPT, memory))
    noted: dict = {}
    built = []
    for name, queue in queues:
        members = _construct(batch, queue, keep, cap, noted)
        value = None if members is None else _price(batch, members, finished=True)
        if value is not None:
            built.append((name, value, members))
            record["routes"].append([name, value])
    if noted:
        record["runtime_capacity"] = True
    if not built:
        return None, record
    low = min(value for _, value, _ in built)
    # Constructions within an epsilon of the faster tie; ``compute_lpt`` then wins.
    name, value, members = next(
        route for route in built if route[1] <= low + time_eps_ms(route[1], low)
    )
    if value > base - time_eps_ms(value, base):
        record["code"] = NO_PREDICTED_IMPROVEMENT
        return None, record
    record.update(code=ADOPTED, order=name, selected_ms=value)
    return [(start, size, ids) for (start, size, _), ids in zip(groups, members)], record


def repack_layout(
    layout: Sequence[Sequence[tuple]],
    lengths: Mapping[int, int],
    tokens: Mapping[int, int],
    costs,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
    fits: Callable[[list, int], bool] | None = None,
):
    """Stage 2 on every microbatch of the layout of one count, one microbatch at a time.

    ``layout`` is ``[[(first rank, size, [sample ids])]]``; the order of ``lengths`` is the
    order of the global batch. Returns ``(layout, records)``: a re-packed microbatch has
    new sample lists on its groups, any other its Stage 1 groups; ``records`` has one
    record of ``repack_microbatch`` per microbatch.

    As upstream's fallback mode, an exception raised while a microbatch is re-packed keeps
    its Stage 1 layout: its record has the code ``PROFILE_EXECUTION_ERROR``, the type of
    the exception as ``error`` and its text as ``reason``, and neither times nor routes.
    """
    position = {sample: index for index, sample in enumerate(lengths)}
    result, records = [], []
    for groups in layout:
        try:
            repacked, record = repack_microbatch(
                groups, lengths, tokens, position, costs, keep_threshold, fits
            )
        except Exception as error:
            repacked = None
            record = _record(
                PROFILE_EXECUTION_ERROR, error=type(error).__name__, reason=error_text(error)
            )
        result.append(list(groups) if repacked is None else repacked)
        records.append(record)
    return result, records

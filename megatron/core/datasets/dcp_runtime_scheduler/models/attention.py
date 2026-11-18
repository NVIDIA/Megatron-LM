# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Time model of causal context-parallel (ring) attention: forward and backward of one pack.

One form covers every CP size and every per-rank capacity. Every sequence of a pack is cut
into ``2 * cp_size`` chunks; ring member ``r`` holds chunk ``r`` (its first half) and chunk
``2 * cp_size - 1 - r`` (its second half) of every sequence. In step ``i`` it attends to the
keys of member ``(r - i) % cp_size``:

    step 0        a causal block on its own tokens                        (diagonal)
    steps 1..r    all its queries x the first-half keys of that member    (lower)
    later steps   its second-half queries x all keys of that member       (upper)

Forward of a member, on two compute streams that take the steps in turn: a prelude; per step
the wait for the keys, their hand-over to the next member, slice helpers, output clear and
kernel set-up, the attention kernel, and the LSE merge of the previous step; then one output
correction per step and an epilogue. The attention kernel fills every SM of the GPU, so the
kernels of the two streams run one after the other. A kernel is a queue of query tiles (one
per head and ``query_tile`` tokens) served by a fixed number of thread blocks; a tile costs a
constant plus a time per key tile it reads, and a kernel takes whole rounds of average tiles.

Backward of a member, on one stream and in reverse step order: a prelude; per step slice
helpers, the kernel (linear in query-key pairs, bytes read and bytes stored), the dQ
correction, the wait for the exchange of the step, and the dK/dV correction; an epilogue.

Causal block (step 0 of every member, and the whole call at ``cp_size`` 1): priced, forward
and backward, as one causal attention call on a single GPU, in place of the output clear,
set-up and kernel of that step: a call boundary, the megabytes of valid and of padded
tensors at their rates, and a kernel. That kernel is a queue of jobs on the thread blocks,
in one or more stages: one job per head and tile of a sequence, with one loop per tile it
reads. Tile sizes, the number of masked loops of a job, the order of the jobs and their
dispatch to the thread blocks are given per hardware, direction and head dim: these kernel
programs and the prices are those of the single-GPU calibration of the benchmark.

Reported times: all members start one forward/backward pair together. In repeated pairs all
cycle with the period of the slowest member, so the member with the shortest backward spends
the difference waiting in its forward:

    backward_ms = longest backward of a member
    forward_ms  = latest end of the pair - shortest backward of a member

Extensions beyond the measured packs (stated, not calibrated):

- A step whose queries are all padding launches an empty kernel: helpers, output clear and
  set-up are paid, the tile queue is empty, and the backward kernel form is evaluated at
  zero queries.
- A payload between two table points is interpolated linearly and one above the largest
  point continues the last slope; below the smallest point the transfer time is
  proportional to the payload.
- ``cp_size`` 1 has no ring: the causal block alone. Its calibration saw up to 64 sequences
  in buffers of 4096, 8192 and 16384 tokens; other packs are priced by the same form.

Provenance: written for this repository from two time models of NVIDIA's DCP attention
benchmark (unpublished); ``attention_<hardware>.json`` holds their frozen coefficients
unchanged, and ``ORIGIN.json`` records the sources and the simplifications. The step order
follows the P2P context-parallel attention of Transformer Engine; no code of it is copied.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Set
from functools import lru_cache
from heapq import heapify, heapreplace
from math import fsum
from operator import index, mul
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

from .base import NVLINK_DOMAIN_SIZES

# (num_attention_heads, num_query_groups, kv_channels) with coefficients for every hardware.
GEOMETRIES = ((32, 4, 128), (64, 4, 128), (16, 2, 256), (32, 2, 256), (32, 8, 128), (64, 8, 128))
# Raw bytes of the coefficient files; ORIGIN.json records the same digests.
PARAMETERS_SHA256 = {
    "gb200": "c2674f121c1c21dd54a7271ea85ecd80a36bd553784f219c06f5f2ceb00b4301",
    "h100": "0954f3fdbe062acdf8324d18f6e836e16a9171a89f40a8e4f86152fb0299d7d4",
}
# Largest ring and largest buffer of a rank (tokens) that are priced: far above what was
# measured (128 members, 16384 tokens); they bound the time and memory of a call.
MAX_CP_SIZE = 1024
MAX_TOKENS_PER_RANK = 2**20
_MIB = float(2**20)
# Bytes of one cumulative-sequence-length vector of a packed batch: 65 int32.
_SEQLENS_BYTES = 260
# Kinds of step of a ring member; they index the per-kind lists of ``_step_costs``.
_DIAGONAL, _LOWER, _UPPER = 0, 1, 2


@lru_cache(maxsize=None)
def _parameters(hardware: str) -> dict:
    """Coefficients of ``hardware``, read from the packaged file after checking its digest."""
    raw = Path(__file__).with_name(f"attention_{hardware}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARAMETERS_SHA256[hardware]:
        raise ValueError(f"Attention coefficient file of {hardware} does not match its digest")
    return json.loads(raw)


def _integers(values, name: str, minimum: int = 1) -> List[int]:
    """``values`` as Python ints; ValueError unless all are integers of at least ``minimum``.

    A bool or a float is no integer here; text, bytes, mappings and sets are refused whole.
    """
    numbers, valid = [], not isinstance(values, (str, bytes, bytearray, Mapping, Set))
    try:
        for value in values if valid else ():
            numbers.append(index(value))
            valid = valid and not isinstance(value, bool) and numbers[-1] >= minimum
    except TypeError:  # Not iterable, or an item without an integer value (2.0, None, "2").
        valid = False
    if not valid:
        raise ValueError(f"{name} must be integers of at least {minimum}, got {values!r}")
    return numbers


def _thread_blocks(par: dict, head_dim: int) -> int:
    """Thread blocks that serve the queue of a kernel: the SMs of the GPU by those of one."""
    return par["sm_count"] // par["sms_per_worker"][str(head_dim)]


def _ceil_to(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _interpolate(xs, ys, x, proportional):
    """Value at ``x`` of a measured table: linear between its points, last slope beyond them."""
    if x <= xs[0]:  # The value of the first point, or one proportional to x.
        return ys[0] * x / xs[0] if proportional else ys[0]
    i = 0
    while i < len(xs) - 2 and xs[i + 1] <= x:
        i += 1
    return (ys[i] * (xs[i + 1] - x) + ys[i + 1] * (x - xs[i])) / (xs[i + 1] - xs[i])


@lru_cache(maxsize=256)
def _step_costs(hardware, query_heads, kv_heads, head_dim, tokens, padded):
    """Every time that depends only on the geometry and the ``tokens`` stored per rank.

    ``padded`` says whether a sequence of the pack carries padding: the sequence-length
    helpers then also read the valid lengths. Lists by kind of step are
    ``[diagonal, lower, upper]``.
    """
    par = _parameters(hardware)
    geometry = f"{query_heads},{kv_heads},{head_dim}"
    helper = par["helper_ms"]

    def ms(name, moved_bytes, merged=0):
        constant, per_mib, per_million_merged = helper[name]
        return constant + per_mib * moved_bytes / _MIB + per_million_merged * merged / 1e6

    def gradient(tensor_bytes, mode):
        # Correction of a gradient kept as two halves; ``mode`` names what happens to the
        # first and to the second half ("none": left alone).
        # Halves moved: a copy reads one and writes one, an addition reads two and writes one.
        halves = sum({"copy": 2, "add": 3, "none": 0}[action] for action in mode.split("_"))
        moved = tensor_bytes // 2 * halves
        return ms("grad_" + mode, moved + _SEQLENS_BYTES)

    # Bytes of one rank: Q (as the output) and K (as V) in BF16, the LSE in FP32.
    q = 2 * tokens * query_heads * head_dim
    k = 2 * tokens * kv_heads * head_dim
    lse = 4 * tokens * query_heads
    divide = ms("seqlens_divide", 2 * _SEQLENS_BYTES)
    seqlens = {
        part: ms("seqlens_" + part, 3 * _SEQLENS_BYTES) if padded else divide
        for part in ("both", "first", "second")
    }
    half_k = ms("half_first", k + _SEQLENS_BYTES)
    half_q = ms("half_second", q + _SEQLENS_BYTES)

    def clear(output_bytes):
        # Output clear and kernel set-up before a kernel that writes ``output_bytes``.
        prefix = par["forward_prefix_ms"]
        return prefix["constant"] + prefix["setup"] + prefix["per_output_mib"] * output_bytes / _MIB

    out_full = ms("out_full", 3 * q + 2 * lse + _SEQLENS_BYTES, lse // 4)
    out_half = ms("out_half", 3 * (q // 2) + lse + _SEQLENS_BYTES, lse // 8)
    costs = SimpleNamespace(query_tile=par["query_tile"], kv_tile=par["kv_tile"])
    costs.workers = _thread_blocks(par, head_dim)
    costs.forward_kernel = par["forward_kernel_ms"][geometry]
    costs.forward_prelude = 2 * (divide + ms("copy", 2 * k))
    # Helpers, output clear and set-up between the arrival of the keys and the kernel. The
    # causal block of the diagonal step adds its own fixed part.
    costs.forward_ready = [
        2 * seqlens["both"],
        seqlens["both"] + seqlens["first"] + 2 * half_k + divide + clear(q),
        seqlens["second"] + seqlens["both"] + half_q + divide + clear(q // 2),
    ]
    costs.lse_merge = [
        ms("lse_copy", 2 * lse) + ms("lse_clone", 2 * lse) + ms("zeros", q),
        ms("lse_copy", 2 * lse) + ms("lse_merge", 3 * lse, lse // 4),
        ms("lse_copy", lse) + ms("lse_half_merge", 3 * (lse // 2) + _SEQLENS_BYTES, lse // 8),
    ]
    costs.output = [out_full, out_full, out_half]
    costs.forward_epilogue = ms("clone", 4 * k)

    # The last member of the ring has no second-half LSE to read back.
    last = costs.backward_prelude_last = ms("lse_copy", 2 * lse) + ms("copy", 4 * k)
    costs.backward_prelude = last + ms("lse_half_read", lse + _SEQLENS_BYTES) + ms("lse_copy", lse)
    costs.backward_ready = [0.0, 2 * half_k + divide, 3 * half_q + divide]
    # Corrections of dQ and of dK with dV (two tensors, hence twice the time of one): on
    # the whole tensor ("add", "copy") or by halves.
    costs.dq = {mode: gradient(q, mode) for mode in ("copy_add", "none_add", "none_copy")}
    costs.dq.update(add=ms("add", 3 * q), copy=ms("copy", 2 * q))
    costs.dkv = {mode: 2 * gradient(k, mode) for mode in ("add_copy", "copy_none", "add_none")}
    costs.dkv.update(add=2 * ms("add", 3 * k), copy=2 * ms("copy", 2 * k))
    costs.backward_epilogue = 2 * ms("copy", 2 * k)

    # Backward kernel of a lower or upper step: [constant, per pair, per byte read, per byte
    # stored]. Fixed per kind of step: the constant, and dQ, dK and dV stored at the physical
    # size of the block (query rows, key rows).
    rect = par["backward_kernel_ms"][geometry]
    blocks = {_LOWER: (tokens, tokens // 2), _UPPER: (tokens // 2, tokens)}
    costs.backward_fixed = {
        kind: rect[0]
        + rect[3] * 2 * head_dim * (q_rows * query_heads + 2 * k_rows * kv_heads) / _MIB
        for kind, (q_rows, k_rows) in blocks.items()
    }
    costs.per_pair = rect[1] * query_heads * head_dim / 1e6
    # Bytes read per valid query token (four BF16 tensors and the LSE) and per valid key token.
    costs.per_query_token = rect[2] * (8 * query_heads * head_dim + 4 * query_heads) / _MIB
    costs.per_key_token = rect[2] * 8 * kv_heads * head_dim / _MIB

    # One exchange carries one K+V copy or two (K+V and dK+dV). Prepare and transfer times
    # come from the measured table of the link class (inside one NVLink domain, or across
    # two) by payload size: {(route, copies): (prepare, transfer)}.
    costs.exchange = {}
    for route, table in par["transport_ms"].items():
        for copies in (1, 2):
            mib = copies * 4 * tokens * kv_heads * head_dim / _MIB
            prepare = _interpolate(table["payload_mib"], table["prepare"], mib, False)
            transfer = _interpolate(table["payload_mib"], table["outbound"], mib, True)
            costs.exchange[route, copies] = (prepare, transfer)
    return costs


@lru_cache(maxsize=128)
def _step_kinds(cp: int) -> tuple:
    """Kind of every step of every ring member: ``[member][step]``."""
    return tuple(
        tuple(_DIAGONAL if not step else _LOWER if step <= member else _UPPER for step in range(cp))
        for member in range(cp)
    )


def _stage_jobs(tokens, stage, heads, prices):
    """Jobs of one kernel stage of a causal block, in the order of the queue.

    One job per head and tile of a sequence, with one loop per tile it reads, the first
    ``masked_loops`` of them on the masked path; a loop counts ``operations`` times, a tile
    has ``copies`` jobs, and ``matrix`` times its loops is the lower bound of a job.
    """
    job_per, query_tile, kv_tile, operations, masked_loops, copies = stage
    loop, masked_loop, launch, matrix = prices
    jobs = []
    for n in tokens:
        if job_per == "query":  # A query tile reads the key tiles up to its own end.
            loops = [-(-min(n, at + query_tile) // kv_tile) for at in range(0, n, query_tile)]
        else:  # A key tile serves the query tiles from its own start on.
            loops = [-(-n // query_tile) - at // query_tile for at in range(0, n, kv_tile)]
        row = []
        for count in loops:
            masked = operations * min(count, masked_loops)
            work = loop * (operations * count - masked) + masked_loop * masked + launch
            row += [max(work, matrix * (operations * count))] * copies
        jobs += row * heads  # Every head repeats the jobs of the sequence.
    return jobs


def _queue_end(jobs, workers, round_robin):
    """End of a queue of jobs on ``workers`` thread blocks that are free at time zero.

    A job goes to the thread block that is free first or, with ``round_robin``, to the
    thread blocks in turn.
    """
    if round_robin:
        return max(fsum(jobs[first::workers]) for first in range(workers))
    free = jobs[:workers]
    heapify(free)
    for job in jobs[workers:]:
        heapreplace(free, free[0] + job)
    return max(free, default=0.0)


@lru_cache(maxsize=4096)
def _causal_block(hardware, query_heads, kv_heads, head_dim, tokens, stored_tokens):
    """Fixed forward time, forward kernel time and backward time of one causal block.

    ``tokens`` are the valid tokens per sequence, ``stored_tokens`` the size of the buffer
    with its padding; the numbers are ``causal_ms`` and ``causal_program`` of the file.
    """
    par = _parameters(hardware)
    workers = _thread_blocks(par, head_dim)
    valid = sum(tokens)
    fixed, kernel = [], []
    # Bytes per token, head and channel: Q, K, V and the output in BF16; the backward pass
    # also moves their gradients.
    for direction, tensor_bytes in (("forward", 4), ("backward", 8)):
        prices = par["causal_ms"][f"{query_heads},{kv_heads},{head_dim}"][direction]
        boundary, per_valid_mb, per_padded_mb, per_span, loop, masked, matrix = prices
        # A unit of span divides into the matrix bound and, of the rest, the share of a loop,
        # then of a masked loop, then the launch of a job.
        rest, other = 1 - matrix, 1 - loop
        job_prices = (rest * loop, rest * (other * masked), rest * (other * (1 - masked)), matrix)
        program = par["causal_program"][direction][str(head_dim)]
        order = sorted(tokens, reverse=True) if program["longest_first"] else tokens
        span = 0.0
        for stage in program["stages"]:  # A stage ends when its last job does.
            jobs = _stage_jobs(order, stage, query_heads, job_prices)
            span += _queue_end(jobs, workers, program["round_robin"])
        # Megabytes per token, with the FP32 LSE. The single-GPU calibration counts tensors
        # in MB (10**6 bytes); the ring terms count MiB.
        megabytes = (head_dim * tensor_bytes * (query_heads + kv_heads) + 4 * query_heads) / 1e6
        padded = stored_tokens - valid
        fixed.append(boundary + megabytes * (per_valid_mb * valid + per_padded_mb * padded))
        kernel.append(per_span * span)
    return fixed[0], kernel[0], fixed[1] + kernel[1]


def _ring_kernels(costs, geometry, lengths, stored, cp):
    """Kernel times ``[member][step]``, forward and backward, and the ready time of step 0.

    ``geometry`` is ``(hardware, query heads, KV heads, head dim)``. Step 0 of a member is
    its causal block, which also brings the fixed part between the helpers and its kernel.
    """
    members, heads = range(cp), geometry[1]
    tokens_per_rank = sum(stored) // cp
    # Valid tokens per member and sequence in its first-half chunk, its second and both.
    width = [n // (2 * cp) for n in stored]
    first = [[min(max(n - r * w, 0), w) for n, w in zip(lengths, width)] for r in members]
    second = [
        [min(max(n - (2 * cp - 1 - r) * w, 0), w) for n, w in zip(lengths, width)] for r in members
    ]
    both = [[a + b for a, b in zip(first[r], second[r])] for r in members]

    def side(rows, tile):
        # Per member: valid tokens per sequence, their tiles, and the totals of the two.
        tiles = [[-(-n // tile) for n in row] for row in rows]
        return [(row, t, sum(row), sum(t)) for row, t in zip(rows, tiles)]

    # A lower step pairs all queries with first-half keys, an upper step second-half
    # queries with all keys.
    queries = {_LOWER: side(both, costs.query_tile), _UPPER: side(second, costs.query_tile)}
    keys = {_LOWER: side(first, costs.kv_tile), _UPPER: side(both, costs.kv_tile)}
    kinds, workers, rect = _step_kinds(cp), costs.workers, costs.forward_kernel
    fixed, per_pair = costs.backward_fixed, costs.per_pair
    per_q, per_k = costs.per_query_token, costs.per_key_token
    forward = [[0.0] * cp for _ in members]
    backward = [[0.0] * cp for _ in members]
    first_ready = []
    for r in members:
        own = tuple(n for n in both[r] if n)  # A sequence without valid tokens here adds nothing.
        block_fixed, forward[r][0], backward[r][0] = _causal_block(*geometry, own, tokens_per_rank)
        first_ready.append(costs.forward_ready[_DIAGONAL] + block_fixed)
        for step in range(1, cp):
            kind = kinds[r][step]
            q_tokens, q_tiles, q_total, q_tile_total = queries[kind][r]
            k_tokens, k_tiles, k_total, _ = keys[kind][(r - step) % cp]
            # Forward kernel: whole rounds of average query tiles on the thread blocks. When
            # every query of the step is padding the tile queue is empty and it stays 0.
            tiles = heads * q_tile_total
            if tiles:
                key_tiles = heads * sum(map(mul, q_tiles, k_tiles))
                rounds = -(-tiles // workers)
                forward[r][step] = rounds * (rect[0] + rect[1] * key_tiles / tiles)
            pairs = sum(map(mul, q_tokens, k_tokens))
            backward[r][step] = fixed[kind] + per_pair * pairs + per_q * q_total + per_k * k_total
    return forward, backward, first_ready


def _exchange(posted, prepare, transfer, direction):
    """Completion per member of one ring exchange; member ``r`` posts it at ``posted[r]``.

    Member ``r`` sends to member ``r + direction`` (+1 or -1). A transfer starts when its
    two ends have prepared; a member is done when its own transfer and the one towards it are.
    """
    ready = [t + p for t, p in zip(posted, prepare)]
    receiver = ready[direction:] + ready[:direction]
    sent = [max(a, b) + s for a, b, s in zip(ready, receiver, transfer)]
    return [max(a, b) for a, b in zip(sent, sent[-direction:] + sent[:-direction])]


def _forward_ends(cp, costs, kernel_ms, first_ready, prepare, transfer):
    """End of the forward of every member; all members start at time zero."""
    members, kinds = range(cp), _step_kinds(cp)
    ready, merge = costs.forward_ready, costs.lse_merge
    streams = [[costs.forward_prelude] * cp, [0.0] * cp]  # End of the work on each stream.
    busy = [0.0] * cp  # The SMs of a member are free again.
    merged = [0.0] * cp  # End of the latest LSE merge of a member.
    arrived = None
    for step in range(cp):
        stream = streams[step % 2]
        if step:  # Wait for the keys of this step.
            stream = [max(a, b) for a, b in zip(stream, arrived)]
        if step < cp - 1:  # Hand the keys on for the next step.
            arrived = _exchange(stream, prepare, transfer, +1)
        for r in members:
            before = ready[kinds[r][step]] if step else first_ready[r]
            stream[r] = busy[r] = max(stream[r] + before, busy[r]) + kernel_ms[r][step]
        streams[step % 2] = stream
        if step:  # The LSE merge of the previous step runs on the stream of that step.
            other = streams[(step - 1) % 2]
            for r in members:
                other[r] = merged[r] = max(other[r], merged[r]) + merge[kinds[r][step - 1]]
    last = streams[(cp - 1) % 2]
    diagonal, lower, upper = costs.output
    ends = []
    for r in members:
        # The merge of the last step, then the output correction of the diagonal step, of
        # the r lower steps and of the cp - 1 - r upper steps. (The other stream ended with
        # the merge before, so it is free by then.)
        done = max(last[r], merged[r]) + merge[kinds[r][cp - 1]]
        corrected = done + diagonal + r * lower + (cp - 1 - r) * upper
        ends.append(corrected + costs.forward_epilogue)
    return ends


def _pair_ends(cp, costs, kernel_ms, forward_ends, exchanges):
    """End of the backward of every member, each starting when its forward has ended.

    Backward step ``s`` undoes forward step ``cp - 1 - s``. The corrections are those of the
    Transformer Engine ring: the first gradient written to a buffer is a copy and later
    ones are additions; an upper step writes the second half of dQ only, a lower step the
    first half of dK and dV only, and member 0 copies their second half in its last step.
    """
    members, kinds = range(cp), _step_kinds(cp)
    ready, dq, dkv = costs.backward_ready, costs.dq, costs.dkv
    preludes = [costs.backward_prelude] * (cp - 1) + [costs.backward_prelude_last]
    now = [end + prelude for end, prelude in zip(forward_ends, preludes)]
    for s in range(cp):
        step = cp - 1 - s
        # K+V travel in the first step, dK+dV in the last, both in between.
        prepare, transfer = exchanges[1 if s in (0, cp - 1) else 2]
        done = _exchange(now, prepare, transfer, -1)
        for r in members:
            first_full = cp - 1 - r  # Backward step in which the member first uses all queries.
            if s > first_full:
                dq_ms = dq["add"]
            elif s < first_full:
                dq_ms = dq["none_add"] if s else dq["none_copy"]
            else:
                dq_ms = dq["copy_add"] if first_full else dq["copy"]
            if s == cp - 1:
                dkv_ms = dkv["add"] if r else dkv["add_copy"]
            elif s >= first_full:
                dkv_ms = dkv["add_none"] if s or first_full else dkv["copy_none"]
            else:
                dkv_ms = dkv["add"] if s else dkv["copy"]
            kernel_end = now[r] + ready[kinds[r][step]] + kernel_ms[r][step] + dq_ms
            now[r] = max(kernel_end, done[r]) + dkv_ms
    return [t + costs.backward_epilogue for t in now]


def predict_pair(
    hardware: str,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    lengths: Sequence[int],
    cp_size: int,
    *,
    sequence_alignment: int,
    pack_alignment: int,
    ranks: Optional[Sequence[int]] = None,
) -> Tuple[float, float]:
    """Forward and backward milliseconds of the core attention of one pack of sequences.

    ``lengths`` are the valid tokens of the sequences. Each is stored padded to
    ``sequence_alignment`` and to the chunks of the ring, and the pack is padded to
    ``pack_alignment`` (``cp_size`` times the tokens per rank); the last sequence takes the
    tail padding. ``ranks`` are the physical ranks of the ring members in ring order
    (default ``0..cp_size-1``); rank ``r`` is a GPU of NVLink domain ``r // N`` with the
    ``N`` of ``NVLINK_DOMAIN_SIZES``.

    Raises ValueError outside the domain that the checks below spell out.
    """
    if not isinstance(hardware, str) or hardware not in PARAMETERS_SHA256:
        raise ValueError(f"hardware must be one of {sorted(PARAMETERS_SHA256)}, got {hardware!r}")
    heads, kv_heads, head_dim, cp, sequence_alignment, pack_alignment = _integers(
        (query_heads, kv_heads, head_dim, cp_size, sequence_alignment, pack_alignment),
        "the attention geometry, cp_size and the alignments",
    )
    geometry = (hardware, heads, kv_heads, head_dim)
    if geometry[1:] not in GEOMETRIES:
        raise ValueError(f"No attention coefficients for heads and head dim {geometry[1:]}")
    if cp > MAX_CP_SIZE:  # Before a ring of that size is laid out.
        raise ValueError(f"cp_size must not exceed {MAX_CP_SIZE}, got {cp}")
    lengths = _integers(lengths, "lengths")
    ranks = list(range(cp)) if ranks is None else _integers(ranks, "ranks", 0)
    if not lengths or len(ranks) != cp or len(set(ranks)) != cp:
        raise ValueError(f"A pack needs a sequence and one distinct rank per member: {ranks}")

    # Stored layout. A ring of one member does not cut its sequences into chunks.
    chunks = 2 * cp if cp > 1 else 1
    stored = [_ceil_to(_ceil_to(n, sequence_alignment), chunks) for n in lengths]
    capacity = _ceil_to(sum(stored), pack_alignment)
    if capacity % chunks:
        raise ValueError("A ring needs 2 x cp_size chunks: an even number of tokens per rank")
    if capacity > cp * MAX_TOKENS_PER_RANK:
        raise ValueError(f"A rank stores up to {MAX_TOKENS_PER_RANK} tokens, got {capacity // cp}")
    if cp == 1:  # No ring: the causal block alone.
        fixed, forward, backward = _causal_block(*geometry, tuple(lengths), capacity)
        return fixed + forward, backward

    stored[-1] += capacity - sum(stored)  # The last sequence takes the tail padding.
    costs = _step_costs(*geometry, capacity // cp, lengths != stored)
    # Link from member r to member r + 1: inside one NVLink domain or across two.
    domain = [rank // NVLINK_DOMAIN_SIZES[hardware] for rank in ranks]
    to_next = ["intra" if domain[r] == domain[(r + 1) % cp] else "cross" for r in range(cp)]
    if "cross" in to_next and ("cross", 1) not in costs.exchange:
        raise ValueError(
            f"{hardware} has no measured cross-domain exchange; keep the ring inside one "
            f"NVLink domain of {NVLINK_DOMAIN_SIZES[hardware]} ranks"
        )
    from_previous = to_next[-1:] + to_next[:-1]

    def links(copies, receive_on, send_on):
        # Prepare time by the link a member receives on, transfer time by the link it sends on.
        prepare = [costs.exchange[route, copies][0] for route in receive_on]
        return prepare, [costs.exchange[route, copies][1] for route in send_on]

    forward_kernel, backward_kernel, first_ready = _ring_kernels(
        costs, geometry, lengths, stored, cp
    )
    # Forward: the keys travel to the next member.
    forward_ends = _forward_ends(
        cp, costs, forward_kernel, first_ready, *links(1, from_previous, to_next)
    )
    # Backward: keys and key gradients travel to the previous member.
    exchanges = {copies: links(copies, to_next, from_previous) for copies in (1, 2)}
    pair_ends = _pair_ends(cp, costs, backward_kernel, forward_ends, exchanges)
    backward = [end - start for end, start in zip(pair_ends, forward_ends)]
    return max(pair_ends) - min(backward), max(backward)

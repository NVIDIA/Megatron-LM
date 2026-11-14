# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``_request_seed_noise_kernel``
(megatron/core/inference/sampling/request_seed_noise.py, launcher ``fill_request_seed_noise``).

**Contract** (from the sampler's semantics and the launcher's documented API, before reading the
kernel). ``TorchSampling.sample_from_logits`` draws a token by the exponential race
``argmax_v(prob[v] / q[v])`` with ``q[v]`` iid Exp(1), which samples ``Categorical(prob)``. A
request with a seed (``SamplingParams.seed``, an int in ``[0, 2**63)``: "request-local RNG,
independent of dynamic batch position") gets its ``q`` from this kernel:

* ``fill_request_seed_noise(noise [R, V] fp32 contiguous, seeds [R] of int | None,
  positions [R] of int >= 0 or None for zeros)``;
* a row with seed ``None`` is not touched (the sampler has filled it from the shared generator);
* a row with seed ``s`` at position ``p`` (the request's absolute next-token position) gets
  ``noise[r, v] = q(s, p, v)``, a function of the seed, the position and the vocabulary index
  only: not of the row, the batch, the other rows, the launch path (one row or many), the
  vocabulary size beyond ``v``, earlier launches or the shared generator, which it does not
  advance;
* ``q(s, p, v)`` is a standard exponential variate, independent across distinct
  ``(s, p, v)``: across requests (seeds, including the server's adjacent seeds for ``n > 1``
  choices), across a request's steps (positions) and across vocabulary entries; strictly
  positive and finite, so ``prob / q`` is defined.

The stream itself is Philox4x32-10 (Random123) keyed by the 64-bit seed with counter
``(v, p mod 2**32, p div 2**32, 0)``, word 0 mapped to ``[0, 1)`` by Triton's
``uint_to_uniform_float`` and ``q = -ln(max(u, FLT_MIN))`` (from the kernel's comments; the
layout is the only part of the oracle taken from the kernel). Pinning it keeps seeded outputs
reproducible across releases.

**Oracles.**

* Bit level: a numpy Philox4x32-10 checked against the Random123 known-answer vectors, and
  Triton's documented uniform map: the uniforms are exact by construction, and ``q`` is
  compared with ``-ln u`` in fp64, bounded by torch's fp32 ``log`` on the same device
  (``assert_close_to_reference``, per binade of ``q``, so small values get relative bounds).
* Metamorphic, exact: the same ``(s, p)`` row in different rows, batches, launch paths,
  vocabulary sizes and repeated launches is bit-identical; unseeded rows and the shared
  generator are untouched.
* Statistical (the harness's chi-square and exact binomial bounds, false-failure rate 1e-6 per
  assertion, fixed inputs): the marginal of 4.2M values is Exp(1) (105 bins); the race samples
  an 8-token distribution across 2**17 requests and across 2**17 steps of one request (enough
  to see a 0.01 shift of a 0.3 token with power 0.99, ``ou.samples_to_detect``); the joint draw
  of adjacent seeds, seeds 2**32 apart, adjacent positions, positions 2**32 apart and adjacent
  vocabulary entries is the product of the marginals (2**16 pairs each: a 0.01 shift of a 0.1
  cell).

**Inputs from the kernel's structure.** Grid ``(R, cdiv(V, 1024))`` with ``BLOCK = 1024``
(``request_seed_noise.py:64-66``), ``V`` and ``SINGLE_ROW`` constexpr: V in
1/1023/1024/1025/2047/2048/2049 and the production 4099, 32000, 128256, 151936 and 262144;
``R = 1`` (scalar seed and position, typed i32, u32 or i64 by value: seeds and positions
0/1/2**31-1/2**31/2**32-1/2**32 and 2**63-1 or 2**40+7) and ``R > 1`` (an int64 metadata
tensor) with seeded and unseeded rows; extreme uniforms (Philox word 0, so ``u = 0`` and
``q = -ln FLT_MIN``; and ``u = 1 - 2**-24``, the smallest ``q``) at searched coordinates;
output offsets ``row * V`` past 2**31 (``large``). Factors combined pairwise.

**Memory.** ``noise`` is guard-banded (one row) and starts NaN-poisoned; unseeded rows must
stay poisoned. One small case runs under compute-sanitizer.

**Hardware.** H100; the stream is architecture independent by design (Blackwell is not
run here). The sampler is eager and builds its metadata on the host, so it is never captured
in a CUDA graph. No open PR changes this kernel; #8875 changes ``torch_sampling.py``'s
filters, which these tests do not use (they apply the race themselves).
"""

import numpy as np
import pytest
import torch

from megatron.core.inference.sampling.request_seed_noise import fill_request_seed_noise
from tests.unit_tests.inference.oracles import oracle_utils as ou

DEVICE = "cuda"

BLOCK = 1024
"""Vocabulary entries per program (``request_seed_noise.py:64-66``)."""

FLT_MIN = torch.finfo(torch.float32).tiny

VOCABS = ou.boundary_sizes(BLOCK, minimum=1, production=(4099, 32000, 128256, 151936, 262144))

SEEDS = (0, 1, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1)
"""Scalar seeds are typed i32 below 2**31, u32 below 2**32 and i64 above; 2**63 - 1 is the
largest valid seed (``sampling_params.py``)."""

POSITIONS = (0, 1, 2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**40 + 7)

RACE_SAMPLES = 2**17
PAIR_SAMPLES = 2**16
MARGINAL_ROWS = 64
MARGINAL_VOCAB = 65536
CHUNK_ROWS = 2**14


# ---------------------------------------------------------------------------------------------
# Reference: Philox4x32-10 (Random123) and Triton's uint32 -> [0, 1) map
# ---------------------------------------------------------------------------------------------

_MASK = np.uint64(0xFFFFFFFF)
_ROUND = (np.uint64(0xD2511F53), np.uint64(0xCD9E8D57))
_KEY_STEP = (np.uint64(0x9E3779B9), np.uint64(0xBB67AE85))


def philox4x32(counter, key, rounds=10):
    """Philox4x32 (Salmon et al., SC'11): ``counter`` 4 and ``key`` 2 arrays of 32-bit words."""
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) & _MASK for c in counter)
    k0, k1 = (np.asarray(k, dtype=np.uint64) & _MASK for k in key)
    for _ in range(rounds):
        p0, p1 = _ROUND[0] * c0, _ROUND[1] * c2  # 32 x 32 -> 64 bits, exact in uint64
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)) ^ c1 ^ k0,
            p1 & _MASK,
            (p0 >> np.uint64(32)) ^ c3 ^ k1,
            p0 & _MASK,
        )
        k0, k1 = (k0 + _KEY_STEP[0]) & _MASK, (k1 + _KEY_STEP[1]) & _MASK
    return c0, c1, c2, c3


def uniform_from_word(word):
    """``triton.language.random.uint_to_uniform_float`` for 32-bit words, in numpy fp32."""
    x = np.asarray(word, dtype=np.uint64).astype(np.uint32).view(np.int32)
    x = np.where(x < 0, ~x, x)
    return x.astype(np.float32) * np.float32(4.6566127342e-10)


def stream_word(seed, position, vocab):
    """Philox word 0 for ``(seed, position, vocab index)`` (vectorized over ``vocab``)."""
    v = np.asarray(vocab, dtype=np.uint64)
    zero = np.zeros_like(v)
    return philox4x32(
        (v, zero + (position & 0xFFFFFFFF), zero + (position >> 32), zero),
        (seed & 0xFFFFFFFF, seed >> 32),
    )[0]


def reference_uniform(seed, position, vocab_size):
    return torch.from_numpy(uniform_from_word(stream_word(seed, position, np.arange(vocab_size))))


def _neg_log(u):
    return -torch.log(torch.clamp(u, min=FLT_MIN))


def assert_noise_matches_stream(noise_row, seed, position, what):
    """``noise_row`` (one seeded row) against the Philox stream, per binade of the reference."""
    u = reference_uniform(seed, position, noise_row.numel()).to(noise_row.device)
    reference = ou.fp64_reference(_neg_log, u)
    trusted = _neg_log(u)  # torch's fp32 log on the same device
    _, exponent = torch.frexp(reference)
    for e in torch.unique(exponent).tolist():
        sel = exponent == e
        ou.assert_close_to_reference(
            noise_row[sel],
            reference[sel],
            trusted[sel],
            what=f"{what}: noise in [2**{e - 1}, 2**{e})",
        )


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _noise(seeds, positions, vocab_size):
    out = torch.empty((len(seeds), vocab_size), dtype=torch.float32, device=DEVICE)
    fill_request_seed_noise(out, list(seeds), list(positions))
    return out


def _race(probs, seeds, positions):
    """Tokens drawn by the sampler's race ``argmax(prob / q)`` (fp32, as in production)."""
    tokens = []
    for lo in range(0, len(seeds), CHUNK_ROWS):
        q = _noise(seeds[lo : lo + CHUNK_ROWS], positions[lo : lo + CHUNK_ROWS], probs.numel())
        tokens.append(torch.argmax(probs[None, :] / q, dim=-1))
    return torch.cat(tokens)


def _race_probs(vocab_size, mass):
    probs = torch.zeros(vocab_size, dtype=torch.float64)
    for token, p in mass.items():
        probs[token] = p
    return probs


# ---------------------------------------------------------------------------------------------
# Bit level against the stream
# ---------------------------------------------------------------------------------------------


def test_philox_reference_reproduces_random123_vectors():
    """The reference is Philox4x32 (the known-answer vectors published with Random123, D. E.
    Shaw Research, ``kat_vectors``: 7 and 10 rounds), and the uniform map's ends are where
    Triton documents them."""
    vectors = [
        ((0, 0, 0, 0), (0, 0), 7, (0x5F6FB709, 0x0D893F64, 0x4F121F81, 0x4F730A48)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, 7, (0x5207DDC2, 0x45165E59, 0x4D8EE751, 0x8C52F662)),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            7,
            (0x4DFCCABA, 0x190A87F0, 0xC47362BA, 0xB6B5242A),
        ),
        ((0, 0, 0, 0), (0, 0), 10, (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF,) * 2,
            10,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            10,
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ]
    for counter, key, rounds, expected in vectors:
        got = tuple(int(w) for w in philox4x32(counter, key, rounds))
        assert got == expected, f"Philox4x32-{rounds}{counter}{key}: {got} != {expected}"
    ends = uniform_from_word([0, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 1])
    top = np.float32(1) - np.float32(2**-24)
    assert ends.tolist() == [0.0, 0.0, top, top, np.float32(4.6566127342e-10)]


def _stream_case_id(c):
    return f"V{c['vocab']}-{c['launch']}-seed{c['seed']}-pos{c['position']}"


STREAM_CASES = ou.pairwise_cases(
    {"vocab": VOCABS, "seed": SEEDS, "position": POSITIONS, "launch": ("single", "batched")}
)
SANITIZER_STREAM = dict(vocab=1025, seed=2**32, position=19, launch="batched")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=_stream_case_id(c)) for c in STREAM_CASES]
    + [
        pytest.param(
            SANITIZER_STREAM,
            id="sanitizer-" + _stream_case_id(SANITIZER_STREAM),
            marks=pytest.mark.sanitizer,
        )
    ],
)
def test_noise_matches_philox_stream(case):
    vocab, seed, position = case["vocab"], case["seed"], case["position"]
    if case["launch"] == "single":
        rows = [(seed, position)]
    else:
        # Another seeded row, an unseeded row, and the first row again.
        other = (SEEDS[(SEEDS.index(seed) + 3) % len(SEEDS)], position ^ 0x5A5A)
        rows = [(seed, position), (None, 7), other, (seed, position)]
    seeds, positions = [r[0] for r in rows], [r[1] for r in rows]
    buf = ou.guarded_empty((len(rows), vocab), torch.float32, DEVICE)
    fill_request_seed_noise(buf.tensor, seeds, positions)
    _sync()
    ou.check_guards(noise=buf)
    for r, (s, p) in enumerate(rows):
        if s is None:
            ou.assert_poisoned(buf.tensor[r], what=f"unseeded row {r}")
        else:
            assert_noise_matches_stream(buf.tensor[r], s, p, what=f"row {r} (seed {s}, pos {p})")


# Coordinates whose Philox word 0 is 0 (u = 0) or rounds to the largest uniform 1 - 2**-24,
# found by searching the reference stream; v < 1024.
EXTREME = [
    pytest.param(2026, 278396, 71, 0x0, id="word-0x00000000-u-0"),
    pytest.param(2029, 609015, 869, 0x0, id="word-0x00000000-u-0-b"),
    pytest.param(2026, 3682, 691, 0x80000011, id="word-0x80000011-u-max"),
    pytest.param(2027, 15914, 882, 0x7FFFFFD2, id="word-0x7fffffd2-u-max"),
]


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("seed, position, vocab, word", EXTREME)
@pytest.mark.parametrize("launch", ["single", "batched"])
def test_extreme_uniforms_give_finite_positive_noise(seed, position, vocab, word, launch):
    """u = 0 must give -ln(FLT_MIN) (finite), the largest u the smallest positive noise."""
    assert int(stream_word(seed, position, [vocab])[0]) == word  # the coordinates are right
    rows = [(seed, position)] if launch == "single" else [(None, 0), (seed, position)]
    noise = torch.full((len(rows), BLOCK), float("nan"), device=DEVICE)
    fill_request_seed_noise(noise, [r[0] for r in rows], [r[1] for r in rows])
    _sync()
    value = noise[-1, vocab]
    assert bool(torch.isfinite(value)) and float(value) > 0.0, f"noise {float(value)!r}"
    assert_noise_matches_stream(noise[-1], seed, position, what=f"seed {seed}, pos {position}")


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize("rows", ou.int32_overflow_params(262144))
def test_noise_offsets_past_int32(rows):
    """Row offsets ``row * V`` up to and past 2**31 elements (V = 262144)."""
    vocab = 262144
    ou.require_free_memory(rows * vocab * 4 + 2**30)
    buf = ou.guarded_empty((rows, vocab), torch.float32, DEVICE)
    seeds = list(range(rows))
    fill_request_seed_noise(buf.tensor, seeds, [3] * rows)
    _sync()
    ou.check_guards(noise=buf)
    for r in (0, rows - 1):
        assert_noise_matches_stream(buf.tensor[r], seeds[r], 3, what=f"row {r} of {rows}")
    assert bool(torch.isfinite(buf.tensor).all()), "a row was not written"


# ---------------------------------------------------------------------------------------------
# Per-request determinism (metamorphic, exact)
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
def test_noise_depends_only_on_seed_position_and_vocab_index():
    requests = [(5, 19), (6, 19), (5, 20), (2**40 + 5, 19), (5, 2**32 + 19), (0, 0)]
    vocab = 4099
    cuda_rng = torch.cuda.get_rng_state() if DEVICE == "cuda" else None
    cpu_rng = torch.random.get_rng_state()

    alone = [_noise([s], [p], vocab)[0] for s, p in requests]  # the single-row path
    batch = _noise([s for s, _ in requests], [p for _, p in requests], vocab)
    for i, (s, p) in enumerate(requests):
        ou.assert_exact(batch[i], alone[i], what=f"request {i}: batched row vs alone")

    # Reordered, with unseeded rows interleaved and a different batch size.
    order = [3, None, 0, 5, None, 2, 1, 4, 0]
    seeds = [None if i is None else requests[i][0] for i in order]
    positions = [0 if i is None else requests[i][1] for i in order]
    shuffled = torch.full((len(order), vocab), float("nan"), device=DEVICE)
    fill_request_seed_noise(shuffled, seeds, positions)
    for row, i in enumerate(order):
        if i is None:
            ou.assert_poisoned(shuffled[row], what=f"unseeded row {row}")
        else:
            ou.assert_exact(shuffled[row], alone[i], what=f"request {i} at row {row}")

    # The stream for vocabulary index v does not depend on the vocabulary size.
    for smaller in (1, 1023, 1025, 2049):
        prefix = _noise([s for s, _ in requests], [p for _, p in requests], smaller)
        ou.assert_exact(
            prefix, batch[:, :smaller], what=f"V = {smaller} vs a prefix of V = {vocab}"
        )

    # Launching again gives the same bits (a retried request replays its draws).
    again = _noise([s for s, _ in requests], [p for _, p in requests], vocab)
    ou.assert_exact(again, batch, what="second launch")

    # Distinct coordinates give distinct streams (the high seed and position words count).
    for i in range(len(requests)):
        for j in range(i):
            assert not torch.equal(batch[i], batch[j]), f"requests {i} and {j} share a stream"

    # The shared generator is neither read into the noise nor advanced.
    if cuda_rng is not None:
        ou.assert_exact(torch.cuda.get_rng_state(), cuda_rng, what="CUDA generator state")
    ou.assert_exact(torch.random.get_rng_state(), cpu_rng, what="CPU generator state")


# ---------------------------------------------------------------------------------------------
# Distribution and independence (statistical)
# ---------------------------------------------------------------------------------------------


@ou.requires_sm(ou.HOPPER)
def test_noise_is_standard_exponential():
    """4.2M values over 64 (seed, position) rows: 100 equiprobable Exp(1) bins, the last split
    at 6, 8, 10, 12 and 14."""
    seeds = [7919 * r + 1 for r in range(MARGINAL_ROWS)]
    positions = [(r * 104729) % 4096 for r in range(MARGINAL_ROWS)]
    noise = _noise(seeds, positions, MARGINAL_VOCAB).flatten().double().cpu()
    k = torch.arange(1, 100, dtype=torch.float64)
    edges = torch.cat([-torch.log1p(-k / 100), torch.tensor([6.0, 8.0, 10.0, 12.0, 14.0])])
    cdf = torch.cat([torch.zeros(1), 1 - torch.exp(-edges), torch.ones(1)]).double()
    probs = cdf[1:] - cdf[:-1]
    counts = torch.bincount(torch.bucketize(noise, edges, right=True), minlength=probs.numel())
    ou.assert_matches_distribution(counts, probs, what="noise vs Exp(1)")
    ou.assert_counts_within_binomial_bounds(counts, probs, what="noise vs Exp(1)")


RACE_VOCAB = 1025
RACE_MASS = {0: 0.30, 1023: 0.20, 1024: 0.15, 1: 0.10, 511: 0.08, 1022: 0.07, 512: 0.05}
"""Tokens on both sides of the 1024-entry tile boundary; 0.05 more spread over tokens 2..500;
tokens 501..510 and 513..1021 have probability 0 and must never be drawn."""


@ou.requires_sm(ou.HOPPER)
def test_race_samples_the_distribution_across_requests_and_steps():
    probs64 = _race_probs(RACE_VOCAB, RACE_MASS)
    probs64[2:501] = 0.05 / 499
    probs = probs64.float().to(DEVICE)
    n = RACE_SAMPLES
    across_requests = _race(probs, [10_000 + i for i in range(n)], [19] * n)
    across_steps = _race(probs, [4242] * n, [100 + i for i in range(n)])
    counts = torch.stack(
        [torch.bincount(t.cpu(), minlength=RACE_VOCAB) for t in (across_requests, across_steps)]
    )
    expected = torch.stack([probs.double().cpu()] * 2)
    what = "race (row 0: across requests, row 1: one request across steps)"
    ou.assert_matches_distribution(counts, expected, what=what)
    ou.assert_counts_within_binomial_bounds(counts, expected, what=what)


PAIR_MASS = {0: 0.4, 1023: 0.3, 1024: 0.2, 512: 0.1}


def _pairs(family, n):
    """Coordinates ``(seed_a, pos_a, seed_b, pos_b)`` of ``n`` disjoint pairs of a family."""
    i = np.arange(n, dtype=np.int64)
    if family == "adjacent_seeds":  # the server's seeds for n > 1 choices of one prompt
        return 5000 + 2 * i, np.full(n, 19), 5000 + 2 * i + 1, np.full(n, 19)
    if family == "seeds_2**32_apart":
        return 5000 + i, np.full(n, 19), 5000 + i + 2**32, np.full(n, 19)
    if family == "adjacent_positions":  # consecutive steps of one request
        return np.full(n, 77), 100 + 2 * i, np.full(n, 77), 100 + 2 * i + 1
    if family == "positions_2**32_apart":
        return np.full(n, 77), 100 + i, np.full(n, 77), 100 + i + 2**32
    raise ValueError(family)


@ou.requires_sm(ou.HOPPER)
@pytest.mark.parametrize(
    "family", ["adjacent_seeds", "seeds_2**32_apart", "adjacent_positions", "positions_2**32_apart"]
)
def test_draws_are_independent_across_requests_and_steps(family):
    """The joint distribution of two draws is the product of the marginals (16 cells)."""
    probs64 = _race_probs(RACE_VOCAB, PAIR_MASS)
    probs = probs64.float().to(DEVICE)
    sa, pa, sb, pb = (x.tolist() for x in _pairs(family, PAIR_SAMPLES))
    ta, tb = _race(probs, sa, pa).cpu(), _race(probs, sb, pb).cpu()
    cats = torch.tensor(sorted(PAIR_MASS))
    index = torch.full((RACE_VOCAB,), -1, dtype=torch.int64)
    index[cats] = torch.arange(cats.numel())
    ia, ib = index[ta], index[tb]
    assert bool((ia >= 0).all() and (ib >= 0).all()), "a draw fell on a probability-0 token"
    k = cats.numel()
    counts = torch.bincount(ia * k + ib, minlength=k * k)
    marginal = probs64[cats]
    expected = torch.outer(marginal, marginal).flatten()
    ou.assert_matches_distribution(counts, expected, what=f"joint draws, {family}")


@ou.requires_sm(ou.HOPPER)
def test_adjacent_vocabulary_entries_are_independent():
    """Pairs ``(q[v], q[v+1])`` binned in 8 equiprobable Exp(1) bins each: uniform 64 cells."""
    rows = 2 * PAIR_SAMPLES // (RACE_VOCAB - 1)
    noise = _noise([31 * r + 3 for r in range(rows)], [19 + r for r in range(rows)], RACE_VOCAB)
    pairs = noise[:, : RACE_VOCAB - 1].reshape(rows, -1, 2).double().cpu()
    edges = -torch.log1p(-torch.arange(1, 8, dtype=torch.float64) / 8)
    bins = torch.bucketize(pairs, edges, right=True)
    counts = torch.bincount((bins[..., 0] * 8 + bins[..., 1]).flatten(), minlength=64)
    expected = torch.full((64,), 1 / 64, dtype=torch.float64)
    ou.assert_matches_distribution(counts, expected, what="adjacent vocabulary entries")

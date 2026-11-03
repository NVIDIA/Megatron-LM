# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``flashinfer.sampling.top_p_sampling_from_probs`` at Megatron's call site.

Call site: ``FlashInferSampling.sample_kernel``
(``megatron/core/inference/sampling/flashinfer_sampling.py:120``), taken when some active request
uses top-p and none uses top-k. Megatron passes ``probs = softmax(l / max(T, 1e-6))`` in fp32,
a per-row fp32 ``top_p`` with the no-op values (``p <= 0``, ``p >= 1``) mapped to 1.0,
``deterministic=True`` and the controller's generator. Until this module, no test ran the real
kernel on this path (the only test mocked it).

Contract (from the call site, FlashInfer's documented API and the definition of nucleus
sampling, written before reading the kernel):

* Computes: for each row, one token drawn from the nucleus distribution: with ``q`` the row's
  probabilities and ``p`` its top-p, the kept set is ``{i : sum of q_j over q_j > q_i < p}``, the
  smallest set whose mass reaches p, every token tied at the cut kept; the draw follows ``q``
  renormalized to that set. ``p = 1`` keeps every token with ``q > 0``.
  (``sampling_oracle.reference_distribution`` steps 1, 3-5.)
* A cut exactly on a cumulative boundary (``sum over the top m tokens == p``) excludes the
  ``(m+1)``-th token: the top m already reach p (FlashInfer's own ``top_p_renorm_probs`` docs
  define the same threshold). Megatron cannot hit it through ``softmax`` except by chance; the
  direct case below builds it from exact dyadic probabilities.
* Rows of one batch take their own ``p`` and temperature; rows with the no-op sentinel are
  sampled unfiltered.
* Outputs, RNG, degenerate rows: as for ``sampling_from_logits`` (int64 ids in ``[0, V)``; the
  same generator state replays bit-exactly, every call advances it).
* Tolerance class: sampler. Chi-square per row group against the fp64 reference, false-failure
  rate ``alpha = 1e-6`` per assertion (Bonferroni over groups), fixed seed; any draw outside the
  kept set fails exactly.

Structure (read from FlashInfer 0.7.2 after the contract): multi-round rejection sampling
without sorting, one block of 1024 threads per row, ``VEC = gcd(4, V)``; ``deterministic=True``
selects the deterministic block scan. Inputs: the vocabulary sweep of ``sampling_from_logits``;
cuts in the middle of a ramp, at ``p`` near 0 and near 1, a three-way tie straddling the cut,
a heavy tail that forces many rejection rounds, and exact dyadic boundaries.

Hardware: H100 (sm_90). Open PRs: #8875 (``fix/sampling-contract``) does not change this path
(no top-k rows); every test here must pass on main and on #8875.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_sampling_rope import sampling_oracle as so

pytestmark = ou.requires_sm(ou.HOPPER)

KERNEL = "top_p_sampling_from_probs"


def _ramp(vocab=32000, count=48, dtype=torch.bfloat16):
    g = ou.make_generator()
    hot = so.spread_positions(vocab, count, g)
    return so.ramp_logits(vocab, hot, dtype=dtype), hot


def _nucleus_specs():
    ramp, hot = _ramp()
    shallow = ramp.clone()
    shallow[shallow == -16.0] = -9.0  # the background now carries 1.6% of the mass at T=1
    straddle = ramp.clone()
    straddle[hot[3]] = straddle[hot[4]]
    straddle[hot[5]] = straddle[hot[4]]  # tokens 3, 4, 5 tie
    # The cut falls inside the tie group (above its 1st member, below its 3rd): all three stay.
    q = so.reference_distribution(straddle)
    top = torch.sort(q, descending=True).values
    straddle_p = float(top[:3].sum() + 1.5 * top[3])
    return [
        so.RowSpec(ramp, 1.0, top_p=1e-6, name="p=1e-6 (top token only)"),
        so.RowSpec(ramp, 1.0, top_p=so.mid_cut_top_p(ramp, 1.0, 0, 4), name="mid cut, 5 kept"),
        so.RowSpec(ramp, 0.3, top_p=so.mid_cut_top_p(ramp, 0.3, 0, 4), name="T=0.3, 5 kept"),
        so.RowSpec(ramp, 2.0, top_p=0.8, name="T=2, p=0.8 (drops the 15% tail)"),
        so.RowSpec(ramp, 5.0, top_p=0.999999, name="T=5, p=0.999999 (keeps all)"),
        so.RowSpec(shallow, 1.0, top_p=0.98, name="p=0.98 over a 1.6% tail"),
        so.RowSpec(straddle, 1.0, top_p=straddle_p, name="three-way tie straddling the cut"),
        so.RowSpec(ramp, 1.0, top_p=0.0, name="sentinel p=0 (unfiltered)"),
        so.RowSpec(ramp, 0.7, top_p=1.0, name="p=1 (unfiltered)"),
    ]


def test_nucleus_rows_of_a_batch(monkeypatch):
    """Rows with different p and temperatures, sentinels among them, each sample their nucleus."""
    specs, launches = _nucleus_specs(), 64
    batch = so.build_batch(specs, replicas=512)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(
        so.make_sampler(batch.vocab), batch, so.batch_context(batch), launches
    )
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=KERNEL)


def _dyadic_row(vocab, g):
    """Exact probabilities: 1/4 on one token, 1/8 on two, 1/16 on four, 2**-12 on 1024 tokens."""
    masses = [2.0**-2] + [2.0**-3] * 2 + [2.0**-4] * 4 + [2.0**-12] * 1024
    row = torch.zeros(vocab, dtype=torch.float64)
    row[torch.randperm(vocab, generator=g)[: len(masses)]] = torch.tensor(
        masses, dtype=torch.float64
    )
    assert float(row.sum()) == 1.0
    return row.to(torch.float32)


# p on, just above and just below each cumulative boundary of the dyadic row (1/4, 1/2, 3/4).
DYADIC_TOP_P = [
    2.0**-2,
    2.0**-2 + 2.0**-10,
    2.0**-1 - 2.0**-10,
    2.0**-1,
    2.0**-1 + 2.0**-10,
    0.75,
    0.75 + 2.0**-12,
    1.0,
]


def test_cut_exactly_on_a_cumulative_boundary():
    """Direct call as Megatron makes it, on exact dyadic probabilities: boundaries are exact."""
    vocab, replicas, launches = 4096, 512, 32
    g = ou.make_generator()
    row = _dyadic_row(vocab, g)
    probs = row.to(so.DEVICE).expand(len(DYADIC_TOP_P) * replicas, vocab).contiguous()
    top_p = torch.tensor(DYADIC_TOP_P, dtype=torch.float32).repeat_interleave(replicas)
    rng = so.make_rng()
    counts = torch.zeros((len(DYADIC_TOP_P), vocab), dtype=torch.int64)
    for _ in range(launches):
        tokens = so.flashinfer.sampling.top_p_sampling_from_probs(
            probs, top_p.to(so.DEVICE), deterministic=True, generator=rng
        ).long()
        tokens = tokens.cpu()
        assert tokens.min() >= 0 and tokens.max() < vocab
        for i in range(len(DYADIC_TOP_P)):
            counts[i] += torch.bincount(tokens[i * replicas : (i + 1) * replicas], minlength=vocab)
    refs = torch.stack(
        [so.filter_distribution(row, 0, p, exact_boundaries=True) for p in DYADIC_TOP_P]
    )
    point = (refs > 0).sum(-1) == 1
    ou.assert_counts_within_binomial_bounds(counts[point], refs[point], what=f"{KERNEL} exact")
    ou.assert_matches_distribution(counts[~point], refs[~point], what=f"{KERNEL} exact")


def test_heavy_tail_needs_many_rejection_rounds(monkeypatch):
    """A nucleus of a few tokens over a heavy tail, and a long geometric ramp, in fp32 logits."""
    vocab, launches = 32000, 64
    g = ou.make_generator()
    heavy = torch.full((vocab,), -8.25, dtype=torch.float32)
    heavy[torch.randperm(vocab, generator=g)[:8]] = 0.0  # 8 tied tokens, about half the mass
    geometric = -torch.arange(vocab, dtype=torch.float32) / 128  # exact in fp32
    geometric = geometric[torch.randperm(vocab, generator=g)]
    specs = [
        so.RowSpec(heavy, 1.0, top_p=0.4, name="8 tied tokens over a heavy tail"),
        so.RowSpec(
            geometric, 1.0, top_p=so.mid_cut_top_p(geometric, 1.0, 0, 300), name="geometric"
        ),
        so.RowSpec(geometric, 3.0, top_p=0.5, name="geometric, T=3"),
    ]
    batch = so.build_batch(specs, replicas=1024)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=KERNEL)


def test_seeded_replay_and_fresh_draws():
    """Same generator state, same tokens; the next call draws new ones."""
    specs = _nucleus_specs()[2:4]
    batch = so.build_batch(specs, replicas=2048)
    context = so.batch_context(batch)
    rng = so.make_rng()
    sampler = so.make_sampler(batch.vocab, rng)
    state = rng.get_state()
    first = so.sample(sampler, batch.logits, batch.rows, context)
    rng.set_state(state)
    ou.assert_exact(so.sample(sampler, batch.logits, batch.rows, context), first, what="replay")
    fresh = so.sample(sampler, batch.logits, batch.rows, context)
    assert not torch.equal(fresh, first), "the generator did not advance between calls"


@pytest.mark.parametrize("vocab", so.SAMPLING_VOCAB_SIZES)
def test_vocabulary_sizes_around_the_block_iteration(vocab, monkeypatch):
    """A mid cut over hot tokens at the first, last and block-edge ids of each vocabulary."""
    launches, count = 16, min(vocab, 24)
    logits, _ = _ramp(vocab, count)
    spec = so.RowSpec(
        logits, 1.0, top_p=so.mid_cut_top_p(logits, 1.0, 0, count // 2), name=f"V={vocab}"
    )
    plain = so.RowSpec(logits, 1.0, top_p=0.0, name="unfiltered row")
    batch = so.build_batch([spec, plain], replicas=1024 if vocab <= 65536 else 256)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, [spec, plain], what=f"{KERNEL} V={vocab}")


@pytest.mark.sanitizer
def test_sanitizer_odd_vocabulary():
    """A small odd-vocabulary case (vec 1) with a guard-banded output, for memcheck."""
    vocab = 4097
    logits, _ = _ramp(vocab, 16)
    spec = so.RowSpec(logits, 1.0, top_p=so.mid_cut_top_p(logits, 1.0, 0, 6))
    batch = so.build_batch([spec], replicas=512)
    sampler, context = so.make_sampler(vocab), so.batch_context(batch)
    counts = torch.zeros((1, vocab), dtype=torch.int64)
    for _ in range(8):
        out = ou.guarded_empty((batch.rows,), torch.int64)
        so.sample(sampler, batch.logits, batch.rows, context, output=out.tensor)
        ou.check_guards(out=out)
        counts[0] += torch.bincount(out.tensor.cpu(), minlength=vocab)
    so.assert_batch_distribution(counts, [spec], what=KERNEL)


@pytest.mark.parametrize("rows", ou.int32_overflow_params(262144))
def test_int32_row_offsets(rows):
    """Rows whose offsets row * V cross 2**31 still sample their own (one-token) nucleus."""
    vocab = 262144
    ou.require_free_memory(rows * vocab * 10 + 2**30)  # bf16 logits, fp32 scaled and probs
    logits = torch.full((rows, vocab), -30.0, dtype=torch.bfloat16, device=so.DEVICE)
    target = (torch.arange(rows, device=so.DEVICE) * 7919 + 13) % vocab
    logits[torch.arange(rows, device=so.DEVICE), target] = 30.0
    context = so.stub_context([1.0] * rows, [0] * rows, [0.5] * rows)
    tokens = so.sample(so.make_sampler(vocab), logits, rows, context)
    ou.assert_exact(tokens, target, what="the nucleus of each row is its one hot token")

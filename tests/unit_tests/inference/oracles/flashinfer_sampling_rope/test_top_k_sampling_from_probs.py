# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``flashinfer.sampling.top_k_sampling_from_probs`` at Megatron's call site.

Call site: ``FlashInferSampling.sample_kernel``
(``megatron/core/inference/sampling/flashinfer_sampling.py:127``), taken when some active request
uses top-k and none uses top-p. Megatron passes ``probs = softmax(l / max(T, 1e-6))`` in fp32,
a per-row int32 ``top_k`` with the disabled sentinel (``k <= 0``) mapped to ``V`` (``k > V`` is
passed through), ``deterministic=True`` and the controller's generator.

Contract (from the call site, FlashInfer's documented API and the definition of top-k sampling,
written before reading the kernel):

* Computes: for each row, one token drawn from ``q`` renormalized to
  ``S_k = {i : #{j : q_j > q_i} < k}``: the k most likely tokens and every token tied with the
  k-th. ``k <= 0`` (sentinel) and ``k >= V`` keep every token. Rows take their own k and
  temperature. (``sampling_oracle.reference_distribution`` steps 1-3, 5.)
* Greedy: a ``top_k == 1`` row samples the lowest-index argmax of its logits, the token the torch
  backend returns. This is the contract of open PR #8875 (``fix/sampling-contract``). On main a
  ``k = 1`` row with tied maxima draws at random among them, by batch row (finding R08-F2), so
  ``test_greedy_rows_take_the_lowest_index_maximum`` fails on main and passes with #8875; the
  ``k = 1`` rows elsewhere have a unique maximum and pass on both.
* Outputs, RNG: int64 ids in ``[0, V)``; the same generator state replays bit-exactly, every
  call advances it.
* Tolerance class: sampler. Chi-square per row group against the fp64 reference, false-failure
  rate ``alpha = 1e-6`` per assertion (Bonferroni over groups), fixed seed; any draw outside
  ``S_k`` fails exactly; a one-token reference is checked with exact binomial bounds.

Structure (read from FlashInfer 0.7.2 after the contract): rejection sampling with a pivot,
accepting a draw when fewer than k probabilities exceed it; one block of 1024 threads per row,
``VEC = gcd(4, V)``. Inputs: k in {1, 2, 5, 31, 32, 33} on a production vocabulary and
{31, 32, 33, V-1, V, V+1, 0 -> V} on a small one where the token at the cut carries visible
mass; exact ties at the k-th value; the vocabulary sweep of ``sampling_from_logits``.

Hardware: H100 (sm_90).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_sampling_rope import sampling_oracle as so

pytestmark = ou.requires_sm(ou.HOPPER)

KERNEL = "top_k_sampling_from_probs"


def _ramp(vocab=32000, count=48):
    g = ou.make_generator()
    hot = so.spread_positions(vocab, count, g)
    return so.ramp_logits(vocab, hot), hot


def _tie_at(ramp, hot, k):
    """The ramp with its (k-1)-th, k-th and (k+1)-th largest tokens (1-based) tied.

    S_k then keeps k + 1 tokens.
    """
    tied = ramp.clone()
    tied[hot[k - 2]] = tied[hot[k - 1]]
    tied[hot[k]] = tied[hot[k - 1]]
    return tied


def test_top_k_rows_of_a_batch(monkeypatch):
    """Rows with different k (ties at the k-th among them) and temperatures sample their top-k."""
    ramp, hot = _ramp()
    specs = [
        so.RowSpec(ramp, 1.0, top_k=1, name="k=1, unique maximum"),
        so.RowSpec(ramp, 1.0, top_k=2, name="k=2"),
        so.RowSpec(ramp, 0.3, top_k=5, name="k=5, T=0.3"),
        so.RowSpec(ramp, 5.0, top_k=31, name="k=31, T=5"),
        so.RowSpec(ramp, 2.0, top_k=32, name="k=32, T=2"),
        so.RowSpec(ramp, 1.0, top_k=33, name="k=33"),
        so.RowSpec(_tie_at(ramp, hot, 2), 1.0, top_k=2, name="k=2, 1st-3rd tied (3 kept)"),
        so.RowSpec(_tie_at(ramp, hot, 32), 3.0, top_k=32, name="k=32, 31st-33rd tied (33 kept)"),
        so.RowSpec(ramp, 0.0, top_k=3, name="k=3, T=0 -> 1e-6 (argmax)"),
        so.RowSpec(ramp, 1.5, top_k=0, name="sentinel k=0 (unfiltered)"),
    ]
    batch = so.build_batch(specs, replicas=400)
    launches = 64
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(
        so.make_sampler(batch.vocab), batch, so.batch_context(batch), launches
    )
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=KERNEL)


def test_k_around_the_vocabulary(monkeypatch):
    """k = V-1 drops exactly the least likely token; V, V+1 and the sentinel keep every token."""
    vocab, launches = 64, 32
    g = ou.make_generator()
    row = (-torch.arange(vocab, dtype=torch.float32) / 8)[torch.randperm(vocab, generator=g)]
    # T = 8: neighboring tokens differ by exp(-1/64), the least likely carries 1% of the mass.
    specs = [
        so.RowSpec(row, 8.0, top_k=k, name=f"k={k}")
        for k in (31, 32, 33, vocab - 1, vocab, vocab + 1, 0)
    ]
    batch = so.build_batch(specs, replicas=512)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=f"{KERNEL} V={vocab}")


def _tied_maximum_logits(vocab):
    """Ids 7 and 11 tie at the maximum (8.0, exact in bf16); every other id is lower."""
    logits = torch.linspace(-4.0, 4.0, vocab).to(torch.bfloat16)
    logits[7] = logits[11] = 8.0
    return logits


@pytest.mark.parametrize("mixed", [False, True], ids=["all-greedy", "greedy-among-top-k"])
def test_greedy_rows_take_the_lowest_index_maximum(mixed):
    """``top_k == 1`` rows return the lowest-index argmax: id 7 of the tied ids 7 and 11.

    Contract of PR #8875. Fails on main (R08-F2): FlashInfer's top-k kernel keeps both tied
    maxima at k = 1 and about half the rows return 11.
    """
    vocab, greedy_rows = 1024, 2048
    logits_row = _tied_maximum_logits(vocab)
    temperature = [0.0 if i % 2 else 0.7 for i in range(greedy_rows)]
    specs = [so.RowSpec(logits_row, t, top_k=1) for t in temperature]
    if mixed:
        specs += [so.RowSpec(logits_row, 1.0, top_k=5, name="k=5")] * 2048
    batch = so.build_batch(specs, replicas=1)
    tokens = so.sample(so.make_sampler(vocab), batch.logits, batch.rows, so.batch_context(batch))
    ou.assert_exact(
        tokens[:greedy_rows],
        torch.full((greedy_rows,), 7, dtype=torch.int64),
        what="greedy rows (top_k == 1) with ids 7 and 11 tied at the maximum",
    )
    if mixed:
        counts = torch.bincount(tokens[greedy_rows:].cpu(), minlength=vocab).reshape(1, -1)
        so.assert_batch_distribution(counts, specs[-1:], what="k=5 rows next to greedy rows")


@pytest.mark.parametrize("vocab", so.SAMPLING_VOCAB_SIZES)
def test_vocabulary_sizes_around_the_block_iteration(vocab, monkeypatch):
    """Top-k over hot tokens at the first, last and block-edge ids of each vocabulary."""
    count = min(vocab, 24)
    logits, _ = _ramp(vocab, count)
    k = max(2, count // 2) if vocab > 1 else 2
    spec = so.RowSpec(logits, 1.0, top_k=k, name=f"V={vocab}, k={k}")
    plain = so.RowSpec(logits, 1.0, top_k=0, name="sentinel row")
    batch = so.build_batch([spec, plain], replicas=1024 if vocab <= 65536 else 256)
    launches = 16
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, [spec, plain], what=f"{KERNEL} V={vocab}")


def test_seeded_replay_and_fresh_draws():
    """Same generator state, same tokens; the next call draws new ones."""
    ramp, _ = _ramp()
    specs = [so.RowSpec(ramp, 2.0, top_k=32), so.RowSpec(ramp, 1.0, top_k=0)]
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


@pytest.mark.sanitizer
def test_sanitizer_odd_vocabulary():
    """A small odd-vocabulary case (vec 1) with a guard-banded output, for memcheck."""
    vocab = 4097
    logits, _ = _ramp(vocab, 16)
    spec = so.RowSpec(logits, 1.0, top_k=6)
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
    """Rows whose offsets row * V cross 2**31 still sample their own top-2 (one hot token)."""
    vocab = 262144
    ou.require_free_memory(rows * vocab * 10 + 2**30)  # bf16 logits, fp32 scaled and probs
    logits = torch.full((rows, vocab), -30.0, dtype=torch.bfloat16, device=so.DEVICE)
    target = (torch.arange(rows, device=so.DEVICE) * 7919 + 13) % vocab
    logits[torch.arange(rows, device=so.DEVICE), target] = 30.0
    context = so.stub_context([1.0] * rows, [2] * rows, [0.0] * rows)
    tokens = so.sample(so.make_sampler(vocab), logits, rows, context)
    ou.assert_exact(tokens, target, what="the hot token of each row")

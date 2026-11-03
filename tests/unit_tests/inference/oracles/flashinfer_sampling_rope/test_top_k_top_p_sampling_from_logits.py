# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``flashinfer.sampling.top_k_top_p_sampling_from_logits`` at Megatron's call site.

Call site: ``FlashInferSampling.sample_kernel``
(``megatron/core/inference/sampling/flashinfer_sampling.py:135``), taken for mixed batches: some
requests use top-k and some top-p, or requests use both. Megatron passes the temperature-scaled
fp32 logits, a per-row int32 ``top_k`` (sentinel ``k <= 0`` -> ``V``), a per-row fp32 ``top_p``
(``p <= 0`` or ``p >= 1`` -> 1.0), ``deterministic=True``, the controller's generator and
FlashInfer's default ``filter_apply_order="top_k_first"``.

Contract (from the call site and FlashInfer's documented API, written before reading the kernel):

* Computes: for each row, one token drawn from ``sampling_oracle.reference_distribution``: the
  tempered softmax restricted to the tie-inclusive top-k, renormalized, then restricted to the
  tie-inclusive nucleus of that renormalized distribution, renormalized again. Top-k first: a
  row with both filters keeps a subset of what either filter alone would keep on the raw
  distribution (the rows below are built so that the two orders differ).
* Rows of one batch take their own k, p and temperature: k only, p only, both, neither.
* Greedy: a ``top_k == 1`` row samples the lowest-index argmax of its logits, whatever its top-p
  (open PR #8875). On main such a row with tied maxima draws at random among them (R08-F2):
  ``test_greedy_rows_in_a_mixed_batch`` fails on main and passes with #8875.
* Outputs, RNG: int64 ids in ``[0, V)``; the same generator state replays bit-exactly, every
  call advances it.
* Tolerance class: sampler. Chi-square per row group against the fp64 reference, false-failure
  rate ``alpha = 1e-6`` per assertion (Bonferroni over groups), fixed seed; any draw outside the
  kept set fails exactly.

Structure (read from FlashInfer 0.7.2 after the contract): Megatron's tensor ``top_k`` skips the
int-only fast path, so the kernel runs ``top_k_mask_logits`` (radix select, keeping logits
``>=`` the k-th) -> softmax -> ``top_p_sampling_from_probs``. The radix mask splits a row over
``ceil(V / 57592)`` CTAs on H100 (``sampling_oracle.RADIX_CHUNK_H100``): vocabularies around one
and two CTAs and the production sizes, with hot tokens on the chunk edges, cover the single- and
multi-CTA kernels and the cross-CTA pivot. Every vocabulary must launch; finding O4a-N1: a row
whose CTA chunk fills the shared memory to within the kernel's static scratch fails with
``invalid argument`` (measured at V = 115183 and 115184 for the multi-CTA mask),
``test_vocabularies_at_the_shared_memory_limit``.

Hardware: H100 (sm_90; the CTA split is derived from H100's 227 KiB opt-in shared memory).
"""

import math

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_sampling_rope import sampling_oracle as so

pytestmark = ou.requires_sm(ou.HOPPER)

KERNEL = "top_k_top_p_sampling_from_logits"


def _ramp(vocab=32000, count=48, positions=None):
    g = ou.make_generator()
    hot = positions if positions is not None else so.spread_positions(vocab, count, g)
    return so.ramp_logits(vocab, hot), hot


def _mixed_specs():
    ramp, hot = _ramp()
    tied = ramp.clone()
    tied[hot[4]] = tied[hot[5]] = tied[hot[3]]  # 4th-6th largest tie
    both_p = so.mid_cut_top_p(ramp, 1.0, 8, 3)  # top-8 renormalized, then 4 kept
    specs = [
        so.RowSpec(ramp, 1.0, top_k=8, top_p=both_p, name="k=8 then p (4 kept)"),
        so.RowSpec(ramp, 1.0, top_k=8, name="k=8 only"),
        so.RowSpec(ramp, 1.0, top_p=both_p, name="same p only (5 kept)"),
        so.RowSpec(ramp, 0.3, top_k=12, top_p=0.95, name="T=0.3, k=12, p=0.95"),
        so.RowSpec(ramp, 4.0, top_k=40, top_p=so.mid_cut_top_p(ramp, 4.0, 40, 20), name="T=4"),
        so.RowSpec(tied, 1.0, top_k=4, top_p=0.99, name="k=4 with 4th-6th tied (6 kept)"),
        so.RowSpec(ramp, 1.0, top_k=1, top_p=0.5, name="k=1 (unique maximum) with p"),
        so.RowSpec(ramp, 0.0, top_k=6, top_p=0.9, name="T=0 -> argmax"),
        so.RowSpec(ramp, 1.0, name="neither filter"),
    ]
    return specs


def test_mixed_rows_of_a_batch(monkeypatch):
    """k only, p only, both and neither in one batch: each row samples top-k first, then top-p."""
    specs, launches = _mixed_specs(), 64
    # The joint row and the p-only row must differ, or the test cannot tell the two orders apart.
    assert not torch.equal(specs[0].reference() > 0, specs[2].reference() > 0)
    batch = so.build_batch(specs, replicas=448)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(
        so.make_sampler(batch.vocab), batch, so.batch_context(batch), launches
    )
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=KERNEL)


def test_greedy_rows_in_a_mixed_batch():
    """``top_k == 1`` rows, with or without top-p, return the lowest tied maximum (id 7).

    Contract of PR #8875. Fails on main (R08-F2): the joint kernel's top-k mask keeps both
    tied maxima (ids 7 and 11) at k = 1, and about half the greedy rows return 11.
    """
    vocab = 1024
    logits_row = torch.linspace(-4.0, 4.0, vocab).to(torch.bfloat16)
    logits_row[7] = logits_row[11] = 8.0
    greedy = [so.RowSpec(logits_row, t, top_k=1, top_p=p) for t in (0.0, 0.8) for p in (0.0, 0.9)]
    others = [
        so.RowSpec(logits_row, 1.0, top_k=5, name="k=5"),
        so.RowSpec(logits_row, 1.0, top_p=0.9, name="p=0.9"),
    ]
    batch = so.build_batch(greedy + others, replicas=512)
    tokens = so.sample(so.make_sampler(vocab), batch.logits, batch.rows, so.batch_context(batch))
    n_greedy = 512 * len(greedy)
    ou.assert_exact(
        tokens[:n_greedy],
        torch.full((n_greedy,), 7, dtype=torch.int64),
        what="greedy rows (top_k == 1) with ids 7 and 11 tied at the maximum",
    )
    counts = torch.stack(
        [
            torch.bincount(tokens[rows].cpu(), minlength=vocab)
            for rows in batch.groups[len(greedy) :]
        ]
    )
    so.assert_batch_distribution(counts, others, what="filtered rows next to greedy rows")


def _chunk_edges(vocab):
    """Hot positions on the radix CTA chunk edges of ``vocab`` (H100) and at its ends."""
    ctas = max(1, math.ceil(vocab / so.RADIX_CHUNK_H100))
    vec = math.gcd(4, vocab)
    chunk = min(-(-math.ceil(vocab / ctas) // vec) * vec, so.RADIX_CHUNK_H100)
    edges = [0, 1, vocab - 2, vocab - 1]
    for c in range(1, ctas):
        edges += [c * chunk - 1, c * chunk]
    edges = [e for e in dict.fromkeys(edges) if 0 <= e < vocab]
    g = ou.make_generator()
    taken = set(edges)
    rest = [p for p in torch.randperm(vocab, generator=g).tolist() if p not in taken]
    positions = (edges + rest)[: max(len(edges), 24)]
    order = torch.randperm(len(positions), generator=g).tolist()
    return [positions[i] for i in order], ctas


@pytest.mark.parametrize("vocab", [1, 2, 1023, 4097] + so.RADIX_VOCAB_SIZES)
def test_radix_mask_cta_split(vocab, monkeypatch):
    """Top-k cuts whose kept tokens sit on the chunk edges of a one-, two- or many-CTA row."""
    positions, ctas = _chunk_edges(vocab)
    logits, _ = _ramp(vocab, positions=positions)
    count = len(positions)
    k = max(1, count // 2)
    specs = [
        so.RowSpec(logits, 1.0, top_k=k, name=f"V={vocab} ({ctas} CTAs), k={k}"),
        so.RowSpec(
            logits,
            1.0,
            top_k=k,
            top_p=so.mid_cut_top_p(logits, 1.0, k, max(0, k // 2 - 1)),
            name="k then p",
        ),
        so.RowSpec(logits, 1.0, top_p=0.9 if vocab > 2 else 0.5, name="p only"),
    ]
    if vocab == 1 or k == 1:
        specs = [specs[0], specs[2]]  # k = 1 is greedy, covered above
    batch = so.build_batch(specs, replicas=1024 if vocab <= 65536 else 256)
    launches = 16
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=f"{KERNEL} V={vocab}")


@pytest.mark.parametrize("vocab", so.RADIX_SMEM_LIMIT_VOCABS)
def test_vocabularies_at_the_shared_memory_limit(vocab, monkeypatch):
    """Mixed batches whose radix mask chunk ends at or just below the shared-memory limit.

    Finding O4a-N1 (see ``sampling_oracle.RADIX_SMEM_LIMIT_VOCABS``): on H100 (pdx 7849977) the
    multi-CTA ``top_k_mask_logits`` fails with ``invalid argument`` at V = 115183 and 115184
    (chunk 57592, no headroom); the single-CTA one runs at 57592.
    """
    positions, ctas = _chunk_edges(vocab)
    logits, _ = _ramp(vocab, positions=positions)
    specs = [
        so.RowSpec(logits, 1.0, top_k=12, name=f"V={vocab} ({ctas} CTAs), k=12"),
        so.RowSpec(logits, 1.0, top_p=0.9, name="p only"),
    ]
    batch = so.build_batch(specs, replicas=256)
    launches = 8
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=f"{KERNEL} V={vocab}")


def test_seeded_replay_and_fresh_draws():
    """Same generator state, same tokens; the next call draws new ones."""
    specs = _mixed_specs()[3:5]
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
def test_sanitizer_two_cta_vocabulary():
    """A two-CTA radix mask (V = 57593) with a guard-banded output, for memcheck."""
    vocab = so.RADIX_CHUNK_H100 + 1
    positions, _ = _chunk_edges(vocab)
    logits, _ = _ramp(vocab, positions=positions)
    specs = [so.RowSpec(logits, 1.0, top_k=6), so.RowSpec(logits, 1.0, top_p=0.8)]
    batch = so.build_batch(specs, replicas=64)
    sampler, context = so.make_sampler(vocab), so.batch_context(batch)
    out = ou.guarded_empty((batch.rows,), torch.int64)
    tokens = so.sample(sampler, batch.logits, batch.rows, context, output=out.tensor)
    ou.check_guards(out=out)
    counts = torch.stack(
        [torch.bincount(tokens[rows].cpu(), minlength=vocab) for rows in batch.groups]
    )
    refs = torch.stack([spec.reference() for spec in specs])
    # 64 draws per row group: exact support and wide per-token binomial bounds.
    ou.assert_counts_within_binomial_bounds(counts, refs, what=f"{KERNEL} V={vocab}")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(262144))
def test_int32_row_offsets(rows):
    """Rows whose offsets row * V cross 2**31 still sample their own hot token."""
    vocab = 262144
    ou.require_free_memory(rows * vocab * 14 + 2**30)  # bf16 logits, fp32 scaled, mask, probs
    logits = torch.full((rows, vocab), -30.0, dtype=torch.bfloat16, device=so.DEVICE)
    target = (torch.arange(rows, device=so.DEVICE) * 7919 + 13) % vocab
    logits[torch.arange(rows, device=so.DEVICE), target] = 30.0
    top_k = [2 if i % 2 else 0 for i in range(rows)]
    top_p = [0.0 if i % 2 else 0.5 for i in range(rows)]
    context = so.stub_context([1.0] * rows, top_k, top_p)
    tokens = so.sample(so.make_sampler(vocab), logits, rows, context)
    ou.assert_exact(tokens, target, what="the hot token of each row")

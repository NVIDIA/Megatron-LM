# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``flashinfer.sampling.top_k_renorm_probs`` and the processed log-probs built on it.

Call site: ``FlashInferSampling.log_probs_kernel``
(``megatron/core/inference/sampling/flashinfer_sampling.py:187``), used for
``logprobs_mode='processed_logprobs'`` when some active request filters. Megatron passes
``probs = softmax(l / max(T, 1e-6))`` in fp32 and a per-row int32 ``top_k`` in which no-op rows
(``k <= 0`` or ``k >= V``) carry ``V``; the result goes to ``top_p_renorm_probs`` and then
``log``. Rows with neither filter take ``log_softmax`` of the scaled logits instead.

Contract (from the call site and FlashInfer's documented API, written before reading the kernel):

* ``top_k_renorm_probs(probs, top_k)``: for each row, ``out_i = q_i / sum over S_k of q`` for
  ``i`` in ``S_k = {i : #{j : q_j > q_i} < k}`` (every token tied with the k-th kept), ``0``
  elsewhere; ``k >= V`` renormalizes the whole row (FlashInfer documents ``k`` in ``(0, V)``;
  Megatron relies on ``k = V`` meaning no filter). fp32 in and out; the input is not modified.
* ``log_probs_kernel``: row ``r`` is ``log pi_r``, with ``pi_r`` the distribution the sampler
  draws row ``r`` from (``sampling_oracle.reference_distribution``): finite exactly on the kept
  set, ``-inf`` elsewhere. Rows with neither filter (``k <= 0`` or ``k >= V``, no top-p) are
  ``log_softmax(l / max(T, 1e-6))`` in fp32, bit for bit. ``T = 0`` is that clamp, not a
  division by zero, also for filtered rows that are not greedy. Greedy rows (``top_k == 1``) are
  one-hot on the lowest-index argmax (``0`` there, ``-inf`` elsewhere): the contract of open PR
  #8875. On main a greedy row with tied maxima puts ``log(1/2)`` on both (R08-F2), so
  ``test_greedy_rows_score_one_hot`` fails on main and passes with #8875.
* Every token the sampler draws has a finite processed log-prob, and the draws follow
  ``exp(log_probs)``: sampling and scoring describe one distribution.
* Every vocabulary launches: a row is split over as many CTAs as its shared memory needs.
  Finding O4a-N1: sizes whose per-CTA chunk ends within a few dozen tokens of the 57592-token
  limit fail with ``invalid argument`` (``test_vocabularies_at_the_shared_memory_limit``).
* Tolerance class: floating point, bounded (``sampling_oracle.assert_renormalized`` and
  ``assert_log_probs``). The kept set is exact: zero (``-inf``) exactly off it, positive (finite)
  on it. A row's scale (its mass) within the fp32 summation error of FlashInfer's reduction tree
  plus its reciprocal; the values after that scale against the fp64 reference within
  ``ou.assert_close_to_reference``'s default (factor 2, one-ulp floor, per row), the trusted
  implementation being plain torch fp32 on the same GPU (softmax, mask the reference's kept
  set, renormalize, log) normalized the same way. (FlashInfer multiplies by ``rcp.approx`` of
  an fp32 sum taken per thread in sequence: on H100 its rows carry a common 1-5 ulp scale error
  that a direct comparison with torch's pairwise sum flags at up to 21x.) Statistical checks
  as for the samplers (``alpha = 1e-6``).

Structure (read from FlashInfer 0.7.2 after the contract): radix top-k with the same CTA split as
``top_k_mask_logits`` (``ceil(V / 57592)`` CTAs per row on H100); ``k >= V`` short-circuits to a
plain renormalization; with Megatron's default ``is_deterministic=False`` a multi-CTA row sums
with float atomics (bounded here, not replayed). Inputs: k in {1, 2, 31, 32, 33, V-1, V} per
row, ties at the k-th value, V around one/two CTAs and the production vocabularies.

Open PR #8572 (s12, ``fix/s12-processed-logprob-determinism``) replaces this call with
``top_k_mask_logits`` + softmax; the ``log_probs_kernel`` tests assert the contract, not the
kernel, so they hold on main, on #8572 and on #8875. The direct kernel tests hold wherever
FlashInfer does.

Hardware: H100 (sm_90).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_sampling_rope import sampling_oracle as so

pytestmark = ou.requires_sm(ou.HOPPER)

KERNEL = "top_k_renorm_probs"


def _rows(vocab, specs):
    """fp32 probabilities of each (logits, T) spec, as Megatron computes them, and k per row."""
    logits = torch.stack([s.logits.to(torch.float32) for s in specs]).to(so.DEVICE)
    temperature = torch.tensor([so.clamped_temperature(s.temperature) for s in specs])
    probs = torch.softmax(logits / temperature.to(so.DEVICE)[:, None], dim=-1)
    top_k = torch.tensor([s.top_k for s in specs], dtype=torch.int32, device=so.DEVICE)
    return probs, top_k


def _specs(vocab, positions):
    ramp = so.ramp_logits(vocab, positions, dtype=torch.float32)
    count = len(positions)
    tied = ramp.clone()
    if count >= 4:
        tied[positions[1]] = tied[positions[2]] = tied[positions[3]]  # 2nd-4th largest tie
    specs = []
    for k in (1, 2, 31, 32, 33, vocab - 1, vocab):
        if 0 < k <= vocab:
            specs.append(so.RowSpec(ramp, 1.0, top_k=k, name=f"k={k}"))
    if count >= 4:
        specs.append(so.RowSpec(tied, 0.5, top_k=3, name="k=3, 2nd-4th tied (4 kept)"))
    specs.append(so.RowSpec(ramp, 2.0, top_k=min(count, vocab), name="T=2, k=#hot"))
    return specs


@pytest.mark.parametrize("vocab", [2, 64, 1023, 4097] + so.RADIX_VOCAB_SIZES)
def test_renormalizes_the_top_k(vocab):
    """Per-row k (ties at the k-th, k = V-1 and V among them) over one-, two- and many-CTA rows."""
    g = ou.make_generator()
    count = min(vocab, 48)
    positions = so.spread_positions(vocab, count, g)
    specs = _specs(vocab, positions)
    if vocab > count:
        # One background token one step lower: k = V-1 drops exactly it.
        hot = set(positions)
        least = next(i for i in range(vocab) if i not in hot)
        for s in specs:
            s.logits = s.logits.clone()
            s.logits[least] = s.logits.min() - 1.0
    probs, top_k = _rows(vocab, specs)
    before = probs.clone()
    out = so.flashinfer.sampling.top_k_renorm_probs(probs, top_k)
    ou.assert_exact(probs, before, what="probs are read-only")
    assert out.dtype == torch.float32 and out.shape == probs.shape
    refs = torch.stack([so.filter_distribution(p, k) for p, k in zip(probs, top_k.tolist())])
    so.assert_renormalized(out, probs, refs, what=f"{KERNEL} V={vocab}")


@pytest.mark.parametrize(
    "vocab",
    [
        v if v in so.RADIX_RENORM_LAUNCHABLE_VOCABS else pytest.param(v, marks=so.XFAIL_O4A_N1)
        for v in so.RADIX_SMEM_LIMIT_VOCABS
    ],
)
def test_vocabularies_at_the_shared_memory_limit(vocab):
    """Rows whose radix CTA chunk ends at or just below the 57592-token shared-memory limit.

    Finding O4a-N1: FlashInfer gives each CTA ``2080 + 4 * chunk`` bytes of dynamic shared
    memory, up to the full 227 KiB opt-in limit, and the kernel's static shared memory (two cub
    block-reduction scratch buffers, 328-332 bytes) on top; it needs 333-336 bytes of headroom.
    On H100 every size here with less headroom fails with ``invalid argument`` (pdx 7849977,
    dfw 21223592: 57591, 57592, 115072, 115183, 115184, 172672, 230272, 287744, 287872), and so
    do the edge probes 57509-57511 and 115018 (pdx 7852166); the probe 115016, with 336 bytes,
    runs (pdx 7876398; ``sampling_oracle.RADIX_SMEM_LIMIT_VOCABS``). No Megatron change can fix
    it: every failing case is a strict xfail up to the FlashInfer the CI image pins
    (``sampling_oracle.XFAIL_O4A_N1``); the Megatron call site is
    ``test_log_probs_kernel_at_the_shared_memory_limit`` (#8572).
    """
    g = ou.make_generator()
    specs = _specs(vocab, so.spread_positions(vocab, 48, g))
    specs = [s for s in specs if s.top_k in (2, 32, vocab)]
    probs, top_k = _rows(vocab, specs)
    out = so.flashinfer.sampling.top_k_renorm_probs(probs, top_k)
    refs = torch.stack([so.filter_distribution(p, k) for p, k in zip(probs, top_k.tolist())])
    so.assert_renormalized(out, probs, refs, what=f"{KERNEL} V={vocab}")


@pytest.mark.sanitizer
def test_sanitizer_two_cta_rows():
    """A two-CTA row (V = 57593, odd: vec 1) inside a guard-banded input, for memcheck."""
    vocab = so.RADIX_CHUNK_H100 + 1
    g = ou.make_generator()
    specs = _specs(vocab, so.spread_positions(vocab, 48, g))[:4]
    probs, top_k = _rows(vocab, specs)
    guarded = ou.guarded_like(probs)
    out = so.flashinfer.sampling.top_k_renorm_probs(guarded.tensor, top_k)
    ou.check_guards(probs=guarded)
    refs = torch.stack([so.filter_distribution(p, k) for p, k in zip(probs, top_k.tolist())])
    so.assert_renormalized(out, probs, refs, what=f"{KERNEL} V={vocab}")


# ---------------------------------------------------------------------------------------------
# The call site: FlashInferSampling.log_probs_kernel
# ---------------------------------------------------------------------------------------------


def _trusted_log_probs(logits, specs, refs):
    """fp32 torch on the GPU: scale, softmax, keep the reference's set, renormalize, log."""
    temperature = torch.tensor([so.clamped_temperature(s.temperature) for s in specs])
    probs = torch.softmax(logits.float() / temperature.to(logits.device)[:, None], dim=-1)
    return torch.log(so.trusted_renorm(probs, refs))


def _log_probs(specs, *, greedy=True):
    batch = so.build_batch(specs, replicas=1)
    sampler = so.make_sampler(batch.vocab)
    log_probs = sampler.log_probs_kernel(batch.logits, so.batch_context(batch))
    refs = torch.stack([s.reference(greedy=greedy) for s in specs])
    return batch, log_probs, refs


@pytest.mark.parametrize("vocab", [32000, so.RADIX_CHUNK_H100 + 1, 151936])
def test_log_probs_kernel_scores_the_top_k_distribution(vocab):
    """Processed log-probs of top-k rows (and unfiltered rows) of one batch are log pi."""
    _check_top_k_log_probs(vocab)


@pytest.mark.parametrize("vocab", so.RENORM_UNLAUNCHABLE_PADDED_VOCABS[:2])
def test_log_probs_kernel_at_the_shared_memory_limit(vocab):
    """Processed log-probs at 128-padded vocabularies where ``top_k_renorm_probs`` cannot launch.

    Finding O4a-N1 at Megatron's call site: fails on main (``TopKRenormProb failed with error
    code invalid argument``, dfw 21223592 for the kernel at these sizes); passes with PR #8572,
    whose ``top_k_mask_logits`` + softmax launches at them (measured through the joint sampler).
    """
    _check_top_k_log_probs(vocab)


def _check_top_k_log_probs(vocab):
    g = ou.make_generator()
    positions = so.spread_positions(vocab, 48, g)
    specs = [s for s in _specs(vocab, positions) if s.top_k != 1]
    plain = so.RowSpec(specs[0].logits, 0.7, name="unfiltered row")
    specs.append(plain)
    batch, log_probs, refs = _log_probs(specs)
    assert log_probs.dtype == torch.float32 and log_probs.shape == batch.logits.shape
    trusted = _trusted_log_probs(batch.logits, specs, refs)
    so.assert_log_probs(log_probs, refs, trusted, what="log_probs_kernel (top-k)")
    _assert_unfiltered_rows_are_log_softmax(batch, log_probs, specs)


def _assert_unfiltered_rows_are_log_softmax(batch, log_probs, specs):
    """Rows with neither filter (k <= 0 or k >= V, p <= 0 or p >= 1) of a filtered batch are
    torch's log_softmax of the fp32-scaled logits, bit for bit."""
    vocab = batch.vocab
    rows = [
        i
        for i, s in enumerate(specs)
        if (s.top_k <= 0 or s.top_k >= vocab) and not 0.0 < s.top_p < 1.0
    ]
    assert rows, "the batch needs a row with neither filter"
    temperature = torch.tensor(batch.temperature, dtype=torch.float32, device=so.DEVICE)
    scaled = batch.logits / temperature.clamp(min=so.MIN_SAMPLING_TEMPERATURE)[:, None]
    names = ", ".join(specs[i].name for i in rows)
    ou.assert_exact(
        log_probs[rows],
        torch.log_softmax(scaled, dim=-1)[rows],
        what=f"rows with neither filter ({names})",
    )


def test_log_probs_kernel_at_zero_temperature():
    """Rows with T = 0 that are not greedy (k != 1) score the distribution at the clamped
    temperature ``max(T, 1e-6)``, not a division by zero.

    Over hot logits ``2**-20`` apart (exact in fp32; about 0.95 apart after the clamp) the
    distribution is not degenerate: a three-token top-k, a top-k then top-p, and an unfiltered row
    of the filtered batch (``log_softmax`` of the clamped scale, bit for bit). Over the ramp
    (logits 0.25 apart) a top-k row is one-hot on its unique maximum. Holds on main, on #8875
    (which maps a request's T = 0 to greedy before the kernel; the kernel still clamps) and on
    #8572; the greedy T = 0 rows are ``test_greedy_rows_score_one_hot``.
    """
    vocab = 32000
    g = ou.make_generator()
    positions = so.spread_positions(vocab, 48, g)
    fine = so.ramp_logits(vocab, positions, top=0.0, step=2.0**-20, dtype=torch.float32)
    ramp = so.ramp_logits(vocab, positions, dtype=torch.float32)
    specs = [
        so.RowSpec(fine, 0.0, top_k=3, name="T=0, k=3"),
        so.RowSpec(
            fine, 0.0, top_k=8, top_p=so.mid_cut_top_p(fine, 0.0, 8, 4), name="T=0, k=8 then p"
        ),
        so.RowSpec(ramp, 0.0, top_k=32, name="T=0, unique maximum (one-hot)"),
        so.RowSpec(fine, 0.0, name="T=0, unfiltered"),
    ]
    batch = so.build_batch(specs, replicas=1)
    log_probs = so.make_sampler(vocab).log_probs_kernel(batch.logits, so.batch_context(batch))
    filtered = specs[:3]
    # The unfiltered row's reference is 0 on the background (exp(-1.6e7)) where its log_softmax is
    # finite: it is checked bit for bit only.
    refs = torch.stack([s.reference() for s in filtered])
    trusted = _trusted_log_probs(batch.logits[:3], filtered, refs)
    so.assert_log_probs(log_probs[:3], refs, trusted, what="log_probs_kernel at T = 0")
    _assert_unfiltered_rows_are_log_softmax(batch, log_probs, specs)


def test_greedy_rows_score_one_hot():
    """``top_k == 1`` rows score the lowest-index argmax only (0 there, -inf elsewhere).

    Contract of PR #8875. Fails on main (R08-F2): ``top_k_renorm_probs`` keeps both maxima tied
    at ids 7 and 11 and the processed log-prob of each is log(1/2).
    """
    vocab = 1024
    logits_row = torch.linspace(-4.0, 4.0, vocab).to(torch.bfloat16)
    logits_row[7] = logits_row[11] = 8.0
    specs = [so.RowSpec(logits_row, t, top_k=1) for t in (0.0, 0.7, 1.0)]
    specs.append(so.RowSpec(logits_row, 1.0, top_k=5, name="k=5"))
    _, log_probs, refs = _log_probs(specs)
    expected = torch.log(refs[:3]).to(torch.float32).to(log_probs.device)
    ou.assert_exact(log_probs[:3], expected, what="greedy rows with ids 7 and 11 tied")


def test_sampled_tokens_have_finite_log_probs():
    """The sampler's draws lie in the scored support and follow exp(processed log-probs)."""
    vocab, launches = 32000, 32
    g = ou.make_generator()
    positions = so.spread_positions(vocab, 48, g)
    specs = [s for s in _specs(vocab, positions) if s.top_k in (2, 32, 3)]
    batch = so.build_batch(specs, replicas=1024)
    context = so.batch_context(batch)
    sampler = so.make_sampler(vocab)
    log_probs = sampler.log_probs_kernel(batch.logits, context)
    counts = so.sample_counts(sampler, batch, context, launches)
    scored = torch.stack([log_probs[rows.start].exp().double().cpu() for rows in batch.groups])
    ou.assert_matches_distribution(counts, scored, what="draws against exp(processed log-probs)")

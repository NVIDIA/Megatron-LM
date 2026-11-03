# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``flashinfer.sampling.top_p_renorm_probs`` and the processed log-probs built on it.

Call site: ``FlashInferSampling.log_probs_kernel``
(``megatron/core/inference/sampling/flashinfer_sampling.py:188``), used for
``logprobs_mode='processed_logprobs'`` when some active request filters. Megatron passes the
top-k-renormalized fp32 probabilities (zeros outside each row's top-k) and a per-row fp32
``top_p`` with the no-op values (``p <= 0``, ``p >= 1``) mapped to 1.0, without
``is_deterministic`` (default False); the result goes to ``log``. Before this module, no test
ran it with ``p < 1``.

Contract (from the call site and FlashInfer's documented API, written before reading the kernel):

* ``top_p_renorm_probs(probs, top_p)``: for each row, ``out_i = q_i / sum over S of q`` on the
  nucleus ``S = {i : sum of q_j over q_j > q_i < p}`` (the threshold v with
  ``mass(q > v) < p <= mass(q >= v)``, as FlashInfer documents it; every token tied at the cut
  kept), ``0`` elsewhere. ``p = 1`` keeps every ``q > 0``. A cut exactly on a cumulative
  boundary excludes the next token. fp32 in and out; the input is not modified; the result does
  not depend on ``is_deterministic`` (Megatron's main passes False, open PR #8572 passes True).
* ``log_probs_kernel``: row ``r`` is ``log pi_r`` (``sampling_oracle.reference_distribution``,
  top-k first, then top-p): finite exactly on the kept set, ``-inf`` elsewhere.
* Determinism: ``log_probs_kernel`` replays bit-exactly (RL compares these log-probs with the
  trainer's). On main, ``top_p_renorm_probs`` builds its radix histograms with float atomics and
  ``top_k_renorm_probs`` sums rows split over several CTAs with float atomics, so a row whose
  cut sits within rounding of a cumulative boundary can keep a different set from run to run:
  ``test_log_probs_kernel_replays_bit_exactly`` can fail on main; open PR #8572 (s12,
  ``fix/s12-processed-logprob-determinism``) makes it pass.
* Tolerance class: floating point, bounded (``sampling_oracle.assert_renormalized`` and
  ``assert_log_probs``). The kept set is exact. A row's scale (its mass) within the fp32
  summation error of FlashInfer's reduction tree plus its reciprocal; the values after that
  scale against the fp64 reference within ``ou.assert_close_to_reference``'s default (factor 2,
  one-ulp floor, per row), the trusted implementation being plain torch fp32 on the same GPU
  (mask the reference's kept set, renormalize, log) normalized the same way. (FlashInfer sums
  the kept mass per thread in sequence and multiplies by its reciprocal, ``rcp.approx`` in the
  ternary search: on H100 a single kept token comes out as ``1 + 2**-23``, and long rows carry
  a common scale error that a direct comparison with torch's pairwise sum flags at up to 24x.)

Structure (read from FlashInfer 0.7.2 after the contract): ``V < 2048`` runs a ternary search
(``csrc/renorm.cu:44``); ``V >= 2048`` runs AIR top-p (``air_top_p.cuh``): three radix passes of
11 bits over the fp32 bits, a candidate buffer of ``max(256, align256(V / 32))`` entries that is
re-read when more candidates share the threshold bucket, and float (default) or integer
(``is_deterministic``) histograms. Inputs: V in {2, 1023, 2047, 2048, 2049, 32000, 151936,
262144}; per-row p near 0, mid cuts, near 1 and 1; three-way ties straddling the cut; zeros from
a top-k stage; exact dyadic boundaries with 8192 tokens tied at the threshold (8x the candidate
buffer at V = 32000).

Hardware: H100 (sm_90).
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_sampling_rope import sampling_oracle as so

pytestmark = ou.requires_sm(ou.HOPPER)

KERNEL = "top_p_renorm_probs"
DETERMINISTIC = pytest.mark.parametrize(
    "is_deterministic", [False, True], ids=["default-float-hist", "deterministic"]
)


def _nucleus_rows(vocab):
    """Rows of fp32 probabilities (as the top-k stage hands them over) and their top-p."""
    g = ou.make_generator()
    count = min(vocab, 48)
    positions = so.spread_positions(vocab, count, g)
    ramp = so.ramp_logits(vocab, positions, dtype=torch.float32)
    straddle = ramp.clone()
    if count >= 6:
        straddle[positions[3]] = straddle[positions[5]] = straddle[positions[4]]
    rows = []  # (logits, temperature, top_k for the stage before, top_p)
    rows.append((ramp, 1.0, 0, 1e-6))
    rows.append((ramp, 1.0, 0, so.mid_cut_top_p(ramp, 1.0, 0, min(4, count - 1))))
    rows.append((ramp, 0.5, 0, 0.5))
    rows.append((ramp, 2.0, 0, 0.9))
    rows.append((ramp, 3.0, 0, 0.999999))
    rows.append((ramp, 1.0, 0, 1.0))
    if count >= 24:
        rows.append((ramp, 1.0, 20, so.mid_cut_top_p(ramp, 1.0, 20, 9)))  # zeros past the top-20
    if count >= 6:
        q = so.reference_distribution(straddle)
        top = torch.sort(q, descending=True).values
        rows.append((straddle, 1.0, 0, float(top[:3].sum() + 1.5 * top[3])))
    probs, refs, top_p = [], [], []
    for logits, t, k, p in rows:
        q = torch.softmax(logits.to(so.DEVICE) / so.clamped_temperature(t), dim=-1)
        if k:  # what the top-k stage hands over: zeros past the k-th, renormalized (torch)
            q = torch.where(q >= q.topk(k).values[-1], q, torch.zeros_like(q))
            q = q / q.sum()
        probs.append(q)
        top_p.append(p)
    probs = torch.stack(probs)
    refs = torch.stack([so.filter_distribution(q, 0, p) for q, p in zip(probs, top_p)])
    return probs, torch.tensor(top_p, dtype=torch.float32, device=so.DEVICE), refs


@DETERMINISTIC
@pytest.mark.parametrize("vocab", [2, 1023, 2047, 2048, 2049, 32000, 151936, 262144])
def test_renormalizes_the_nucleus(vocab, is_deterministic):
    """Per-row top-p over ternary-search and AIR vocabularies, with ties, zeros and p near 0/1."""
    probs, top_p, refs = _nucleus_rows(vocab)
    before = probs.clone()
    out = so.flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=is_deterministic)
    ou.assert_exact(probs, before, what="probs are read-only")
    assert out.dtype == torch.float32 and out.shape == probs.shape
    so.assert_renormalized(out, probs, refs, what=f"{KERNEL} V={vocab}")


def _dyadic_row(vocab, tied, small, g):
    """Exact probabilities: 4 tokens of 1/8, ``tied`` tokens sharing 1/4, ``small`` sharing 1/4."""
    masses = [2.0**-3] * 4 + [0.25 / tied] * tied + [0.25 / small] * small
    row = torch.zeros(vocab, dtype=torch.float64)
    row[torch.randperm(vocab, generator=g)[: len(masses)]] = torch.tensor(
        masses, dtype=torch.float64
    )
    assert float(row.sum()) == 1.0
    return row.to(torch.float32)


# On, just below and just above the cumulative boundaries 1/2 and 3/4 of the dyadic rows.
DYADIC_TOP_P = [0.5 - 2.0**-20, 0.5, 0.5 + 2.0**-20, 0.6, 0.75, 0.75 + 2.0**-20, 1.0]


@DETERMINISTIC
@pytest.mark.parametrize(
    "vocab,tied,small", [(1024, 256, 512), (32000, 8192, 16384)], ids=["ternary", "air"]
)
def test_cut_exactly_on_a_cumulative_boundary(vocab, tied, small, is_deterministic):
    """Exact dyadic rows: a cut on a boundary drops the tie group below it; just above keeps it.

    At V = 32000 the 8192 tokens tied at the 1/2 boundary are 8x AIR's candidate buffer.
    """
    g = ou.make_generator()
    row = _dyadic_row(vocab, tied, small, g)
    probs = row.to(so.DEVICE).expand(len(DYADIC_TOP_P), vocab).contiguous()
    top_p = torch.tensor(DYADIC_TOP_P, dtype=torch.float32, device=so.DEVICE)
    out = so.flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=is_deterministic)
    refs = torch.stack(
        [so.filter_distribution(row, 0, p, exact_boundaries=True) for p in DYADIC_TOP_P]
    )
    so.assert_renormalized(out, probs, refs, what=f"{KERNEL} exact boundaries V={vocab}")


@pytest.mark.sanitizer
def test_sanitizer_smallest_air_vocabulary():
    """V = 2049 (AIR, odd) inside a guard-banded input, for memcheck."""
    probs, top_p, refs = _nucleus_rows(2049)
    guarded = ou.guarded_like(probs)
    out = so.flashinfer.sampling.top_p_renorm_probs(guarded.tensor, top_p)
    ou.check_guards(probs=guarded)
    so.assert_renormalized(out, probs, refs, what=f"{KERNEL} V=2049")


# ---------------------------------------------------------------------------------------------
# The call site: FlashInferSampling.log_probs_kernel
# ---------------------------------------------------------------------------------------------


def _log_prob_specs(vocab):
    g = ou.make_generator()
    positions = so.spread_positions(vocab, 48, g)
    ramp = so.ramp_logits(vocab, positions, dtype=torch.float32)
    return [
        so.RowSpec(ramp, 1.0, top_p=so.mid_cut_top_p(ramp, 1.0, 0, 6), name="p (7 kept)"),
        so.RowSpec(ramp, 0.5, top_p=0.9, name="T=0.5, p=0.9"),
        so.RowSpec(ramp, 2.0, top_k=30, top_p=so.mid_cut_top_p(ramp, 2.0, 30, 14), name="k then p"),
        so.RowSpec(ramp, 1.0, top_k=12, name="k only"),
        so.RowSpec(ramp, 1.5, top_p=0.999999, name="p=0.999999"),
        so.RowSpec(ramp, 1.0, name="unfiltered"),
    ]


@pytest.mark.parametrize("vocab", [2047, 32000, 151936])
def test_log_probs_kernel_scores_the_nucleus(vocab):
    """Processed log-probs of top-p, top-k + top-p, top-k and plain rows of one batch are log pi."""
    specs = _log_prob_specs(vocab)
    batch = so.build_batch(specs, replicas=1)
    log_probs = so.make_sampler(vocab).log_probs_kernel(batch.logits, so.batch_context(batch))
    refs = torch.stack([s.reference() for s in specs])
    temperature = torch.tensor([so.clamped_temperature(s.temperature) for s in specs])
    probs = torch.softmax(batch.logits / temperature.to(so.DEVICE)[:, None], dim=-1)
    trusted = torch.log(so.trusted_renorm(probs, refs))
    so.assert_log_probs(log_probs, refs, trusted, what="log_probs_kernel")


def test_log_probs_kernel_replays_bit_exactly():
    """Two calls on the same rows return the same bits, also for cuts on a cumulative boundary.

    Can fail on main (float-atomic histograms and multi-CTA sums, see the module docstring);
    passes with PR #8572 (s12).
    """
    vocab, rows = 152064, 64
    g = ou.make_generator(device=so.DEVICE)
    g_cpu = ou.make_generator()
    logits = torch.randn((rows, vocab), generator=g, device=so.DEVICE) * 3.0
    probs = torch.softmax(logits, dim=-1)
    # Half the rows cut exactly on a cumulative mass of their own sorted probabilities.
    sorted_probs = probs.double().sort(dim=-1, descending=True).values
    cumulative = sorted_probs.cumsum(-1) / sorted_probs.sum(-1, keepdim=True)
    index = torch.randint(8, 2048, (rows,), generator=g, device=so.DEVICE)
    top_p = cumulative.gather(1, index[:, None])[:, 0].float().cpu()
    top_p[rows // 2 :] = 0.5 + 0.4 * torch.rand(rows - rows // 2, generator=g_cpu)
    top_k = torch.where(torch.arange(rows) % 4 == 0, 1000, 0).tolist()
    context = so.stub_context([1.0] * rows, top_k, top_p.tolist())
    sampler = so.make_sampler(vocab)
    first = sampler.log_probs_kernel(logits, context)
    for _ in range(4):
        ou.assert_exact(sampler.log_probs_kernel(logits, context), first, what="log_probs replay")

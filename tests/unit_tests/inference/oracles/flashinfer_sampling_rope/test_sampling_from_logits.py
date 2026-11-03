# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for ``flashinfer.sampling.sampling_from_logits`` at Megatron's call site.

Call site: ``FlashInferSampling.sample_kernel``
(``megatron/core/inference/sampling/flashinfer_sampling.py:113``), taken when no active request
uses top-k or top-p (batch-level flags read from the pinned CPU metadata).

Contract (from the call site and FlashInfer's documented API, written before reading the kernel):

* Computes: for each of the first ``n`` rows, one token drawn from ``softmax(l / max(T, 1e-6))``,
  where ``l`` is ``logits[i]`` (``logits[gather_indices[i]]`` when given) and ``T`` the row's
  temperature (``gpu_view.temperature[i]``, or its request's under ``token_to_request_index``).
  Rows draw independently.
* Inputs: logits ``[>= n, V]`` in bf16 or fp32, scaled in fp32; fp32 temperatures; the
  controller's seeded CUDA ``torch.Generator``.
* Outputs: int64 ``[n]`` token ids in ``[0, V)``; written into ``output`` (all ``n`` entries,
  nothing outside) and returned when ``output`` is given.
* RNG: the same generator state replays the same tokens; every call advances the generator, so
  the next call draws afresh. Draws also depend on the batch row (documented, not asserted).
* Degenerate: ``T = 0`` is clamped to 1e-6: the argmax, uniform among exact ties (finding
  O4a-N2: FlashInfer skews the choice among ties at such temperatures,
  ``test_near_zero_temperature_splits_exact_ties_evenly``). A token whose
  logit is ``-inf`` is never drawn. A row of NaN logits has no distribution: its token is
  unspecified (S11; Megatron requests no NaN check) but must be a valid id and must not disturb
  the other rows.
* Tolerance class: sampler. Chi-square of each row group's counts against the fp64 reference
  (``sampling_oracle.reference_distribution``), false-failure rate ``alpha = 1e-6`` per
  assertion split over the groups (Bonferroni), fixed seed; a draw outside the reference
  support fails exactly; a one-token reference is checked with exact binomial bounds.

Structure (read from FlashInfer 0.7.2 after the contract): one compiled variant per
vec size; 1024 threads per row on sm >= 80, ``VEC = gcd(4, V)``, so a block iteration covers
4096 tokens (4 | V), 2048 (V = 2 mod 4) or 1024 (odd V). The vocabulary sweep takes V around
1024/2048/4096/8192, V = 4094 (vec 2) and the production vocabularies. Megatron-side branches:
plain rows, ``gather_indices``, ``token_to_request_index`` and ``output``.

Memory: ``output`` in a guard-banded, poisoned buffer; ``gather_indices`` entries past ``n``
poisoned (a read of them indexes out of range); a ``sanitizer`` case at V = 4095 and 4097.
int32 offsets: rows x V crossing 2**31 (``large``, opt-in).

Hardware: H100 (sm_90). Open PRs: #8875 (``fix/sampling-contract``) leaves this path unchanged
for batches without top-k; every test here must pass on main and on #8875.
"""

import pytest
import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.flashinfer_sampling_rope import sampling_oracle as so

pytestmark = ou.requires_sm(ou.HOPPER)

KERNEL = "sampling_from_logits"


def _tempered_specs(vocab):
    g = ou.make_generator()
    hot = so.spread_positions(vocab, 48, g)
    ramp = so.ramp_logits(vocab, hot)
    tied = ramp.clone()
    tied[hot[1]] = tied[hot[0]]
    tied[hot[2]] = tied[hot[0]]
    masked = ramp.clone()
    masked[torch.randperm(vocab, generator=g)[: vocab // 2]] = float("-inf")
    return [
        so.RowSpec(ramp, 1.0, name="T=1"),
        so.RowSpec(ramp, 0.3, name="T=0.3"),
        so.RowSpec(ramp, 5.0, name="T=5 (tail-dominated)"),
        so.RowSpec(tied, 1.0, name="T=1 with tied maxima"),
        so.RowSpec(masked, 0.7, name="half the vocabulary -inf"),
    ]


def test_draws_follow_the_tempered_softmax(monkeypatch):
    """Rows of different temperatures in one batch each sample their tempered softmax."""
    vocab, launches = 32000, 64
    specs = _tempered_specs(vocab)
    batch = so.build_batch(specs, replicas=512)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=KERNEL)


@pytest.mark.parametrize(
    "top,temperature",
    [(4.0, 0.0), (32.0, 0.0), (4.0, 1e-4)],
    ids=["logit4-T0", "logit32-T0", "logit4-T1e-4-control"],
)
def test_near_zero_temperature_splits_exact_ties_evenly(top, temperature, monkeypatch):
    """Three exactly tied maxima at a temperature near the clamp are drawn 1/3 each.

    Finding O4a-N2: fails on H100 (pdx 7849977, top logit 4, T = 0 -> 1e-6: counts 10061 /
    11652 / 11055 of 32768, chi2 118 on 2 dof, p = 2e-26). FlashInfer's Gumbel race adds the
    noise to the fp32 scaled logit (``sampling.cuh:777``), whose spacing at 4e6 is 0.25, so
    tied tokens tie again in the race and ``DataAndIndex`` (``:670``) breaks those ties by
    position. The control (T = 1e-4: spacing 0.004) should pass. PR #8875 maps a request's
    T = 0 to greedy, so the case remains for explicit temperatures near the clamp.
    """
    vocab, launches = 32000, 16
    g = ou.make_generator()
    hot = so.spread_positions(vocab, 48, g)
    tied = so.ramp_logits(vocab, hot, top=top)
    tied[hot[1]] = tied[hot[0]]
    tied[hot[2]] = tied[hot[0]]
    spec = so.RowSpec(tied, temperature, name=f"three tied maxima at {top}, T={temperature}")
    batch = so.build_batch([spec], replicas=4096)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, [spec], what=KERNEL)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("vocab", so.SAMPLING_VOCAB_SIZES)
def test_vocabulary_sizes_around_the_block_iteration(vocab, dtype, monkeypatch):
    """Hot tokens on the first, last and block-edge ids of each vocabulary are drawn correctly."""
    launches = 16
    g = ou.make_generator()
    hot = so.spread_positions(vocab, min(vocab, 24), g)
    spec = so.RowSpec(so.ramp_logits(vocab, hot, dtype=dtype), 0.8, name=f"V={vocab}")
    replicas = 2048 if vocab <= 65536 else 512
    batch = so.build_batch([spec], replicas=replicas)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = so.sample_counts(so.make_sampler(vocab), batch, so.batch_context(batch), launches)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, [spec], what=f"{KERNEL} V={vocab}")


def test_seeded_replay_and_fresh_draws():
    """The same generator state replays bit-exactly; the next call draws new tokens."""
    vocab, rows = 4096, 4096
    logits = torch.zeros((rows, vocab), dtype=torch.bfloat16, device=so.DEVICE)
    context = so.stub_context([1.0] * rows, [0] * rows, [0.0] * rows)
    rng = so.make_rng()
    sampler = so.make_sampler(vocab, rng)
    state = rng.get_state()
    first = so.sample(sampler, logits, rows, context)
    rng.set_state(state)
    replay = so.sample(sampler, logits, rows, context)
    ou.assert_exact(replay, first, what="replay from the same generator state")
    fresh = so.sample(sampler, logits, rows, context)
    # 4096 uniform draws over 4096 tokens: equal to the previous call with probability 4096**-4096.
    assert not torch.equal(fresh, replay), "the generator did not advance between calls"
    again = so.sample(so.make_sampler(vocab), logits, rows, context)
    ou.assert_exact(again, first, what="a new sampler with the same seed")
    # Both calls are uniform draws: 16 bins of 256 tokens, 512 expected per bin.
    coarse = torch.bincount(torch.cat([first, fresh]) // 256, minlength=16).reshape(1, -1).cpu()
    ou.assert_matches_distribution(coarse, torch.ones(1, 16), what="uniform logits")


def test_gather_indices_and_output_buffer(monkeypatch):
    """Row i samples logits[gather_indices[i]] with its own temperature into ``output``.

    Every source row is gathered by rows of both temperatures, in a seeded random order, so a row
    that took the temperature of the row it gathers (or of any fixed row) samples the wrong
    distribution in at least one (source, temperature) group.
    """
    vocab, n, launches = 4097, 3000, 32
    temperatures = (0.5, 1.0)
    g = ou.make_generator()
    sources = [
        so.ramp_logits(vocab, so.spread_positions(vocab, 16, g), dtype=torch.float32)
        for _ in range(3)
    ]
    logits = torch.stack(sources).to(so.DEVICE)
    combos = [(s, t) for s in range(len(sources)) for t in temperatures]
    combo_of_row = torch.arange(len(combos)).repeat(n // len(combos))
    combo_of_row = combo_of_row[torch.randperm(n, generator=g)]
    gather = ou.poisoned_empty((n + 64,), torch.int64)  # entries past n index out of range
    gather[:n] = torch.tensor([combos[c][0] for c in combo_of_row.tolist()], device=so.DEVICE)
    temperature = [combos[c][1] for c in combo_of_row.tolist()]
    context = so.stub_context(temperature, [0] * n, [0.0] * n)
    specs = [so.RowSpec(sources[s], t, name=f"source {s}, T={t}") for s, t in combos]
    sampler = so.make_sampler(vocab)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    counts = torch.zeros((len(combos), vocab), dtype=torch.int64)
    for _ in range(launches):
        out = ou.guarded_empty((n,), torch.int64)
        returned = so.sample(sampler, logits, n, context, gather_indices=gather, output=out.tensor)
        ou.check_guards(out=out)
        assert returned.data_ptr() == out.tensor.data_ptr(), "tokens not returned in `output`"
        tokens = out.tensor.cpu()
        for c in range(len(combos)):
            counts[c] += torch.bincount(tokens[combo_of_row == c], minlength=vocab)
    spy.assert_only(KERNEL, launches)
    so.assert_batch_distribution(counts, specs, what=f"{KERNEL} gather_indices")


def test_token_to_request_index(monkeypatch):
    """Under ``token_to_request_index`` each row takes its request's temperature."""
    vocab, per_request, launches = 32000, 768, 32
    g = ou.make_generator()
    logits_row = so.ramp_logits(vocab, so.spread_positions(vocab, 32, g))
    temperatures = [0.3, 1.0, 2.0, 0.0]
    request_of_row = torch.arange(4).repeat_interleave(per_request)
    request_of_row = request_of_row[torch.randperm(len(request_of_row), generator=g)]
    n = len(request_of_row)
    logits = logits_row.to(so.DEVICE).expand(n, vocab).contiguous()
    context = so.stub_context(temperatures, [0] * 4, [0.0] * 4)
    sampler = so.make_sampler(vocab)
    spy = so.KernelSpy(monkeypatch, so.SAMPLER_KERNELS)
    index = request_of_row.to(so.DEVICE)
    counts = torch.zeros((4, vocab), dtype=torch.int64)
    for _ in range(launches):
        tokens = so.sample(sampler, logits, n, context, token_to_request_index=index).cpu()
        for r in range(4):
            counts[r] += torch.bincount(tokens[request_of_row == r], minlength=vocab)
    spy.assert_only(KERNEL, launches)
    specs = [so.RowSpec(logits_row, t, name=f"request T={t}") for t in temperatures]
    so.assert_batch_distribution(counts, specs, what=f"{KERNEL} token_to_request_index")


def test_nan_rows_do_not_disturb_other_rows(monkeypatch):
    """NaN rows return valid ids (unspecified which) and leave every other row's draws intact."""
    vocab, replicas, launches = 32000, 512, 32
    specs = _tempered_specs(vocab)[:3]
    batch = so.build_batch(specs, replicas=replicas)
    nan_rows = torch.arange(0, batch.rows, 7, device=so.DEVICE)
    batch.logits[nan_rows] = float("nan")
    one_nan = torch.arange(3, batch.rows, 11, device=so.DEVICE)
    batch.logits[one_nan, 5] = float("nan")
    poisoned = torch.zeros(batch.rows, dtype=torch.bool)
    poisoned[nan_rows.cpu()] = True
    poisoned[one_nan.cpu()] = True
    sampler = so.make_sampler(vocab)
    context = so.batch_context(batch)
    counts = torch.zeros((len(specs), vocab), dtype=torch.int64)
    for _ in range(launches):
        tokens = so.sample(sampler, batch.logits, batch.rows, context).cpu()  # checks the id range
        for gi, rows in enumerate(batch.groups):
            keep = ~poisoned[rows]
            counts[gi] += torch.bincount(tokens[rows][keep], minlength=vocab)
    so.assert_batch_distribution(counts, specs, what=f"{KERNEL} rows next to NaN rows")


@pytest.mark.sanitizer
@pytest.mark.parametrize("vocab", [4095, 4097])
def test_sanitizer_odd_vocabulary(vocab):
    """A small odd-vocabulary case (vec 1) for compute-sanitizer memcheck."""
    g = ou.make_generator()
    spec = so.RowSpec(so.ramp_logits(vocab, so.spread_positions(vocab, 8, g)), 1.0)
    batch = so.build_batch([spec], replicas=1024)
    sampler = so.make_sampler(vocab)
    context = so.batch_context(batch)
    counts = torch.zeros((1, vocab), dtype=torch.int64)
    for _ in range(8):
        out = ou.guarded_empty((batch.rows,), torch.int64)
        so.sample(sampler, batch.logits, batch.rows, context, output=out.tensor)
        ou.check_guards(out=out)
        counts[0] += torch.bincount(out.tensor.cpu(), minlength=vocab)
    so.assert_batch_distribution(counts, [spec], what=f"{KERNEL} V={vocab}")


@pytest.mark.parametrize("rows", ou.int32_overflow_params(262144))
def test_int32_row_offsets(rows):
    """Rows whose offsets row * V cross 2**31 still sample their own (one-token) row."""
    vocab = 262144
    ou.require_free_memory(rows * vocab * 6 + 2**30)  # bf16 logits + fp32 scaled copy
    logits = torch.full((rows, vocab), float("-inf"), dtype=torch.bfloat16, device=so.DEVICE)
    target = (torch.arange(rows, device=so.DEVICE) * 7919 + 13) % vocab
    logits[torch.arange(rows, device=so.DEVICE), target] = 0.0
    context = so.stub_context([1.0] * rows, [0] * rows, [0.0] * rows)
    tokens = so.sample(so.make_sampler(vocab), logits, rows, context)
    ou.assert_exact(tokens, target, what="the only finite logit of each row")

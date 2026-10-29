# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Float64 race noise draws low-probability tokens at their softmax rate."""

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.sampling.torch_sampling import TorchSampling

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

VOCAB = 131072
ROWS = 4096


def _poisson_interval(mean, tail=1e-6):
    """Smallest [lo, hi] with P(X < lo) <= tail and P(X > hi) <= tail for X ~ Poisson(mean)."""

    def pmf(k):
        return math.exp(k * math.log(mean) - mean - math.lgamma(k + 1))

    lo, cdf = 0, pmf(0)
    while cdf + pmf(lo + 1) <= tail:
        lo += 1
        cdf += pmf(lo)
    lo = lo + 1 if cdf <= tail else 0
    hi, sf = int(mean), 1.0 - sum(pmf(k) for k in range(int(mean) + 1))
    while sf > tail:
        hi += 1
        sf -= pmf(hi)
    return lo, hi


def _tail_misses(gap, num_rows, *, fp64_noise, seeded):
    """Draw from rows [0, -gap, ..., -gap]; count draws that are not token 0.

    Under the exact law each draw misses token 0 with probability
    (V - 1) r / (1 + (V - 1) r), r = exp(-gap).
    """
    logits = torch.full((ROWS, VOCAB), -gap, device="cuda")
    logits[:, 0] = 0.0
    generator = torch.Generator(device="cuda").manual_seed(1234)
    misses = 0
    for call in range(num_rows // ROWS):
        kwargs = {}
        if seeded:
            kwargs = dict(
                row_seeds=[7] * ROWS, row_positions=list(range(call * ROWS, (call + 1) * ROWS))
            )
        tokens = TorchSampling.sample_from_logits(
            logits, 1.0, 0, 0.0, generator=generator, fp64_noise=fp64_noise, **kwargs
        )
        misses += int((tokens != 0).sum())
    r = math.exp(-gap)
    rate = (VOCAB - 1) * r / (1 + (VOCAB - 1) * r)
    return misses, num_rows * rate


@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("gap", [16.118, 18.421, 20.723])  # r = 1e-7, 1e-8, 1e-9
def test_fp64_noise_matches_tail_rate(gap, seeded):
    misses, expected = _tail_misses(gap, 64 * ROWS, fp64_noise=True, seeded=seeded)
    lo, hi = _poisson_interval(expected)
    assert lo <= misses <= hi, (misses, expected)


@pytest.mark.parametrize("seeded", [False, True])
def test_fp32_noise_misses_tail(seeded):
    """Non-vacuity: the default float32 noise cannot draw a token 20.7 nats down."""
    misses, expected = _tail_misses(20.723, 64 * ROWS, fp64_noise=False, seeded=seeded)
    lo, _ = _poisson_interval(expected)
    assert expected > 30 and misses < lo, (misses, expected)


@pytest.mark.parametrize("seeded", [False, True])
def test_fp64_noise_chi_square(seeded):
    torch.manual_seed(0)
    vocab, rows, calls = 16, 1 << 16, 16
    row_logits = torch.randn(vocab, device="cuda") * 2
    logits = row_logits.expand(rows, vocab).contiguous()
    generator = torch.Generator(device="cuda").manual_seed(5)
    counts = torch.zeros(vocab, device="cuda", dtype=torch.float64)
    for call in range(calls):
        kwargs = {}
        if seeded:
            kwargs = dict(row_seeds=[3] * rows, row_positions=range(call * rows, (call + 1) * rows))
            kwargs["row_positions"] = list(kwargs["row_positions"])
        tokens = TorchSampling.sample_from_logits(
            logits, 1.0, 0, 0.0, generator=generator, fp64_noise=True, **kwargs
        )
        counts += torch.bincount(tokens, minlength=vocab).double()
    expected = torch.softmax(row_logits.double(), -1) * rows * calls
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 56.49, chi2  # chi-square(15) upper 1e-6 quantile


@pytest.mark.parametrize(
    "top_k, top_p, temperature", [(0, 0.0, 1.0), (0, 0.9, 1.0), (50, 0.0, 0.7)]
)
def test_fp64_seeded_draws_are_batch_position_invariant(top_k, top_p, temperature):
    torch.manual_seed(0)
    rows = 256
    logits = (torch.randn(rows, VOCAB, device="cuda") * 3).bfloat16()
    seeds = [1000 + i for i in range(rows)]
    positions = [17 * i + 3 for i in range(rows)]
    generator = torch.Generator(device="cuda").manual_seed(0)

    def sample(x, row_seeds, row_positions):
        return TorchSampling.sample_from_logits(
            x,
            temperature,
            top_k,
            top_p,
            generator=generator,
            vocab_size=VOCAB,
            row_seeds=row_seeds,
            row_positions=row_positions,
            fp64_noise=True,
        )

    reference = sample(logits, seeds, positions)
    assert torch.equal(sample(logits, seeds, positions), reference)
    for i in range(0, rows, 51):
        assert torch.equal(
            sample(logits[i : i + 1], [seeds[i]], [positions[i]]), reference[i : i + 1]
        )
    perm = torch.randperm(rows, device="cuda")
    mates = (torch.randn(32, VOCAB, device="cuda") * 3).bfloat16()
    mixed = sample(
        torch.cat([logits[perm], mates]),
        [seeds[i] for i in perm.tolist()] + [None] * 32,
        [positions[i] for i in perm.tolist()] + [0] * 32,
    )
    assert torch.equal(mixed[:rows], reference[perm])


def test_sample_kernel_uses_configured_noise():
    rows = 8
    logits = torch.full((rows, VOCAB), -20.723, device="cuda")
    logits[:, 0] = 0.0
    context = SimpleNamespace(
        total_request_count=rows,
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=0),
        active_request_metadata={
            "seed": torch.full((rows,), 9, dtype=torch.int64),
            "temperature": torch.ones(rows),
            "top_k": torch.zeros(rows, dtype=torch.int32),
            "top_p": torch.zeros(rows),
        },
        get_active_sequence_lengths=lambda: torch.arange(rows),
    )
    generator = torch.Generator(device="cuda").manual_seed(0)
    tokens = TorchSampling(generator, VOCAB, fp64_noise=True).sample_kernel(
        logits, rows, context, no_top_k=True, no_top_p=True
    )
    expected = TorchSampling.sample_from_logits(
        logits,
        1.0,
        0,
        0.0,
        generator=generator,
        vocab_size=VOCAB,
        row_seeds=[9] * rows,
        row_positions=list(range(rows)),
        fp64_noise=True,
    )
    assert torch.equal(tokens, expected)


def test_fp64_noise_requires_torch_backend():
    pytest.importorskip("flashinfer")
    with pytest.raises(ValueError, match="fp64_sampling_noise"):
        InferenceConfig(sampling_backend="flashinfer", fp64_sampling_noise=True)

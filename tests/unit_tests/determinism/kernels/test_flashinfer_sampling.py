# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the FlashInfer processed log-probs (``FlashInferSampling.log_probs_kernel``).

RL consumes these log-probs, so every row must replay bit-exactly. Two FlashInfer defaults
do not: ``top_k_renorm_probs`` adds the partial sums of a row split over three or more CTAs
(vocabularies above ~115K on H100) with a float ``atomicAdd``, and the radix
``top_p_renorm_probs`` of FlashInfer >= 0.6.7 builds float-atomic histograms unless
``is_deterministic=True``, so a row whose cumulative mass lies within rounding of ``top_p``
keeps a different token set from run to run. The negative controls show that the replay
assertions catch both at these sizes.
"""

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from megatron.core.inference.sampling_params import is_no_op_top_k, is_no_op_top_p
from megatron.core.utils import is_flashinfer_min_version
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    count_differing_replays,
)

try:
    import flashinfer.sampling

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_FLASHINFER), reason="needs a GPU and FlashInfer"
)

# >= 2048 selects the radix top-p kernel; one top-k CTA per row on H100.
SMALL_VOCAB = 32000
# Qwen2's vocabulary: three top-k CTAs per row on H100.
LARGE_VOCAB = 152064


def _context(temperature, top_k, top_p):
    """Stand-in dynamic context: the GPU sampling view and the real filter-flag gate."""
    context = SimpleNamespace(
        gpu_view=SimpleNamespace(
            temperature=temperature.cuda(), top_k=top_k.cuda(), top_p=top_p.cuda()
        ),
        active_request_metadata={"temperature": temperature, "top_k": top_k, "top_p": top_p},
        total_request_count=temperature.numel(),
        paused_request_count=0,
    )
    context.active_sampling_filter_flags = lambda count=None: (
        DynamicInferenceContext.active_sampling_filter_flags(context, count)
    )
    return context


def _boundary_top_p(probs, targets):
    """Per-row top_p on a cumulative-mass boundary of the row, so that the kernel's
    threshold test compares quantities that are equal in exact arithmetic."""
    cum = probs.double().sort(dim=-1, descending=True).values.cumsum(dim=-1)
    cum = cum / cum[:, -1:]
    j = torch.searchsorted(cum, targets.double().unsqueeze(1).to(cum.device))
    return cum.gather(1, j).squeeze(1).float().cpu()


def _top_p_tie_case():
    """64 rows of SMALL_VOCAB: 16 top-p rows tied to a mass boundary, 32 random top-p rows
    and 16 unfiltered rows. Only the top-p stage can race at this size."""
    g = torch.Generator().manual_seed(1234)
    logits = (torch.randn(64, SMALL_VOCAB, generator=g) * 3.0).cuda()
    top_p = torch.ones(64)
    top_p[16:48] = 0.5 + 0.49 * torch.rand(32, generator=g)
    targets = 0.5 + 0.45 * torch.rand(16, generator=g)
    top_p[:16] = _boundary_top_p(torch.softmax(logits[:16], dim=-1), targets)
    return logits, torch.ones(64), torch.zeros(64, dtype=torch.int32), top_p


def _large_vocab_case():
    """32 rows of LARGE_VOCAB: 16 top-k rows (k=50) and 16 random top-p rows."""
    g = torch.Generator().manual_seed(4321)
    logits = (torch.randn(32, LARGE_VOCAB, generator=g) * 3.0).cuda()
    top_k = torch.zeros(32, dtype=torch.int32)
    top_k[:16] = 50
    top_p = torch.ones(32)
    top_p[16:] = 0.5 + 0.49 * torch.rand(16, generator=g)
    return logits, torch.full((32,), 0.8), top_k, top_p


@pytest.mark.parametrize(
    "case,replays", [(_top_p_tie_case, 32), (_large_vocab_case, 8)], ids=["top_p_ties", "large"]
)
def test_log_probs_replay_bit_exact(case, replays):
    """Top-k and top-p rows replay bit-exactly; unfiltered rows are the exact log_softmax."""
    logits, temperature, top_k, top_p = case()
    sampler = FlashInferSampling(logits.size(1), torch.Generator(device="cuda"))
    context = _context(temperature, top_k, top_p)
    reference, _ = assert_replays_bit_exact(
        lambda logits: sampler.log_probs_kernel(logits, context),
        (logits,),
        replays=replays,
        backward=False,
        contention=True,
        what="FlashInferSampling.log_probs_kernel",
    )
    # Unfiltered rows bypass the renorm kernels: exactly the log_softmax of the scaled logits.
    unfiltered = (is_no_op_top_k(top_k) & is_no_op_top_p(top_p)).cuda()
    expected = torch.log_softmax(logits / temperature.cuda().unsqueeze(1), dim=-1)
    assert torch.equal(reference["out"][unfiltered], expected[unfiltered])


def test_log_probs_request_deterministic_top_p_renorm():
    """The sampler sets the top-p renorm's determinism switch whenever FlashInfer has one."""
    params = inspect.signature(flashinfer.sampling.top_p_renorm_probs).parameters
    logits = torch.randn(4, 4096, device="cuda")
    context = _context(torch.ones(4), torch.zeros(4, dtype=torch.int32), torch.full((4,), 0.9))
    sampler = FlashInferSampling(4096, torch.Generator(device="cuda"))
    renorm = flashinfer.sampling.top_p_renorm_probs
    with mock.patch.object(flashinfer.sampling, "top_p_renorm_probs", wraps=renorm) as spy:
        sampler.log_probs_kernel(logits, context)
    assert spy.call_count == 1
    assert spy.call_args.kwargs.get("is_deterministic", False) == ("is_deterministic" in params)


@pytest.mark.skipif(
    not (HAVE_FLASHINFER and is_flashinfer_min_version("0.6.7")),
    reason="FlashInfer < 0.6.7 renormalizes top-p with a one-block-per-row kernel",
)
@pytest.mark.parametrize("stage", ["top_k_renorm_probs", "top_p_renorm_probs"])
def test_default_renorm_is_not_bit_exact(stage):
    """Negative control: the FlashInfer defaults the sampler avoids race at these sizes."""
    if stage == "top_k_renorm_probs":
        logits, temperature, top_k, _ = _large_vocab_case()
        logits = logits / temperature.cuda().unsqueeze(1)
        threshold = top_k.masked_fill(is_no_op_top_k(top_k), LARGE_VOCAB).cuda()
    else:
        logits, _, _, top_p = _top_p_tie_case()
        threshold = top_p.cuda()
    renorm = getattr(flashinfer.sampling, stage)
    differing = count_differing_replays(
        lambda probs: renorm(probs, threshold),
        (torch.softmax(logits, dim=-1),),
        replays=16,
        backward=False,
    )
    assert differing > 0, (
        f"default {stage} replayed bit-exactly 15 times; either FlashInfer made it "
        "deterministic (drop this control) or the shape no longer contends"
    )

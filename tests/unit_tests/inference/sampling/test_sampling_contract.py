# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The sampling contract both backends implement, checked against an fp64 reference.

* Top-k and top-p together: top-p applies to the renormalized top-k set (top-k first).
* Greedy (`top_k == 1`, which temperature 0 maps to): the lowest-index argmax on every row, in
  all-greedy and mixed batches, with or without a top-p; its processed log-probs are one-hot.
* A top-k at or above the vocabulary filters nothing: the engine admits it as no top-k.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling.torch_sampling import TorchSampling
from megatron.core.inference.sampling_params import SamplingParams

try:
    import flashinfer  # noqa: F401

    from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

BACKENDS = [
    "torch",
    pytest.param(
        "flashinfer",
        marks=pytest.mark.skipif(
            not (HAVE_FLASHINFER and torch.cuda.is_available()),
            reason="FlashInfer sampling needs FlashInfer and CUDA",
        ),
    ),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ids 0..9 carry 10 - 0.25 i (exact in bf16), every other id -30. At T = 1, top-4 then top-0.7
# keeps {0, 1, 2}; top-p 0.7 alone would keep {0..4} and top-4 alone {0..3}.
VOCAB = 128
REAL_LOGITS = [10.0 - 0.25 * i for i in range(10)]
JOINT_TOP_K, JOINT_TOP_P = 4, 0.7


def _fixed_rows(rows):
    logits = torch.full((rows, VOCAB), -30.0, dtype=torch.bfloat16, device=DEVICE)
    logits[:, : len(REAL_LOGITS)] = torch.tensor(REAL_LOGITS, dtype=torch.bfloat16)
    return logits


def _reference_probs(logits_row, temperature, top_k, top_p):
    """fp64 top-k-then-top-p distribution of one row (top-p keeps the crossing token)."""
    x = logits_row.double().cpu() / temperature
    order = torch.argsort(x, descending=True)
    kept = order[:top_k] if top_k > 0 else order
    probs = torch.softmax(x[kept], dim=-1)
    if 0.0 < top_p < 1.0:
        keep = torch.cumsum(probs, 0) - probs <= top_p
        kept, probs = kept[keep], probs[keep] / probs[keep].sum()
    return torch.zeros_like(x).index_put_((kept,), probs)


def _context(temperature, top_k, top_p):
    """Stand-in dynamic context: CPU metadata, its GPU view, and the real filter flags."""
    metadata = {
        "temperature": torch.tensor(temperature, dtype=torch.float32),
        "top_k": torch.tensor(top_k, dtype=torch.int32),
        "top_p": torch.tensor(top_p, dtype=torch.float32),
    }
    context = SimpleNamespace(
        active_request_metadata=metadata,
        gpu_view=SimpleNamespace(**{key: value.to(DEVICE) for key, value in metadata.items()}),
        total_request_count=len(temperature),
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=0),
    )
    context.active_sampling_filter_flags = lambda count=None: (
        DynamicInferenceContext.active_sampling_filter_flags(context, count)
    )
    return context


def _sampler(backend, vocab_size):
    rng = torch.Generator(device=DEVICE).manual_seed(1234)
    if backend == "torch":
        return TorchSampling(rng, vocab_size)
    return FlashInferSampling(vocab_size, rng)


def _sample(sampler, logits, context):
    no_top_k, no_top_p = context.active_sampling_filter_flags()
    return sampler.sample_kernel(
        logits, logits.shape[0], context, no_top_k=no_top_k, no_top_p=no_top_p
    ).cpu()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "top_k, top_p", [(JOINT_TOP_K, 0.0), (0, JOINT_TOP_P), (JOINT_TOP_K, JOINT_TOP_P)]
)
def test_top_k_and_top_p_sample_from_reference_support(backend, top_k, top_p):
    """Filtered rows next to plain rows sample, and score, the fp64 reference distribution."""
    plain, filtered = 64, 256
    logits = _fixed_rows(plain + filtered)
    context = _context(
        [1.0] * (plain + filtered),
        [0] * plain + [top_k] * filtered,
        [0.0] * plain + [top_p] * filtered,
    )
    sampler = _sampler(backend, VOCAB)
    reference = _reference_probs(logits[0], 1.0, top_k, top_p)
    support = set(torch.nonzero(reference).flatten().tolist())

    tokens = _sample(sampler, logits, context)
    assert set(tokens[plain:].tolist()) == support
    assert set(tokens[:plain].tolist()) <= set(range(len(REAL_LOGITS)))

    log_probs = sampler.log_probs_kernel(logits.float(), context)[plain].double().cpu()
    torch.testing.assert_close(log_probs.exp(), reference, rtol=1e-5, atol=1e-6)


def _tied_rows(rows, tie):
    """Identical bf16 rows whose ids 7 and 11 tie at the maximum (or 11 is one bf16 ulp below)."""
    generator = torch.Generator().manual_seed(0)
    row = (torch.randn(32000, generator=generator) * 2).clamp(max=16.0)
    row[7] = 24.0
    row[11] = 24.0 if tie else 24.0 - 0.125
    return row.to(torch.bfloat16).to(DEVICE).expand(rows, -1).contiguous()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mixed", [False, True], ids=["all-greedy", "mixed"])
@pytest.mark.parametrize("greedy_top_p", [0.0, 0.9], ids=["temperature-0", "do-sample-false"])
@pytest.mark.parametrize("tie", [False, True], ids=["strict-max", "tied-max"])
def test_greedy_rows_take_lowest_tied_index(backend, mixed, greedy_top_p, tie):
    """Greedy rows return the lowest-index maximum on every batch row, not a draw among ties.

    `temperature=0` arrives as `top_k=1, top_p=0`; `do_sample: false` with a model-card top-p
    as `top_k=1, top_p=0.9`. A mixed batch adds one top-p row, which routes FlashInfer to its
    joint kernel.
    """
    greedy = 1024
    rows = greedy + int(mixed)
    temperature, top_k, top_p = [0.0] * rows, [1] * rows, [greedy_top_p] * rows
    if mixed:
        temperature[-1], top_k[-1], top_p[-1] = 1.0, 0, 0.9
    context = _context(temperature, top_k, top_p)
    logits = _tied_rows(rows, tie)
    sampler = _sampler(backend, logits.shape[1])

    tokens = _sample(sampler, logits, context)
    assert torch.equal(tokens[:greedy], torch.full((greedy,), 7))

    # The processed distribution of a greedy row is one-hot on the token it samples.
    log_probs = sampler.log_probs_kernel(logits.float(), context)[:greedy]
    one_hot = torch.full_like(log_probs[0], float("-inf"))
    one_hot[7] = 0.0
    assert torch.equal(log_probs, one_hot.expand_as(log_probs))


@pytest.mark.parametrize(
    "top_k, admitted_top_k",
    [(VOCAB - 1, VOCAB - 1), (VOCAB, 0), (VOCAB + 1000, 0)],
    ids=["below-vocab", "vocab", "beyond-vocab"],
)
def test_admission_disables_a_top_k_that_keeps_the_whole_vocabulary(top_k, admitted_top_k):
    """The engine admits a top-k at or above the vocabulary as no top-k, next to its top-p.

    Such a top-k filters nothing, as a top-p of 1.0 does, which SamplingParams already maps to
    no top-p. The samplers only apply a top-k below the vocabulary; the torch backend asserted
    on it inside the engine step, failing the whole batch.
    """
    engine = object.__new__(DynamicInferenceEngine)
    engine.context = SimpleNamespace(
        config=SimpleNamespace(sampling_backend="torch", num_speculative_tokens=0),
        max_sequence_length=16,
        max_tokens=16,
        num_speculative_tokens=0,
        block_size_tokens=8,
        kv_block_allocator=SimpleNamespace(pool_size=32),
    )
    engine.controller = SimpleNamespace(vocab_size=VOCAB)
    engine.requests = {}
    engine._loop = asyncio.new_event_loop()
    engine._generation_epoch = None
    engine.rank = 0
    engine.enable_chunked_prefill = False
    engine.waiting_request_ids = []
    request = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2]),
        sampling_params=SamplingParams(
            top_k=top_k, top_p=0.7, termination_id=0, num_tokens_to_generate=4
        ),
    )
    try:
        engine._add_request(request)
    finally:
        engine._loop.close()

    assert engine.waiting_request_ids == [1]
    assert request.sampling_params.top_k == admitted_top_k
    assert request.sampling_params.top_p == 0.7

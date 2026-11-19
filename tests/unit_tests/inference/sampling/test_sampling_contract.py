# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The sampling contract both backends implement, checked against an fp64 reference.

* Top-k and top-p together: top-p applies to the renormalized top-k set (top-k first).
* Greedy (`is_greedy`, which temperature 0 and top_k=1 set): the lowest-index argmax of the
  logits as given, on every row, in all-greedy and mixed batches, whatever its filters; its
  processed log-probs are one-hot. Its sampling metadata is neutral, so it never changes the
  kernel the other rows sample with.
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


SAMPLING_COLUMNS = ("temperature", "top_k", "top_p", "is_greedy")


def _sampling_row(params):
    """The sampling metadata a dynamic context tracks for a request with these params."""
    request = DynamicInferenceRequest(
        request_id=0, prompt="", prompt_tokens=torch.tensor([1]), sampling_params=params
    )
    names = [name for name, _ in request.get_metadata_types()]
    return dict(zip(names, request.tracked_metadata))


def _context(groups):
    """Stand-in dynamic context for `(SamplingParams, count)` groups of requests, in order: the
    CPU sampling metadata the real context tracks, its GPU view, and the real filter flags."""
    dtypes = dict(DynamicInferenceRequest.get_metadata_types())
    rows = []
    for params, count in groups:
        params.termination_id = 0
        rows += [_sampling_row(params)] * count
    metadata = {
        name: torch.tensor([row[name] for row in rows], dtype=dtypes[name])
        for name in SAMPLING_COLUMNS
    }
    gpu_view = {name: value.to(DEVICE) for name, value in metadata.items()}
    gpu_view["is_greedy"] = gpu_view["is_greedy"].to(torch.int32)
    context = SimpleNamespace(
        active_request_metadata=metadata,
        gpu_view=SimpleNamespace(**gpu_view),
        total_request_count=len(rows),
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
        [(SamplingParams(), plain), (SamplingParams(top_k=top_k, top_p=top_p), filtered)]
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


# The ways a request asks for greedy decoding: `temperature=0`; `do_sample: false` with a
# model-card top-p (`top_k=1, top_p=0.9`); and the flag itself, next to a sampled temperature.
GREEDY = {
    "temperature-0": dict(temperature=0.0),
    "do-sample-false": dict(top_k=1, top_p=0.9),
    "is-greedy": dict(is_greedy=True, temperature=0.7, top_p=0.9),
}


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mixed", [False, True], ids=["all-greedy", "mixed"])
@pytest.mark.parametrize("greedy", list(GREEDY.values()), ids=list(GREEDY))
@pytest.mark.parametrize("tie", [False, True], ids=["strict-max", "tied-max"])
def test_greedy_rows_take_lowest_tied_index(backend, mixed, greedy, tie):
    """Greedy rows return the lowest-index maximum on every batch row, not a draw among ties.

    A mixed batch adds one top-p row, which routes FlashInfer to its top-p kernel.
    """
    greedy_rows = 1024
    groups = [(SamplingParams(**greedy), greedy_rows)]
    if mixed:
        groups.append((SamplingParams(top_p=0.9), 1))
    context = _context(groups)
    logits = _tied_rows(greedy_rows + int(mixed), tie)
    sampler = _sampler(backend, logits.shape[1])

    tokens = _sample(sampler, logits, context)
    assert torch.equal(tokens[:greedy_rows], torch.full((greedy_rows,), 7))

    # The processed distribution of a greedy row is one-hot on the token it samples.
    log_probs = sampler.log_probs_kernel(logits.float(), context)[:greedy_rows]
    one_hot = torch.full_like(log_probs[0], float("-inf"))
    one_hot[7] = 0.0
    assert torch.equal(log_probs, one_hot.expand_as(log_probs))


@pytest.mark.skipif(
    not (HAVE_FLASHINFER and torch.cuda.is_available()),
    reason="FlashInfer sampling needs FlashInfer and CUDA",
)
@pytest.mark.parametrize("greedy", list(GREEDY.values()), ids=list(GREEDY))
def test_greedy_rows_leave_the_other_rows_kernel_alone(greedy):
    """Greedy rows change neither the FlashInfer kernel nor the draws of the rows that sample.

    Their filters never reach the batch's filter flags, so plain rows next to greedy rows sample
    and score exactly as next to plain rows; the greedy rows still take their argmax.
    """
    sampled = 64
    logits = _fixed_rows(2 * sampled)

    def serve(neighbours):
        context = _context([(SamplingParams(), sampled), (neighbours, sampled)])
        sampler = _sampler("flashinfer", VOCAB)
        tokens = _sample(sampler, logits, context)
        return tokens, sampler.log_probs_kernel(logits.float(), context).cpu()

    plain_tokens, plain_log_probs = serve(SamplingParams())
    assert len(set(plain_tokens[:sampled].tolist())) > 1, "the plain rows must draw"
    tokens, log_probs = serve(SamplingParams(**greedy))
    assert torch.equal(tokens[:sampled], plain_tokens[:sampled])
    assert torch.equal(log_probs[:sampled], plain_log_probs[:sampled])
    assert torch.equal(tokens[sampled:], torch.zeros(sampled, dtype=tokens.dtype))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("top_p", [0.0, 0.9])
def test_greedy_ignores_a_temperature_that_would_tie_its_maximum(backend, top_p):
    """A greedy row's token and one-hot log-probs come from its logits as given.

    Dividing the two logits below by the request's temperature 1.5, as `filter_logits` scales
    them on this device, rounds them into an fp32 tie whose lowest index is not the argmax.
    (CUDA divides by a scalar as a multiply by its reciprocal, so the pair is found here.)
    """
    grid = 1.5 + torch.arange(16, dtype=torch.float32, device=DEVICE) * 2**-23
    scaled = grid / 1.5
    first = int((scaled[1:] == scaled[:-1]).nonzero()[0])
    logits = grid[first : first + 2].unsqueeze(0)
    assert logits[0, 1] > logits[0, 0]

    context = _context([(SamplingParams(temperature=1.5, top_k=1, top_p=top_p), 1)])
    sampler = _sampler(backend, 2)
    assert _sample(sampler, logits, context).tolist() == [1]
    log_probs = sampler.log_probs_kernel(logits, context).cpu()
    assert torch.equal(log_probs, torch.tensor([[float("-inf"), 0.0]]))


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

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling
from megatron.core.inference.sampling.torch_sampling import TorchSampling


def _oracle_scores(logits, temperature, top_k, top_p, *, compose_filters):
    """Independently compute raw and filtered distributions for one row."""
    raw = torch.log_softmax(logits.float(), dim=-1)
    scaled = logits.float() / max(float(temperature), 1.0e-6)
    probabilities = torch.softmax(scaled, dim=-1)

    if top_k > 0:
        # Keep the full cutoff tie set. The non-tie matrix below therefore agrees
        # with exact-k kernels without relying on either production implementation.
        cutoff = torch.topk(probabilities, int(top_k)).values[-1]
        probabilities = probabilities.masked_fill(probabilities < cutoff, 0.0)
        probabilities /= probabilities.sum()

    if 0.0 < top_p < 1.0:
        if top_k > 0 and not compose_filters:
            raise AssertionError("the Torch backend rejects simultaneous top-k and top-p")
        values, indices = torch.sort(probabilities, descending=True, stable=True)
        # Nucleus support includes the token that crosses p: a token is retained
        # exactly when the mass before it is below the threshold.
        keep_sorted = values.cumsum(0) - values < float(top_p)
        keep = torch.zeros_like(keep_sorted).scatter(0, indices, keep_sorted)
        probabilities = probabilities.masked_fill(~keep, 0.0)
        probabilities /= probabilities.sum()

    processed = torch.full_like(probabilities, float("-inf"))
    support = probabilities > 0
    processed[support] = probabilities[support].log()
    return raw, processed


def _make_context(parameters):
    temperature = torch.tensor([row[0] for row in parameters], dtype=torch.float32)
    top_k = torch.tensor([row[1] for row in parameters], dtype=torch.int32)
    top_p = torch.tensor([row[2] for row in parameters], dtype=torch.float32)
    context = SimpleNamespace(
        total_request_count=len(parameters),
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=0),
        active_request_metadata={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": torch.full((len(parameters),), -1, dtype=torch.int64),
        },
        gpu_view=SimpleNamespace(
            temperature=temperature.cuda(), top_k=top_k.cuda(), top_p=top_p.cuda()
        ),
    )
    context.active_sampling_filter_flags = lambda count=None: (
        DynamicInferenceContext.active_sampling_filter_flags(context, count)
    )
    return context


def _make_sampler(backend, vocab_size, seed=17):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if backend == "torch":
        return TorchSampling(generator, vocab_size)
    pytest.importorskip("flashinfer")
    return FlashInferSampling(vocab_size, generator)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real sampling kernels require CUDA")
@pytest.mark.parametrize(
    ("backend", "dtype"),
    [
        pytest.param("torch", torch.float32, id="dtype0-torch"),
        pytest.param("flashinfer", torch.float32, id="dtype0-flashinfer"),
        pytest.param("flashinfer", torch.bfloat16, id="dtype1-flashinfer"),
    ],
)
def test_known_logits_match_independent_filter_and_score_oracle(backend, dtype):
    """Heterogeneous rows must agree with an independent analytic distribution."""
    parameters = [(1.0, 0, 0.0), (0.0, 1, 0.0), (0.75, 3, 0.0), (1.25, 0, 0.72), (0.5, 0, 0.0)]
    if backend == "flashinfer":
        parameters.append((0.9, 4, 0.83))
    logits = torch.tensor(
        [
            [-1.2, 0.3, 2.1, 0.8, -0.4, 1.4],
            [0.1, 1.7, -0.5, 0.6, 2.8, 1.1],
            [2.0, -0.7, 0.4, 1.1, 3.2, -1.3],
            [-1.0, 0.2, 0.8, 2.4, 1.3, -0.2],
            [1.3, -0.4, 0.9, -1.1, 0.2, 2.0],
            [0.5, 2.7, -0.6, 1.6, 0.1, 1.0],
        ],
        device="cuda",
        dtype=dtype,
    )[: len(parameters)]
    context = _make_context(parameters)
    sampler = _make_sampler(backend, logits.shape[1])

    actual_scores = sampler.log_probs_kernel(logits, context)
    expected_raw = []
    expected_processed = []
    for row, (temperature, top_k, top_p) in zip(logits, parameters):
        raw, processed = _oracle_scores(
            row.cpu(), temperature, top_k, top_p, compose_filters=backend == "flashinfer"
        )
        expected_raw.append(raw)
        expected_processed.append(processed)
    expected_raw = torch.stack(expected_raw).cuda()
    expected_processed = torch.stack(expected_processed).cuda()

    raw_context = SimpleNamespace(config=SimpleNamespace(logprobs_mode="raw_logprobs"))
    actual_raw = DynamicInferenceContext._processed_log_probs(
        raw_context, logits.float(), sampling=None
    )

    assert actual_scores.dtype == torch.float32
    assert actual_raw.dtype == torch.float32
    assert torch.equal(torch.isfinite(actual_scores), torch.isfinite(expected_processed))
    torch.testing.assert_close(actual_raw, expected_raw, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(
        actual_scores[torch.isfinite(actual_scores)],
        expected_processed[torch.isfinite(expected_processed)],
        rtol=2.0e-4 if dtype == torch.bfloat16 else 2.0e-5,
        atol=2.0e-4 if dtype == torch.bfloat16 else 2.0e-6,
    )
    torch.testing.assert_close(actual_raw.exp().sum(1), torch.ones(len(parameters), device="cuda"))
    torch.testing.assert_close(
        actual_scores.exp().sum(1), torch.ones(len(parameters), device="cuda"), atol=2.0e-6, rtol=0
    )
    torch.testing.assert_close(
        expected_raw.exp().sum(1), torch.ones(len(parameters), device="cuda")
    )
    torch.testing.assert_close(
        expected_processed.exp().sum(1), torch.ones(len(parameters), device="cuda")
    )
    assert not torch.allclose(actual_raw[4], actual_scores[4])

    no_top_k, no_top_p = context.active_sampling_filter_flags()
    sampled = sampler.sample_kernel(
        logits, len(parameters), context, no_top_k=no_top_k, no_top_p=no_top_p
    )
    rows = torch.arange(len(parameters), device="cuda")
    assert torch.isfinite(expected_processed[rows, sampled]).all()
    assert torch.isfinite(expected_raw[rows, sampled]).all()
    assert sampled[1].item() == logits[1].argmax().item()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real sampling kernels require CUDA")
@pytest.mark.parametrize("backend", ["torch", "flashinfer"])
def test_stochastic_top_p_matches_analytic_distribution(backend):
    """A controlled frequency oracle checks probabilities, not backend trajectories."""
    draws = 16384
    row = torch.tensor([2.0, 1.0, 0.0, -1.0], dtype=torch.float32)
    parameters = [(1.0, 0, 0.8)] * draws
    context = _make_context(parameters)
    sampler = _make_sampler(backend, row.numel(), seed=982451653)
    logits = row.cuda().repeat(draws, 1)
    _, expected_scores = _oracle_scores(row, 1.0, 0, 0.8, compose_filters=True)

    sampled = sampler.sample_kernel(logits, draws, context, no_top_k=True, no_top_p=False)
    observed = torch.bincount(sampled, minlength=row.numel()).float() / draws
    expected = expected_scores.exp().cuda()

    excluded = ~torch.isfinite(expected_scores).cuda()
    assert torch.equal(observed[excluded], torch.zeros(2, device="cuda"))
    torch.testing.assert_close(observed, expected, rtol=0, atol=0.02)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real sampling kernels require CUDA")
def test_exact_tie_contract_does_not_invent_cross_backend_identity():
    """Torch promises first-index argmax; FlashInfer promises only tied-max membership."""
    pytest.importorskip("flashinfer")
    logits = torch.tensor([[3.0, 3.0, 0.0, -1.0]], device="cuda")
    context = _make_context([(0.0, 1, 0.0)])
    torch_token = _make_sampler("torch", 4).sample_kernel(
        logits, 1, context, no_top_k=False, no_top_p=True
    )
    flashinfer_token = _make_sampler("flashinfer", 4).sample_kernel(
        logits, 1, context, no_top_k=False, no_top_p=True
    )

    assert torch_token.item() == 0
    assert flashinfer_token.item() in {0, 1}

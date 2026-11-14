# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.sampling.torch_sampling import TorchSampling


def _make_context(temperature, top_k, top_p, *, paused_request_count=0):
    """Minimal stand-in for `DynamicInferenceContext` covering the fields
    `log_probs_kernel` reads: `total_request_count`, `paused_request_count`,
    and `active_request_metadata`."""
    num_requests = temperature.numel()
    return SimpleNamespace(
        total_request_count=num_requests + paused_request_count,
        paused_request_count=paused_request_count,
        active_request_metadata={"temperature": temperature, "top_k": top_k, "top_p": top_p},
    )


class TestFilterLogitsPrecision:
    """Regression tests for issue #7857: temperature scaling must run in fp32.

    Scaling in the logits dtype rounds every scaled logit before the fp32
    softmax, which can change top-p membership; fp16 division by the
    temperature floor can also overflow to inf.
    """

    def test_bf16_scaling_keeps_top_p_membership(self):
        logits = torch.tensor([[26.625, 26.375]], dtype=torch.bfloat16)
        filtered = TorchSampling.filter_logits(logits, temperature=0.1, top_k=0, top_p=0.9)
        expected = TorchSampling.filter_logits(logits.float(), temperature=0.1, top_k=0, top_p=0.9)
        assert filtered.dtype == torch.float32
        torch.testing.assert_close(filtered, expected)
        # In bf16 the scaled logits round to [266, 264], whose fp32 softmax leaves
        # the runner-up inside the 0.9 nucleus. In fp32 it falls outside.
        assert filtered[0, 1] == float("-inf")

    @pytest.mark.parametrize("top_k,top_p", [(0, 0.9), (1, 0.0)])
    def test_fp16_greedy_temperature_stays_finite(self, top_k, top_p):
        logits = torch.tensor([[26.625, 26.375]], dtype=torch.float16)
        filtered = TorchSampling.filter_logits(logits, temperature=0.0, top_k=top_k, top_p=top_p)
        kept = filtered[filtered != float("-inf")]
        assert torch.isfinite(kept).all()
        assert filtered.argmax(dim=-1).item() == 0

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("temperature", [1.0, 0.7, 0.1, 0.0])
    def test_matches_float32_reference(self, dtype, temperature):
        torch.manual_seed(0)
        logits = torch.randn(3, 64, dtype=dtype)
        for top_k, top_p in ((0, 0.9), (8, 0.0)):
            actual = TorchSampling.filter_logits(logits, temperature, top_k, top_p)
            expected = TorchSampling.filter_logits(logits.float(), temperature, top_k, top_p)
            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual, expected)


class TestLogProbsKernelDtype:
    """Regression test for the fp32 log-probs precision fix.

    `log_probs_kernel` computes `torch.log_softmax(..., dtype=torch.float32)`
    specifically to avoid the tail erosion low precision causes on wide
    vocabularies. The returned tensor must stay fp32 even when the caller's
    raw logits are bf16/fp16 -- allocating the output buffer with
    `torch.empty_like(logits)` would silently downcast the fp32 result back
    to the logits dtype on assignment and defeat the fix.
    """

    @pytest.mark.parametrize("logits_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_returns_float32_for_low_precision_logits(self, logits_dtype):
        torch.manual_seed(0)
        vocab_size = 128
        num_rows = 4

        logits = torch.randn(num_rows, vocab_size, dtype=logits_dtype)
        temperature = torch.ones(num_rows)
        top_k = torch.zeros(num_rows, dtype=torch.long)
        top_p = torch.zeros(num_rows)

        context = _make_context(temperature, top_k, top_p)
        sampling = TorchSampling(rng=torch.Generator(), vocab_size=vocab_size)

        log_probs = sampling.log_probs_kernel(logits, context)

        assert log_probs.dtype == torch.float32
        assert log_probs.shape == (num_rows, vocab_size)

        # The fp32 result should match computing log_softmax directly on the
        # fp32-upcast logits -- not the (lossy) low-precision computation.
        expected = torch.log_softmax(logits.float(), dim=-1)
        torch.testing.assert_close(log_probs, expected)

    def test_low_precision_logits_do_not_collapse_tail_precision(self):
        """A regression on the specific failure mode: casting the fp32
        log-softmax result down to bf16 loses resolution in the tail that a
        genuinely fp32 buffer preserves."""
        torch.manual_seed(0)
        vocab_size = 4096
        num_rows = 2

        logits = torch.randn(num_rows, vocab_size, dtype=torch.bfloat16)
        temperature = torch.ones(num_rows)
        top_k = torch.zeros(num_rows, dtype=torch.long)
        top_p = torch.zeros(num_rows)

        context = _make_context(temperature, top_k, top_p)
        sampling = TorchSampling(rng=torch.Generator(), vocab_size=vocab_size)

        log_probs = sampling.log_probs_kernel(logits, context)
        assert log_probs.dtype == torch.float32

        # Downcasting to bf16 and back must lose information relative to the
        # fp32 buffer; if it doesn't, the kernel is (again) only pretending
        # to compute in fp32.
        downcast_roundtrip = log_probs.to(torch.bfloat16).to(torch.float32)
        assert not torch.equal(log_probs, downcast_roundtrip)


def test_uniform_top_one_sampling_with_speculative_row_mapping():
    """Uniform greedy policy samples every speculative row in its original order."""
    torch.manual_seed(7)
    request_count = 3
    context = _make_context(
        torch.tensor([0.5, 1.0, 2.0]),
        torch.ones(request_count, dtype=torch.long),
        torch.zeros(request_count),
    )
    sampling = TorchSampling(rng=torch.Generator(), vocab_size=257)
    logits = torch.randn(12, 257, device="cuda", dtype=torch.bfloat16)
    gather = torch.tensor([8, 3, 7, 1, 10, 4, 5, 11], device="cuda")
    token_to_request = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], device="cuda")
    output = torch.full((8,), -1, device="cuda", dtype=torch.long)

    result = sampling.sample_kernel(
        logits,
        8,
        context,
        no_top_k=False,
        no_top_p=True,
        gather_indices=gather,
        token_to_request_index=token_to_request,
        output=output,
    )

    assert result is output
    torch.testing.assert_close(result, logits[gather].argmax(dim=-1), rtol=0, atol=0)

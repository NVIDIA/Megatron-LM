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

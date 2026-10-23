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


@pytest.mark.parametrize(argnames="dtype", argvalues=[torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    argnames="device",
    argvalues=[
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                condition=not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
class TestTemperatureScaling:
    @pytest.mark.parametrize(
        argnames="values,temperature,top_p,expected",
        argvalues=[((26.625, 26.375), 0.1, 0.9, (1.0, 0.0)), ((1.0, 2.0), 1e-5, 0.0, (0.0, 1.0))],
        ids=["nucleus-boundary", "fp16-overflow"],
    )
    def test_sampling_distribution(
        self,
        dtype: torch.dtype,
        device: torch.device,
        values: tuple[float, float],
        temperature: float,
        top_p: float,
        expected: tuple[float, float],
    ) -> None:
        logits = torch.tensor([values], dtype=dtype, device=device)
        rng = torch.Generator(device=device).manual_seed(123)
        sampling = TorchSampling(rng=rng, vocab_size=2)
        context: SimpleNamespace = _make_context(
            temperature=torch.tensor([temperature]),
            top_k=torch.tensor([0]),
            top_p=torch.tensor([top_p]),
        )

        torch.testing.assert_close(
            actual=sampling.log_probs_kernel(logits=logits, context=context).exp(),
            expected=torch.tensor([expected], dtype=torch.float32, device=device),
            rtol=0,
            atol=0,
        )
        sampled: torch.Tensor = TorchSampling.sample_from_logits(
            last_token_logits=logits.repeat((256, 1)),
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            generator=rng,
            vocab_size=2,
        )
        assert (sampled == expected.index(1.0)).all()

    @pytest.mark.parametrize(
        argnames="temperature,top_k,top_p",
        argvalues=[(1.0, 0, 0.0), (0.1, 0, 0.0), (1.0, 2, 0.0), (1.0, 0, 0.75)],
    )
    def test_filter_preserves_input(
        self, dtype: torch.dtype, device: torch.device, temperature: float, top_k: int, top_p: float
    ) -> None:
        logits = torch.tensor([[0.0, 2.0, 1.0]], dtype=dtype, device=device)
        original = logits.clone()

        TorchSampling.filter_logits(
            last_token_logits=logits, temperature=temperature, top_k=top_k, top_p=top_p
        ).zero_()

        torch.testing.assert_close(actual=logits, expected=original, rtol=0, atol=0)

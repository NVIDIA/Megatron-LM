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


class TestFilterLogitsTemperaturePrecision:
    """Regression tests for ``TorchSampling.filter_logits`` (issue #7857).

    ``Tensor.div_(scalar)`` runs in the tensor's dtype. For BF16/FP16 inputs
    that (a) loses precision in the scaled logit and shifts top-p nucleus
    membership at moderate temperatures like 0.1, and (b) can overflow FP16
    at temperature=0 (floored to MIN_SAMPLING_TEMPERATURE=1e-6). The fix
    promotes the working buffer to FP32 when scaling; the filters below run
    on FP32 and the downstream softmax already runs in FP32.
    """

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
    def test_temperature_scaling_does_not_shift_top_p_nucleus(self, input_dtype):
        """REGRESSION (issue #7857, BF16 column): with the BF16-divisor bug,
        BF16 rounding of 26.625 / 0.1 = 266.25 -> 266 (loss of 0.25) makes
        the BF16 gap between leading and runner-up 2.0 vs the FP32 gap 2.5,
        raising the runner-up's softmax mass enough that top-p=0.9 retains
        it where the FP32 reference masks it. The fix must reproduce the
        FP32-reference filter."""
        # Issue #7857 reproducer inputs: leading and runner-up logits that
        # differ by 0.25 in BF16 / 0.5 in FP16 -- small enough that divisor
        # rounding matters.
        leading = 26.625 if input_dtype is torch.bfloat16 else 26.5
        runner_up = 26.375 if input_dtype is torch.bfloat16 else 26.0
        logits = torch.tensor([[leading, runner_up] + [-100.0] * 126], dtype=input_dtype)

        filtered = TorchSampling.filter_logits(logits, temperature=0.1, top_k=0, top_p=0.9)

        # The leading token must be retained (not masked to -inf).
        assert filtered[0, 0] != float("-inf"), (
            "leading token unexpectedly masked out -- top-p nucleus should " "always include it"
        )
        # The runner-up must be masked (-inf). The pre-fix BF16 path used to
        # retain it because the BF16-rounded divisor (266, 264) inflated its
        # softmax mass past the top-p cutoff.
        assert filtered[0, 1] == float("-inf"), (
            f"{input_dtype} rounding in the temperature divisor retained the "
            "runner-up where the FP32 reference would have masked it "
            "(issue #7857)."
        )

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
    def test_temperature_scaling_matches_fp32_reference(self, input_dtype):
        """The fixed filter must match the FP32-computed reference for
        BF16/FP16 inputs across the temperatures and top-p values that
        exercise the divisor rounding path."""
        torch.manual_seed(0)
        vocab_size = 128
        logits = torch.randn(4, vocab_size, dtype=input_dtype)

        for temperature, top_p in (
            (0.1, 0.0),
            (0.1, 0.5),
            (0.1, 0.9),
            (0.5, 0.9),
            (0.7, 0.5),
            (1.5, 0.5),  # temperature > 1 also exercises the divisor path
        ):
            low_prec = TorchSampling.filter_logits(
                logits, temperature=temperature, top_k=0, top_p=top_p
            )
            fp32_ref = TorchSampling.filter_logits(
                logits.float(), temperature=temperature, top_k=0, top_p=top_p
            )
            # The mask must agree exactly: any divergence means the BF16/
            # FP16 divisor truncation shifted the nucleus boundary.
            low_prec_mask = torch.isinf(low_prec)
            fp32_ref_mask = torch.isinf(fp32_ref)
            assert torch.equal(low_prec_mask, fp32_ref_mask), (
                f"{input_dtype} filter mask diverged from the FP32 reference "
                f"at temperature={temperature}, top_p={top_p}: "
                f"{int((low_prec_mask != fp32_ref_mask).sum())} entries differ"
            )
            # Finite entries should match within FP32 precision.
            finite = ~low_prec_mask
            torch.testing.assert_close(
                low_prec[finite],
                fp32_ref[finite],
                msg=(
                    f"{input_dtype} filter_logits diverged from the FP32 "
                    f"reference at temperature={temperature}, top_p={top_p}"
                ),
            )

    def test_fp16_does_not_overflow_at_zero_temperature(self):
        """REGRESSION (issue #7857, FP16 column): at temperature=0 the
        divisor is floored to MIN_SAMPLING_TEMPERATURE=1e-6, so
        ``logits / 1e-6 = logits * 1e6`` overflows FP16 (max ~65504).
        Scaling in FP32 keeps the result finite and the top-p mask valid."""
        logits = torch.tensor([[30.0, 29.0, 28.0] + [-100.0] * 125], dtype=torch.float16)

        filtered = TorchSampling.filter_logits(logits, temperature=0.0, top_k=0, top_p=0.9)

        kept = filtered[0][filtered[0] != float("-inf")]
        assert torch.isfinite(kept).all(), (
            "FP16 logits overflowed to inf during temperature scaling at "
            "temperature=0 (issue #7857)."
        )
        # Sanity: the leading token survives the top-p=0.9 filter.
        assert filtered[0, 0] != float("-inf")

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_filter_logits_returns_fp32_when_temperature_not_one(self, input_dtype):
        """When temperature != 1.0, the working buffer is promoted to FP32
        so the divisor and downstream filters run in FP32. This matches the
        FlashInfer sampling path, which asserts the scaled logits are FP32."""
        logits = torch.randn(2, 128, dtype=input_dtype)
        filtered = TorchSampling.filter_logits(logits, temperature=0.7, top_k=0, top_p=0.9)
        assert filtered.dtype == torch.float32, (
            f"filter_logits dtype should be FP32 with temperature=0.7 for "
            f"{input_dtype} input, got {filtered.dtype}"
        )

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_filter_logits_preserves_dtype_when_temperature_is_one(self, input_dtype):
        """When temperature == 1.0 the divisor path is skipped and the
        filter preserves the input dtype; we still clone so the in-place
        filter does not mutate the caller."""
        logits = torch.randn(2, 128, dtype=input_dtype)
        logits_snapshot = logits.clone()

        filtered = TorchSampling.filter_logits(logits, temperature=1.0, top_k=0, top_p=0.0)

        assert filtered.dtype == input_dtype, (
            f"filter_logits dtype should preserve {input_dtype} with "
            f"temperature=1.0, got {filtered.dtype}"
        )
        # In-place filter must not mutate the caller-provided logits.
        assert torch.equal(
            logits, logits_snapshot
        ), "filter_logits mutated the caller's logits tensor"


class TestFilterLogitsDoesNotMutateInput:
    """Regression tests for the aliasing bug in ``TorchSampling.filter_logits``.

    ``Tensor.to(dtype)`` returns ``self`` when the dtype already matches, so
    promoting the working buffer to FP32 with a bare ``.to(torch.float32)``
    left the caller's tensor aliased and the following ``div_`` mutated it in
    place. For an FP32 input that is the framework default
    (``params_dtype`` defaults to ``torch.float32``), so the static batching
    path -- which passes a *view* of the model's logits
    (``logits[:, -1, :]``) -- had its prompt log-probs computed from
    temperature-scaled, top-k/top-p-masked logits.
    """

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 0.7, 2.0])
    def test_filter_logits_leaves_the_input_untouched(self, input_dtype, temperature):
        """The input tensor must be bit-identical after the call, for every
        temperature on the divisor path and for every input dtype."""
        torch.manual_seed(0)
        logits = torch.randn(4, 128, dtype=input_dtype)
        logits_snapshot = logits.clone()

        TorchSampling.filter_logits(logits, temperature=temperature, top_k=0, top_p=0.9)

        assert torch.equal(logits, logits_snapshot), (
            f"filter_logits mutated the caller's {input_dtype} logits at "
            f"temperature={temperature}"
        )

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("temperature", [0.0, 0.7, 2.0])
    def test_filter_logits_leaves_a_view_of_the_input_untouched(self, input_dtype, temperature):
        """The static batching path passes a view (``logits[:, -1, :]``) of the
        model's logits. A view aliases the parent buffer, so an in-place
        ``div_`` on the view corrupts the parent's last row."""
        torch.manual_seed(0)
        # Same [batch, seq, vocab] layout the static path produces.
        logits = torch.randn(4, 7, 128, dtype=input_dtype)
        logits_snapshot = logits.clone()

        TorchSampling.filter_logits(logits[:, -1, :], temperature=temperature, top_k=0, top_p=0.9)

        assert torch.equal(logits, logits_snapshot), (
            f"filter_logits mutated the caller's {input_dtype} logits through "
            f"a view at temperature={temperature}"
        )

    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("temperature", [0.0, 1.0, 0.7, 2.0])
    def test_filter_logits_returns_a_fresh_buffer(self, input_dtype, temperature):
        """``filter_logits`` documents "Returns a new tensor (input unmodified)".
        The result must never share storage with the input, whichever branch
        (divisor path or clone path) produced it."""
        torch.manual_seed(0)
        logits = torch.randn(4, 128, dtype=input_dtype)

        filtered = TorchSampling.filter_logits(logits, temperature=temperature, top_k=0, top_p=0.0)

        assert filtered.data_ptr() != logits.data_ptr(), (
            f"filter_logits returned a tensor aliasing the caller's {input_dtype} "
            f"logits at temperature={temperature}"
        )

    @pytest.mark.parametrize("temperature", [0.0, 0.7, 2.0])
    def test_sample_from_logits_leaves_the_input_untouched(self, temperature):
        """``sample_from_logits`` forwards the caller's tensor to
        ``filter_logits``; it must not modify it either."""
        torch.manual_seed(0)
        logits = torch.randn(4, 128, dtype=torch.float32)
        logits_snapshot = logits.clone()

        TorchSampling.sample_from_logits(
            logits,
            temperature=temperature,
            top_k=0,
            top_p=0.9,
            generator=torch.Generator().manual_seed(0),
        )

        assert torch.equal(
            logits, logits_snapshot
        ), "sample_from_logits mutated the caller's logits tensor"


class TestSamplerMatchesFP32Reference:
    """Sampler-level regressions for the two symptoms issue #7857 reports.

    The filter-level tests pin the intermediate values; these pin what the
    engine actually emits, which is what a user observes.
    """

    def test_bf16_sampler_never_emits_outside_the_fp32_nucleus(self):
        """REGRESSION (issue #7857): with the BF16 divisor the nucleus kept a
        second candidate (softmax mass 0.119 vs the FP32 reference's 0.076),
        so the engine emitted token 2136 where the FP32 reference emitted
        1317 with the same prefix and RNG state.

        The FP32 nucleus holds the leading token only, so every draw must land
        on it. Pre-fix the runner-up survived the filter and was drawn in
        roughly one draw in eight.
        """
        # The issue's step-29 reproducer: two candidates whose scaled logits
        # land 2.0 apart in BF16 but 2.5 apart in FP32.
        logits = torch.tensor([[26.625, 26.375] + [-100.0] * 126], dtype=torch.bfloat16)

        for seed in range(64):
            sampled = TorchSampling.sample_from_logits(
                logits,
                temperature=0.1,
                top_k=0,
                top_p=0.9,
                generator=torch.Generator().manual_seed(seed),
            )
            assert sampled.item() == 0, (
                f"seed {seed} emitted token {sampled.item()}, which the FP32 "
                "reference nucleus (leading token only) excludes (issue #7857)."
            )

    def test_fp16_zero_temperature_emits_the_argmax(self):
        """REGRESSION (issue #7857, FP16 column): at temperature=0 the divisor
        is floored to 1e-6, so FP16 scaling overflowed to ``inf`` and the
        softmax produced NaN. The engine then emitted token 0 instead of the
        greedy token (1124 in the issue's report).

        Scaling in FP32 keeps the distribution finite and one-hot on the
        argmax, so the sampler must agree with greedy.
        """
        logits = torch.tensor([[0.0, 1.0, 2.0, -1.0]], dtype=torch.float16)

        filtered = TorchSampling.filter_logits(logits, temperature=0.0, top_k=0, top_p=0.0)
        probabilities = filtered.softmax(dim=-1, dtype=torch.float32)
        assert torch.isfinite(probabilities).all(), (
            "FP16 scaling at temperature=0 produced non-finite probabilities " "(issue #7857)."
        )

        sampled = TorchSampling.sample_from_logits(
            logits,
            temperature=0.0,
            top_k=0,
            top_p=0.0,
            generator=torch.Generator().manual_seed(123),
        )
        assert sampled.item() == int(logits.argmax()), (
            f"zero-temperature sampling emitted token {sampled.item()} instead "
            f"of the greedy token {int(logits.argmax())} (issue #7857)."
        )

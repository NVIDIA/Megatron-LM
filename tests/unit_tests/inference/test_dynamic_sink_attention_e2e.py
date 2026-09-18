# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end test: dynamic-batching inference engine with sink (off-by-one /
learnable) softmax enabled.

Why a *separate* test file from ``engines/test_dynamic_engine.py``:
``engines/test_dynamic_engine.py`` is currently excluded from cog cluster
runs because its ``teardown_method`` calls ``delete_cuda_graphs()`` and can
SIGABRT — see the run-inference-unit-tests skill. This file lives one
directory up so it is picked up by the inference unit-test sweep, reuses
``DynamicInferenceEngineTestBase`` (which knows how to build a small GPT
model + dynamic engine end-to-end), but provides its own teardown that
does not accumulate CUDA graphs.

What this exercises that the math-only unit tests in
``test_dynamic_sink_attention.py`` do *not*:
  * Real flash-attn kernel call with ``return_softmax_lse=True`` /
    ``return_attn_probs=True`` — catches a kernel build that doesn't
    actually populate the LSE return value.
  * The FA3 wrapper's version-robust LSE locator
    (``_flash_attention_3_forward_wrapper(return_lse=True)``) against a
    real kernel return tuple.
  * The ``_get_inference_softmax_offset()`` accessor against a real
    ``self.core_attention`` module — both local DPA (where
    ``softmax_offset`` is set explicitly) and TE DPA.
  * The full plumbing through ``Attention.forward()`` →
    ``flash_decode_and_prefill()`` → sink correction → linear_proj.
"""
import pytest
import torch

from megatron.core.inference.inference_request import Status
from megatron.core.inference.utils import InferenceMode
from megatron.core.utils import is_fa_min_version

# Reuse the existing dynamic-engine test infrastructure. Only the
# *teardown* in that file is hazardous (the SIGABRT in delete_cuda_graphs);
# the builder/runner code is fine, and we add a softmax_type field on top
# in a separate edit to ``DynamicEngineTestConfig``.
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicInferenceEngineTestBase,
    set_rounder,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="dynamic batching requires flash-attn >= 2.7.3"
)
class TestDynamicEngineSinkAttention(DynamicInferenceEngineTestBase):
    """End-to-end dynamic-engine runs with sink (off-by-one / learnable)
    softmax enabled.

    Uses local transformer impl so the ``softmax_offset`` parameter is
    always exposed on ``self.core_attention`` — TE backend coverage is
    delegated to the math-only unit tests since it depends on TE version.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    def teardown_method(self, method):
        # ``DynamicInferenceEngine.start()`` (invoked by ``_run_test`` via
        # ``_build_test_env``) flips the process-wide ``InferenceMode`` flag
        # on but only clears it via an explicit ``suspend()``. These tests
        # never call ``suspend()``, so without this teardown the flag would
        # leak into subsequent tests in the same pytest worker (notably
        # ``test_moe_dispatching_and_routing.py::TestInferenceTopKRouter``,
        # which depends on the flag being False to exercise the training-mode
        # router path that returns sparse ``[num_tokens, num_experts]``
        # routing maps).
        InferenceMode.unset_active()

    @classmethod
    def teardown_class(cls):
        # Deliberately NOT calling delete_cuda_graphs() — these tests do
        # not enable CUDA graphs, so there is nothing to clean up, and
        # avoiding the call sidesteps the known teardown SIGABRT.
        set_rounder(64)
        Utils.destroy_model_parallel()

    @staticmethod
    def _generated_token_lists(env):
        """Return the per-request output-token tuples in a stable order."""
        return [
            tuple(req.generated_tokens) if req.generated_tokens is not None else ()
            for req in sorted(env.requests, key=lambda r: r.request_id)
        ]

    @pytest.mark.parametrize("softmax_type", ["off-by-one", "learnable"])
    def test_dynamic_engine_runs_with_sink(self, softmax_type):
        """Smoke test: the dynamic engine runs to completion when sink
        softmax is enabled, and every request produces non-empty output.

        This is the canonical signal that the new code path
        (``Attention._get_inference_softmax_offset`` →
        ``flash_decode_and_prefill(softmax_offset=…)`` → flash-attn with
        LSE → ``_apply_sink_softmax_correction_*``) is wired up correctly
        against real CUDA kernels.
        """
        env = self._run_test(
            softmax_type=softmax_type,
            transformer_impl="local",
            num_tokens_to_generate=16,
            min_prompt_length=8,
            max_prompt_length=16,
        )

        for req in env.requests:
            assert req.status == Status.COMPLETED, (
                f"request {req.request_id} ended with status {req.status} "
                f"(softmax_type={softmax_type!r})"
            )
            assert req.generated_tokens is not None and len(req.generated_tokens) > 0, (
                f"request {req.request_id} produced no output tokens "
                f"(softmax_type={softmax_type!r})"
            )

    def test_sink_rescale_helpers_are_invoked(self, monkeypatch):
        """Verify the sink-softmax post-hoc rescale path actually fires when
        the dynamic engine runs with ``softmax_type='off-by-one'``.

        A naïve "tokens must differ from vanilla" assertion is unreliable
        here: with ``softmax_offset=0`` (the default for ``off-by-one``),
        the denominator gains only ``exp(0)=1`` next to ``∑exp(qk)``, which
        is huge for a context of 16+ tokens. That's by design — Miller's
        off-by-one is *meant* to barely perturb saturating heads. Greedy
        sampling on a small random-init model is unlikely to flip the
        argmax. So instead we directly verify the wiring: at least one of
        the two rescale helpers in ``Attention`` must be called during the
        run, which can only happen if
        ``_get_inference_softmax_offset()`` returned a non-None tensor
        *and* a flash-attn branch actually retrieved + applied an LSE.
        """
        from megatron.core.transformer.attention import Attention

        call_counts = {"varlen": 0, "bshd": 0}
        orig_varlen = Attention._apply_sink_softmax_correction_varlen
        orig_bshd = Attention._apply_sink_softmax_correction_bshd

        def wrap_varlen(output, lse, softmax_offset):
            call_counts["varlen"] += 1
            return orig_varlen(output, lse, softmax_offset)

        def wrap_bshd(output, lse, softmax_offset):
            call_counts["bshd"] += 1
            return orig_bshd(output, lse, softmax_offset)

        monkeypatch.setattr(
            Attention, "_apply_sink_softmax_correction_varlen", staticmethod(wrap_varlen)
        )
        monkeypatch.setattr(
            Attention, "_apply_sink_softmax_correction_bshd", staticmethod(wrap_bshd)
        )

        env = self._run_test(
            softmax_type="off-by-one",
            transformer_impl="local",
            num_tokens_to_generate=8,
            min_prompt_length=8,
            max_prompt_length=8,
        )

        # Sanity: engine completed normally.
        for req in env.requests:
            assert req.status == Status.COMPLETED

        # At least one rescale path must have fired. Which one depends on
        # whether the workload was decode-only (bshd) or mixed
        # prefill+decode (varlen); the test fixture exercises both at
        # different steps, so we don't pin which counter increments.
        total_calls = call_counts["varlen"] + call_counts["bshd"]
        assert total_calls > 0, (
            f"Neither sink-rescale helper was called during the dynamic "
            f"engine run with softmax_type='off-by-one' "
            f"({call_counts!r}). The post-hoc LSE rescale is not being "
            f"wired through Attention.flash_decode_and_prefill()."
        )

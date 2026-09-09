# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``megatron.core.telemetry.fallbacks``.

Call sites import these five names unconditionally, so they must exist and be
callable whether or not ``nemo-lens`` is installed. The no-op behaviour
assertions only hold for the local stubs, so they skip when lens is present.
"""

import pytest

from megatron.core.telemetry import fallbacks

try:
    import nemo.lens  # noqa: F401

    HAVE_NEMO_LENS = True
except ImportError:
    HAVE_NEMO_LENS = False

requires_no_lens = pytest.mark.skipif(
    HAVE_NEMO_LENS, reason="local no-op stubs are only used without nemo-lens"
)


class TestPublicSurface:
    """Holds regardless of which implementation was imported."""

    @pytest.mark.parametrize(
        "name",
        [
            "trace_fn",
            "managed_span",
            "is_span_group_enabled",
            "safe_set_span_attributes",
            "span_cm",
        ],
    )
    def test_name_is_exported_and_callable(self, name):
        assert callable(getattr(fallbacks, name))


@requires_no_lens
class TestTraceFn:
    def test_returns_the_function_unchanged(self):
        def original(a, b):
            return a + b

        decorated = fallbacks.trace_fn("job", "megatron.train")(original)
        assert decorated is original

    def test_decorated_function_still_works(self):
        @fallbacks.trace_fn("job", "megatron.train")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_accepts_an_explicit_tracer(self):
        def original():
            return None

        assert fallbacks.trace_fn("job", "megatron.train", tracer=object())(original) is original


@requires_no_lens
class TestManagedSpan:
    def test_yields_none(self):
        with fallbacks.managed_span("job", "megatron.train") as span:
            assert span is None

    def test_accepts_arbitrary_attributes(self):
        with fallbacks.managed_span("job", "megatron.train", iteration=7, rank=0) as span:
            assert span is None

    def test_propagates_exceptions_from_the_body(self):
        with pytest.raises(ValueError, match="boom"):
            with fallbacks.managed_span("job", "megatron.train"):
                raise ValueError("boom")


@requires_no_lens
class TestSpanCm:
    def test_yields_none(self):
        with fallbacks.span_cm("megatron.train") as span:
            assert span is None

    def test_accepts_record_exception_and_attributes(self):
        with fallbacks.span_cm("megatron.train", record_exception=False, rank=3) as span:
            assert span is None

    def test_propagates_exceptions_from_the_body(self):
        with pytest.raises(ValueError, match="boom"):
            with fallbacks.span_cm("megatron.train"):
                raise ValueError("boom")


@requires_no_lens
class TestIsSpanGroupEnabled:
    @pytest.mark.parametrize("group", ["job", "step", "microbatch", "not_a_real_group"])
    def test_always_false(self, group):
        """Every group is off, so gated instrumentation stays dormant."""
        assert fallbacks.is_span_group_enabled(group) is False


@requires_no_lens
class TestSafeSetSpanAttributes:
    def test_accepts_a_none_span(self):
        assert fallbacks.safe_set_span_attributes(None, {"iteration": 7}) is None

    def test_accepts_redact_keys(self):
        assert (
            fallbacks.safe_set_span_attributes(None, {"token": "x"}, redact_keys=["token"]) is None
        )

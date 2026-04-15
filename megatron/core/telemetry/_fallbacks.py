# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""No-op fallbacks for when nemo-lens is not installed.

When nemo-lens IS installed, re-exports from nemo.lens.fallbacks for
consistency. When it is NOT installed, provides identical local no-ops.
"""

try:
    from nemo.lens.fallbacks import (  # noqa: F401
        is_span_group_enabled,
        managed_span,
        safe_set_span_attributes,
        span_cm,
        trace_fn,
    )
except ImportError:
    from contextlib import contextmanager

    def trace_fn(group, name, tracer=None):
        """No-op decorator — returns the function unchanged."""
        def decorator(func):
            return func
        return decorator

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        """No-op context manager — yields None."""
        yield None

    def is_span_group_enabled(group):
        """Always returns False when nemo-lens is not installed."""
        return False

    def safe_set_span_attributes(span, attributes, redact_keys=None):
        """No-op."""
        pass

    @contextmanager
    def span_cm(name, tracer=None, record_exception=True, **attributes):
        """No-op context manager — yields None."""
        yield None

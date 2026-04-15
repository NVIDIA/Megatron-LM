# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""No-op fallbacks for when nemo-lens is not installed.

These match the nemo.lens API so instrumented code works unchanged.
When nemo-lens IS installed, these are never used.
"""

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

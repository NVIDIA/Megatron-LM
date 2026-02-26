# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures for telemetry unit tests.

The OpenTelemetry API treats ``set_tracer_provider`` / ``set_meter_provider``
as set-once operations — calling them a second time logs a warning instead of
replacing the provider.  In a test suite that creates fresh providers per-test,
we must reset the internal ``Once`` guards so each test starts from a clean
slate.  The helper below does exactly that.
"""

import pytest

from opentelemetry import metrics, trace
from opentelemetry.util._once import Once

# Internal module references used to reset the set-once guards.
import opentelemetry.trace as _trace_mod
import opentelemetry.metrics._internal as _metrics_mod


def _reset_otel_globals() -> None:
    """Reset the global OTel tracer/meter providers to an unset state.

    This allows ``set_tracer_provider`` / ``set_meter_provider`` to be called
    again without emitting the "Overriding of current … is not allowed" warning.
    """
    # Reset tracer provider
    _trace_mod._TRACER_PROVIDER = None  # noqa: SLF001
    _trace_mod._TRACER_PROVIDER_SET_ONCE = Once()  # noqa: SLF001

    # Reset meter provider
    _metrics_mod._METER_PROVIDER = None  # noqa: SLF001
    _metrics_mod._METER_PROVIDER_SET_ONCE = Once()  # noqa: SLF001


@pytest.fixture(autouse=True)
def reset_otel_providers():
    """Reset global OTel providers before and after each test.

    This prevents cross-test pollution and eliminates the
    "Overriding of current TracerProvider/MeterProvider is not allowed"
    warnings that would otherwise fire on every test.
    """
    _reset_otel_globals()
    yield
    _reset_otel_globals()

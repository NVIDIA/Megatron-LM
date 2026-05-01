# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lightweight metrics abstraction for the inference coordinator.

Provides a backend-agnostic interface so that instrumentation code is
decoupled from any specific metrics system (Prometheus, StatsD, etc.).

Usage
-----
Pass a ``CoordinatorMetrics`` implementation into the coordinator at
construction time.  When no implementation is provided the coordinator
defaults to ``NoOpMetrics``, which adds near-zero overhead.

Metric naming conventions
--------------------------
- ``coordinator_*`` — system-level metrics (errors, active engines, latency).
- ``routing_*``     — routing-quality metrics (cache hits, misses, fallbacks).
"""

from abc import ABC, abstractmethod


class CoordinatorMetrics(ABC):
    """Abstract interface for coordinator observability metrics.

    Implement this class to plug in any metrics backend without modifying
    coordinator logic.
    """

    @abstractmethod
    def inc(self, name: str, value: int = 1) -> None:
        """Increment a counter by *value*.

        Args:
            name:  Metric name, e.g. ``"routing_cache_hit_total"``.
            value: Amount to add (default 1).
        """

    @abstractmethod
    def observe(self, name: str, value: float) -> None:
        """Record a latency or distribution sample.

        Args:
            name:  Metric name, e.g. ``"coordinator_routing_latency_seconds"``.
            value: Observed value in the metric's natural unit.
        """

    @abstractmethod
    def gauge(self, name: str, value: float) -> None:
        """Set an instantaneous gauge value.

        Args:
            name:  Metric name, e.g. ``"coordinator_active_engines"``.
            value: Current value.
        """


class NoOpMetrics(CoordinatorMetrics):
    """No-op implementation.  Adds near-zero overhead when observability is disabled.

    This is the default used by the coordinator when no metrics backend is
    supplied.
    """

    def inc(self, name: str, value: int = 1) -> None:
        pass

    def observe(self, name: str, value: float) -> None:
        pass

    def gauge(self, name: str, value: float) -> None:
        pass

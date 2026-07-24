# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Central definitions for MLite local-test execution markers."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MIN_ARCHITECTURE = "hopper"
DEFAULT_TIMEOUT_SECONDS = 1800

ARCHITECTURE_ORDER = {"hopper": 0, "blackwell": 1}
ARCHITECTURE_CAPABILITIES = {"hopper": (9, 0), "blackwell": (10, 0)}

# Keep this allowlist small. A test that needs another per-process environment
# variable must add it here so misspellings cannot silently change test behavior.
ENVIRONMENT_VARIABLES = {"CUDA_DEVICE_MAX_CONNECTIONS"}

MARKER_DESCRIPTIONS = (
    "gpus(count, min_architecture='hopper'): request count GPUs and declare the minimum "
    "supported GPU architecture; an absent marker means a CPU test",
    "env(**variables): set an allowlisted per-test environment variable to a string, or "
    "use None to explicitly unset it",
    "timeout(seconds=1800): set the positive per-suite timeout in seconds; the marker is "
    "optional and defaults to 1800 seconds",
    "optional: exclude a test from the standard run_tests.sh workflow",
)


class MarkerError(ValueError):
    """Raised when an MLite execution marker is malformed."""


@dataclass(frozen=True)
class ExecutionSpec:
    gpus: int
    min_architecture: str | None
    environment: tuple[tuple[str, str | None], ...]
    timeout_seconds: int
    optional: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "gpus": self.gpus,
            "min_architecture": self.min_architecture,
            "environment": dict(self.environment),
            "timeout_seconds": self.timeout_seconds,
            "optional": self.optional,
        }


def register(config) -> None:
    """Register every MLite-owned marker with pytest in one place."""
    for description in MARKER_DESCRIPTIONS:
        config.addinivalue_line("markers", description)


def _gpus_marker(item) -> tuple[int, str | None]:
    marker = next(item.iter_markers(name="gpus"), None)
    if marker is None:
        return 0, None
    if len(marker.args) != 1 or set(marker.kwargs) - {"min_architecture"}:
        raise MarkerError(
            "gpus requires one positional count and optional min_architecture keyword"
        )
    count = marker.args[0]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise MarkerError("gpus count must be a positive integer")
    architecture = marker.kwargs.get("min_architecture", DEFAULT_MIN_ARCHITECTURE)
    if architecture not in ARCHITECTURE_ORDER:
        supported = ", ".join(ARCHITECTURE_ORDER)
        raise MarkerError(f"min_architecture must be one of: {supported}")
    return count, architecture


def _environment_markers(item) -> tuple[tuple[str, str | None], ...]:
    environment: dict[str, str | None] = {}
    # pytest yields closest markers first. Apply outer scopes first so function
    # and parameter markers can override module or class defaults.
    for marker in reversed(list(item.iter_markers(name="env"))):
        if marker.args:
            raise MarkerError("env accepts keyword arguments only")
        unknown = set(marker.kwargs) - ENVIRONMENT_VARIABLES
        if unknown:
            raise MarkerError(
                "env contains unsupported variables: " + ", ".join(sorted(unknown))
            )
        for name, value in marker.kwargs.items():
            if value is not None and not isinstance(value, str):
                raise MarkerError(f"env value for {name} must be a string or None")
            environment[name] = value
    return tuple(sorted(environment.items()))


def _timeout_marker(item) -> int:
    marker = next(item.iter_markers(name="timeout"), None)
    if marker is None:
        return DEFAULT_TIMEOUT_SECONDS
    if marker.args or set(marker.kwargs) != {"seconds"}:
        raise MarkerError("timeout requires exactly one keyword argument: seconds")
    seconds = marker.kwargs["seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
        raise MarkerError("timeout seconds must be a positive integer")
    return seconds


def execution_for_item(item) -> ExecutionSpec:
    """Resolve and validate the effective execution contract for one item."""
    gpus, architecture = _gpus_marker(item)
    environment = _environment_markers(item)
    timeout_seconds = _timeout_marker(item)
    return ExecutionSpec(
        gpus=gpus,
        min_architecture=architecture,
        environment=environment,
        timeout_seconds=timeout_seconds,
        optional=next(item.iter_markers(name="optional"), None) is not None,
    )


def architecture_supports(actual: str, minimum: str) -> bool:
    return ARCHITECTURE_ORDER[actual] >= ARCHITECTURE_ORDER[minimum]

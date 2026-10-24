"""Train-side lightweight protocols and defaults."""

from __future__ import annotations

from collections.abc import Callable

from torch.distributed.tensor import Replicate  # pyright: ignore[reportMissingImports]

ExpertClassifierFn = Callable[[str], bool]
PlacementFn = Callable[[str], list]


def default_expert_classifier(name: str) -> bool:
    """Default: params with 'experts' (but not 'router' or 'shared') are expert params."""
    return "experts" in name and "router" not in name and "shared" not in name


def default_placement_fn(name: str) -> list:
    """Default: all Replicate (safe but no resharding benefit)."""
    return [Replicate(), Replicate(), Replicate(), Replicate()]


__all__ = [
    "ExpertClassifierFn",
    "PlacementFn",
    "default_expert_classifier",
    "default_placement_fn",
]

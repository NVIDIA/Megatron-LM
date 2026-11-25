# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resolve recipe targets when the model exists, not at config construction."""

from fnmatch import fnmatchcase

from .config import LoRA


def matches(pattern: str, path: str) -> bool:
    """Match a full module path or an unqualified leaf name."""
    return fnmatchcase(path, pattern) or (
        "." not in pattern and fnmatchcase(path.rsplit(".", 1)[-1], pattern)
    )


def selected_components(config: LoRA, path: str) -> tuple[set[str], set[str]]:
    """Return selected logical projections and the user patterns that selected them."""
    if any(matches(pattern, path) for pattern in config.exclude_modules):
        return set(), set()
    parent, _, leaf = path.rpartition(".")
    if leaf == "linear_qkv" and config.canonical_qkv_projections:
        components = ("linear_q", "linear_k", "linear_v")
    elif leaf == "linear_fc1" and config.canonical_mlp_projections:
        components = ("linear_fc1_gate", "linear_fc1_up")
    else:
        components = (leaf,)
    selected, matched = set(), set()
    for component in components:
        component_path = f"{parent}.{component}" if parent else component
        if any(matches(pattern, component_path) for pattern in config.exclude_modules):
            continue
        for pattern in config.target_modules:
            if matches(pattern, path) or matches(pattern, component_path):
                selected.add(component)
                matched.add(pattern)
    return selected, matched

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Plan deterministic corpus blends into immutable disk-backed reader catalogs."""

import json
from typing import Any

from .disk_plan import MANIFEST_VERSION, plan

DEFAULTS = {"seed": 1234, "lanes": 64}


def load_config(path: str) -> dict[str, Any]:
    """Read and complete a blend configuration."""
    with open(path) as stream:
        config = {**DEFAULTS, **json.load(stream)}
    if not isinstance(config.get("corpus"), str) or not config.get("sources"):
        raise ValueError("A corpus blend config needs a corpus directory and weighted sources")
    if any(not isinstance(w, (int, float)) or w <= 0 for w in config["sources"].values()):
        raise ValueError("Corpus blend weights must be positive numbers")
    if type(config["lanes"]) is not int or config["lanes"] < 1:
        raise ValueError("Corpus blend lanes must be a positive integer")
    return config


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Return the identity streamed over catalog contents and schedule during planning."""
    return manifest["hash"]


def describe(manifest: dict[str, Any]) -> str:
    """Describe a plan's source allocations without opening its catalog."""
    lines = [
        f"Offline KD corpus blend {manifest['hash'][:12]}: {manifest['lanes']} lanes x "
        f"{manifest['lane_samples']} samples"
    ]
    for source in manifest["sources"]:
        weight = manifest["config"]["sources"][source["name"]]
        lines.append(
            f"  {source['name']}: weight {weight}, {source['samples']} samples "
            f"({source['share']:.1%}; {source['supervised_share']:.1%} of supervised tokens) "
            f"from {source['available']} available, {source['passes']} pass(es)"
        )
    return "\n".join(lines)


__all__ = ["describe", "load_config", "manifest_hash", "plan"]

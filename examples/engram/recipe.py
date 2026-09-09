# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure-Python definitions for the offline FineWeb comparison recipe."""

import json
import uuid
from dataclasses import asdict, dataclass
from math import isqrt
from pathlib import Path
from typing import Any


def archive_invocation_manifest(artifacts: Path, manifest: dict[str, Any]) -> Path:
    """Preserve the original training identity and archive every invocation separately."""
    sessions = artifacts / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{uuid.uuid4().hex}.json"
    payload = json.dumps(manifest, indent=2) + "\n"
    path.write_text(payload)
    original = artifacts / "recipe.json"
    if not manifest.get("execution", {}).get("full_eval_label") and not original.exists():
        with original.open("x") as stream:
            stream.write(payload)
    return path


@dataclass(frozen=True)
class Schedule:
    """Training schedule, indexed by the number of already completed updates."""

    train_steps: int = 36754
    warmup_steps: int = 1000
    decay_steps: int = 3675
    stable_mtp_weight: float = 0.3
    decay_mtp_weight: float = 0.15

    def phase(self, completed_steps: int) -> str:
        """Return the phase of the next forward, including after checkpoint loading."""
        if completed_steps < 0:
            raise ValueError("Completed steps cannot be negative")
        if completed_steps < self.warmup_steps:
            return "warmup"
        if completed_steps < self.train_steps - self.decay_steps:
            return "stable"
        return "decay"

    def mtp_weight(self, completed_steps: int) -> float:
        """Return the coefficient used by every microbatch of the next update."""
        return (
            self.decay_mtp_weight
            if self.phase(completed_steps) == "decay"
            else self.stable_mtp_weight
        )


def prime_capacities(minimum: int, heads: int = 16) -> list[int]:
    """Independently calculate the distinct prime row counts for this single memory layer."""
    result = []
    candidate = minimum
    while len(result) < heads:
        if candidate >= 2 and all(
            candidate % divisor for divisor in range(2, isqrt(candidate) + 1)
        ):
            result.append(candidate)
        candidate += 1
    return result


def expected_parameters(engram: bool) -> dict[str, Any]:
    """Return analytic counts, excluding embedding, output head and the separate MTP branch."""
    hidden, blocks, heads, kv, dense_ffn, moe_ffn = 512, 8, 12, 128, 1344, 256
    routed, active, shared = (208 if engram else 256), 8, 1
    attention = hidden * (heads * kv + 2 * kv + heads * kv + heads * kv)
    norms = 2 * hidden * blocks + hidden
    dense = 3 * hidden * dense_ffn
    expert = 3 * hidden * moe_ffn
    routers = (blocks - 1) * hidden * routed
    capacities = prime_capacities(205700) if engram else []
    table = sum(capacities) * 40
    fusion = 2 * (640 * hidden + hidden) + 4 * hidden + 3 * hidden if engram else 0
    memory_active = 16 * 40 + fusion if engram else 0
    common = blocks * attention + norms + dense + routers
    return {
        "scope_excludes": ["embedding", "lm_head", "mtp"],
        "backbone_total": common + (blocks - 1) * expert * (routed + shared) + table + fusion,
        "activated": common + (blocks - 1) * expert * (active + shared) + memory_active,
        "engram_total": table + fusion,
        "engram_table": table,
        "engram_fusion": fusion,
        "table_capacities": capacities,
        "table_rows": sum(capacities),
    }


def recipe_manifest(engram: bool) -> dict[str, Any]:
    """Return the fixed comparison configuration, independent of machine paths."""
    return {
        "name": "Engram-0.73A0.05B-0.8" if engram else "MoE-0.73A0.05B",
        "seed": 2026,
        "holdout_seed": 1234,
        "hybrid_pattern": "*-" + "*E" * 7 + "/*E",
        "global_batch_size": 64,
        "sequence_length": 4096,
        "samples": 36754 * 64,
        "tokens": 36754 * 64 * 4096,
        "maximum_lr": 8e-4,
        "minimum_lr": 8e-5,
        "schedule": asdict(Schedule()),
        "expected_parameters": expected_parameters(engram),
    }


def set_mtp_weight(model: Any, weight: float) -> None:
    """Update the actual configurations read by all MTP modules in one model chunk."""
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "mtp_loss_scaling_factor"):
            config.mtp_loss_scaling_factor = weight


class PaddedEvaluationDataset:
    """Pad a GPT evaluation dataset with zero-loss samples to equalize DP rank lengths.

    Megatron's full-validation sampler emits one sample per rank. Padding avoids
    its cyclic iterator repeating real samples on ranks with shorter tails.
    """

    def __init__(self, dataset: Any, data_parallel_size: int) -> None:
        self.dataset = dataset
        self.index_split = dataset.index_split
        self.size = (
            (len(dataset) + data_parallel_size - 1) // data_parallel_size * data_parallel_size
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index if index < len(self.dataset) else None]

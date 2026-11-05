# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layer-independent parsing and execution policy for depth recurrence."""

import math
from dataclasses import dataclass
from typing import Iterator

import torch


def parse_recurrence_pattern(pattern: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Remove loop boundaries and return half-open physical layer ranges.

    In ``prefix:body1:body2:...:suffix``, each interior section is a separate,
    non-empty recurrent region. Prefix and suffix may be empty. All regions use
    the same loop count. Ranges refer to unique layers, never to expanded depth.
    """
    if ":" not in pattern:
        return pattern, ()
    if "|" in pattern:
        raise ValueError("Hybrid recurrence does not support pipeline ('|') boundaries")
    parts = pattern.split(":")
    if len(parts) < 3 or any(not body for body in parts[1:-1]):
        raise ValueError(
            "Hybrid recurrence requires non-empty interior regions: prefix:body1:...:suffix"
        )
    ranges = []
    offset = len(parts[0])
    for body in parts[1:-1]:
        end = offset + len(body)
        ranges.append((offset, end))
        offset = end
    return "".join(parts), tuple(ranges)


@dataclass(frozen=True)
class LoopDistribution:
    """A categorical distribution over positive loop counts, independent of model weights."""

    counts: tuple[int, ...]
    """Distinct positive loop counts in sampling order."""

    probabilities: tuple[float, ...]
    """Positive finite probabilities, aligned with counts and summing to one."""

    def __post_init__(self) -> None:
        if not self.counts or len(self.counts) != len(self.probabilities):
            raise ValueError("Loop counts and probabilities must be non-empty and aligned")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in self.counts
        ) or len(set(self.counts)) != len(self.counts):
            raise ValueError("Loop counts must be unique positive integers")
        if any(not math.isfinite(p) or p <= 0 for p in self.probabilities):
            raise ValueError("Loop probabilities must be finite and positive")
        if not math.isclose(math.fsum(self.probabilities), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Loop probabilities must sum to 1.0")

    @classmethod
    def from_string(cls, value: str | None) -> "LoopDistribution":
        """Parse ``count:probability,...``; an omitted policy executes one loop."""
        if value is None:
            return cls((1,), (1.0,))
        try:
            entries = [entry.split(":") for entry in value.split(",")]
            counts = tuple(int(count) for count, _ in entries)
            probabilities = tuple(float(probability) for _, probability in entries)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "n_loops_distribution must have the form '1:0.5,2:0.5'; "
                "per-region pipe syntax is not supported"
            ) from error
        return cls(counts, probabilities)

    def sample(
        self, training: bool, device: torch.device, group: torch.distributed.ProcessGroup | None
    ) -> int:
        """Sample once per training forward, synchronizing the supplied rank domain.

        Evaluation uses the largest configured count and does not consume random state.
        Stochastic draws use the checkpointed CPU torch RNG. Every rank consumes one
        draw, then the group's first rank broadcasts its count so TP, DP, and expert
        collectives follow the same execution order even if their RNG states differ.
        """
        if not training:
            return max(self.counts)
        if len(self.counts) == 1:
            return self.counts[0]
        if group is None:
            raise ValueError("Stochastic hybrid recurrence requires pg_collection.tp_dp_cp")
        draw = torch.rand((), device="cpu").item()
        cumulative = 0.0
        count = self.counts[-1]
        for candidate, probability in zip(self.counts, self.probabilities, strict=True):
            cumulative += probability
            if draw < cumulative:
                count = candidate
                break
        if group.size() > 1:
            comm_device = device if torch.distributed.get_backend(group) == "nccl" else "cpu"
            sampled = torch.tensor(count, dtype=torch.int64, device=comm_device)
            torch.distributed.broadcast(
                sampled, src=torch.distributed.get_global_rank(group, 0), group=group
            )
            count = sampled.item()
        return count


def recurrent_layer_indices(
    num_layers: int, ranges: tuple[tuple[int, int], ...], n_loops: int
) -> Iterator[int]:
    """Repeat each region in order with one shared count and no duplicated modules."""
    if isinstance(n_loops, bool) or not isinstance(n_loops, int) or n_loops < 1:
        raise ValueError("n_loops must be a positive integer")
    previous_end = 0
    for start, end in ranges:
        if not previous_end <= start < end <= num_layers:
            raise ValueError("Recurrent layer ranges must be non-empty, ordered, and in bounds")
        yield from range(previous_end, start)
        for _ in range(n_loops):
            yield from range(start, end)
        previous_end = end
    yield from range(previous_end, num_layers)


def validate_recurrence_config(config) -> None:
    """Reject execution modes that need a recurrence-aware schedule or per-call state.

    These restrictions apply only to marked recurrent stacks. Ordinary HybridModel
    configurations retain their existing behavior.
    """
    unsupported = {
        "pipeline parallelism": config.pipeline_model_parallel_size != 1
        or config.virtual_pipeline_model_parallel_size is not None
        or config.pipeline_model_parallel_layout is not None,
        "context parallelism": config.context_parallel_size != 1 or config.dynamic_context_parallel,
        "activation recomputation": config.recompute_granularity is not None,
        "CUDA graphs": config.cuda_graph_impl != "none",
        "FP8/FP4 quantization": bool(config.fp8 or config.fp4 or config.quant_recipe),
        "shortcut MoE": config.moe_shortcut_connection,
        "expert communication overlap": config.overlap_moe_expert_parallel_comm,
        "activation offloading": config.cpu_offloading
        or config.fine_grained_activation_offloading
        or config.moe_paged_stash,
        "delayed weight gradients": config.delay_wgrad_compute,
        "weight rematerialization": config.gtp_weight_remat_size != 1
        or config.expert_gtp_weight_remat_size != 1,
    }
    for feature, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"Hybrid recurrence does not yet support {feature}")

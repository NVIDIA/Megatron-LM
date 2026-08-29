# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Decide once how DSA indexer scoring will run, instead of re-deriving it per call.

Choosing between the fused indexer scorer and the per-head fallback loop used to be
spread across two files: ``DSAttention.forward`` assembled a handful of booleans
(``use_local_indexer_varlen``, ``single_packed_thd_sequence``, the packed ``cu_seqlens``
metadata), threaded them through the kernel dispatcher, and the dispatcher re-combined
them into a branch. The decision existed only as the emergent result of those
conditions, so a configuration that no fused kernel claimed simply fell through to the
slow path with nothing reported.

This module makes the decision a value. :func:`resolve_indexer_scoring_plan` is total --
every configuration maps to some :class:`IndexerScoringPlan`, including
:attr:`IndexerScoringPlan.UNFUSED_BOUNDS` -- and carries the reason it landed there, so
callers can surface an unfused configuration instead of silently paying for it.

Splitting the decision from its execution also separates two questions that the old
conditions conflated: *what layout is this* versus *which kernel do we happen to have*.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class IndexerScoringPlan(Enum):
    """How indexer scores and top-k are produced for one attention call."""

    PLAIN_CAUSAL = "plain_causal"
    """Whole-sequence causal bounds. Reconstructible, so the single-kernel scorer runs."""

    PACKED_SINGLE_SEGMENTS = "packed_single_segments"
    """One sequence per pack, scored as one or more causal query/key segments."""

    PACKED_THD_GENERAL = "packed_thd_general"
    """General packed THD scoring driven by query/key cu_seqlens metadata."""

    UNFUSED_BOUNDS = "unfused_bounds"
    """No fused kernel claims this layout; scoring falls back to the per-head loop."""

    DECLINE = "decline"
    """Bounds cannot be built at all, so the caller must not attempt fused scoring."""

    @property
    def is_fused(self) -> bool:
        """Whether this plan reaches a fused scoring kernel."""
        return self in (
            IndexerScoringPlan.PLAIN_CAUSAL,
            IndexerScoringPlan.PACKED_SINGLE_SEGMENTS,
            IndexerScoringPlan.PACKED_THD_GENERAL,
        )


@dataclass(frozen=True)
class IndexerScoringDecision:
    """A resolved plan plus the reason it was chosen.

    ``reason`` is meant to be read by a human staring at a slow run, so it names the
    condition that decided the outcome rather than restating the plan.
    """

    plan: IndexerScoringPlan
    reason: str

    @property
    def is_fused(self) -> bool:
        """Whether the chosen plan reaches a fused scoring kernel."""
        return self.plan.is_fused


def resolve_indexer_scoring_plan(
    *,
    bounds_available: bool,
    varlen_is_plain_causal: bool,
    packed_thd: bool,
    cp_size: int,
    mask_is_causal: bool,
    explicit_key_positions: bool,
    single_sequence_pack: bool,
    packed_metadata_available: bool,
    segment_kernel_applicable: bool = True,
) -> IndexerScoringDecision:
    """Resolve how indexer scoring should run for one attention call.

    The arguments are facts about the layout, not capability flags: whoever calls this
    does not need to know which kernels exist. Ordering matters only in that the
    earlier clauses describe strictly more specific layouts.

    Args:
        bounds_available: The caller managed to build ``(starts, ends)`` key bounds.
        varlen_is_plain_causal: Those bounds are the trivial whole-sequence causal ones,
            as flagged by the mask builder, so a kernel can rebuild them itself.
        packed_thd: Sequences are packed into a THD buffer.
        cp_size: Context-parallel world size.
        mask_is_causal: The attention mask type is plain causal.
        explicit_key_positions: Custom key positions are in play, so bounds alone do not
            describe the layout.
        single_sequence_pack: This microbatch's pack holds exactly one sequence. This is
            the only input here that varies batch to batch.
        packed_metadata_available: ``cu_seqlens`` and max-seqlen metadata are present.
        segment_kernel_applicable: The single-sequence segment kernel's structural
            preconditions hold for this call. It raises rather than declining when they
            do not, so the plan must not promise it.

    Returns:
        The resolved plan and the reason for it.
    """
    if explicit_key_positions:
        return IndexerScoringDecision(
            IndexerScoringPlan.DECLINE, "custom key positions are not expressible as bounds"
        )
    if not bounds_available:
        return IndexerScoringDecision(
            IndexerScoringPlan.DECLINE, "key bounds could not be built for this mask"
        )

    if varlen_is_plain_causal and not packed_thd and cp_size == 1:
        return IndexerScoringDecision(
            IndexerScoringPlan.PLAIN_CAUSAL, "whole-sequence causal bounds are reconstructible"
        )

    if packed_thd and mask_is_causal:
        if single_sequence_pack and segment_kernel_applicable:
            return IndexerScoringDecision(
                IndexerScoringPlan.PACKED_SINGLE_SEGMENTS,
                "one sequence per pack, scored as causal segments",
            )
        # General THD scoring is independent of context-parallel world size. It is the
        # normal multi-sequence path and also the safe fallback when the specialized
        # single-sequence segment executor cannot take this call's shape.
        if packed_metadata_available:
            return IndexerScoringDecision(
                IndexerScoringPlan.PACKED_THD_GENERAL,
                (
                    "single-sequence pack falling back to cu_seqlens metadata"
                    if single_sequence_pack
                    else "multi-sequence pack with cu_seqlens metadata"
                ),
            )
        return IndexerScoringDecision(
            IndexerScoringPlan.UNFUSED_BOUNDS,
            (
                "single-sequence pack whose segment shape is unsupported and whose "
                "cu_seqlens metadata is unavailable or incompatible"
                if single_sequence_pack
                else "packed THD without compatible cu_seqlens metadata"
            ),
        )

    # Everything below is a layout no fused kernel currently claims. Name the specific
    # reason: these are the configurations worth widening a gate for.
    if not packed_thd and cp_size > 1:
        return IndexerScoringDecision(
            IndexerScoringPlan.UNFUSED_BOUNDS,
            "context parallelism without sequence packing has no fused scoring path",
        )
    if not mask_is_causal:
        return IndexerScoringDecision(
            IndexerScoringPlan.UNFUSED_BOUNDS, "non-causal mask has no fused scoring path"
        )
    return IndexerScoringDecision(
        IndexerScoringPlan.UNFUSED_BOUNDS, "bounds are not reconstructible by any fused kernel"
    )


_REPORTED_UNFUSED: set = set()


def report_unfused_scoring_once(decision: IndexerScoringDecision, context: str) -> Optional[str]:
    """Log an unfused scoring plan the first time each distinct reason appears.

    A configuration that misses every fused kernel is a throughput cliff with no other
    symptom, so it should say so once rather than never. Returns the emitted message, or
    ``None`` if this reason was already reported.
    """
    if decision.is_fused or decision.plan is IndexerScoringPlan.DECLINE:
        return None
    key = f"{context}|{decision.reason}"
    if key in _REPORTED_UNFUSED:
        return None
    _REPORTED_UNFUSED.add(key)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    message = (
        f"DSA indexer scoring falls back to the unfused per-head loop ({context}, "
        f"rank {rank}): {decision.reason}."
    )
    # Deliberately not rank-0-only: with heterogeneous packs a non-zero rank can sit
    # on the unfused cliff while rank 0 stays fused, which is exactly the silent case
    # this exists to surface. The once-per-reason set bounds the volume per process.
    logging.getLogger(__name__).warning(message)
    return message

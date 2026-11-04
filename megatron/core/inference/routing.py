# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stateless engine selection shared by ordinary and disaggregated routing."""

from collections.abc import Sequence

import numpy as np


def select_engine(
    identities: Sequence[bytes],
    loads: np.ndarray,
    *,
    affinity_scores: np.ndarray | None = None,
    available_fractions: np.ndarray | None = None,
    routing_alpha: float = 0.0,
) -> bytes:
    """Select from an ordered pool without mutating its state.

    All arrays align with ``identities``; their order is the final tie-break.
    ``affinity_scores`` contains normalized reusable-work scores, or is None
    when there are no eligible affinity hits. Without hits, prefer available
    capacity (if supplied), then the lowest load. With hits, subtract the
    weighted relative load from affinity and break score ties by lowest load.
    """
    if not identities:
        raise RuntimeError("No engines connected")
    if affinity_scores is None:
        if available_fractions is None:
            return identities[int(np.argmin(loads))]
        scores = available_fractions
    else:
        # Floor the fleet mean at one so near-idle pools do not amplify noise.
        mean_load = float(loads.mean())
        relative_load = (loads - mean_load) / max(1.0, mean_load)
        scores = affinity_scores - routing_alpha * relative_load
    order = np.lexsort((np.arange(len(identities)), loads, -scores))
    return identities[int(order[0])]

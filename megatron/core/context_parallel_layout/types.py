# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Leaf type definitions for context-parallel layout helpers."""

from dataclasses import dataclass
from typing import List, Literal, Optional

import torch

CpPartitionMode = Literal["zigzag", "contiguous"]


@dataclass
class ThdCpRoute:
    """Rank-local route plan for THD zigzag/contiguous CP layout conversion.

    The route stores each layout's local communication view exactly once. A
    directional conversion interprets the source layout fields as send metadata
    and the target layout fields as receive metadata. The same plan describes both
    the CP-only route of a CP rank's packed sequence and the fused TP x CP route of
    one sequence-parallel shard: split sizes are ordered by the rank order of the
    communication group the route was built for (the CP group or the TP x CP group),
    and the caller supplies that group at conversion time. A microbatch carries
    exactly one of the two on its ``PackedSeqParams`` (see
    ``prebuild_thd_cp_partition_routes``).
    """

    zigzag_index: Optional[torch.Tensor]
    zigzag_split_sizes: List[int]
    contiguous_index: Optional[torch.Tensor]
    contiguous_split_sizes: List[int]

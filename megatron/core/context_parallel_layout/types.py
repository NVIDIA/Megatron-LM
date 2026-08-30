# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Leaf type definitions for context-parallel layout helpers."""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch

CpPartitionMode = Literal["zigzag", "contiguous"]


@dataclass
class ThdCpRoute:
    """Rank-local route plan for THD zigzag/contiguous CP layout conversion.

    The route stores each layout's local communication view exactly once. A
    directional conversion interprets the source layout fields as send metadata
    and the target layout fields as receive metadata.
    """

    zigzag_index: Optional[torch.Tensor]
    zigzag_split_sizes: List[int]
    contiguous_index: Optional[torch.Tensor]
    contiguous_split_sizes: List[int]
    cu_seqlens: Tuple[int, ...] = ()
    cp_size: int = 0
    cp_rank: int = 0
    source_cu_seqlens_id: int = 0

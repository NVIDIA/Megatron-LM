# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from megatron.core.ssm.gdn_layer_config import GDNLayerConfig


@dataclass(kw_only=True)
class KDALayerConfig(GDNLayerConfig):
    """Configuration for a KDA (Kimi Delta Attention) layer in a hybrid stack."""

    linear_num_value_heads: Optional[int] = 16
    """KDA uses equal key and value head counts by default."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_kda()

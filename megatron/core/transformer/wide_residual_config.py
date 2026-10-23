# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass


@dataclass
class WideResidualConfig:
    """Configuration for streamwise wide residuals around ordinary-width branches.

    The model carries ``num_streams`` contiguous residual streams, each with
    ``TransformerConfig.hidden_size`` features. Attention, MLP, and MoE branches
    continue to operate at the ordinary hidden size.
    """

    num_streams: int
    """Number of ordinary-width streams carried by the model; must be greater than one."""

    streamwise_sigmoid_init_scale: float = 0.01
    """Symmetric initialization spread for streamwise write logits."""

    learned_retention: bool = False
    """Apply one bounded learned carry factor to each residual stream."""

    retention_init: float = 0.999
    """Initial retention factor; values near one preserve the initial function."""

    retention_max_forget: float = 0.10
    """Maximum forget rate in ``1 - max_forget * sigmoid(-logit)``."""

    def __post_init__(self) -> None:
        if isinstance(self.num_streams, bool) or not isinstance(self.num_streams, int):
            raise TypeError("wide residual num_streams must be an integer.")
        if self.num_streams <= 1:
            raise ValueError(
                f"wide residual num_streams must be greater than one, got {self.num_streams}."
            )
        if self.streamwise_sigmoid_init_scale < 0.0:
            raise ValueError(
                "streamwise_sigmoid_init_scale must be non-negative, got "
                f"{self.streamwise_sigmoid_init_scale}."
            )
        if self.learned_retention:
            if not 0.0 < self.retention_max_forget < 1.0:
                raise ValueError(
                    f"retention_max_forget must be in (0, 1), got {self.retention_max_forget}."
                )
            if not 1.0 - self.retention_max_forget < self.retention_init < 1.0:
                raise ValueError(
                    "retention_init must satisfy 1 - retention_max_forget < "
                    f"retention_init < 1, got {self.retention_init}."
                )

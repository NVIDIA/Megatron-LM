# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-Bridge backend configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig, pick_fields


@dataclass
class BridgeConfig:
    """Config for ``BridgeRuntime``.

    The backend is intentionally thin: it lowers Megatron Lite's runtime
    contract into Megatron-Bridge / Megatron-Core objects, while model-specific
    mutations stay in examples as explicit benchmark hooks.
    """

    model_name: str = "auto"

    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    seed: int = 42
    param_offload: bool = False
    optimizer_offload: bool = False
    load_hf_weights: bool = True
    build_optimizer: bool = True

    # When False, the bridge feeds a dense [b=1, s] forward (no THD packing).
    # Used for deterministic layout-matched parity vs models whose Megatron-Core
    # kernel is dense-only (e.g. GatedDeltaNet). Default True keeps THD packing.
    use_thd: bool = True

    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    override_optimizer_config: dict[str, Any] = field(default_factory=dict)

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Bench-only hook. Kept callable so examples can trim model configs without
    # making the runtime know about benchmark profiles.
    bridge_post_init: Any = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BridgeConfig:
        """Construct ``BridgeConfig`` from a flat or nested mapping."""
        if "num_microbatches" in cfg:
            raise ValueError(
                "BridgeConfig does not accept `num_microbatches`; "
                "pass it to Runtime.forward_backward(..., num_microbatches=...) instead."
            )

        parallel_src = cfg.get("parallel")
        parallel_data = (
            pick_fields(ParallelConfig, parallel_src) if isinstance(parallel_src, dict) else {}
        )
        parallel_data.update(pick_fields(ParallelConfig, cfg))
        parallel = ParallelConfig(**parallel_data)

        optimizer_src = cfg.get("optimizer", {})
        lr_src = cfg.get("lr_scheduler", {})
        optimizer_data: dict[str, Any] = {}
        if isinstance(optimizer_src, dict):
            optimizer_data.update(optimizer_src)
        if isinstance(lr_src, dict):
            optimizer_data.update(lr_src)
        optimizer = OptimizerConfig(**pick_fields(OptimizerConfig, optimizer_data))

        skip = {"parallel", "optimizer", "lr_scheduler"}
        return cls(
            **{k: v for k, v in pick_fields(cls, cfg).items() if k not in skip},
            parallel=parallel,
            optimizer=optimizer,
        )


__all__ = ["BridgeConfig"]

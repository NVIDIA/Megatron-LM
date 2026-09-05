"""Megatron Lite backend configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig, pick_fields


@dataclass(slots=True)
class DebugConfig:
    """Lite backend debug flags. Not exposed to end users."""

    param_update: bool = False
    optimizer_state: bool = False
    grad_phases: bool = False
    router_summary: bool = False
    moe_io: bool = False
    attn_io: bool = False


@dataclass
class LiteConfig:
    """Config for MegatronLiteRuntime.

    Runtime-specific training features live in ``impl_cfg`` (a plain dict);
    each model implementation reads the keys it needs through its typed
    ``ImplConfig``.
    """

    # ── identity ──
    model_name: str = "auto"
    impl: str = "lite"
    hf_path: str = ""

    # ── shared runtime knobs ──
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # ── common runtime/model fields ──
    attention_backend_override: str | None = "flash"
    router_aux_loss_coef: float | None = None
    load_hf_weights: bool = True

    # ── impl-specific (each impl reads its own keys) ──
    impl_cfg: dict[str, Any] = field(default_factory=dict)

    # ── debug ──
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ── bench-only hook: mutate model_cfg after build (e.g. expert truncation) ──
    model_config_hook: Any = None

    @classmethod
    def from_dict(cls, hf_path: str, cfg: dict[str, Any]) -> LiteConfig:
        """Construct LiteConfig from a flat dict (legacy / OmegaConf path)."""
        if "num_microbatches" in cfg:
            raise ValueError(
                "LiteConfig no longer accepts `num_microbatches`; "
                "pass it to Runtime.forward_backward(..., num_microbatches=...) instead"
            )
        parallel = ParallelConfig(**pick_fields(ParallelConfig, cfg))

        opt_d = cfg.get("optimizer", {})
        optimizer = (
            OptimizerConfig(**pick_fields(OptimizerConfig, opt_d))
            if isinstance(opt_d, dict)
            else OptimizerConfig()
        )

        # impl_cfg: merge nested dict + top-level overrides
        impl_cfg: dict[str, Any] = {}
        nested = cfg.get("impl_cfg")
        if isinstance(nested, dict):
            impl_cfg.update(nested)
        for k in list(impl_cfg):
            if k in cfg:
                impl_cfg[k] = cfg[k]
        for k in ("recompute", "use_thd", "use_deepep", "precision_aware_opt"):
            if k in cfg and k not in impl_cfg:
                impl_cfg[k] = cfg[k]

        skip = {"parallel", "optimizer", "impl_cfg", "debug"}
        return cls(
            **{k: v for k, v in pick_fields(cls, cfg).items() if k not in skip},
            hf_path=hf_path,
            parallel=parallel,
            optimizer=optimizer,
            impl_cfg=impl_cfg,
        )


__all__ = [
    "LiteConfig",
    "DebugConfig",
]

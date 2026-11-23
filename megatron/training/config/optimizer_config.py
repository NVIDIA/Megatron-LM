# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recipe integration for Core's existing optimizer override mappings."""

from collections.abc import Callable
from dataclasses import fields, replace
from typing import Literal

from megatron.core.optimizer import (
    OptimizerConfig,
    get_mup_config_overrides,
    get_standard_config_overrides,
    merge_config_overrides,
)
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model

OptimizerConfigOverrides = dict[ParamKey, ParamGroupOverride]
OptimizerConfigOverrideHook = Callable[..., OptimizerConfigOverrides]


def build_optimizer_config_overrides(
    optimizer_config: OptimizerConfig,
    model: list[MegatronModule],
    overrides: OptimizerConfigOverrides | OptimizerConfigOverrideHook | None = None,
    *,
    mode: Literal["extend", "replace"] = "extend",
) -> OptimizerConfigOverrides:
    """Resolve a mapping or callable after the actual model has been constructed.

    Callables receive keyword arguments ``optimizer_config`` and ``model`` and return a
    ParamKey-to-ParamGroupOverride dictionary. Dataclass callables need no base class and
    can use the training configuration's existing serialization/target allowlist.

    Extend mode retains standard WD/decoupled-LR rules; replace mode replaces those rules
    explicitly. An empty mapping in extend mode makes no changes. Enabled model features
    (muP and LSQ) are composed in either mode and conflicts raise rather than taking the
    value from the last dictionary. Feature policy comes from the constructed model.
    """
    if mode not in ("extend", "replace"):
        raise ValueError(f"Unknown optimizer override mode: {mode!r}; use 'extend' or 'replace'.")
    custom = (
        overrides(optimizer_config=optimizer_config, model=model)
        if callable(overrides)
        else overrides
    )
    if custom is None and overrides is None:
        custom = {}
    if not isinstance(custom, dict):
        raise TypeError("Optimizer override hooks must return a ParamKey-to-override dictionary.")
    if any(not isinstance(key, ParamKey) for key in custom):
        raise TypeError("Optimizer override dictionary keys must be ParamKey objects.")

    standard = get_standard_config_overrides(optimizer_config) if mode == "extend" else {}
    model_chunks = unwrap_model(model)
    # Core's MIMO entry point delegates to per-module factories and does not consume
    # the ordinary config_overrides argument. Do not silently discard a recipe hook.
    from megatron.core.models.mimo.model.base import MimoModel

    if isinstance(model_chunks[0], MimoModel) and (custom or mode == "replace"):
        raise NotImplementedError("MIMO optimizer overrides must be configured per module.")
    # Plain torch modules have no model-specific policy; retain the standard rules.
    model_config = getattr(model_chunks[0], "config", None)
    mup = (
        get_mup_config_overrides(
            optimizer_config, model_config.mup_width_mult, optimizer_config.optimizer
        )
        if getattr(model_config, "use_mup", False)
        else {}
    )
    # Flextron and other model builders may use a transform config different from cfg.model.
    # The LSQ hook records the actual setting on the model without a process-global patch.
    from megatron.post_training.optimizer import get_lsq_config_overrides

    lsq = [
        get_lsq_config_overrides(getattr(chunk, "_lsq_scale_lr", None), base_lr=optimizer_config.lr)
        for chunk in model_chunks
    ]
    return merge_config_overrides(standard, mup, *lsq, custom)


def optimizer_config_overrides_from_entries(entries: list) -> OptimizerConfigOverrides:
    """Reconstruct a mapping whose dataclass selectors cannot be YAML mapping keys.

    This is an ordinary importable config target. JSON/YAML turn tuples into lists;
    normalize ParamKey collections back to tuples so the selectors remain hashable.
    """
    mappings = []
    for selector, overrides in entries:
        selector = replace(
            selector,
            **{
                field.name: tuple(getattr(selector, field.name))
                for field in fields(selector)
                if isinstance(getattr(selector, field.name), list)
            },
        )
        mappings.append({selector: overrides})
    return merge_config_overrides(*mappings)


def serialize_optimizer_config_overrides(overrides: OptimizerConfigOverrides) -> dict:
    """Encode Core selector keys using the configuration system's existing targets."""
    from megatron.training.config.utils import serialize_dataclass, serialize_value

    return {
        "_target_": f"{__name__}.optimizer_config_overrides_from_entries",
        "entries": [
            [serialize_dataclass(key, encode_callables=True), serialize_value(value)]
            for key, value in overrides.items()
        ],
    }

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Recipe integration for Core's existing optimizer override mappings."""

import math
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from fnmatch import fnmatchcase
from typing import Literal

from megatron.core.optimizer import (
    OptimizerConfig,
    get_mup_config_overrides,
    get_standard_config_overrides,
    merge_config_overrides,
)
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.optimizer_param_scheduler import (
    ParamGroupOverride,
    combine_param_group_overrides,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model

OptimizerConfigOverrides = dict[ParamKey, ParamGroupOverride]
OptimizerConfigOverrideHook = Callable[..., OptimizerConfigOverrides]


def get_lora_plus_lr_overrides(
    optimizer_config: OptimizerConfig, ratio: float
) -> OptimizerConfigOverrides:
    """Build explicit A/B learning-rate bounds, without composing standard rules.

    Keeping both entries preserves the parameter-group structure of existing LoRA+
    callers. New recipe callables may omit A's redundant contribution. ``None`` for
    the base minimum LR is interpreted as zero, matching the legacy Bridge helper.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("LoRA+ ratio must be finite and positive.")
    lr = optimizer_config.lr
    minimum = optimizer_config.min_lr if optimizer_config.min_lr is not None else 0.0
    if lr is None or not math.isfinite(lr) or not 0 <= minimum <= lr:
        raise ValueError("LoRA+ requires finite base LR bounds with 0 <= min_lr <= lr.")
    if not math.isfinite(lr * ratio):
        raise ValueError("LoRA+ scaled learning rate must be finite.")
    return {
        ParamKey(name="*.linear_in.weight"): {"max_lr": lr, "min_lr": minimum},
        ParamKey(name="*.linear_out.weight"): {"max_lr": lr * ratio, "min_lr": minimum * ratio},
    }


@dataclass
class LoRAPlusOptimizerOverrides:
    """Scale adapter B's entire LR schedule relative to A through the recipe hook.

    Adapter A uses the base optimizer LR. Adapter B matches ``*.linear_out.weight``;
    A uses the corresponding ``*.linear_in.weight`` naming convention. This callable
    configures existing adapters; it does not insert or freeze model parameters.
    Standard WD and decoupled-LR rules are composed by the caller in extend mode.
    """

    ratio: float
    """B-to-A learning-rate ratio; must be finite and positive."""

    def __call__(
        self, *, optimizer_config: OptimizerConfig, model: list[MegatronModule]
    ) -> OptimizerConfigOverrides:
        """Return only the contribution that differs from the base optimizer policy."""
        overrides = get_lora_plus_lr_overrides(optimizer_config, self.ratio)
        if any(
            getattr(getattr(chunk, "config", None), "use_mup", False)
            for chunk in unwrap_model(model)
        ):
            raise ValueError("LoRA+ with muP requires an explicit combined optimizer mapping.")
        if self.ratio == 1.0:
            return {}
        key = ParamKey(name="*.linear_out.weight")
        defaults = {"max_lr": optimizer_config.lr, "min_lr": optimizer_config.min_lr}
        values = {name: value for name, value in overrides[key].items() if value != defaults[name]}
        return {key: values} if values else {}


def validate_lora_plus_warmup(
    overrides: OptimizerConfigOverrides | None,
    optimizer_config: OptimizerConfig,
    initial_lr: float,
    *,
    require_adapter_selector: bool = True,
) -> None:
    """Check composed name-based adapter bounds without depending on rank-local parameters.

    MLM can receive arbitrary model policies. By default, only check when a name-only
    selector explicitly targets adapter A or B. Bridge historically checks all named
    policies and can opt into that behavior. Predicates cannot be evaluated from a name.
    """
    if not overrides or initial_lr == 0:
        return
    named = [
        (selector, (selector.name,) if isinstance(selector.name, str) else selector.name, value)
        for selector, value in overrides.items()
        if selector.name
    ]
    if require_adapter_selector and not any(
        pattern.endswith(("linear_in.weight", "linear_out.weight"))
        for selector, patterns, _ in named
        if not (selector.attr or selector.predicate or selector.with_name_predicate)
        for pattern in patterns
    ):
        return
    bounds = []
    for name in ("adapter.linear_in.weight", "adapter.linear_out.weight"):
        matched = [
            value
            for _, patterns, value in named
            if any(fnmatchcase(name, pattern) for pattern in patterns)
        ]
        policy = combine_param_group_overrides(matched)
        bounds.append(
            (
                policy.get("max_lr", optimizer_config.lr),
                policy.get("min_lr", optimizer_config.min_lr),
            )
        )
    if bounds[0] != bounds[1]:
        raise ValueError("LoRA+ with unequal A/B LR bounds requires lr_warmup_init=0.")


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
    combined = merge_config_overrides(standard, mup, *lsq, custom)
    # Validate overlapping selectors before removing no-op fields: explicitly asking
    # for the base LR must still conflict with a matching muP/custom LR rule.
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if param.requires_grad:
                combine_param_group_overrides(
                    [value for selector, value in combined.items() if selector.matches(param, name)]
                )
    defaults = {
        "max_lr": optimizer_config.lr,
        "min_lr": optimizer_config.min_lr,
        "wd_mult": 1.0,
        "lr_mult": 1.0,
        "is_decoupled_lr": False,
        "is_dsa_indexer": False,
    }
    # Core groups by the raw override tuple, while checkpoints use effective values.
    # Canonicalize this new recipe API without changing legacy factory callers' groups.
    normalized = {}
    for selector, values in combined.items():
        values = {
            key: value
            for key, value in values.items()
            if key not in defaults or value != defaults[key]
        }
        if values:
            normalized[selector] = values
    return normalized


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

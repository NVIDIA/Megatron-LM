# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import copy
import logging
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import astuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.optim import SGD as CPUSGD
from torch.optim import AdamW as CPUAdam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD

    USING_PYTORCH_OPTIMIZER = False
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD

        USING_PYTORCH_OPTIMIZER = False
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        # See https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/optimizers/fused_adam.py#L16.
        from torch.optim import SGD
        from torch.optim import AdamW as Adam

        USING_PYTORCH_OPTIMIZER = True

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    _eo_ver = tuple(int(x) for x in _pkg_version('emerging-optimizers').split('.')[:2])
except (ImportError, PackageNotFoundError):
    _eo_ver = (0, 0)

HAVE_EMERGING_OPTIMIZERS = _eo_ver >= (0, 2)

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.scalar_optimizers import Lion

from megatron.core import parallel_state
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer_param_scheduler import (
    ParamGroupOverride,
    canonicalize_optimizer_config_value,
    combine_param_group_overrides,
    param_group_override_to_tuple,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.fsdp_dtensor_checkpoint import get_global_unique_param_name

from ..distributed.param_and_grad_buffer import _ParamAndGradBuffer
from ..transformer.module import MegatronModule
from ..utils import get_model_config, get_pg_rank, get_pg_size, is_te_min_version, log_single_rank
from .distrib_optimizer import DistributedOptimizer
from .emerging_optimizers import (
    _EMERGING_OPTIMIZERS,
    HAVE_EMERGING_OPTIMIZERS,
    _create_emerging_optimizer,
    _get_qkv_split_shapes,
)
from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .layer_wise_optimizer import LayerWiseDistributedOptimizer, is_managed_by_layer_wise_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
    param_group_identifier_keys,
)

# Subclass aliases kept for backward compatibility; all are OptimizerConfig.
from .optimizer_config import (
    AdamOptimizerConfig,
    OptimizerConfig,
    OptimizerInstanceConfig,
    OptimizerOverrideRecipe,
    OptimizerParamGroupTarget,
    ParamKey,
    ParamPredicate,
    ParamWithNamePredicate,
    SGDOptimizerConfig,
)

logger = logging.getLogger(__name__)

try:
    import yaml

    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def get_standard_config_overrides(config: OptimizerConfig) -> Dict[ParamKey, ParamGroupOverride]:
    """Get standard config overrides for the optimizer, handling decoupled LR and common wd skips.

    Args:
        config (OptimizerConfig): optimizer configuration object.

    Returns:
        Dict[ParamKey, ParamGroupOverride]: standard config overrides.
    """
    warnings.warn(
        "get_standard_config_overrides is deprecated and superseded by "
        "OptimizerConfig.overrides_config.",
        DeprecationWarning,
        stacklevel=2,
    )

    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = {}
    # First, figure out how we are going to do wd skipping. The two main approaches are:
    #  1. The classic megatron approach of skipping all len 1 and bias parameters.
    #  2. The Qwen3-Next approach of doing 1, other than qk layernorm parameters.
    if config.apply_wd_to_qk_layernorm:
        shape_1_not_qkln_param = ParamWithNamePredicate(
            name="s1_not_qkln",
            fn=lambda param, name: (len(param.shape) == 1 or name.endswith(".bias"))
            and not ("q_layernorm." in name or "k_layernorm." in name),
        )
        param_wd_mult_key = ParamKey(with_name_predicate=shape_1_not_qkln_param)
    else:
        param_length_1_match = ParamPredicate(
            name="param_len_1", fn=lambda param: len(param.shape) == 1
        )
        param_wd_mult_key = ParamKey(name="*.bias", predicate=param_length_1_match)

    config_overrides[param_wd_mult_key] = ParamGroupOverride(wd_mult=0.0)

    if config.decoupled_lr is not None:
        decoupled_lr_config: ParamGroupOverride = {"max_lr": config.decoupled_lr}
        decoupled_param_key = ParamKey(attr="is_embedding_or_output_parameter")
        if config.decoupled_min_lr is not None:
            decoupled_lr_config["min_lr"] = config.decoupled_min_lr
        config_overrides[decoupled_param_key] = decoupled_lr_config

    return config_overrides


def get_optimizer_overrides_from_config(
    config: OptimizerConfig,
    legacy_config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
) -> Union[Dict[ParamKey, ParamGroupOverride], OptimizerOverrideRecipe]:
    """Build optimizer overrides from ``OptimizerConfig.overrides_config``.

    When ``overrides_config`` is unset, this returns :func:`get_standard_config_overrides`.
    Otherwise, the configured mapping is authoritative. Like the quantization recipe, named
    optimizer instances own named parameter groups, while enabled matchers select a parameter
    group by optimizer alias and group name. Unmatched parameters use the explicit default target.

    Example::

        overrides_config = {
            "optimizers": {
                "adam": {
                    "type": "adam",
                    "kwargs": {},
                    "param_groups": {"default": {"eps": 1.0e-12}},
                },
                "newon_fast": {
                    "type": "newon",
                    "kwargs": {"new_mode": "fast"},
                    "param_groups": {
                        "hidden": {
                            "max_lr": 1.0e-3,
                            "min_lr": 1.0e-5,
                            "wd_mult": 1.0,
                            "new_scale": 0.5,
                        }
                    },
                }
            },
            "default": {"optimizer": "adam", "param_group": "default"},
            "matchers": {
                "fc1": {
                    "type": "glob",
                    "pattern": "*.linear_fc1.weight",
                    "optimizer": "newon_fast",
                    "param_group": "hidden",
                    "enabled": True,
                }
            },
        }

    Args:
        config (OptimizerConfig): Optimizer configuration containing parsed override data.
        legacy_config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]]): Deprecated standard
            layernorm-weight-decay overrides, optionally composed with MuP overrides.

    Returns:
        Union[Dict[ParamKey, ParamGroupOverride], OptimizerOverrideRecipe]: Legacy programmatic
            overrides or the parsed optimizer recipe.
    """

    def as_mapping(value: Any, path: str) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return value
        if hasattr(value, "__dict__"):
            return vars(value)
        raise TypeError(f"{path} must be a mapping, got {type(value).__name__}")

    if config.overrides_config is None:
        if legacy_config_overrides is not None:
            return legacy_config_overrides
        return get_standard_config_overrides(config)

    assert legacy_config_overrides is None, (
        "OptimizerConfig.overrides_config requires standard layernorm-weight-decay and MuP "
        "config overrides to be None."
    )

    if isinstance(config.overrides_config, str):
        if not HAVE_YAML:
            raise ImportError(
                "PyYAML is required to load optimizer overrides from a YAML file. "
                "Install it with `pip install pyyaml`."
            )
        with open(config.overrides_config, "r", encoding="utf-8") as config_file:
            raw_recipe = yaml.load(config_file, Loader=yaml.SafeLoader)
        log_single_rank(
            logger,
            logging.INFO,
            f"Loaded optimizer overrides from path '{config.overrides_config}'.",
        )
    else:
        raw_recipe = config.overrides_config

    recipe = as_mapping(raw_recipe, "overrides_config")
    unknown_recipe_fields = set(recipe) - {"optimizers", "default", "matchers"}
    if unknown_recipe_fields:
        raise ValueError(
            f"overrides_config has unsupported fields: {sorted(unknown_recipe_fields)}"
        )

    optimizers = as_mapping(recipe.get("optimizers", {}), "overrides_config.optimizers")
    if not optimizers:
        raise ValueError("overrides_config.optimizers must not be empty")
    matchers = as_mapping(recipe.get("matchers", {}), "overrides_config.matchers")
    protected_param_group_fields = {
        "params",
        "optimizer",
        "optimizer_instance",
        "optimizer_kwargs",
        "param_group",
        "param_group_kwargs",
        "default_config",
        "is_expert_parallel",
    }
    parsed_optimizers: Dict[str, OptimizerInstanceConfig] = {}

    for optimizer_name, raw_optimizer in optimizers.items():
        optimizer_path = f"overrides_config.optimizers.{optimizer_name}"
        if not isinstance(optimizer_name, str) or not optimizer_name:
            raise TypeError("overrides_config.optimizers keys must be non-empty strings")
        optimizer = as_mapping(raw_optimizer, optimizer_path)
        unknown_optimizer_fields = set(optimizer) - {"type", "kwargs", "param_groups"}
        if unknown_optimizer_fields:
            raise ValueError(
                f"{optimizer_path} has unsupported fields: {sorted(unknown_optimizer_fields)}"
            )
        optimizer_type = optimizer.get("type")
        if not isinstance(optimizer_type, str) or not optimizer_type:
            raise TypeError(f"{optimizer_path}.type must be a non-empty string")
        kwargs = as_mapping(optimizer.get("kwargs", {}), f"{optimizer_path}.kwargs")
        if any(not isinstance(key, str) for key in kwargs):
            raise TypeError(f"{optimizer_path}.kwargs keys must be strings")
        if "params" in kwargs:
            raise ValueError(
                f"{optimizer_path}.kwargs cannot override the protected 'params' argument"
            )
        raw_param_groups = as_mapping(
            optimizer.get("param_groups", {}), f"{optimizer_path}.param_groups"
        )
        if not raw_param_groups:
            raise ValueError(f"{optimizer_path}.param_groups must not be empty")
        parsed_param_groups: Dict[str, Dict[str, Any]] = {}
        for param_group_name, raw_param_group in raw_param_groups.items():
            param_group_path = f"{optimizer_path}.param_groups.{param_group_name}"
            if not isinstance(param_group_name, str) or not param_group_name:
                raise TypeError(f"{optimizer_path}.param_groups keys must be non-empty strings")
            param_group = as_mapping(raw_param_group, param_group_path)
            if any(not isinstance(key, str) for key in param_group):
                raise TypeError(f"{param_group_path} keys must be strings")
            protected_fields = set(param_group) & protected_param_group_fields
            if protected_fields:
                raise ValueError(
                    f"{param_group_path} cannot override protected fields: "
                    f"{sorted(protected_fields)}"
                )
            parsed_param_groups[param_group_name] = copy.deepcopy(dict(param_group))

        parsed_optimizers[optimizer_name] = OptimizerInstanceConfig(
            optimizer_type=optimizer_type,
            kwargs=copy.deepcopy(dict(kwargs)),
            param_groups=parsed_param_groups,
        )

    def parse_target(raw_target: Any, path: str) -> OptimizerParamGroupTarget:
        target = as_mapping(raw_target, path)
        unknown_target_fields = set(target) - {"optimizer", "param_group"}
        if unknown_target_fields:
            raise ValueError(f"{path} has unsupported fields: {sorted(unknown_target_fields)}")
        optimizer_name = target.get("optimizer")
        if not isinstance(optimizer_name, str):
            raise TypeError(f"{path}.optimizer must be a string")
        if optimizer_name not in parsed_optimizers:
            raise ValueError(f"{path}.optimizer references unknown optimizer {optimizer_name!r}")
        param_group_name = target.get("param_group")
        if not isinstance(param_group_name, str):
            raise TypeError(f"{path}.param_group must be a string")
        if param_group_name not in parsed_optimizers[optimizer_name].param_groups:
            raise ValueError(
                f"{path}.param_group references unknown parameter group "
                f"{optimizer_name}.{param_group_name}"
            )
        return OptimizerParamGroupTarget(optimizer_name, param_group_name)

    if "default" not in recipe:
        raise ValueError("overrides_config.default is required")
    default_target = parse_target(recipe["default"], "overrides_config.default")
    parsed_matchers: Dict[ParamKey, OptimizerParamGroupTarget] = {}

    for matcher_name, raw_matcher in matchers.items():
        matcher_path = f"overrides_config.matchers.{matcher_name}"
        if not isinstance(matcher_name, str):
            raise TypeError("overrides_config.matchers keys must be strings")
        matcher = as_mapping(raw_matcher, matcher_path)
        unknown_matcher_fields = set(matcher) - {
            "type",
            "pattern",
            "optimizer",
            "param_group",
            "enabled",
        }
        if unknown_matcher_fields:
            raise ValueError(
                f"{matcher_path} has unsupported fields: {sorted(unknown_matcher_fields)}"
            )
        if not matcher.get("enabled", False):
            continue

        match_type = matcher.get("type")
        if match_type != "glob":
            raise ValueError(f"{matcher_path}.type must be 'glob', got {match_type!r}")
        pattern = matcher.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise TypeError(f"{matcher_path}.pattern must be a non-empty string")
        target = parse_target(
            {"optimizer": matcher.get("optimizer"), "param_group": matcher.get("param_group")},
            matcher_path,
        )

        param_key = ParamKey(name=pattern)
        if param_key in parsed_matchers:
            raise ValueError(f"{matcher_path}.pattern duplicates an earlier enabled matcher")
        parsed_matchers[param_key] = target

    parsed_recipe = OptimizerOverrideRecipe(
        optimizers=parsed_optimizers, matchers=parsed_matchers, default=default_target
    )
    for optimizer_name, optimizer in parsed_recipe.optimizers.items():
        if optimizer.optimizer_type not in _EMERGING_OPTIMIZERS:
            continue
        constructor_only_kwargs = _EMERGING_OPTIMIZERS[
            optimizer.optimizer_type
        ].constructor_only_kwargs
        for param_group_name, param_group in optimizer.param_groups.items():
            invalid_group_kwargs = set(param_group) & constructor_only_kwargs
            if invalid_group_kwargs:
                raise ValueError(
                    "Constructor-only optimizer arguments cannot be set on parameter group "
                    f"{optimizer_name}.{param_group_name}: {sorted(invalid_group_kwargs)}. "
                    f"Move them to overrides_config.optimizers.{optimizer_name}.kwargs or "
                    "define another optimizer alias."
                )
    return parsed_recipe


def get_mup_config_overrides(
    config: OptimizerConfig, mup_width_mult: float, optimizer_type: str = 'adam'
) -> Dict[ParamKey, ParamGroupOverride]:
    """Get MuP config overrides for per-layer LR and Adam epsilon scaling.

    In MuP, optimizer learning rates are adjusted by parameter class to ensure
    stable update scales across model widths and enable hyperparameter transfer.

    MuP optimizer scaling rules (as implemented here):
    - Adam/AdamW:
      - hidden (matrix-like) lr = base_lr / width_mult
      - hidden (matrix-like) eps = base_eps / width_mult
      - vector-like params keep base lr and eps
    - SGD:
      - vector-like lr = base_lr * width_mult
      - hidden (matrix-like) lr keeps base_lr in the current uniform-width setup
      - no eps override is applied
    - Non-Adam optimizers:
      - hidden (matrix-like) lr = base_lr / width_mult
      - no eps override is applied.
      - for Muon optimizers, matrix-like params managed by Muon itself are
        excluded from these Adam-style MuP overrides.

    With decoupled_lr enabled, embedding/output params continue using decoupled LR
    and MuP will not override those explicit decoupled values.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        mup_width_mult (float): Width multiplier (hidden_size / base_hidden_size).
        optimizer_type (str): Optimizer type string from config.optimizer.

    Returns:
        Dict[ParamKey, ParamGroupOverride]: MuP optimizer overrides.
    """
    warnings.warn(
        "get_mup_config_overrides is deprecated and superseded by "
        "OptimizerConfig.overrides_config.",
        DeprecationWarning,
        stacklevel=2,
    )

    optimizer_type_lower = optimizer_type.lower()
    is_sgd_optimizer = optimizer_type_lower == 'sgd'
    is_adam_optimizer = 'adam' in optimizer_type_lower
    is_muon_optimizer = 'muon' in optimizer_type_lower

    decoupled_lr_enabled = config.decoupled_lr is not None
    if decoupled_lr_enabled:
        message = (
            "Both decoupled_lr and MuP LR scaling are enabled. decoupled_lr sets an "
            "absolute LR for embedding+output params, and MuP LR scaling will not "
            "override those parameters."
        )
        if is_adam_optimizer:
            message += " MuP Adam epsilon scaling remains applied to hidden matrix-like parameters."
        log_single_rank(logger, logging.WARNING, message)

    if is_muon_optimizer:
        muon_scale_mode = getattr(config, 'muon_scale_mode', 'spectral')
        if muon_scale_mode == 'spectral':
            log_single_rank(
                logger,
                logging.WARNING,
                "Both MuP and muon_scale_mode=spectral are enabled. "
                "Muon-managed matrix parameters will continue using spectral Muon scaling. "
                "Set --muon-scale-mode unit_rms_norm to use unit_rms_norm scaling for "
                "Muon-managed matrices with MuP.",
            )

    if mup_width_mult == 1.0:
        # No scaling needed when width_mult is 1
        return {}

    hidden_lr_mult = 1.0 / mup_width_mult
    base_lr = config.lr
    base_min_lr = config.min_lr

    # Hidden matrix-like layers get scaled LR/eps; vector-like params keep base values.
    # Prefer the explicit parameter attribute set by LanguageModule. Fall back to
    # a conservative name check for older or non-language modules.
    def is_embedding_parameter(param: torch.nn.Parameter, param_name: str) -> bool:
        if getattr(param, 'shared_embedding', False):
            return True
        if hasattr(param, 'is_embedding_parameter'):
            return bool(param.is_embedding_parameter)
        return 'embedding' in param_name.lower()

    def is_vector_like_parameter(param: torch.nn.Parameter, param_name: str) -> bool:
        if is_embedding_parameter(param, param_name):
            return True
        if param.dim() <= 1:
            return True
        return False

    def is_muon_managed_matrix_parameter(param: torch.nn.Parameter, _: str) -> bool:
        if not is_muon_optimizer:
            return False
        return is_managed_by_layer_wise_optimizer(param)

    def should_scale_lr_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if decoupled_lr_enabled and getattr(param, 'is_embedding_or_output_parameter', False):
            return False
        if is_muon_managed_matrix_parameter(param, param_name):
            return False
        return not is_vector_like_parameter(param, param_name)

    def should_scale_vector_like_lr_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if decoupled_lr_enabled and getattr(param, 'is_embedding_or_output_parameter', False):
            return False
        return is_vector_like_parameter(param, param_name)

    def should_scale_eps_with_mup(param: torch.nn.Parameter, param_name: str) -> bool:
        if is_vector_like_parameter(param, param_name):
            return False
        if is_muon_managed_matrix_parameter(param, param_name):
            return False
        # MuP Appendix B.3: eps scales with fan_in when non-negligible.
        # This implementation follows the common denominator form: sqrt(v) + eps.
        return True

    mup_overrides: Dict[ParamKey, ParamGroupOverride] = {}

    if is_sgd_optimizer:
        vector_like_lr_mult = mup_width_mult
        vector_like_lr_override: ParamGroupOverride = {}
        if base_lr is not None:
            vector_like_lr_override["max_lr"] = base_lr * vector_like_lr_mult
        if base_min_lr is not None:
            vector_like_lr_override["min_lr"] = base_min_lr * vector_like_lr_mult

        if vector_like_lr_override:
            vector_like_predicate = ParamWithNamePredicate(
                name="mup_sgd_vector_like_excluding_embedding_output",
                fn=should_scale_vector_like_lr_with_mup,
            )
            mup_overrides[ParamKey(with_name_predicate=vector_like_predicate)] = (
                vector_like_lr_override
            )

        return mup_overrides

    lr_override: ParamGroupOverride = {}
    if base_lr is not None:
        lr_override["max_lr"] = base_lr * hidden_lr_mult
    if base_min_lr is not None:
        lr_override["min_lr"] = base_min_lr * hidden_lr_mult

    eps_override: ParamGroupOverride = {}
    if is_adam_optimizer and config.adam_eps is not None:
        eps_override["eps"] = config.adam_eps * hidden_lr_mult

    if decoupled_lr_enabled:
        if lr_override:
            hidden_predicate = ParamWithNamePredicate(
                name="mup_hidden_only_excluding_embedding_output", fn=should_scale_lr_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_predicate)] = lr_override

        if eps_override:
            hidden_output_predicate = ParamWithNamePredicate(
                name="mup_hidden_only_for_adam_eps", fn=should_scale_eps_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_output_predicate)] = eps_override
    else:
        combined_override: ParamGroupOverride = {}
        combined_override.update(lr_override)
        combined_override.update(eps_override)
        if combined_override:
            hidden_output_predicate = ParamWithNamePredicate(
                name="mup_hidden_and_output", fn=should_scale_eps_with_mup
            )
            mup_overrides[ParamKey(with_name_predicate=hidden_output_predicate)] = combined_override

    return mup_overrides


def _get_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Union[Dict[ParamKey, ParamGroupOverride], OptimizerOverrideRecipe]],
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups from provided optimizer config object.

    Programmatic ``ParamKey`` overrides retain their historical composition behavior. A serialized
    optimizer recipe instead resolves every parameter to exactly one named parameter group. Multiple
    recipe matchers may match only when they select the same target.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer overrides,
            specified on a per-layer basis. NOTE: if you want to skip applying weight decay on bias
            and length 1 parameters, and also do not want to do any other overrides, set this to an
            empty dictionary rather than the default value of None.
    Returns:
        List of parameter groups.
    """

    # Map (canonical pg_overrides, is_expert_parallel) to params and the original override.
    params_map = {}
    param_overrides_map = {}

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            # Get optimizer config overrides for this parameter.
            if isinstance(config_overrides, OptimizerOverrideRecipe):
                matching_targets = {
                    target
                    for param_key, target in config_overrides.matchers.items()
                    if param_key.matches(param, name)
                }
                if len(matching_targets) > 1:
                    formatted_targets = sorted(
                        f"{target.optimizer}.{target.param_group}" for target in matching_targets
                    )
                    raise ValueError(
                        f"Parameter {name!r} matches conflicting optimizer parameter groups: "
                        f"{formatted_targets}"
                    )
                target = (
                    next(iter(matching_targets)) if matching_targets else config_overrides.default
                )
                param_override: ParamGroupOverride | None = ParamGroupOverride(
                    **config_overrides.resolve(target)
                )
            else:
                param_overrides_list: list[ParamGroupOverride] = []
                if config_overrides is not None:
                    for param_key, candidate_override in config_overrides.items():
                        if param_key.matches(param, name):
                            param_overrides_list.append(candidate_override)

                if param_overrides_list:
                    param_override = combine_param_group_overrides(param_overrides_list)
                else:
                    param_override = None

            is_expert_parallel = not getattr(param, 'allreduce', True)

            # Create config_tuple that is hash-able, and has a consistent ordering of the keys.
            param_override_tuple: tuple[tuple[str, Any], ...] | None = (
                param_group_override_to_tuple(param_override)
            )
            key = (param_override_tuple, is_expert_parallel)
            if key not in params_map:
                params_map[key] = []
                param_overrides_map[key] = copy.deepcopy(param_override)
            params_map[key].append(param)

    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    local_param_group_specs = list(param_overrides_map.items())
    gathered_param_group_specs = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered_param_group_specs, local_param_group_specs)
    param_group_specs = {}
    for rank_specs in gathered_param_group_specs:
        for key, param_override in rank_specs:
            param_group_specs.setdefault(key, param_override)

    param_groups = []
    # Sort keys, None first.
    for key in sorted(param_group_specs, key=lambda x: (x[0] is not None, x[0], x[1])):
        param_override_tuple, is_expert_parallel = key
        params = params_map[key] if key in params_map else []
        param_override: ParamGroupOverride = copy.deepcopy(param_group_specs[key] or {})

        # False if param_group_override is None or empty tuple or if we do not modify the
        #  LR schedule.
        #  NOTE: "default_config" is used for logging the learning rate in training.py.
        #   so set to True if we do not modify the learning rate.
        #  if param_group['default_config']:
        #    learning_rate = param_group['lr']
        uses_default_lr_schedule: bool = (not bool(param_override_tuple)) or not any(
            ["lr" in k for k in param_override]
        )

        # TODO: Remove "backwards compatible" fields below eventually.
        default_config: ParamGroupOverride = {
            'wd_mult': 1.0,
            'lr_mult': 1.0,
            'is_decoupled_lr': False,
            # The following two fields may be important to keep even when we remove the
            #   above "backwards compatible" fields.
            "max_lr": config.lr,  # user may override this in param_override
            "min_lr": config.min_lr,  # user may override this in param_override
        }
        assert (
            "params" not in param_override
        ), "'params' should not be in param_override, this is a protected key"
        param_group_kwargs = param_override.get("param_group_kwargs", {})
        protected_fields = set(param_group_kwargs) & (
            set(ParamGroupOverride.__annotations__)
            | set(default_config)
            | {"params", "default_config", "is_expert_parallel"}
        )
        if protected_fields:
            raise ValueError(
                "param_group_kwargs cannot override protected parameter-group fields: "
                f"{sorted(protected_fields)}"
            )
        param_group = {
            'params': params,
            'is_expert_parallel': is_expert_parallel,
            'default_config': uses_default_lr_schedule,
            **default_config,
            **param_override,  # keep **param_override last so that users can override other fields.
            **param_group_kwargs,
        }
        param_groups.append(param_group)

    return param_groups


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    config_overrides: Optional[Union[Dict[ParamKey, ParamGroupOverride], OptimizerOverrideRecipe]],
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer/scheduler
            overrides, specified on the basis of ParamKey matches with each parameter.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    param_groups = _get_param_groups(model_chunks, config, config_overrides)
    param_groups = list(filter(filter_fn, param_groups))
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name)

    return param_groups, buffers


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    distributed_optimizer_instance_id: Optional[int] = 0,
    pg_collection: Optional[ProcessGroupCollection] = None,
    skip_megatron_wrapping: bool = False,
) -> Union[MegatronOptimizer, Tuple[Optional[torch.optim.Optimizer], Optional[Callable]]]:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.
        distributed_optimizer_instance_id (int, optional): Distributed optimizer instance. Defaults
            0.
        skip_megatron_wrapping (bool): if True, return a
            ``(optimizer, init_state_fn)`` tuple of the raw PyTorch optimizer
            without any Megatron wrapping. Useful when the caller
            (e.g. LayerWiseDistributedOptimizer) performs its own wrapping.

    Returns:
        Instance of MegatronOptimizer, or ``(optimizer, init_state_fn)`` when
        *skip_megatron_wrapping=True*.
    """
    # All param_groups passed here must belong to the same optimizer type (adam / sgd).
    # Callers are responsible for splitting by optimizer type before calling this function.

    if skip_megatron_wrapping and config.use_precision_aware_optimizer:
        raise ValueError(
            "skip_megatron_wrapping=True is incompatible with use_precision_aware_optimizer."
        )
    if skip_megatron_wrapping and config.optimizer_cpu_offload:
        raise ValueError("skip_megatron_wrapping=True is incompatible with optimizer_cpu_offload.")

    # When freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions.
    if param_groups:
        if config.optimizer_cpu_offload:
            if torch.__version__ < '2.3.0':
                warnings.warn(
                    "CPU offload is recommended for PyTorch >= 2.3.0, "
                    "untested versions below this may have convergence issues."
                )
            assert (
                config.decoupled_weight_decay
            ), "CPU offloading only supported with decoupled_weight_decay enabled (AdamW mode)."
            gpu_optimizer_cls = Adam if config.optimizer == 'adam' else SGD
            cpu_optimizer_cls = CPUAdam if config.optimizer == 'adam' else CPUSGD
            if config.use_torch_optimizer_for_cpu_offload:
                gpu_optimizer_cls = cpu_optimizer_cls
            if config.optimizer == 'adam':
                gpu_optimizer_cls = Adam
                cpu_optimizer_cls = CPUAdam
                optimizer_defaults = dict(
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_eps,
                    bias_correction=True,
                    fused=True,  # this flag is used to improve the performance of the cpu optimizer
                )
            else:
                gpu_optimizer_cls = SGD
                cpu_optimizer_cls = CPUSGD
                optimizer_defaults = dict(
                    lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum
                )
            optimizer = HybridDeviceOptimizer(
                param_groups,
                offload_fraction=config.optimizer_offload_fraction,
                cpu_optimizer_cls=cpu_optimizer_cls,
                gpu_optimizer_cls=gpu_optimizer_cls,
                overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
                pin_cpu_grads=config.pin_cpu_grads,
                pin_cpu_params=config.pin_cpu_params,
                param_update_in_fp32=True,
                **optimizer_defaults,
            )
            init_state_fn = None
        elif config.optimizer == 'adam':
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
                "capturable": config.optimizer_cuda_graph,
            }

            # set Adam class and weight decay mode depending
            # on source of optimizer (Torch or TE/Apex)
            if USING_PYTORCH_OPTIMIZER:
                adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
            else:
                kwargs["adam_w_mode"] = config.decoupled_weight_decay
                adam_cls = Adam

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )
                # Master weight is managed by MCore when main_params_dtype is fp32. This is
                # because we want to use fp8 primary weight with precision aware optimizer.
                # Otherwise, master weight will be managed by TransformerEngine.
                # Delayed scaling is an exception because casting as well as the computation
                # of the scaling factor can be conducted in the adam kernel.
                if config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                    kwargs.update(
                        {
                            "master_weights": True,
                            "use_decoupled_grad": True,
                            "master_weight_dtype": config.main_params_dtype,
                        }
                    )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": config.store_param_remainders})

            optimizer = adam_cls(**kwargs)

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

        elif config.optimizer == 'lion':
            if not HAVE_EMERGING_OPTIMIZERS:
                raise ImportError(
                    "Lion optimizer requires emerging_optimizers >= 0.2. "
                    "Please install or upgrade it to use --optimizer lion."
                )
            optimizer = Lion(  # pylint: disable=possibly-used-before-assignment
                param_groups,
                lr=config.lr,
                betas=(config.lion_beta1, config.lion_beta2),
                weight_decay=config.weight_decay,
            )

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            opt.state[p]['exp_avg'] = torch.zeros_like(p.data)

        elif config.optimizer == 'sgd':
            optimizer = SGD(
                param_groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                momentum=config.sgd_momentum,
            )
            init_state_fn = None
        else:
            raise Exception('{} optimizer is not supported.'.format(config.optimizer))
    else:
        optimizer = None
        init_state_fn = None

    if skip_megatron_wrapping:
        return optimizer, init_state_fn

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                model_chunks=model_chunks,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
            # This is needed for case where num_distributed_optimizer_instances > 1. In this case,
            # weight gradients are all-reduced across optimizer instances, so each instance has
            # the duplicated weight gradients, need to reduce gradient stats inside each instance.
            setattr(optimizer, 'grad_stats_parallel_group', intra_dist_opt_group)
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
            setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    if pg_collection is None or not hasattr(pg_collection, 'tp'):
        tp_group = parallel_state.get_tensor_model_parallel_group()
    else:
        tp_group = pg_collection.tp
    # TODO(M4): plumb tp_group through optimizer constructors so this setattr disappears.
    setattr(optimizer, 'tp_group', tp_group)

    return optimizer


def check_config_overrides_consistency(
    config: OptimizerConfig,
    config_overrides: Optional[Union[Dict[ParamKey, ParamGroupOverride], OptimizerOverrideRecipe]],
):
    """Check if the config overrides are consistent with the config."""

    # Only fields that cannot vary across chained optimizers remain global consistency checks.
    if config_overrides is not None:
        fields_to_check_for_consistency = [
            'overlap_param_gather_with_optimizer_step',
            'optimizer_cpu_offload',
        ]
        for field_name in fields_to_check_for_consistency:
            base_field = getattr(config, field_name, None)
            all_config_overrides = list(config_overrides.values())
            for config_override in all_config_overrides:
                if field_name in config_override:
                    field = config_override[field_name]
                    if field != base_field:
                        raise ValueError(
                            f"Field {field_name} should not be overriden in a config override."
                        )
    return True


def _get_megatron_emerging_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Union[Dict[ParamKey, Any], OptimizerOverrideRecipe]] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Build a mixed or emerging optimizer for the given model chunks.

    Parameter separation (e.g., linear weights -> Muon, rest -> Adam) is expressed as a
    config_override, the same mechanism used for weight-decay and learning-rate overrides.
    Adam/SGD groups are delegated to _get_megatron_optimizer_based_on_param_groups so they
    go through the exact same code path as the standard optimizer factory.

    When ``config.use_layer_wise_distributed_optimizer`` is True, the underlying optimizers
    are wrapped with :class:`LayerWiseDistributedOptimizer`.
    """
    eopt_name = config.optimizer
    use_layer_wise = config.use_layer_wise_distributed_optimizer
    if config_overrides is None:
        config_overrides = {}
    configured_optimizer_names = {
        override['optimizer'] for override in config_overrides.values() if 'optimizer' in override
    }

    # Handle legacy "dist_*" optimizer names (e.g. "dist_muon" → "muon" + layer-wise).
    if eopt_name.startswith('dist_'):
        bare_name = eopt_name[len('dist_') :]
        warnings.warn(
            f"optimizer='{eopt_name}' is deprecated. "
            f"Use optimizer='{bare_name}' with use_layer_wise_distributed_optimizer=True.",
            DeprecationWarning,
            stacklevel=3,
        )
        eopt_name = bare_name
        use_layer_wise = True

    builtin_optimizer_names = {'adam', 'sgd'}
    selected_optimizer_names = (
        configured_optimizer_names
        if isinstance(config_overrides, OptimizerOverrideRecipe)
        else configured_optimizer_names | {eopt_name}
    )
    if not HAVE_EMERGING_OPTIMIZERS and selected_optimizer_names - builtin_optimizer_names:
        raise ImportError(
            "emerging-optimizers package is required for configured optimizer routing. "
            "Install it with: pip install emerging-optimizers"
        )
    assert not (use_layer_wise and config.overlap_param_gather_with_optimizer_step), (
        "overlap_param_gather_with_optimizer_step is not supported with "
        "use_layer_wise_distributed_optimizer: the emerging-optimizer path does not "
        "split model_chunks into (first, rest) groups, so the per-chunk param-gather "
        "dispatch never fires. Disable one of the two flags."
    )
    supported_optimizer_names = builtin_optimizer_names | set(_EMERGING_OPTIMIZERS)
    unsupported_optimizer_names = selected_optimizer_names - supported_optimizer_names
    if unsupported_optimizer_names:
        raise ValueError(
            f"Unsupported per-parameter optimizers: {sorted(unsupported_optimizer_names)}. "
            f"Supported optimizers: {sorted(supported_optimizer_names)}"
        )
    if config.fp16 and selected_optimizer_names - builtin_optimizer_names:
        raise ValueError('emerging optimizer with fp16 is not supported.')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up mixed optimizer with config {config}')

    # Tag parameters with optimizer-specific attributes (expert_tp, is_qkv).
    for model_chunk in model_chunks:
        qkv_split_shapes = None
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                if qkv_split_shapes is None:
                    qkv_split_shapes = _get_qkv_split_shapes(model_chunk.config)
                if param.shape[0] % sum(qkv_split_shapes) == 0:
                    param.is_qkv = True
                    param.qkv_split_shapes = qkv_split_shapes
                else:
                    log_single_rank(
                        logger,
                        logging.DEBUG,
                        f"Emerging optimizer QKV split skipped for {name}: "
                        f"shape={tuple(param.shape)}, split_shapes={qkv_split_shapes}",
                    )

    # Apply optimizer-specific routing defaults only when the caller has not supplied routing.
    # Once any group selects an optimizer, the configured routing is authoritative and unmatched
    # parameters use its explicit default. Copy registry defaults before rewriting the Muon scalar
    # optimizer so shared registry state remains unchanged.
    if (
        not isinstance(config_overrides, OptimizerOverrideRecipe)
        and not configured_optimizer_names
        and eopt_name in _EMERGING_OPTIMIZERS
    ):
        default_param_overrides = copy.deepcopy(
            _EMERGING_OPTIMIZERS[eopt_name].default_param_overrides
        )
        if eopt_name in ('muon', 'adaptive_muon'):
            for override in default_param_overrides.values():
                if override.get('optimizer') in ('adam', 'lion'):
                    override['optimizer'] = config.muon_scalar_optimizer
        config_overrides.update(default_param_overrides)

    # Build param groups and bucket by configured optimizer instance and expert status. A recipe
    # creates one runtime optimizer per alias; legacy programmatic overrides continue to distinguish
    # constructor-kwargs buckets implicitly.
    all_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    grouped_param_groups = defaultdict(list)
    configured_instance_kwargs = {}
    for group in all_param_groups:
        opt_name = group.get('optimizer', eopt_name)
        optimizer_kwargs = group.get("optimizer_kwargs", {})
        if optimizer_kwargs and opt_name not in _EMERGING_OPTIMIZERS:
            raise ValueError(
                f"optimizer_kwargs is only supported for emerging optimizers, got {opt_name!r}"
            )
        optimizer_instance = group.get("optimizer_instance")
        optimizer_kwargs_key = canonicalize_optimizer_config_value(optimizer_kwargs)
        if optimizer_instance is not None:
            instance_key = ("configured", optimizer_instance)
            previous_descriptor = configured_instance_kwargs.setdefault(
                instance_key, (opt_name, optimizer_kwargs_key)
            )
            if previous_descriptor != (opt_name, optimizer_kwargs_key):
                raise ValueError(
                    f"Optimizer instance {optimizer_instance!r} resolves to inconsistent "
                    "optimizer types or constructor kwargs"
                )
        else:
            instance_key = ("legacy", opt_name, optimizer_kwargs_key)
        is_expert = group['is_expert_parallel'] and not use_layer_wise
        grouped_param_groups[(instance_key, opt_name, is_expert)].append(group)

    # Set up DistOpt process groups + filtered buffers once, only if we'll
    # construct a DistributedOptimizer for non-Muon groups in layer-wise mode.
    # The DistOpt-vs-LayerWise buffer split only happens when DDP was wrapped
    # with ``use_distributed_optimizer=True`` (i.e. the layout-based path); in
    # legacy ping-pong mode all params share one unpadded DDP buffer that
    # DistOpt cannot manage, so we keep non-Muon params inside LayerWise.
    ddp_uses_distributed_optimizer = (
        bool(getattr(model_chunks[0], 'ddp_config', None))
        and model_chunks[0].ddp_config.use_distributed_optimizer
    )
    distopt_process_groups = None
    distopt_per_model_buffers = None
    use_separate_distributed_optimizer = ddp_uses_distributed_optimizer and use_layer_wise
    if use_separate_distributed_optimizer:
        ddp_config = model_chunks[0].ddp_config
        assert ddp_config.num_distributed_optimizer_instances == 1, (
            "Layer-wise + DistributedOptimizer split path does not yet support "
            "num_distributed_optimizer_instances > 1: distributed_optimizer_instance_id "
            "is hardcoded to 0 in this path. Disable use_layer_wise_param_layout to "
            "fall back to the legacy LayerWise ping-pong path."
        )
    if use_separate_distributed_optimizer and any(
        # A separate DistributedOptimizer with byte-level sharding handles any group
        # whose optimizer is not the primary emerging optimizer (stored in ``eopt_name``,
        # e.g., Muon). This includes scalar optimizers like Adam or Lion.
        not (opt_name == eopt_name and opt_name in _EMERGING_OPTIMIZERS)
        for (_, opt_name, _), groups in grouped_param_groups.items()
        if groups
    ):
        # ``setup_process_groups_for_optimizer`` rejects Gloo groups whenever
        # an explicit ``pg_collection`` is supplied, so the only legal value
        # here is False.
        distopt_process_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
            pg_collection, model_chunks, use_gloo_process_groups=False
        )
        # DistOpt should only manage non-LayerWise buffers (those holding
        # embeddings, biases, layernorm, etc.). Filter out the LayerWise
        # shard-aligned buffers that the LayerWiseDistributedOptimizer owns.
        distopt_per_model_buffers = {}
        for model_chunk_idx, model_chunk in enumerate(model_chunks):
            if not hasattr(model_chunk, 'buffers'):
                continue
            non_layer_wise_buffers = [
                buffer
                for buffer in model_chunk.buffers
                if buffer.params
                and not getattr(buffer.params[0], 'is_managed_by_layer_wise_optimizer', False)
            ]
            if non_layer_wise_buffers:
                distopt_per_model_buffers[model_chunk_idx] = non_layer_wise_buffers

    # Build an optimizer for each (optimizer_instance, is_expert) bucket and combine.
    # In layer-wise mode, emerging-optimizer (Muon) groups feed into LayerWise,
    # while non-emerging (Adam) groups are managed by a separate DistributedOptimizer
    # — that is, the LayerWise optimizer only owns Muon-managed matrix parameters,
    # and the rest go through DistOpt's standard byte-level shard machinery.
    results = []
    layer_wise_base_results = []  # (raw_optimizer, init_state_fn) feeding LayerWise.
    for (_, opt_name, is_expert), groups in grouped_param_groups.items():
        if not groups:
            continue

        model_parallel_group = pg_collection.tp_ep_pp if is_expert else pg_collection.mp

        # Non-layer-wise recipes may construct any registered emerging optimizer. Layer-wise
        # mode retains the existing primary-optimizer split: scalar optimizers such as Lion use
        # the DistributedOptimizer fallback instead of joining the Muon layer-wise optimizer.
        use_emerging_factory = opt_name in _EMERGING_OPTIMIZERS and (
            not use_layer_wise or opt_name == eopt_name
        )
        if use_emerging_factory:
            optimizer, init_state_fn = _create_emerging_optimizer(
                config,
                groups,
                opt_name,
                model_chunks,
                pg_collection,
                optimizer_kwargs=groups[0].get("optimizer_kwargs"),
            )
            if use_layer_wise:
                layer_wise_base_results.append((optimizer, init_state_fn))
                continue
            if config.bf16:
                optimizer = Float16OptimizerWithFloat16Params(
                    optimizer, config, None, init_state_fn
                )
            else:
                optimizer = FP32Optimizer(optimizer, config, init_state_fn)
            setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
            if pg_collection is None or not hasattr(pg_collection, 'tp'):
                tp_group = parallel_state.get_tensor_model_parallel_group()
            else:
                tp_group = pg_collection.tp
            setattr(optimizer, 'tp_group', tp_group)
            results.append(optimizer)
            continue
        else:
            fallback_config = copy.copy(config)
            fallback_config.optimizer = opt_name
            if use_separate_distributed_optimizer:
                # Route non-emerging params (adam/lion) through a real DistributedOptimizer
                # (byte-level sharding) instead of stuffing them inside LayerWise.
                for group in groups:
                    assert not group['is_expert_parallel'], (
                        "Non-emerging expert-parallel param groups are not yet "
                        "supported on the layer-wise + DistributedOptimizer "
                        "path: they need a separate DistOpt instance with the "
                        "expert-DP process group, which is not wired up yet. "
                        "Disable use_layer_wise_param_layout to fall back to "
                        "the legacy LayerWise ping-pong path for MoE models."
                    )
                fallback_config.use_distributed_optimizer = True
                result = _get_megatron_optimizer_based_on_param_groups(
                    config=fallback_config,
                    model_chunks=model_chunks,
                    param_groups=groups,
                    per_model_buffers=distopt_per_model_buffers,
                    model_parallel_group=distopt_process_groups['mp_group'],
                    data_parallel_group=distopt_process_groups['intra_dp_cp_group'],
                    data_parallel_group_gloo=distopt_process_groups['intra_dp_cp_group_gloo'],
                    data_parallel_group_idx=get_pg_rank(distopt_process_groups['mp_group']),
                    intra_dist_opt_group=distopt_process_groups['intra_dist_opt_group'],
                    distributed_optimizer_instance_id=0,
                    pg_collection=pg_collection,
                    skip_megatron_wrapping=False,
                )
                # TODO(deyuf): ChainedOptimizer currently asserts all sub-optimizers
                # share the same config. Reset to the top-level config so the
                # assertion holds when DistOpt+LayerWise are chained.
                if hasattr(result, 'config'):
                    result.config = config
                results.append(result)
            else:
                # Legacy ping-pong layer-wise path (use_layer_wise=True) or the
                # non-layer-wise standard chain: keep ``use_distributed_optimizer``
                # off; in layer-wise mode the raw torch optimizer (returned as a
                # ``(optimizer, init_state_fn)`` tuple via ``skip_megatron_wrapping``)
                # feeds into ``LayerWiseDistributedOptimizer``.
                fallback_config.use_distributed_optimizer = False
                result = _get_megatron_optimizer_based_on_param_groups(
                    config=fallback_config,
                    model_chunks=model_chunks,
                    param_groups=groups,
                    model_parallel_group=model_parallel_group,
                    pg_collection=pg_collection,
                    skip_megatron_wrapping=use_layer_wise,
                )
                if use_layer_wise:
                    layer_wise_base_results.append(result)
                else:
                    if hasattr(result, 'config'):
                        result.config = config
                    results.append(result)

    if use_layer_wise:
        log_single_rank(
            logger, logging.INFO, f'Using LayerWiseDistributedOptimizer for {eopt_name}'
        )
        base_optimizers, init_fns = (), ()
        if layer_wise_base_results:
            base_optimizers, init_fns = zip(*layer_wise_base_results)
        layer_wise_optimizer = LayerWiseDistributedOptimizer(
            list(base_optimizers),
            config,
            pg_collection,
            init_state_fn_list=list(init_fns),
            model_chunks=model_chunks,
        )
        # LayerWise owns Muon-managed params; DistOpt instances in ``results``
        # own the rest. Chain them so the training loop sees one optimizer.
        if results:
            return ChainedOptimizer([layer_wise_optimizer] + results)
        return layer_wise_optimizer

    return ChainedOptimizer(results)


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
    dump_param_to_param_group_map: Optional[str] = None,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    Handles both standard optimizers (Adam, SGD) and emerging optimizers (e.g. Muon).
    We use separate optimizers for expert parameters and non-expert parameters.
    For emerging optimizers with ``config.use_layer_wise_distributed_optimizer=True``,
    the optimizer is automatically wrapped with :class:`LayerWiseDistributedOptimizer`.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        config_overrides (Optional[Dict[ParamKey, OptimizerConfig]]): optional dictionary of
            optimizer configuration objects to override default optimizer behavior for different
            subsets of parameters (identified by ParamKey).
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        pg_collection: Optional unified process group for distributed training.
        dump_param_to_param_group_map (Optional[str]): path to dump parameter to param group map.

    Returns:
        Instance of MegatronOptimizer.
    """

    # A MimoModel routes to the heterogeneous per-module optimizer builder.
    from megatron.core.models.mimo.model.base import MimoModel

    if isinstance(model_chunks[0], MimoModel):
        from megatron.core.models.mimo.optimizer import get_mimo_optimizer

        assert (
            len(model_chunks) == 1
        ), "MimoModel does not support virtual pipeline parallelism (multiple model chunks)"
        return get_mimo_optimizer(model_chunks[0], config)

    config_overrides = get_optimizer_overrides_from_config(
        config, legacy_config_overrides=config_overrides
    )

    check_config_overrides_consistency(config, config_overrides)

    # TODO: the standard and emerging optimizer paths handle pg_collection differently;
    # unify them so both use a single pg_collection-based flow.
    has_mixed_optimizer_routing = any(
        override.get('optimizer', config.optimizer) != config.optimizer
        for override in config_overrides.values()
    )
    if (
        isinstance(config_overrides, OptimizerOverrideRecipe)
        or config.optimizer not in ('adam', 'sgd')
        or has_mixed_optimizer_routing
    ):
        return _get_megatron_emerging_optimizer(
            config=config,
            model_chunks=model_chunks,
            config_overrides=config_overrides,
            pg_collection=pg_collection,
        )

    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    # Separate out first model chunk if overlapping param AG with optimizer step.
    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]

    # Setup process groups using helper method
    process_groups_dict = ProcessGroupCollection.setup_process_groups_for_optimizer(
        pg_collection, model_chunks, use_gloo_process_groups
    )

    dp_cp_group = process_groups_dict['dp_cp_group']
    intra_dp_cp_group = process_groups_dict['intra_dp_cp_group']
    intra_expt_dp_group = process_groups_dict['intra_expt_dp_group']
    mp_group = process_groups_dict['mp_group']
    expt_tp_pp_group = process_groups_dict['expt_tp_pp_group']
    intra_dp_cp_group_gloo = process_groups_dict['intra_dp_cp_group_gloo']
    intra_expt_dp_group_gloo = process_groups_dict['intra_expt_dp_group_gloo']
    intra_dist_opt_group = process_groups_dict['intra_dist_opt_group']

    model_parallel_rank = get_pg_rank(mp_group)

    if get_pg_size(dp_cp_group) > get_pg_size(intra_dp_cp_group):
        inter_dist_opt_group = process_groups_dict['inter_dist_opt_group']
        distributed_optimizer_instance_id = get_pg_rank(inter_dist_opt_group)
    else:
        distributed_optimizer_instance_id = 0

    optimizers = []
    model_chunk_offset = 0
    ddp_config = model_chunks[0].ddp_config  # Use the first model chunk's DDP config
    if ddp_config.use_megatron_fsdp:
        # For no_shard, gradients are replicated across DP ranks after all-reduce, so grad stats
        # should only be reduced over TP/PP (model_parallel_group) to avoid inflating the norm.
        effective_intra_dist_opt_group = (
            mp_group
            if ddp_config.data_parallel_sharding_strategy == 'no_shard'
            else intra_dist_opt_group
        )
        for model_chunk, overlap_param_gather_with_optimizer_step in zip(
            all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
        ):
            param_groups, buffers = _get_param_groups_and_buffers(
                model_chunk,
                model_chunk_offset=model_chunk_offset,
                config=config,
                config_overrides=config_overrides,
                filter_fn=lambda g: True,
                buffer_name='buffers',
            )

            optimizer_part = _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunk,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=effective_intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
            if (
                not USING_PYTORCH_OPTIMIZER
                and config.use_precision_aware_optimizer
                and getattr(optimizer_part.optimizer, "master_weights", None) is not None
            ):
                # NOTE(@cspades): FusedAdam is provided Megatron-FSDP's main weights as
                # non-quantized DTensor(s). Megatron-FSDP should NEVER use FusedAdam's
                # main weights, complete waste of memory as the optimizer step is still
                # applied to the Megatron-FSDP main weight and extended to FusedAdam
                # main weights. Override this here.
                setattr(optimizer_part.optimizer, "master_weights", False)
                # Megatron-FSDP always uses a decoupled gradient when using FusedAdam.
                setattr(optimizer_part.optimizer, "use_decoupled_grad", True)

            optimizers.append(optimizer_part)
            model_chunk_offset += 1

        if len(optimizers) == 1:
            return optimizers[0]

        return ChainedOptimizer(optimizers)

    if dump_param_to_param_group_map is not None:
        param_to_param_group = {}
        param_group_id = 0
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            config_overrides=config_overrides,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
        )
        for model_chunk in dense_model_chunks:
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )
        if dump_param_to_param_group_map is not None:
            for param_group in param_groups:
                for param in param_group["params"]:
                    param_name = get_global_unique_param_name(model_chunks, param)
                    param_to_param_group[param_name] = param_group_id
                param_group_id += 1

        # Pass Gloo process groups into optimizer only if needed.
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=intra_dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )
        model_chunk_offset += 1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )
    if dump_param_to_param_group_map is not None:
        for param_group in moe_param_groups:
            for param in param_group["params"]:
                param_name = get_global_unique_param_name(model_chunks, param)
                param_to_param_group[param_name] = param_group_id
            param_group_id += 1
    if len(moe_param_groups) > 0:
        expt_model_parallel_rank = get_pg_rank(expt_tp_pp_group)
        # Pass Gloo process groups into optimizer only if needed.
        if use_gloo_process_groups:
            expt_data_parallel_group_gloo = intra_expt_dp_group_gloo
        else:
            expt_data_parallel_group_gloo = None
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=expt_tp_pp_group,
                data_parallel_group=intra_expt_dp_group,
                data_parallel_group_gloo=expt_data_parallel_group_gloo,
                data_parallel_group_idx=expt_model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )

    if dump_param_to_param_group_map is not None:
        torch.distributed.checkpoint.save(
            state_dict=param_to_param_group, checkpoint_id=dump_param_to_param_group_map
        )

    return ChainedOptimizer(optimizers)

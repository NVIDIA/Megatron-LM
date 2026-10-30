# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import functools
import importlib
import inspect
import logging
from functools import lru_cache
from dataclasses import is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec

logger = logging.getLogger(__name__)

# Runtime services attached after construction are not checkpoint configuration.
_TRANSFORMER_RUNTIME_FIELDS = frozenset(
    {
        "timers", "finalize_model_grads_func", "grad_scale_func", "moe_grad_scale_func",
        "mtp_grad_scale_func", "no_sync_func", "grad_sync_func", "param_sync_func",
    }
)


def serialize_dataclass(
    config: Any, *, skip_fields: frozenset[str] = frozenset(), encode_callables: bool = False
) -> dict[str, Any]:
    """Serialize dataclass fields recursively, with an importable class target.

    Args:
        config: Dataclass instance to serialize.
        skip_fields: Runtime-only field names to omit throughout this subtree.
        encode_callables: Encode importable callables for dictionary round trips.
            Unresolvable callable fields are omitted. Otherwise callables pass
            through unchanged for the existing YAML representers.

    Returns:
        The dataclass fields and their class's ``_target_``.
    """
    if isinstance(config, TransformerConfig):
        skip_fields = skip_fields | _TRANSFORMER_RUNTIME_FIELDS
        encode_callables = True
    if isinstance(config, ModuleSpec):
        # Specs contain partials whose keyword arguments can themselves be dataclasses.
        encode_callables = True
    convert = functools.partial(
        serialize_value, skip_fields=skip_fields, encode_callables=encode_callables
    )
    result = {"_target_": f"{config.__class__.__module__}.{config.__class__.__qualname__}"}
    for field in dataclass_fields(config):
        if field.name.startswith("_") or field.name in skip_fields:
            continue
        value = getattr(config, field.name)
        if encode_callables and callable(value) and (
            isinstance(value, type) or not is_dataclass(value)
        ):
            encoded = _encode_callable(value, convert)
            if encoded is not None:
                result[field.name] = encoded
        else:
            result[field.name] = convert(value)
    return result


def serialize_value(
    value: Any, *, skip_fields: frozenset[str] = frozenset(), encode_callables: bool = False
) -> Any:
    """Recursively serialize configs, dataclasses, collections, and model-spec inputs."""
    # Imported lazily: the container and model configs also use these helpers.
    from megatron.training.config.container import ConfigContainerBase
    from megatron.training.models.base import Serializable

    convert = functools.partial(
        serialize_value, skip_fields=skip_fields, encode_callables=encode_callables
    )
    if not isinstance(value, type):
        if isinstance(value, ConfigContainerBase):
            return value.to_dict()
        if isinstance(value, Serializable):
            return value.as_dict()
        if hasattr(value, "to_cfg_dict"):
            return value.to_cfg_dict()
        if is_dataclass(value):
            return serialize_dataclass(
                value, skip_fields=skip_fields, encode_callables=encode_callables
            )
    if encode_callables and callable(value):
        encoded = _encode_callable(value, convert)
        return value if encoded is None else encoded
    if isinstance(value, (list, tuple)):
        return [convert(item) for item in value]
    if isinstance(value, dict):
        return {key: convert(item) for key, item in value.items()}
    return value


def _encode_callable(value, convert):
    """Encode only callables whose import paths resolve back to the original object."""
    if isinstance(value, functools.partial):
        inner = _encode_callable(value.func, convert)
        if inner is None:
            return None
        encoded = {
            "_target_": inner["_target_"],
            "_partial_": True,
            "_args_": [convert(arg) for arg in value.args],
        }
        encoded.update({key: convert(arg) for key, arg in value.keywords.items()})
        return encoded
    module = inspect.getmodule(value)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None:
        return None
    resolved = module
    for part in qualname.split("."):
        resolved = getattr(resolved, part, None)
    if resolved != value:
        return None
    return {"_target_": f"{module.__name__}.{qualname}", "_call_": False}


def sanitize_dataclass_config(config: dict[str, Any], _visited: set | None = None) -> dict[str, Any]:
    """Remove init=False fields from a dataclass config dict for backward compatibility.

    This function automatically detects fields with init=False by inspecting the
    target class specified in the config's _target_ field. This handles cases where
    older checkpoints serialized computed fields that should not be passed to __init__.

    The function recursively processes nested dicts that may also be dataclass configs.

    Args:
        config: A configuration dictionary, potentially with a _target_ field.
        _visited: Internal set to track visited objects and prevent infinite recursion.

    Returns:
        The sanitized configuration with init=False fields removed.
    """
    if not isinstance(config, dict):
        return config

    if _visited is None:
        _visited = set()
    config_id = id(config)
    if config_id in _visited:
        return config
    _visited.add(config_id)

    target = config.get("_target_")
    init_false_fields: frozenset[str] = frozenset()

    if isinstance(target, str):
        target_class = _resolve_target_class(target)
        if target_class is not None:
            init_false_fields = _get_init_false_fields(target_class)

    # Process all values, filtering init=False fields and recursing into nested dicts
    sanitized = {}
    for key, value in config.items():
        if key in init_false_fields:
            if target_class is not None:
                logger.debug(
                    f"Removing init=False field '{key}' from {target_class.__name__} config for backward compatibility"
                )
            continue

        # Recursively sanitize nested dicts (which may be nested dataclass configs)
        if isinstance(value, dict):
            value = sanitize_dataclass_config(value, _visited)
        elif isinstance(value, list):
            value = [sanitize_dataclass_config(item, _visited) if isinstance(item, dict) else item for item in value]

        sanitized[key] = value

    return sanitized


def _resolve_target_class(target: str) -> type | None:
    """Resolve a _target_ string to a class.

    Args:
        target: A fully qualified class path (e.g., "module.submodule.ClassName").

    Returns:
        The resolved class, or None if resolution fails.
    """
    from megatron.training.config.instantiate_utils import target_allowlist

    if not target_allowlist.is_allowed(target):
        logger.warning(f"Target '{target}' is not in the allowlist. Skipping resolution.")
        return None

    try:
        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except (ValueError, ImportError, AttributeError) as e:
        logger.warning(f"Could not resolve target '{target}': {e}")
        return None


@lru_cache(maxsize=128)
def _get_init_false_fields(target_class: type) -> frozenset[str]:
    """Get the set of field names with init=False for a dataclass.

    Args:
        target_class: A dataclass type to inspect.

    Returns:
        A frozenset of field names that have init=False.
    """
    if not is_dataclass(target_class):
        return frozenset()

    return frozenset(f.name for f in dataclass_fields(target_class) if not f.init)

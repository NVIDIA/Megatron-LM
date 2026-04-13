# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import importlib
import logging
from functools import lru_cache
from dataclasses import is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

logger = logging.getLogger(__name__)


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

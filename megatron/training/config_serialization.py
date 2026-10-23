# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Serialization shared by model and transformer configurations."""

import functools
import inspect
from dataclasses import fields, is_dataclass
from typing import Any, Callable

# Hooks and callbacks that each run attaches to the live config. Not to be serialized.
_RUNTIME_FIELDS = frozenset(
    {
        "pre_wrap_hooks",
        "post_wrap_hooks",
        "timers",
        "finalize_model_grads_func",
        "grad_scale_func",
        "moe_grad_scale_func",
        "mtp_grad_scale_func",
        "no_sync_func",
        "grad_sync_func",
        "param_sync_func",
    }
)


def _encode_callable(value: Callable) -> dict[str, Any] | None:
    """Serialize a callable so that it can be deserialized with `instantiate()`."""
    if isinstance(value, functools.partial):
        inner = _encode_callable(value.func)
        if inner is None:
            return None
        encoded: dict[str, Any] = {
            "_target_": inner["_target_"],
            "_partial_": True,
            "_args_": [_value_to_dict(arg) for arg in value.args],
        }
        encoded.update({key: _value_to_dict(arg) for key, arg in value.keywords.items()})
        return encoded

    module = inspect.getmodule(value)
    qualname = getattr(value, "__qualname__", None)
    if module is None or qualname is None:
        return None
    # Encode only paths that resolve back to `value`; a bound method's path drops its instance.
    resolved = module
    for part in qualname.split("."):
        resolved = getattr(resolved, part, None)
    if resolved != value:
        return None
    return {"_target_": f"{module.__name__}.{qualname}", "_call_": False}


def _value_to_dict(value: Any) -> Any:
    """Preserve nested model-spec inputs, including dataclasses in partial kwargs."""
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_model_dataclass(value)
    if callable(value):
        encoded = _encode_callable(value)
        return value if encoded is None else encoded
    if isinstance(value, dict):
        return {key: _value_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value_to_dict(item) for item in value]
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return value


def serialize_model_dataclass(config: Any) -> dict[str, Any]:
    """Encode model/transformer fields, excluding run-local hooks and callbacks."""
    result = {"_target_": f"{config.__class__.__module__}.{config.__class__.__qualname__}"}
    for f in fields(config):
        if f.name.startswith("_") or f.name in _RUNTIME_FIELDS:
            continue
        value = getattr(config, f.name)
        if callable(value) and (isinstance(value, type) or not is_dataclass(value)):
            encoded = _encode_callable(value)
            if encoded is not None:
                result[f.name] = encoded
            continue
        result[f.name] = _value_to_dict(value)
    return result

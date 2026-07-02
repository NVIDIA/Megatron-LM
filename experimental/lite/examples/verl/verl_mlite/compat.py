# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small compatibility patches for dependency-version gaps in examples."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from typing import Any


def _patch_transformers_rope_ignore_keys() -> None:
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return

    for cls in vars(rope_utils).values():
        if not isinstance(cls, type):
            continue
        if getattr(cls, "_verl_mlite_rope_ignore_keys_patch", False):
            continue
        descriptor = vars(cls).get("_check_received_keys")
        if descriptor is None:
            continue

        is_staticmethod = isinstance(descriptor, staticmethod)
        is_classmethod = isinstance(descriptor, classmethod)
        original = descriptor.__func__ if is_staticmethod or is_classmethod else descriptor

        def build_wrapper(check_received_keys: Any) -> Any:
            @wraps(check_received_keys)
            def patched(*args: Any, **kwargs: Any) -> Any:
                ignore_keys = kwargs.get("ignore_keys")
                if isinstance(ignore_keys, list):
                    kwargs["ignore_keys"] = set(ignore_keys)
                elif ignore_keys is not None and not isinstance(ignore_keys, set):
                    if isinstance(ignore_keys, Iterable) and not isinstance(
                        ignore_keys, (str, bytes)
                    ):
                        kwargs["ignore_keys"] = set(ignore_keys)
                return check_received_keys(*args, **kwargs)

            return patched

        patched = build_wrapper(original)
        if is_staticmethod:
            cls._check_received_keys = staticmethod(patched)
        elif is_classmethod:
            cls._check_received_keys = classmethod(patched)
        else:
            cls._check_received_keys = patched
        cls._verl_mlite_rope_ignore_keys_patch = True


def apply_runtime_patches() -> None:
    _patch_transformers_rope_ignore_keys()


def _load_verl_file(relative_path: str, module_name: str):
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError("No module named 'verl'")

    path = Path(next(iter(spec.submodule_search_locations))) / relative_path
    file_spec = importlib.util.spec_from_file_location(module_name, path)
    if file_spec is None or file_spec.loader is None:
        raise ImportError(f"Unable to load VERL module from {path}")

    module = importlib.util.module_from_spec(file_spec)
    sys.modules[module_name] = module
    file_spec.loader.exec_module(module)
    return module


def load_verl_engine_api():
    # Prefer the canonical package import so the MLite engine registers into the
    # SAME EngineRegistry that verl's trainers resolve against. Loading base.py as
    # a standalone module (below) creates a *duplicate* registry, which silently
    # drops the mlite backend ("Unknown backend: mlite"). The file-load path is
    # only a fallback for environments where verl isn't importable as a package.
    try:
        from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
        from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
    except (ModuleNotFoundError, ImportError):
        base = _load_verl_file("workers/engine/base.py", "_verl_mlite_verl_engine_base")
        utils = _load_verl_file("workers/engine/utils.py", "_verl_mlite_verl_engine_utils")
        BaseEngine = base.BaseEngine
        BaseEngineCtx = base.BaseEngineCtx
        EngineRegistry = base.EngineRegistry
        postprocess_batch_func = utils.postprocess_batch_func
        prepare_micro_batches = utils.prepare_micro_batches

    return BaseEngine, BaseEngineCtx, EngineRegistry, postprocess_batch_func, prepare_micro_batches

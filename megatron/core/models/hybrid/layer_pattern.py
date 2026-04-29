# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Pattern-flattening and recipe-loading helpers for the HybridModel DSL.

The Python DSL composes a model from a (possibly nested) list of
:class:`LayerConfig` instances. :func:`flatten_decoder_pattern` flattens the
*decoder body* (i.e. between the :class:`EmbeddingLayerConfig` and
:class:`CrossEntropyLayerConfig` markers) into a flat list of layer configs.
:func:`load_recipe` resolves a recipe spec — either a dotted Python module
path or a filesystem path, with an optional ``:func`` suffix — and returns
a compiled :class:`CompiledRecipe` ready to feed into :class:`HybridModel`.
"""

import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from megatron.core.models.hybrid.hybrid_model_config import CompiledRecipe, HybridModelConfig
from megatron.core.models.hybrid.layer_configs import (
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    LayerConfig,
    PipelineSplit,
)

# Canonical recipe entry point. When a recipe module / file omits the
# ``:func`` suffix, this function name is preferred as the default recipe.
RECIPE_ENTRY_POINT = "make_recipe"


def flatten_decoder_pattern(pattern: Any) -> List[LayerConfig]:
    """Recursively flatten the decoder portion of a layer pattern.

    The decoder body is everything between the :class:`EmbeddingLayerConfig`
    and :class:`CrossEntropyLayerConfig` markers. Lists and tuples are
    descended into; every leaf must be a :class:`LayerConfig` (Mamba,
    Attention, MoE, MLP, GDN, DSA). Embedding/Loss/PipelineSplit at this
    point are an error — they should have been handled earlier.

    Raises :class:`TypeError` (with the offending index path) on
    non-conforming leaves.
    """
    flat: List[LayerConfig] = []
    _flatten_into(pattern, flat, path=())
    return flat


def _flatten_into(node: Any, out: List[LayerConfig], path: tuple) -> None:
    if isinstance(node, LayerConfig):
        out.append(node)
        return
    if isinstance(node, (EmbeddingLayerConfig, CrossEntropyLayerConfig, PipelineSplit)):
        raise TypeError(
            f"Encountered {type(node).__name__} at path {list(path)} inside the "
            f"decoder body; embedding/loss markers may only appear at the "
            f"start/end of layer_pattern, and PipelineSplit is not yet supported "
            f"in the Python DSL."
        )
    if isinstance(node, (list, tuple)):
        for i, child in enumerate(node):
            _flatten_into(child, out, path + (i,))
        return
    raise TypeError(
        f"layer_pattern leaf at path {list(path)} has unsupported type "
        f"{type(node).__name__!r}; expected a LayerConfig instance or a "
        f"nested list/tuple of LayerConfigs."
    )


# ──────────────────────────────────────────────────────────────────────────
# Recipe loading
# ──────────────────────────────────────────────────────────────────────────


def _split_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split ``module-or-path[:func]`` into ``(module-or-path, func or None)``.

    Special-cases Windows drive letters in file paths (``C:/foo``) so the
    drive-letter colon isn't confused with the function-name separator.
    """
    if ":" not in spec:
        return spec, None
    head, tail = spec.rsplit(":", 1)
    # Windows drive-letter check: ``C:/foo`` or ``C:\foo`` would have a
    # one-char head and a tail that starts with ``/`` or ``\``.
    if len(head) == 1 and tail.startswith(("/", "\\")):
        return spec, None
    return head, tail or None


def _is_file_path(s: str) -> bool:
    """Heuristic: treat ``s`` as a filesystem path when it ends in ``.py`` or
    contains a path separator."""
    return s.endswith(".py") or os.sep in s or "/" in s


def _import_from_spec(spec: str):
    """Import a module from either a dotted path or a filesystem path."""
    if _is_file_path(spec):
        path = Path(spec).expanduser().resolve()
        if not path.is_file():
            raise ImportError(f"--model-recipe file path {spec!r} does not exist or is not a file.")
        # Use a unique synthetic module name so multiple recipes loaded by
        # path don't collide in ``sys.modules``.
        module_name = f"_megatron_recipe_{abs(hash(str(path)))}"
        spec_obj = importlib.util.spec_from_file_location(module_name, path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"could not build import spec for {path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
        return module
    try:
        return importlib.import_module(spec)
    except ImportError as e:
        raise ImportError(
            f"--model-recipe {spec!r} could not be imported. Ensure the module "
            f"path is correct and on PYTHONPATH (or pass a filesystem path "
            f"ending in ``.py``). Underlying error: {e}"
        ) from e


def _resolve_recipe_object(module, func_name: Optional[str], origin: str) -> HybridModelConfig:
    """Find the :class:`HybridModelConfig` to compile.

    Resolution order, given a loaded module:

    1. If ``func_name`` is given (i.e. the user wrote ``--model-recipe foo:bar``),
       call that function and return its result.
    2. Otherwise call :data:`RECIPE_ENTRY_POINT`, the canonical recipe
       convention.
    """
    # 1. Explicit :func selection
    if func_name is not None:
        if not hasattr(module, func_name):
            raise AttributeError(
                f"--model-recipe {origin!r}: module has no attribute " f"{func_name!r}."
            )
        return _call_recipe_function(getattr(module, func_name), origin, source=func_name)

    # 2. Canonical default: a function literally named ``make_recipe``.
    if hasattr(module, RECIPE_ENTRY_POINT):
        return _call_recipe_function(
            getattr(module, RECIPE_ENTRY_POINT), origin, source=RECIPE_ENTRY_POINT
        )
    raise AttributeError(
        f"--model-recipe {origin!r}: no {RECIPE_ENTRY_POINT}() function found. "
        f"Define {RECIPE_ENTRY_POINT}() in the recipe module, or pass an "
        f"explicit function with ``--model-recipe {origin}:<func_name>``."
    )


def _call_recipe_function(fn, origin: str, source: str) -> HybridModelConfig:
    if not callable(fn):
        raise TypeError(f"--model-recipe {origin!r}: {source!r} is not callable.")
    return _ensure_hybrid_model_config(fn(), origin, source=source)


def _ensure_hybrid_model_config(value, origin: str, source: str) -> HybridModelConfig:
    if not isinstance(value, HybridModelConfig):
        raise TypeError(
            f"--model-recipe {origin!r}: {source!r} returned "
            f"{type(value).__name__}, not HybridModelConfig."
        )
    return value


def load_recipe(spec: str) -> CompiledRecipe:
    """Resolve a recipe spec and return a compiled :class:`CompiledRecipe`.

    ``spec`` accepts three forms:

    1. Dotted Python path::

           --model-recipe examples.nemotron3.nano

    2. Dotted path with explicit function selection::

           --model-recipe examples.nemotron3.nano:make_recipe

    3. Filesystem path (anywhere on disk; no PYTHONPATH manipulation needed),
       with optional ``:func`` suffix::

           --model-recipe /home/me/recipes/qwen3_next.py
           --model-recipe /home/me/recipes/qwen3_next.py:my_pretrain_recipe

    When no ``:func`` is given, the loader calls ``make_recipe()``.

    Raises:
        ImportError: if the module / file cannot be loaded.
        AttributeError: if the selected recipe function is missing.
        TypeError: if the resolved value is not a :class:`HybridModelConfig`.
    """
    module_or_path, func_name = _split_spec(spec)
    module = _import_from_spec(module_or_path)
    recipe = _resolve_recipe_object(module, func_name, origin=spec)
    return recipe.compile()

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transformer Engine custom-recipe construction utilities."""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from transformer_engine.common.recipe import CustomRecipe

try:
    from transformer_engine.common import recipe as te_recipe
except ImportError:
    te_recipe = None


_CUSTOM_RECIPE_CACHE_ATTRIBUTE = "_mcore_custom_recipe_cache"
_WARNED_LEGACY_CUSTOM_RECIPE_MODES: set[str] = set()


def resolve_quantizer_factory(dotted_path: str) -> Callable[..., Any]:
    """Resolve a dotted Python import path to a quantizer factory.

    Args:
        dotted_path: Import path in ``package.module.callable`` form.

    Returns:
        The resolved callable.

    Raises:
        ValueError: If the path is malformed or does not resolve to a callable.
    """
    if not isinstance(dotted_path, str) or not dotted_path.strip():
        raise ValueError(
            "quantizer_factory must be a non-empty string with format 'package.module.callable'."
        )

    module_path, separator, attribute = dotted_path.rpartition(".")
    if not separator or not module_path or not attribute:
        raise ValueError(
            f"Invalid quantizer_factory '{dotted_path}'. Expected 'package.module.callable'."
        )

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to import module '{module_path}' for quantizer_factory '{dotted_path}': {exc}"
        ) from exc

    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"Attribute '{attribute}' not found in module '{module_path}' for "
            f"quantizer_factory '{dotted_path}'."
        ) from exc
    if not callable(factory):
        raise ValueError(f"Resolved quantizer_factory '{dotted_path}' is not callable.")
    return factory


def build_custom_recipe(
    quantizer_factory: Callable[..., Any], *, fp8_dpa: bool = False, fp8_mha: bool = False
) -> CustomRecipe:
    """Construct a Transformer Engine ``CustomRecipe``.

    Args:
        quantizer_factory: Callable that creates a fresh quantizer for each requested tensor role.
        fp8_dpa: Whether to quantize dot-product attention through Transformer Engine.
        fp8_mha: Whether to quantize multi-head attention through Transformer Engine.

    Returns:
        A Transformer Engine custom recipe.

    Raises:
        ValueError: If Transformer Engine lacks ``CustomRecipe`` or the requested attention flags.
    """
    if te_recipe is None or not hasattr(te_recipe, "CustomRecipe"):
        raise ValueError(
            "CustomRecipe is not available in this Transformer Engine version. "
            "Please use Transformer Engine >= 2.9."
        )
    if not callable(quantizer_factory):
        raise ValueError("quantizer_factory must be callable.")

    recipe_class = te_recipe.CustomRecipe
    recipe_parameters = inspect.signature(recipe_class).parameters
    kwargs: dict[str, Any] = {"qfactory": quantizer_factory}
    for name, enabled in (("fp8_dpa", fp8_dpa), ("fp8_mha", fp8_mha)):
        if name in recipe_parameters:
            kwargs[name] = enabled
        elif enabled:
            raise ValueError(
                f"CustomRecipe with {name}=True requires a Transformer Engine version that "
                f"supports the '{name}' constructor argument."
            )

    return recipe_class(**kwargs)


def get_cached_custom_recipe(
    owner: Any, quantizer_factory_path: str, *, fp8_dpa: bool = False, fp8_mha: bool = False
) -> CustomRecipe:
    """Resolve and materialize one custom recipe per owning configuration.

    Custom factories may be stateful, and TE associates module recipe state with
    the recipe object. Reconstructing the recipe for every layer or forward pass
    would therefore repeatedly invoke the factory and reset that association.
    """
    key = (quantizer_factory_path, fp8_dpa, fp8_mha)
    cache = getattr(owner, _CUSTOM_RECIPE_CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = {}
        setattr(owner, _CUSTOM_RECIPE_CACHE_ATTRIBUTE, cache)
    if key not in cache:
        cache[key] = build_custom_recipe(
            resolve_quantizer_factory(quantizer_factory_path), fp8_dpa=fp8_dpa, fp8_mha=fp8_mha
        )
    return cache[key]


def warn_deprecated_legacy_custom_recipe(mode: str) -> None:
    """Warn once per process for a legacy format-scoped custom recipe."""
    if mode in _WARNED_LEGACY_CUSTOM_RECIPE_MODES:
        return
    _WARNED_LEGACY_CUSTOM_RECIPE_MODES.add(mode)
    warnings.warn(
        f"--{mode}-recipe custom and --{mode}-quantizer-factory are deprecated and will be "
        "removed in a future release. Pass the factory path to --custom-recipe instead.",
        FutureWarning,
        stacklevel=3,
    )

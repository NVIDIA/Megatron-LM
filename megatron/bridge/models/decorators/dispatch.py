# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/decorators/dispatch.py
# Source license: Apache-2.0.

"""Simplified dispatch system for Python, based on classes' typeclass implementation.

This module provides a dispatch-based polymorphism system allowing extensible
behavior for different types using the `impl` decorator.
"""

from functools import _find_impl  # type: ignore
from typing import Any, Callable, Dict, Optional, TypeVar

_SignatureType = TypeVar("_SignatureType", bound=Callable)


class _Dispatch:
    """Internal dispatch representation with type-based routing logic."""

    __slots__ = ("_signature", "_name", "_exact_types", "_dispatch_cache", "_doc", "_module")

    def __init__(self, signature: Callable) -> None:
        self._signature = signature
        self._name = signature.__name__
        self._exact_types: Dict[Any, Callable] = {}
        self._dispatch_cache: Dict[Any, Callable] = {}

        # Extract docstring and module info for rich repr
        self._doc = signature.__doc__
        self._module = signature.__module__

    def __call__(self, instance: Any, *args, **kwargs) -> Any:
        """Dispatch to the appropriate implementation based on instance type."""
        # Special case for tuple-based keys.
        if isinstance(instance, tuple):
            key = tuple(v if isinstance(v, (type, str)) else type(v) for v in instance)

            # Direct match
            impl = self._exact_types.get(key)
            if impl is not None:
                # NOTE: This path is not cached for simplicity
                return impl(instance, *args, **kwargs)

            # Subclass match for tuples of types
            for registered_key, callback in self._exact_types.items():
                if (
                    not isinstance(registered_key, tuple)
                    or len(registered_key) != len(key)
                    or not all(isinstance(t, type) for t in registered_key)
                ):
                    continue

                try:
                    # For subclass checks, operate on the instance types only
                    key_types = tuple(v if isinstance(v, type) else type(v) for v in instance)
                    if all(issubclass(k, rk) for k, rk in zip(key_types, registered_key)):
                        # NOTE: not caching tuple subclass matches for simplicity
                        return callback(instance, *args, **kwargs)
                except TypeError:
                    continue  # issubclass can fail

            # Normalize both sides to names so tuples of types and/or strings can match.
            def _name(obj):
                return obj if isinstance(obj, str) else getattr(obj, "__name__", None) or str(obj)

            key_names = tuple(_name(v) for v in key)
            for registered_key, callback in self._exact_types.items():
                if not isinstance(registered_key, tuple) or len(registered_key) != len(key):
                    continue
                reg_names = tuple(_name(rk) for rk in registered_key)
                if reg_names == key_names:
                    return callback(instance, *args, **kwargs)

            # No implementation found for this tuple, raise a specific error.
            error_msg = self._format_no_implementation_error(instance)
            raise NotImplementedError(error_msg)

        # For class dispatch, we use the class (or string of class name) itself as the key
        if isinstance(instance, type):
            cache_key = instance
            instance_type = instance
        elif isinstance(instance, str):
            cache_key = instance
            instance_type = str
        else:
            cache_key = type(instance)
            instance_type = cache_key

        # Try cache
        impl = self._dispatch_cache.get(cache_key)
        if impl is None:
            impl = self._dispatch(instance, instance_type)
            if impl is None:
                error_msg = self._format_no_implementation_error(instance)
                raise NotImplementedError(error_msg)
            self._dispatch_cache[cache_key] = impl

        return impl(instance, *args, **kwargs)

    def impl(self, *target_types: Any) -> Callable[[Callable], Callable]:
        """Register an implementation for one or more types.

        Usage:
          @mydispatch.impl(int)          # Register for a single type
          @mydispatch.impl(int, str)     # Register for multiple types
          @mydispatch.impl((list, str))  # Register for a tuple of types as a key
        """
        if not target_types:
            raise ValueError(
                "\n✗ Missing argument to .impl()\n\n"
                "You must specify at least one target type.\n\n"
                "Examples:\n"
                f"  @{self._name}.impl(str)  # Single type\n"
                f"  @{self._name}.impl(int, float)  # Multiple types\n"
                f"  @{self._name}.impl((list, str))  # Tuple key\n"
            )

        def decorator(func: Callable) -> Callable:
            if len(target_types) == 1:
                # This handles both `@impl(int)` and `@impl((int, str))`
                self._exact_types[target_types[0]] = func
            else:
                # This handles `@impl(int, str)`
                for typ in target_types:
                    self._exact_types[typ] = func

            self._dispatch_cache.clear()
            return func

        return decorator

    def __repr__(self) -> str:
        """Rich representation showing all implementations."""
        # Build signature string
        import inspect

        sig = inspect.signature(self._signature)
        sig_str = f"{self._name}{sig}"

        lines = [f"Dispatch({sig_str})("]

        # Add regular implementations
        for typ, impl in self._exact_types.items():
            if isinstance(typ, tuple):
                type_name = (
                    f"({', '.join(t.__name__ if hasattr(t, '__name__') else str(t) for t in typ)})"
                )
            else:
                type_name = typ.__name__ if hasattr(typ, "__name__") else str(typ)
            impl_loc = self._format_location(impl)
            lines.append(f"  ({type_name}): {impl.__name__} at {impl_loc}")

        lines.append(")")
        return "\n".join(lines)

    def _dispatch(self, instance: Any, instance_type: type) -> Optional[Callable]:
        """Find the implementation for a given type.

        Fallback order:
          1) Exact type match
          2) issubclass match (when instance is a type)
          3) MRO-based match via functools._find_impl
          4) Name-based fallback: match by class __name__ for dynamically generated
             classes (e.g., HF transformers auto_map dynamic modules)
        """
        # Direct type match
        impl = self._exact_types.get(instance_type, None)
        if impl is not None:
            return impl

        # For class dispatch, check issubclass relationships
        if isinstance(instance, type):
            for registered_type, callback in self._exact_types.items():
                if not isinstance(registered_type, type):
                    continue
                try:
                    if issubclass(instance, registered_type):
                        return callback
                except TypeError:
                    # issubclass can fail for some types
                    pass

        # Use functools._find_impl for MRO-based dispatch, only for single types
        single_type_impls = {k: v for k, v in self._exact_types.items() if isinstance(k, type)}
        impl = _find_impl(instance_type, single_type_impls)
        if impl is not None:
            return impl

        # Name-based fallback for dynamic HF classes and string registrations.
        def _name(obj):
            return obj if isinstance(obj, str) else getattr(obj, "__name__", None)

        if isinstance(instance, str):
            inst_name = instance
        elif isinstance(instance, type):
            inst_name = _name(instance)
        else:
            inst_name = _name(type(instance))

        if inst_name:
            for registered_type, callback in self._exact_types.items():
                reg_name = _name(registered_type)
                if reg_name and str(reg_name) == inst_name:
                    return callback

        return None

    def _format_location(self, func: Callable) -> str:
        """Format the location of a function for display."""
        try:
            import inspect

            filename = inspect.getfile(func)
            _, lineno = inspect.getsourcelines(func)
            # Shorten the path to be more readable
            import os

            filename = os.path.relpath(filename)
            return f"{filename}:{lineno}"
        except Exception:
            return "<unknown location>"

    def _format_no_implementation_error(self, instance: Any) -> str:
        """Format a helpful error message when no implementation is found."""
        type_name_for_header: str
        type_name_for_suggestion: str
        type_name_for_func: str
        instance_type_hint: str

        if isinstance(instance, tuple):
            instance_types = tuple(v if isinstance(v, type) else type(v) for v in instance)
            type_names_str = ", ".join(
                t.__qualname__ if hasattr(t, "__qualname__") else str(t) for t in instance_types
            )
            type_name_for_header = f"tuple of types ({type_names_str})"

            suggestion_names = ", ".join(
                t.__name__ if hasattr(t, "__name__") else str(t) for t in instance_types
            )
            type_name_for_suggestion = f"({suggestion_names})"
            type_name_for_func = "tuple"
            instance_type_hint = f"Tuple[{', '.join(t.__name__ for t in instance_types)}]"
        else:
            instance_type = instance if isinstance(instance, type) else type(instance)
            qualname = (
                instance_type.__qualname__
                if hasattr(instance_type, "__qualname__")
                else str(instance_type)
            )
            type_name_for_header = f"type '{qualname}'"
            type_name_for_suggestion = (
                instance_type.__name__ if hasattr(instance_type, "__name__") else str(instance_type)
            )
            type_name_for_func = type_name_for_suggestion.lower().replace(".", "_")
            instance_type_hint = type_name_for_suggestion

        # Build error message
        lines = [
            f"\n✗ No implementation found for {type_name_for_header}",
            "",
            f"The dispatch function '{self._name}' has no implementation for this type.",
            "",
        ]

        # Add available implementations
        if self._exact_types:
            lines.append("Available implementations:")

            # Add registered types
            sorted_keys = sorted(self._exact_types.keys(), key=str)
            for typ in sorted_keys:
                if isinstance(typ, tuple):
                    type_display = f"({', '.join(t.__name__ if hasattr(t, '__name__') else str(t) for t in typ)})"
                else:
                    type_display = typ.__name__ if hasattr(typ, "__name__") else str(typ)
                lines.append(f"  • {type_display}")
        else:
            lines.append("No implementations registered yet.")

        # Generate help based on existing implementations
        if self._exact_types:
            # Get a sample implementation to show the pattern
            _, sample_impl = next(iter(self._exact_types.items()))

            lines.extend(
                [
                    "",
                    "To add support for this type, register an implementation:",
                    f"  @{self._name}.impl({type_name_for_suggestion})",
                    f"  def _{self._name}_{type_name_for_func}(instance: {instance_type_hint}) -> ...:",
                    "      # Your implementation here",
                ]
            )

            # Try to extract parameter info from the sample implementation
            import inspect

            try:
                sig = inspect.signature(sample_impl)
                params = list(sig.parameters.keys())[1:]  # Skip first param (instance)
                if params:
                    param_hints = ", ".join(params)
                    lines.append(f"      # Expected parameters: {param_hints}")
            except Exception:
                pass
        else:
            lines.extend(
                [
                    "",
                    "To add support for this type:",
                    f"  @{self._name}.impl({type_name_for_suggestion})",
                    f"  def _{self._name}_{type_name_for_func}(instance: {instance_type_hint}, ...) -> ...:",
                    "      # Your implementation here",
                ]
            )

        return "\n".join(lines)


def dispatch(func: _SignatureType) -> _Dispatch:
    """Create a new dispatch function from a signature.

    Args:
        func: Function defining the dispatch signature and default behavior

    Returns:
        A dispatch object that can be extended with implementations

    Example:
        >>> @dispatch
        ... def to_string(instance) -> str:
        ...     '''Convert instance to string representation.'''
        ...
        >>> @to_string.impl(int)
        ... def _to_string_int(instance: int) -> str:
        ...     return str(instance)
        ...
        >>> @to_string.impl(list, tuple)
        ... def _to_string_sequence(instance) -> str:
        ...     return ', '.join(map(str, instance))
        ...
        >>> assert to_string(42) == "42"
        >>> assert to_string([1, 2, 3]) == "1, 2, 3"
    """
    return _Dispatch(func)


__all__ = ["dispatch"]

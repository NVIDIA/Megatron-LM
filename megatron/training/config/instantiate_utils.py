# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import copy
import functools
import inspect
import logging
from enum import Enum
from textwrap import dedent
from typing import Any, Callable, Sequence

from omegaconf import OmegaConf
from omegaconf._utils import is_structured_config


class InstantiationException(Exception):
    """Custom exception type for instantiation errors."""

    ...


class InstantiationMode(Enum):
    """Enum for instantiation modes."""

    STRICT = "strict"
    LENIENT = "lenient"


class _Keys(str, Enum):
    """Special keys in configs used by instantiate."""

    TARGET = "_target_"
    PARTIAL = "_partial_"
    CALL = "_call_"
    ARGS = "_args_"
    NAME = "_name_"


_DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "megatron.training.",
    "megatron.core.",
    "torch.",
    "transformers.",
    "signal.",
)

_DEFAULT_ALLOWED_EXACT: frozenset[str] = frozenset({
    "functools.partial",
})


class TargetAllowlist:
    """Controls which ``_target_`` strings are permitted for instantiation.

    Security: prevents arbitrary code execution from untrusted YAML configs
    by gating which module paths can be imported and called.
    """

    def __init__(self) -> None:
        self._allowed_prefixes: list[str] = list(_DEFAULT_ALLOWED_PREFIXES)
        self._allowed_exact: set[str] = set(_DEFAULT_ALLOWED_EXACT)
        self._enabled: bool = True

    def is_allowed(self, target: str) -> bool:
        """Check whether *target* is permitted by the allowlist."""
        if not self._enabled:
            return True
        if target in self._allowed_exact:
            return True
        return any(target.startswith(prefix) for prefix in self._allowed_prefixes)

    def add_prefix(self, prefix: str) -> None:
        """Add an allowed module prefix (must end with ``'.'``)."""
        if not prefix.endswith("."):
            raise ValueError(f"Prefix must end with '.': got '{prefix}'")
        if prefix not in self._allowed_prefixes:
            self._allowed_prefixes.append(prefix)

    def remove_prefix(self, prefix: str) -> None:
        """Remove an allowed module prefix."""
        self._allowed_prefixes.remove(prefix)

    def add_exact(self, target: str) -> None:
        """Add an exact target string to the allowlist."""
        self._allowed_exact.add(target)

    def remove_exact(self, target: str) -> None:
        """Remove an exact target string from the allowlist."""
        self._allowed_exact.discard(target)

    def disable(self) -> None:
        """Disable the allowlist check (allows all targets)."""
        logging.warning(
            "Target allowlist has been disabled. "
            "Arbitrary _target_ values will be permitted."
        )
        self._enabled = False

    def enable(self) -> None:
        """Re-enable the allowlist check."""
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def allowed_prefixes(self) -> tuple[str, ...]:
        return tuple(self._allowed_prefixes)

    @property
    def allowed_exact(self) -> frozenset[str]:
        return frozenset(self._allowed_exact)


target_allowlist = TargetAllowlist()


def instantiate(
    config: Any,
    *args: Any,
    mode: InstantiationMode = InstantiationMode.LENIENT,
    **kwargs: Any,
) -> Any:
    """Instantiate an object or callable from a config object.

    This function takes a configuration object (dictionary, list, OmegaConf config,
    or Structured Config instance) and instantiates the target specified within it.

    The config object must contain:
        _target_ (str): The fully qualified name of the class or callable to instantiate.

    The config object may also contain:
        _args_ (list): Positional arguments for the target.
        _partial_ (bool): If True, return a functools.partial object instead of calling
                         the target. Defaults to False.
        _call_ (bool): If False, simply resolves and returns the target without calling it.
                       Defaults to True.
        Additional keyword arguments to pass to the target.

    Args:
        config: The configuration object describing the target and its parameters.
        *args: Optional positional arguments that will override _args_ in the config
               if provided.
        mode: Instantiation mode (STRICT or LENIENT). Controls how config keys that
              do not match the target's signature are handled: LENIENT (default)
              drops them with a warning, STRICT raises ``InstantiationException``.
              Errors resolving a ``_target_`` propagate in both modes.
        **kwargs: Optional keyword arguments that will override parameters in the config.
                  Note: Dataclass instances in kwargs are treated as nested configs.

    Returns:
        The instantiated object or the return value of the callable.
        If config._partial_ is True, returns a functools.partial object.
        If config._call_ is False, returns the resolved target callable/class itself.
        Returns None if the input config is None.

    Raises:
        InstantiationException: If the config is invalid, the target cannot be resolved,
                                or instantiation fails in STRICT mode.
        TypeError: If the _partial_ flag is not a boolean.
    """

    # Return None if config is None
    if config is None:
        return None

    if isinstance(config, (dict, list)):
        config = _prepare_input_dict_or_list(config)

    kwargs = _prepare_input_dict_or_list(kwargs)

    # Structured Config always converted first to OmegaConf
    if is_structured_config(config) or isinstance(config, (dict, list)):
        config = OmegaConf.structured(config, flags={"allow_objects": True})

    if OmegaConf.is_dict(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(flags=["allow_objects", "struct", "readonly"], values=[True, False, False])
        config_copy._set_parent(config._get_parent())
        config = config_copy

        if kwargs:
            config = OmegaConf.merge(config, kwargs)

        OmegaConf.resolve(config)

        _partial_ = config.pop(_Keys.PARTIAL, False)

        return instantiate_node(config, *args, partial=_partial_, mode=mode)
    elif OmegaConf.is_list(config):
        # Finalize config (convert targets to strings, merge with kwargs)
        config_copy = copy.deepcopy(config)
        config_copy._set_flag(flags=["allow_objects", "struct", "readonly"], values=[True, False, False])
        config_copy._set_parent(config._get_parent())
        config = config_copy

        OmegaConf.resolve(config)

        _partial_ = kwargs.pop(_Keys.PARTIAL, False)

        if _partial_:
            raise InstantiationException("The _partial_ keyword is not compatible with top-level list instantiation")

        return instantiate_node(config, *args, partial=_partial_, mode=mode)
    else:
        raise InstantiationException(
            dedent(
                f"""\
                Cannot instantiate config of type {type(config).__name__}.
                Top level config must be an OmegaConf DictConfig/ListConfig object,
                a plain dict/list, or a Structured Config class or instance."""
            )
        )


def instantiate_node(
    node: Any,
    *args: Any,
    partial: bool = False,
    mode: InstantiationMode = InstantiationMode.LENIENT,
) -> Any:
    """Recursively instantiates a node within a configuration structure.

    This function handles the instantiation of individual nodes (dictionaries,
    lists, or primitive values) within a larger configuration tree, typically
    managed by OmegaConf.

    If the node is a dictionary containing a `_target_` key, it resolves and
    instantiates the target callable/class using the other items in the
    dictionary as keyword arguments. Nested nodes are recursively instantiated.

    If the node is a list, it recursively instantiates each item in the list.

    If the node is not an OmegaConf config node (e.g., a primitive type), it's
    returned directly.

    Args:
        node: The configuration node to instantiate (can be DictConfig, ListConfig,
              or a primitive type).
        *args: Positional arguments passed down from the top-level `instantiate` call,
               used primarily for the final target call if the node is a dictionary
               with `_target_`.
        partial: Boolean flag indicating whether to return a `functools.partial` object
                 instead of calling the target. This can be overridden by a
                 `_partial_` key within the node itself.
        mode: Instantiation mode (STRICT or LENIENT). Determines error handling
              behavior for nested instantiations.

    Returns:
        The instantiated object, list, or the original node if it wasn't a config.
        Returns None if the input node is None or represents a None value in OmegaConf.

    Raises:
        InstantiationException: If instantiation fails in STRICT mode, or if there are
                                issues like incompatible arguments or non-callable targets.
        TypeError: If a `_partial_` flag within the config is not a boolean.
    """
    # Return None if config is None
    if node is None or (OmegaConf.is_config(node) and node._is_none()):
        return None

    if not OmegaConf.is_config(node):
        return node

    if OmegaConf.is_dict(node):
        partial = node[_Keys.PARTIAL] if _Keys.PARTIAL in node else partial

    full_key = node._get_full_key(None)

    if not isinstance(partial, bool):
        msg = f"Instantiation: _partial_ flag must be a bool, got {type(partial)}"
        if node and full_key:
            msg += f"\nfull_key: {full_key}"
        raise TypeError(msg)

    if OmegaConf.is_list(node):
        items = [instantiate_node(item, mode=mode) for item in node._iter_ex(resolve=True)]

        return items
    elif OmegaConf.is_dict(node):
        exclude_keys = set(item.value for item in _Keys if item != _Keys.ARGS)
        if _is_target(node):
            should_call_target = node.get(_Keys.CALL, True)
            _target_ = _resolve_target(node.get(_Keys.TARGET), full_key, check_callable=should_call_target)
            kwargs = {}
            is_partial = node.get(_Keys.PARTIAL, False) or partial

            if not should_call_target:
                if len(set(node.keys()) - {_Keys.TARGET, _Keys.CALL}) != 0:
                    extra_keys = set(node.keys()) - {_Keys.TARGET, _Keys.CALL}
                    raise InstantiationException(
                        f"_call_ was set to False for target {_convert_target_to_string(_target_)},"
                        f" but extra keys were found: {extra_keys}"
                    )
                else:
                    return _target_

            for key in node.keys():
                if key not in exclude_keys:
                    if OmegaConf.is_missing(node, key) and is_partial:
                        continue
                    value = node[key]
                    value = instantiate_node(value, mode=mode)
                    kwargs[key] = _convert_node(value)

            assert callable(_target_)
            # Drop unexpected kwargs in lenient mode or raise in strict mode
            kwargs = _filter_kwargs_for_target(_target_, kwargs, full_key, mode)
            return _call_target(_target_, partial, args, kwargs, full_key)
        else:
            dict_items = {}
            for key, value in node.items():
                dict_items[key] = instantiate_node(value, mode=mode)
            return dict_items

    else:
        raise InstantiationException(f"Unexpected config type: {type(node).__name__}")


def _locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This function attempts to import modules starting from the most specific path
    (back to front), making it possible to import objects where the final component
    could be either a module or an attribute of the previous module.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(f"Error loading '{path}': invalid dotstring." + "\nRelative imports are not supported.")
    assert len(parts) > 0

    # Try importing from the most specific path first (back to front)
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = import_module(module_path)

            # If this isn't the full path, get the remaining attributes
            remaining_parts = parts[i:]
            for part in remaining_parts:
                try:
                    obj = getattr(obj, part)
                except AttributeError as exc_attr:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_attr)}"
                        + f"\nAre you sure that '{part}' is an attribute of '{module_path}'?"
                    ) from exc_attr

            # Successfully found the object
            return obj

        except ModuleNotFoundError:
            # Module not found, try a less specific path
            continue
        except Exception as exc_import:
            # If we hit a different exception, it's likely an issue with the module itself
            raise ImportError(f"Error loading '{path}':\n{repr(exc_import)}") from exc_import

    # If we've tried all paths and nothing worked, report failure with the base module
    raise ImportError(
        f"Error loading '{path}': Unable to import any module in the path. "
        f"Are you sure that module '{parts[0]}' is installed?"
    )


def _is_target(x: Any) -> bool:
    if isinstance(x, dict):
        return _Keys.TARGET in x
    if OmegaConf.is_dict(x):
        return _Keys.TARGET in x
    return False


def _call_target(
    _target_: Callable[..., Any],
    _partial_: bool,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    full_key: str,
) -> Any:
    """Call target (type) with args and kwargs."""
    args, kwargs = _extract_pos_args(args, kwargs)
    if _partial_:
        try:
            return functools.partial(_target_, *args, **kwargs)
        except Exception as e:
            msg = f"Error in creating partial({_convert_target_to_string(_target_)}, ...) object:" + f"\n{repr(e)}"
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    else:
        try:
            return _target_(*args, **kwargs)
        except Exception as e:
            msg = f"Error in call to target '{_convert_target_to_string(_target_)}':\n{repr(e)}"
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e


def _convert_target_to_string(t: Any) -> Any:
    if callable(t):
        return f"{t.__module__}.{t.__qualname__}"
    else:
        return t


def _filter_kwargs_for_target(
    target: Callable[..., Any] | type,
    kwargs: dict[str, Any],
    full_key: str,
    mode: InstantiationMode,
) -> dict[str, Any]:
    """Drop unexpected keyword arguments for a target and warn.

    If the target accepts ``**kwargs`` we forward everything. Otherwise we
    inspect the signature and remove keys not present as keyword-capable
    parameters, emitting a warning with the dropped keys.
    """
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        # Some builtins or C-extensions may not have an inspectable signature.
        return kwargs

    parameters = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs

    allowed_keys = {
        name
        for name, param in parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    unexpected = set(kwargs.keys()) - allowed_keys
    if _Keys.ARGS in unexpected:
        unexpected.remove(_Keys.ARGS)

    if not unexpected:
        return kwargs

    target_str = _convert_target_to_string(target)
    if mode == InstantiationMode.LENIENT:
        # Warn and drop the unexpected keys
        warning_msg = f"Dropping unexpected config keys for target '{target_str}': {sorted(unexpected)}"
        if full_key:
            warning_msg += f"\nfull_key: {full_key}"
        logging.warning(warning_msg)
        filtered = {k: v for k, v in kwargs.items() if k in allowed_keys}
        if _Keys.ARGS in kwargs:
            filtered[_Keys.ARGS] = kwargs[_Keys.ARGS]
        return filtered
    else:
        msg = f"Unexpected config keys for target '{target_str}': {sorted(unexpected)}"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise InstantiationException(msg)


def _prepare_input_dict_or_list(d: dict[Any, Any] | list[Any]) -> Any:
    res: Any
    if isinstance(d, dict):
        res = {}
        for k, v in d.items():
            if k == _Keys.TARGET:
                v = _convert_target_to_string(d[_Keys.TARGET])
            elif isinstance(v, (dict, list)):
                v = _prepare_input_dict_or_list(v)
            res[k] = v
    elif isinstance(d, list):
        res = []
        for v in d:
            if isinstance(v, (list, dict)):
                v = _prepare_input_dict_or_list(v)
            res.append(v)
    else:
        raise InstantiationException(f"Expected a dict or list, got {type(d).__name__}")
    return res


def _resolve_target(
    target: str | type | Callable[..., Any],
    full_key: str,
    check_callable: bool = True,
) -> type | Callable[..., Any] | object:
    """Resolve target string, type or callable into type or callable."""
    if isinstance(target, str):
        # Security: check allowlist BEFORE importing to prevent
        # arbitrary code execution from untrusted _target_ strings.
        if not target_allowlist.is_allowed(target):
            msg = (
                f"Target '{target}' is not in the allowlist for _target_ instantiation.\n"
                f"Allowed module prefixes: {', '.join(target_allowlist.allowed_prefixes)}\n"
                f"Allowed exact targets: {', '.join(sorted(target_allowlist.allowed_exact))}\n"
                f"To allow this target, call:\n"
                f"  target_allowlist.add_prefix('{target.rsplit('.', 1)[0] + '.'}')\n"
                f"  or: target_allowlist.add_exact('{target}')"
            )
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg)
        try:
            target = _locate(target)
        except Exception as e:
            msg = f"Error locating target '{target}'."
            if full_key:
                msg += f"\nfull_key: {full_key}"
            raise InstantiationException(msg) from e
    if check_callable and not callable(target):
        msg = f"Expected a callable target, got '{target}' of type '{type(target).__name__}'"
        if full_key:
            msg += f"\nfull_key: {full_key}"
        raise InstantiationException(msg)
    return target


def _extract_pos_args(input_args: Any, kwargs: Any) -> tuple[Any, Any]:
    config_args = kwargs.pop(_Keys.ARGS, ())
    output_args = config_args

    if isinstance(config_args, Sequence):
        if len(input_args) > 0:
            output_args = input_args
    else:
        raise InstantiationException(
            f"Unsupported _args_ type: '{type(config_args).__name__}'. value: '{config_args}'"
        )

    return output_args, kwargs


def _convert_node(node: Any) -> Any:
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=True)

    return node

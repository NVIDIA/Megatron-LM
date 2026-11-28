# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Connect module and method calls to their installed activation strategies."""

from __future__ import annotations

import copy
import inspect
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.activation_policy.strategies import Strategy

When = Literal["enter", "exit"]


@dataclass(frozen=True)
class Event:
    """Entering or leaving a region of the same owner.

    An enter event observes positional and keyword inputs. An exit event observes outputs.
    """

    region: Any  # a child module, or a method of the owner
    when: When

    @classmethod
    def enter(cls, region: Any) -> Event:
        """Observe the inputs to a call."""
        return cls(region, "enter")

    @classmethod
    def exit(cls, region: Any) -> Event:
        """Observe the outputs of a call."""
        return cls(region, "exit")


_INSTALLED_ATTR = "_activation_policy"


class _Installed:
    def __init__(self, strategies: Iterable[Strategy]):
        self.strategies = {}
        self.observed = set()
        self.pending: dict[Event, dict[Strategy, Callable]] = {}
        for strategy in strategies:
            region = strategy.region
            if isinstance(region, str):
                raise TypeError(f"A point must be a child module or a method, got {region!r}.")
            if region in self.strategies:
                raise ValueError(f"Multiple activation strategies for region {region!r}.")
            self.strategies[region] = strategy
            self.observed.add(region)
            self.observed.update(strategy.observes())

    def defer(self, event, source, callback):
        """Replace any callback left by the same strategy at this boundary."""
        self.pending.setdefault(event, {})[source] = callback

    def fire(self, region, when, value):
        if not self.pending:
            return
        callbacks = self.pending.pop(Event(region, when), {})
        for callback in callbacks.values():
            callback(value)


def _installed(owner: torch.nn.Module) -> Optional[_Installed]:
    return owner.__dict__.get(_INSTALLED_ATTR)


# ---------------------------------------------------------------------------------------------
# Region execution
# ---------------------------------------------------------------------------------------------


def run_point(owner: torch.nn.Module, point, forward: Callable, args, kwargs):
    """Run a call of ``point`` of ``owner`` through its strategies (called by the point hooks)."""
    installed = _installed(owner)
    if installed is None:
        return forward(*args, **kwargs)
    installed.fire(point, "enter", (args, kwargs))
    strategy = installed.strategies.get(point)
    if strategy is None:
        out = forward(*args, **kwargs)
    else:
        out = strategy.run(owner, installed.defer, forward, args, kwargs)
    installed.fire(point, "exit", out)
    return out


def fire_boundary(owner: torch.nn.Module, point, when: When, value: Any):
    """Notify pending callbacks when an executor bypasses the original call."""
    installed = _installed(owner)
    if installed is not None:
        installed.fire(point, when, value)
    return None if when == "enter" else value


def has_strategy(owner: torch.nn.Module, point, strategy_type: type) -> bool:
    """Whether ``point`` of ``owner`` has a strategy of ``strategy_type`` installed (for tests and
    diagnostics; model code does not depend on which strategies are installed)."""
    return installed_strategy(owner, point, strategy_type) is not None


def install_activation_strategies(
    owner: torch.nn.Module, strategies: Iterable[Strategy] = ()
) -> None:
    """Replace the strategies installed on this owner's modules and methods."""
    installed = _Installed(strategies)
    previous = _installed(owner)
    previous_points = previous.observed if previous is not None else set()
    attached = []
    try:
        for point in installed.observed - previous_points:
            attach_activation_point(owner, point)
            attached.append(point)
    except Exception:
        for point in attached:
            detach_activation_point(owner, point)
        raise

    for point in previous_points - installed.observed:
        detach_activation_point(owner, point)
    owner.__dict__[_INSTALLED_ATTR] = installed


def _uninstall(owner: torch.nn.Module) -> None:
    installed = owner.__dict__.pop(_INSTALLED_ATTR, None)
    if installed is None:
        return
    for point in installed.observed:
        detach_activation_point(owner, point)


def installed_strategies(owner: torch.nn.Module) -> dict:
    """The strategies installed on ``owner``, by region, in the order they apply."""
    installed = _installed(owner)
    return (
        {point: (strategy,) for point, strategy in installed.strategies.items()}
        if installed
        else {}
    )


def installed_strategy(owner: torch.nn.Module, point, strategy_type: type = object):
    """Return the matching strategy on this point, if installed."""
    installed = _installed(owner)
    if installed is None:
        return None
    strategy = installed.strategies.get(point)
    return strategy if isinstance(strategy, strategy_type) else None


def update_strategy(owner: torch.nn.Module, point, strategy_type: type = object, **changes):
    """Replace fields of ``owner``'s own strategy on ``point`` and re-hook the owner (the points
    a strategy observes, such as its ``release_on`` event, may change).

    Returns the updated strategy, or ``None`` when no such strategy is installed (nothing to
    change, for example when the recompute it configures is off).
    """
    strategy = installed_strategy(owner, point, strategy_type)
    if strategy is None:
        return None
    _amend_strategy(owner, strategy, **changes)
    return strategy


def _amend_strategy(owner, strategy, **changes):
    installed = _installed(owner)
    if installed.pending:
        raise RuntimeError("Activation strategies can only be amended between forwards")
    previous = {}
    for field in changes:
        if not hasattr(strategy, field):
            raise AttributeError(f"{type(strategy).__name__} has no field {field!r}.")
        previous[field] = getattr(strategy, field)
    try:
        for field, value in changes.items():
            setattr(strategy, field, value)
        install_activation_strategies(owner, installed.strategies.values())
    except Exception:
        for field, value in previous.items():
            setattr(strategy, field, value)
        raise


class _ActivationPointHook:
    """Base of the per-point hooks: routes a call of the point to the owner's strategies.

    The owner is held weakly: a recomputed call keeps itself alive from autograd hooks (which the
    garbage collector cannot traverse) until backward. If that backward never happens, e.g. for
    a tensor owned by a CUDA graph, a strong reference would pin the whole owner.
    """

    def _dispatch(self, point, forward, args, kwargs):
        owner = self._owner()
        if owner is None:
            return forward(*args, **kwargs)
        return run_point(owner, point, forward, args, kwargs)

    @property
    def owner(self):
        """The owner module (``None`` once it has been freed)."""
        return self._owner()


class _ModulePointHook(_ActivationPointHook):
    """Wrap the module call outside PyTorch's pre- and post-forward hooks."""

    def __init__(self, owner, module: torch.nn.Module):
        self._owner = weakref.ref(owner)
        self.module, self.call = module, module._call_impl
        # An instance-level ``_call_impl`` the module had before (restored on removal).
        self.instance_call = module.__dict__.get("_call_impl")
        self.bypass = False

    def __deepcopy__(self, memo):
        # When the owner is copied along with this child, route the copy through the owner's
        # copy. A child copied on its own gets a plain forward: its copy has no owner.
        owner = self.owner
        owner_copy = memo.get(id(owner)) if owner is not None else None
        if owner_copy is None:
            return copy.deepcopy(self.call, memo)
        clone = _ModulePointHook.__new__(_ModulePointHook)
        memo[id(self)] = clone
        clone._owner = weakref.ref(owner_copy)
        clone.module = copy.deepcopy(self.module, memo)
        clone.call = copy.deepcopy(self.call, memo)
        clone.instance_call = copy.deepcopy(self.instance_call, memo)
        clone.bypass = False
        return clone

    def call_bypassed(self, *args, **kwargs):
        """Call the module through ``__call__`` (its hooks run) without its strategies."""
        if self.bypass:
            return self.module(*args, **kwargs)
        self.bypass = True
        try:
            return self.module(*args, **kwargs)
        finally:
            self.bypass = False

    def __call__(self, *args, **kwargs):
        if self.bypass:
            return self.call(*args, **kwargs)
        return self._dispatch(self.module, self.call, args, kwargs)


class _MethodPointHook(_ActivationPointHook):
    """Instance attribute shadowing a hooked method of its owner (``self.m(...)`` still works).

    It compares and hashes equal to the bound method it shadows, so a point named as ``self.m``
    before or after the hook is attached is the same point.
    """

    def __init__(self, owner, method):
        self._owner = weakref.ref(owner)
        if isinstance(method, _MethodPointHook):  # named after its hook was attached
            self.func, self.name = method.func, method.name
        else:
            self.func, self.name = method.__func__, method.__name__

    def _bound(self):
        return self.func.__get__(self._owner())

    def __eq__(self, other):
        if isinstance(other, _MethodPointHook):
            return self.func is other.func and self._owner() is other._owner()
        if inspect.ismethod(other):
            return other.__func__ is self.func and other.__self__ is self._owner()
        return NotImplemented

    def __hash__(self):
        return hash(self._bound())

    def __deepcopy__(self, memo):
        owner = self.owner
        clone = _MethodPointHook.__new__(_MethodPointHook)
        memo[id(self)] = clone
        owner_copy = memo.get(id(owner)) if owner is not None else None
        clone._owner = weakref.ref(owner_copy) if owner_copy is not None else lambda: None
        clone.func, clone.name = self.func, self.name
        return clone

    def __call__(self, *args, **kwargs):
        if self._owner() is None:
            raise RuntimeError(f"The owner of hooked method {self.name!r} was freed.")
        method = self._bound()
        return self._dispatch(method, method, args, kwargs)


def activation_point_owner(point):
    """Resolve a point's owner from its bound method or installed child-module hook."""
    if isinstance(point, torch.nn.Module):
        point = point.__dict__.get("_call_impl")
    if isinstance(point, _ActivationPointHook):
        return point.owner
    return point.__self__ if inspect.ismethod(point) else None


def is_method_point(point) -> bool:
    """Whether ``point`` is a method (bound, or already hooked) rather than a child module."""
    return inspect.ismethod(point) or isinstance(point, _MethodPointHook)


def call_without_activation_strategies(module: torch.nn.Module, *args, **kwargs):
    """Call ``module`` without the activation strategies installed on it as a point (its own
    module hooks still run).

    Transitional: for an owner path that still checkpoints the point itself (TransformerLayer's
    residual-stream / mHC replay checkpoint), until that path is a strategy too.
    """
    hook = module.__dict__.get("_call_impl")
    if not isinstance(hook, _ModulePointHook):
        return module(*args, **kwargs)
    return hook.call_bypassed(*args, **kwargs)


def attach_activation_point(owner: torch.nn.Module, point) -> None:
    """Hook ``point`` (a child module, or a method of ``owner``) for ``owner``'s strategies."""
    if is_method_point(point):
        bound_owner = point.__self__ if inspect.ismethod(point) else point.owner
        if bound_owner is not owner:
            raise ValueError(
                f"Method point {point.__name__ if inspect.ismethod(point) else point.name!r} "
                f"must be a method of its owner {type(owner).__name__}."
            )
        name = point.__name__ if inspect.ismethod(point) else point.name
        if not isinstance(owner.__dict__.get(name), _MethodPointHook):
            owner.__dict__[name] = _MethodPointHook(owner, point)
        return
    if not isinstance(point, torch.nn.Module):
        raise TypeError(
            f"An activation point is a child module or a method, got {type(point).__name__}."
        )
    if point is owner:
        raise ValueError("A module cannot be a point of itself.")
    hook = point.__dict__.get("_call_impl")
    if isinstance(hook, _ModulePointHook):
        if hook.owner is owner:
            return
        raise RuntimeError(f"Module {type(point).__name__} is already hooked by another owner.")
    point._call_impl = _ModulePointHook(owner, point)


def detach_activation_point(owner: torch.nn.Module, point) -> None:
    """Remove the hook :func:`attach_activation_point` put on ``point`` for ``owner``."""
    if is_method_point(point):
        name = point.__name__ if inspect.ismethod(point) else point.name
        if isinstance(owner.__dict__.get(name), _MethodPointHook):
            del owner.__dict__[name]
        return
    if not isinstance(point, torch.nn.Module):
        return
    hook = point.__dict__.get("_call_impl")
    if isinstance(hook, _ModulePointHook) and hook.owner is owner:
        if hook.instance_call is None:
            del point.__dict__["_call_impl"]
        else:
            point._call_impl = hook.instance_call

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in NVTX ranges around model components (``MCORE_COMPONENT_NVTX=1``).

The forward range is pushed on the calling thread. The backward range covers the
autograd nodes the component created: it opens at the first of their pre-hooks and
closes after the last of their post-hooks, on the autograd thread. Nodes created by a
nested component belong to the nested one only. Ranges are also recorded while CUDA
graphs are captured, so a profiler can attribute replayed graph kernels through the
capture-time node creation events.

The switch is read once, at import. When it is off (the default), ``component`` and
``layer_scope`` return the decorated function itself and ``forward_range`` does nothing,
so the instrumented modules run exactly the code they would run without it. When it is
on, only host-side NVTX calls and autograd hooks that return ``None`` are added; tensors
and kernels are unchanged.
"""

import functools

import torch

from megatron.core.env_flags import env_flag

ENABLED = env_flag("MCORE_COMPONENT_NVTX", False)
PREFIX = "comp:"

_probe = None
# One set of claimed node ids per active component call, innermost last.
_claimed_stack = []
_open = {}


def _sequence_nr():
    """Return the next autograd sequence number on this thread (consumes one)."""
    global _probe
    if _probe is None:
        _probe = torch.zeros((), requires_grad=True)
    return (_probe * 1).grad_fn._sequence_nr()


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


class _BackwardRange:
    def __init__(self, name, nodes) -> None:
        self.name = name
        self.pending = len(nodes)
        self.handle = None
        for node in nodes:
            node.register_prehook(self._pre)
            node.register_hook(self._post)

    def _pre(self, grad_outputs):
        if self.handle is None:
            stale = _open.pop(self.name, None)
            if stale is not None and stale.handle is not None:
                torch.cuda.nvtx.range_end(stale.handle)
                stale.handle = None
            self.handle = torch.cuda.nvtx.range_start(self.name)
            _open[self.name] = self

    def _post(self, grad_inputs, grad_outputs):
        self.pending -= 1
        if self.pending == 0 and self.handle is not None:
            torch.cuda.nvtx.range_end(self.handle)
            self.handle = None
            if _open.get(self.name) is self:
                del _open[self.name]


def _track_backward(name, outputs, first, claimed, transparent=False):
    last = _sequence_nr()
    stack = [t.grad_fn for t in _tensors(outputs) if t.grad_fn is not None]
    seen, nodes = set(), []
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if not first <= node._sequence_nr() < last:
            continue
        if id(node) not in claimed:
            nodes.append(node)
        stack.extend(next_node for next_node, _ in node.next_functions)
    if transparent:
        # A layer scope spans every node of the layer, nested components included, and claims none.
        if nodes:
            _BackwardRange(name, nodes)
        return
    if _claimed_stack:
        # The enclosing call owns neither these nodes nor those of components nested here.
        _claimed_stack[-1].update(claimed)
        _claimed_stack[-1].update(id(node) for node in nodes)
    if nodes:
        _BackwardRange(f"{PREFIX}{name}.bwd", nodes)


def component(name):
    """Decorate a forward-like callable with forward and backward NVTX ranges."""

    def decorate(fn):
        """Return ``fn`` itself when disabled, else ``fn`` wrapped in the ranges."""
        if not ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            track = torch.is_grad_enabled()
            first = _sequence_nr() if track else None
            _claimed_stack.append(set())
            torch.cuda.nvtx.range_push(f"{PREFIX}{name}")
            try:
                outputs = fn(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
                claimed = _claimed_stack.pop()
            if track:
                _track_backward(name, outputs, first, claimed)
            return outputs

        return wrapper

    return decorate


def layer_scope(fn=None, *, layer_arg=False):
    """Decorate a method with "layer:<n>" ranges.

    ``n`` is ``self.layer_number`` or, with ``layer_arg``, the ``layer_number`` of the first
    positional argument, for runners that take the layer.

    Layer ranges do not take part in component attribution: they tag every kernel of the
    layer, including those of nested components, so a profile can be broken down per layer
    and component at once.
    """
    if fn is None:
        return functools.partial(layer_scope, layer_arg=layer_arg)
    if not ENABLED:
        return fn

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        name = f"layer:{(args[0] if layer_arg else self).layer_number}"
        track = torch.is_grad_enabled()
        first = _sequence_nr() if track else None
        torch.cuda.nvtx.range_push(name)
        try:
            outputs = fn(self, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
        if track:
            _track_backward(f"{name}.bwd", outputs, first, set(), transparent=True)
        return outputs

    return wrapper


class forward_range:
    """Forward-only NVTX range for a block inside a component (e.g. an autograd Function)."""

    def __init__(self, name) -> None:
        self.name = f"{PREFIX}{name}"

    def __enter__(self):
        if ENABLED:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *exc):
        if ENABLED:
            torch.cuda.nvtx.range_pop()

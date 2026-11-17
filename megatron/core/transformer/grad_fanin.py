# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gradient fan-in: sum the gradients of a tensor with several consumers in one buffer.

When a tensor feeds several ops, autograd sums their input gradients with one elementwise add
per extra consumer. A :class:`GradFanIn` collects the contributions itself so a consumer whose
input gradient comes out of a GEMM can add it into the running sum in the GEMM epilogue, and a
fused-norm consumer can add the running sum in its norm-backward kernel. Either way the
separate ``[tokens, hidden]`` add disappears.

Protocol between a fan-in and the consumers that join it (also implemented by Transformer
Engine modules that advertise ``supports_grad_fanin``):

* In forward, every consumer that joins calls :meth:`GradFanIn.add_consumer` once. A
  consumer that does not join is an ordinary autograd consumer of the source tensor.
* In backward, every joined consumer calls :meth:`GradFanIn.finish` exactly once with its
  input gradient. Before computing that gradient it may ask for :meth:`accum_target` (a
  buffer it may add into in place, for example as GEMM output with ``accumulate=True``) or
  :meth:`partial_sum` (read only, for example the extra input of ``rmsnorm_bwd_add``); it then
  passes ``accumulated=True`` to :meth:`finish`.
* :meth:`finish` returns ``None`` to all joined consumers but the last one, which receives the
  complete sum to hand back to autograd.

Contributions may arrive in any order; the result is the same sum. The fused paths are taken
when the first contribution comes from an identity tap (:class:`GradFanInTap`) placed on a
branch whose gradient arrives first in backward.
"""

import weakref
from typing import Optional

import torch


def _storage_use_count(tensor: torch.Tensor) -> int:
    # The temporary Python storage object holds a reference of its own.
    return torch._C._storage_Use_Count(tensor.untyped_storage()._cdata)


_EXCLUSIVE_STORAGE_USE_COUNT: Optional[int] = None


def _storage_is_exclusive(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is safe to update in place as a running gradient sum.

    True if ``tensor`` spans its whole storage and no tensor other than itself (and, for a
    view such as the gradient of a reshape, its base) references that storage. A gradient that
    shares its storage with another live tensor, e.g. the same gradient object routed to
    another branch or a slice of a persistent communication buffer, is not written in place.
    """
    global _EXCLUSIVE_STORAGE_USE_COUNT
    try:
        if _EXCLUSIVE_STORAGE_USE_COUNT is None:
            _EXCLUSIVE_STORAGE_USE_COUNT = _storage_use_count(torch.empty(1))
        if (
            tensor.storage_offset() != 0
            or tensor.numel() * tensor.element_size() != tensor.untyped_storage().nbytes()
        ):
            return False
        expected = _EXCLUSIVE_STORAGE_USE_COUNT + (1 if tensor._is_view() else 0)
        return _storage_use_count(tensor) == expected
    except (AttributeError, RuntimeError):  # pragma: no cover - older PyTorch
        return False


class GradFanIn:
    """Sums the input gradients that several consumers of one tensor produce.

    Args:
        source (torch.Tensor): the tensor whose consumers join this fan-in. Only a weak
            reference is kept; consumers use :meth:`matches` to make sure their input is
            this exact tensor.
    """

    def __init__(self, source: torch.Tensor):
        self._source = weakref.ref(source)
        self.num_consumers = 0
        self.num_finished = 0
        self._acc: Optional[torch.Tensor] = None
        self._acc_writable = False

    def matches(self, tensor: torch.Tensor) -> bool:
        """True if ``tensor`` is the source tensor and the fan-in still accepts consumers."""
        return self.num_finished == 0 and self._source() is tensor

    def add_consumer(self) -> None:
        """Register a consumer (forward). It must call :meth:`finish` once in backward."""
        assert self.num_finished == 0, "Cannot join a gradient fan-in after backward started."
        self.num_consumers += 1

    def accum_target(self) -> Optional[torch.Tensor]:
        """Running sum a consumer may add into in place, or ``None``."""
        return self._acc if self._acc_writable else None

    def partial_sum(self) -> Optional[torch.Tensor]:
        """Running sum for read-only use, or ``None`` before the first contribution."""
        return self._acc

    def finish(self, grad: Optional[torch.Tensor], accumulated: bool) -> Optional[torch.Tensor]:
        """Record one consumer's input gradient (backward).

        Args:
            grad (torch.Tensor, optional): the consumer's input gradient, ``None`` if it has
                none.
            accumulated (bool): ``grad`` already contains the running sum (the consumer used
                :meth:`accum_target` or :meth:`partial_sum`).

        Returns:
            The summed gradient for the last consumer, ``None`` for all others.
        """
        if grad is not None:
            if accumulated or self._acc is None:
                self._acc = grad
                # A consumer that accumulated wrote a buffer of its own. A plain contribution
                # may still be referenced elsewhere (e.g. the same tensor is the gradient of
                # another branch), so it is only written in place if nothing else shares it.
                self._acc_writable = accumulated or _storage_is_exclusive(grad)
            elif self._acc_writable:
                self._acc.add_(grad)
            else:
                self._acc = self._acc + grad
                self._acc_writable = True
        self.num_finished += 1
        assert (
            self.num_finished <= self.num_consumers
        ), "More gradient fan-in contributions than registered consumers."
        if self.num_finished == self.num_consumers:
            out, self._acc = self._acc, None
            return out
        return None


class GradFanInTap(torch.autograd.Function):
    """Identity whose backward reports the incoming gradient to a :class:`GradFanIn`."""

    @staticmethod
    def forward(ctx, inp: torch.Tensor, fanin: GradFanIn) -> torch.Tensor:
        """Join ``fanin`` and return a view of ``inp``."""
        fanin.add_consumer()
        ctx.fanin = fanin
        return inp.view_as(inp)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        """Report ``grad``; return the sum if this is the last contribution."""
        fanin, ctx.fanin = ctx.fanin, None
        return fanin.finish(grad, accumulated=False), None


def grad_fanin_tap(inp: torch.Tensor, fanin: GradFanIn) -> torch.Tensor:
    """Route the gradient of this use of ``inp`` through ``fanin``."""
    return GradFanInTap.apply(inp, fanin)


class _ModuleFanInAttach:
    """Forward pre-hook that hands a fan-in to a module that supports it.

    The module joins only if its input is the fan-in's source tensor and grad is enabled (so a
    no-grad checkpoint forward, which never runs backward, does not join).
    """

    def __init__(self, fanin: GradFanIn):
        self.fanin = fanin
        self.joined = False

    def __call__(self, module, args):
        if (
            not self.joined
            and args
            and torch.is_grad_enabled()
            and isinstance(args[0], torch.Tensor)
            and self.fanin.matches(args[0])
        ):
            module._grad_fanin = self.fanin
            self.fanin.add_consumer()
            self.joined = True


class attach_grad_fanin:
    """Context manager: let ``module`` join ``fanin`` for the forward calls in the block.

    No-op if ``module`` does not advertise ``supports_grad_fanin`` (Transformer Engine
    ``Linear`` / ``LayerNormLinear`` with gradient fan-in support). The module consumes
    ``module._grad_fanin`` in its forward; if it did not (e.g. it took a path without
    fan-in support), the registration is undone so the fan-in never waits for it.
    """

    def __init__(self, module: Optional[torch.nn.Module], fanin: Optional[GradFanIn]):
        self.module = module
        self.fanin = fanin
        self._hook = None
        self._attach = None

    def __enter__(self):
        if (
            self.fanin is not None
            and self.module is not None
            and getattr(self.module, "supports_grad_fanin", False)
        ):
            self._attach = _ModuleFanInAttach(self.fanin)
            self._hook = self.module.register_forward_pre_hook(self._attach)
        return self

    @property
    def joined(self) -> bool:
        """Whether the module joined the fan-in."""
        return self._attach is not None and self._attach.joined

    def __exit__(self, *exc):
        if self._hook is not None:
            self._hook.remove()
            if self.module.__dict__.pop("_grad_fanin", None) is not None:
                # Not consumed by the module's forward: it will not report in backward.
                self.fanin.num_consumers -= 1
                self._attach.joined = False
        return False

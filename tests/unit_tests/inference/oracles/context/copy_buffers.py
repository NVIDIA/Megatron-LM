# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Buffers for the copy and scatter oracles in this package.

* ``random_bits``: arbitrary bit patterns, so a copy must move bits, not values.
* ``Placed``: a view of a logical shape inside a guard-banded, NaN-poisoned allocation, in one
  of the layouts a launcher accepts: contiguous, one element in (a base pointer that is not
  16-byte aligned, which Triton compiles separately), every other row with padded trailing
  dimensions (non-default strides), or with the last two dimensions transposed (a non-unit
  innermost stride). The whole allocation is compared, so a stray write into the gaps shows.
* ``behind_front_guard`` / ``changed``: a tensor behind a poisoned guard of ``2**31`` elements,
  for the int32-offset (``large``) cases. An element offset in ``[2**31, 2**32)`` computed in
  int32 wraps to ``offset - 2**32``, which lands in the guard instead of faulting, so the case
  fails on an assertion instead of on a sticky CUDA error.
"""

import math
from typing import Sequence, Tuple

import torch

from tests.unit_tests.inference.oracles import oracle_utils as ou

_INT_OF_SIZE = {2: torch.int16, 4: torch.int32}

LAYOUTS = ("contiguous", "offset", "padded", "transposed")


def random_bits(shape: Sequence[int], dtype: torch.dtype, generator: torch.Generator):
    """Arbitrary ``dtype`` bit patterns on the generator's device: NaN payloads, infinities,
    denormals and -0.0 included."""
    ints = _INT_OF_SIZE[dtype.itemsize]
    info = torch.iinfo(ints)
    raw = torch.randint(
        info.min, info.max, tuple(shape), generator=generator, device=generator.device, dtype=ints
    )
    return raw.view(dtype)


def full_shape(layout: str, shape: Sequence[int]) -> Tuple[int, ...]:
    """Shape of the allocation that holds a ``layout`` view of the logical ``shape``."""
    n, *trail = shape
    if layout == "contiguous":
        return tuple(shape)
    if layout == "offset":
        return (math.prod(shape) + 1,)
    if layout == "padded":
        return (2 * n, *(t + 1 for t in trail))
    if layout == "transposed":
        if len(trail) == 1:
            return (n, 2 * trail[0])
        return (n, *trail[:-2], trail[-1], trail[-2])
    raise ValueError(f"unknown layout {layout}")


def view(full: torch.Tensor, layout: str, shape: Sequence[int]) -> torch.Tensor:
    """The ``layout`` view of the logical ``shape`` inside ``full`` (same storage)."""
    n, *trail = shape
    if layout == "contiguous":
        return full
    if layout == "offset":
        return full[1:].view(tuple(shape))
    if layout == "padded":
        return full[(slice(None, None, 2), *(slice(0, t) for t in trail))]
    if layout == "transposed":
        if len(trail) == 1:
            return full[:, ::2]
        return full.transpose(-1, -2)
    raise ValueError(f"unknown layout {layout}")


class Placed:
    """A ``layout`` view (``.tensor``) of the logical ``shape`` inside a guard-banded allocation
    (``.buf``, NaN-poisoned). ``.full`` is the whole allocation; ``view_of(clone)`` gives the
    same view of a clone of it, for building the expected allocation."""

    def __init__(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        layout: str = "contiguous",
        *,
        guard_bytes: int,
        device="cuda",
    ):
        self.shape, self.layout = tuple(shape), layout
        self.buf = ou.guarded_empty(
            full_shape(layout, shape), dtype, device, guard_bytes=guard_bytes
        )
        self.tensor = view(self.buf.tensor, layout, self.shape)

    @property
    def full(self) -> torch.Tensor:
        return self.buf.tensor

    def view_of(self, full: torch.Tensor) -> torch.Tensor:
        return view(full, self.layout, self.shape)


INT32_GUARD = 2**31
"""Elements in front of a ``large`` case's buffer: every wrapped int32 offset lands in them."""


def behind_front_guard(
    shape: Sequence[int], dtype: torch.dtype, *, guard_elems: int = INT32_GUARD, device="cuda"
):
    """``(tensor, guard)`` in one allocation: ``guard_elems`` poisoned elements, then ``tensor``
    (uninitialized)."""
    raw = torch.empty(guard_elems + math.prod(shape), dtype=dtype, device=device)
    guard = ou.poison_(raw[:guard_elems])
    return raw[guard_elems:].view(tuple(shape)), guard


def changed(guard: torch.Tensor) -> int:
    """How many elements of a poisoned guard no longer hold the poison."""
    poison = ou.poison_(torch.empty(1, dtype=guard.dtype, device=guard.device))
    ints = _INT_OF_SIZE[guard.dtype.itemsize]
    return int((guard.view(ints) != poison.view(ints)).sum())

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Head-count negotiation of the indexer top-k selectors.

An indexer sums the weighted scores of all its query heads, so a selector needs every head of a
layer, and score kernels exist for some head counts only: a LiteTopK plugin route declares its
head counts in ``plugin_info()``, and the reference score kernel (DeepGEMM
``fp8_fp4_mqa_logits``) is probed for the head counts it accepts (see
:func:`~.reference.score_kernel_heads`). :func:`negotiate_indexer_heads` decides, when a layer is
bound, how many heads each selector of the layer scores with:

* LiteTopK runs a head count of its route as is. With ``IndexerTopKConfig.head_padding`` it runs
  a head count that is a multiple of four with the smallest larger head count of the route
  (:meth:`~.plugins.abi.RouteCapability.padded_heads`), for example 16 with 32 and 48 with 64;
  any other head count fails the binding.
* The reference selector pads when its score kernel has no kernel for the layer's head count
  (it then scores with the next head count DeepGEMM supports). When LiteTopK pads, the reference
  selector scores the plugin's padded operands as well, so that the rows it selects, recomputes
  and votes seeds with see the same bytes as the plugin's rows; with precision ``fast`` and a
  head count DeepGEMM supports, it keeps the layer's own head count.

Zero heads are appended after quantization: FP8 query codes 0 with folded weights 0; MXFP4 query
codes 0 with group scales 127 (the UE8M0 code of 1.0; 255 encodes NaN) and weights 0. A zero
head adds ``relu(+-0) * 0``, a zero, to every score. The score kernels reduce the heads in four
float32 FMA chains (heads ``j`` with ``j % 4`` equal, accumulated in head order, rounded to
nearest, starting at +0), so the appended heads end every chain, and ``x + (+-0) == x`` bit for
bit for every ``x`` but -0. A chain holds -0 only after a negative product or partial sum below
half the smallest float32 subnormal rounded to zero, which needs a product below 2**-102 in
magnitude (every larger product is a multiple of the smallest subnormal). Padding therefore leaves
every score bit for bit unchanged, except that a score of -0 (every term underflowed) can become
+0, the same value. It multiplies the scoring work by ``kernel heads / H``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from megatron.lite.primitive.kernels.indexer_topk.config import (
    IndexerTopKConfigError,
    IndexerTopKPrecision,
)

if TYPE_CHECKING:
    from megatron.lite.primitive.kernels.indexer_topk.layout import IndexerGeometry
    from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability

__all__ = ["IndexerHeads", "ReferenceHeads", "negotiate_indexer_heads"]

# Maps a head count to the head count the reference score kernel scores it with (the count
# itself, or the larger count the queries are padded to); raises IndexerTopKConfigError when the
# kernel supports no such count.
ReferenceHeads = Callable[[int], int]


@dataclass(frozen=True)
class IndexerHeads:
    """The head counts the selectors of one layer score with.

    Attributes:
        num_heads: The indexer heads of the layer.
        litetopk_heads: Heads the LiteTopK kernels run with: ``num_heads``, or the head count of
            the route the layer is padded to; None without a plugin route.
        reference_heads: Heads the reference score kernel runs with while LiteTopK selects rows
            of the layer: the plugin's kernel heads when LiteTopK pads (except with precision
            ``fast`` when DeepGEMM supports ``num_heads``), else ``baseline_heads``. None when
            the module's upstream selector selects the rows LiteTopK does not cover.
        baseline_heads: Heads the reference score kernel needs for the layer on its own:
            ``num_heads`` when DeepGEMM supports it, else its next supported head count. The
            reference backend scores with this count, and so does a binding whose plan gives
            LiteTopK no row. None with the upstream selector.
    """

    num_heads: int
    litetopk_heads: int | None = None
    reference_heads: int | None = None
    baseline_heads: int | None = None

    def __post_init__(self) -> None:
        if type(self.num_heads) is not int or self.num_heads < 1:
            raise ValueError(
                f"IndexerHeads.num_heads must be a positive integer, got {self.num_heads!r}"
            )
        for name in ("litetopk_heads", "reference_heads", "baseline_heads"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < self.num_heads):
                raise ValueError(
                    f"IndexerHeads.{name} must be None or an integer >= num_heads "
                    f"({self.num_heads}), got {value!r}"
                )

    @property
    def litetopk_padded(self) -> bool:
        """Whether LiteTopK scores zero-padded heads."""
        return self.litetopk_heads is not None and self.litetopk_heads > self.num_heads

    @property
    def reference_padded(self) -> bool:
        """Whether the reference selector scores zero-padded heads while LiteTopK runs."""
        return self.reference_heads is not None and self.reference_heads > self.num_heads

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy, for logs and provenance records."""
        return {
            "num_heads": self.num_heads,
            "litetopk_heads": self.litetopk_heads,
            "reference_heads": self.reference_heads,
            "baseline_heads": self.baseline_heads,
            "litetopk_padded": self.litetopk_padded,
            "reference_padded": self.reference_padded,
        }


def _topk_text(route: RouteCapability) -> str:
    return f"<= {route.max_topk}" if route.topk is None else str(sorted(route.topk))


def _litetopk_heads(
    geometry: IndexerGeometry,
    *,
    name: str,
    route: RouteCapability,
    source_id: str,
    head_padding: bool,
    reference_serves: Callable[[int], bool],
) -> int:
    """The heads the plugin runs the layer with, or raise why the route cannot serve it."""
    heads, head_dim = geometry.num_heads, geometry.head_dim
    if head_dim in route.head_dims:
        if heads in route.heads:
            return heads
        padded = route.padded_heads(heads)
        if padded is not None and head_padding:
            return padded
    else:
        padded = None
    message = (
        f"LiteTopK route {route.name} (source {source_id}) supports indexer heads "
        f"{sorted(route.heads)}, head_dim {sorted(route.head_dims)}, topk {_topk_text(route)}; "
        f"layer {name} has H={heads}, D={head_dim}, K={geometry.topk}."
    )
    reference = reference_serves(heads)
    if padded is not None:
        message += (
            f" Set indexer_topk.head_padding=True to pad heads to {padded} "
            f"({padded / heads:.2f}x scoring work)"
        )
        message += " or use backend='reference'." if reference else "."
    elif reference:
        message += " Use backend='reference'."
    else:
        message += (
            f" The reference score kernel cannot score {heads} heads either; keep "
            "backend='default'."
        )
    raise IndexerTopKConfigError(message)


def negotiate_indexer_heads(
    geometry: IndexerGeometry,
    *,
    name: str,
    precision: IndexerTopKPrecision,
    head_padding: bool,
    route: RouteCapability | None,
    source_id: str | None,
    reference_heads: ReferenceHeads | None,
) -> IndexerHeads:
    """Decide the head counts the selectors of one layer score with (see the module docstring).

    A host computation, apart from what ``reference_heads`` probes.

    Args:
        geometry: The indexer geometry of the layer.
        name: The name of the layer, for error messages.
        precision: ``exact`` or ``fast``.
        head_padding: Whether LiteTopK may pad the layer's heads
            (``IndexerTopKConfig.head_padding``).
        route: The plugin route that serves the layer, or None without LiteTopK.
        source_id: The plugin's source id, for error messages; None without LiteTopK.
        reference_heads: The head counts of the reference score kernel (see
            :data:`ReferenceHeads`), or None when the module's upstream selector selects the rows
            LiteTopK does not cover.

    Returns:
        The negotiated head counts.

    Raises:
        IndexerTopKConfigError: If the route cannot serve the layer (its head count is not one
            of the route's and cannot be padded to one, or padding was not enabled, or its head
            dimension is not the route's), the reference score kernel supports no head count
            for the layer, or precision ``exact`` needs the reference selector to score the
            plugin's padded operands and it cannot.
    """
    probed: dict[int, int] = {}

    def reference(heads: int) -> int:
        if heads not in probed:
            probed[heads] = reference_heads(heads)
        return probed[heads]

    def reference_serves(heads: int) -> bool:
        if reference_heads is None:
            return True  # the upstream selector selects any head count
        try:
            reference(heads)
        except IndexerTopKConfigError:
            return False
        return True

    litetopk = None
    if route is not None:
        litetopk = _litetopk_heads(
            geometry,
            name=name,
            route=route,
            source_id=source_id,
            head_padding=head_padding,
            reference_serves=reference_serves,
        )
    if reference_heads is None:
        return IndexerHeads(geometry.num_heads, litetopk_heads=litetopk)
    baseline = reference(geometry.num_heads)
    selected = baseline
    if litetopk is not None and not (precision == "fast" and baseline == geometry.num_heads):
        selected = reference(litetopk)
        if precision == "exact" and selected != litetopk:
            raise IndexerTopKConfigError(
                f"precision='exact' scores the rows LiteTopK does not cover on the plugin's "
                f"operands, but the reference score kernel cannot score {litetopk} heads (layer "
                f"{name}, H={geometry.num_heads}; it would pad them to {selected}); use "
                "precision='fast'"
            )
    return IndexerHeads(
        geometry.num_heads,
        litetopk_heads=litetopk,
        reference_heads=selected,
        baseline_heads=baseline,
    )

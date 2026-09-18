# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer-role resolution and static validation for DeepSeek-V4.1 CSA2.

CSA2 ("compressed sparse attention, version 2") differs from the DeepSeek-V4 CSA by
sharing compressed KV, indexer keys, top-k selections and candidate blocks across layers.
Every model layer is assigned one of four roles from three configuration lists:

* ``window``   compress ratio 0: sliding-window attention only, nothing shared.
* ``full``     a KV-source layer: runs its own compressor, publishes compressed KV and
               indexer keys, runs an indexer and publishes its top-k.
* ``reindex``  an index-source layer that is not a KV source: reads the shared compressed KV
               and indexer keys, runs its own indexer (query side) and publishes top-k.
* ``reuse``    everything else: reads shared compressed KV and the shared top-k.

This module is pure Python (no torch / megatron imports) so it can be used from
``TransformerConfig.__post_init__`` and from CPU-only unit tests.

Design provenance: the role split and the configuration field names follow the public
reference implementation shipped with the DeepSeek-V4.1-Flash weights (``inference/model.py``,
``ModelArgs`` / ``Attention`` / ``Indexer``). The code here is an independent
implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

# A DeepSeek-V4.1 model layer is expressed as two consecutive hybrid-pattern symbols:
# one attention symbol ('W' or 'D') followed by one MoE symbol ('E'). ``layer_number``
# in Megatron-Core counts pattern positions (1-based), so the model layer id is derived
# by integer division. The V4.1 hybrid stack validates that the pattern really alternates.
SYMBOLS_PER_MODEL_LAYER = 2


class CSA2LayerMode(str, Enum):
    """Role of a model layer inside the CSA2 sharing scheme."""

    WINDOW = "window"
    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


@dataclass(frozen=True)
class CSA2LayerPlan:
    """Static description of what one model layer computes and what it reads."""

    layer_id: int
    compress_ratio: int
    mode: CSA2LayerMode
    # Model layer whose compressed KV and indexer keys this layer consumes (itself for FULL).
    kv_source: Optional[int]
    # Model layer whose top-k selection this layer consumes (itself for FULL / REINDEX).
    index_source: Optional[int]
    is_candidate_source: bool
    uses_candidates: bool

    @property
    def has_compressed_path(self) -> bool:
        """True when the layer attends to compressed positions at all."""
        return self.compress_ratio > 0

    @property
    def runs_compressor(self) -> bool:
        """True when the layer owns a compressor (KV source)."""
        return self.mode == CSA2LayerMode.FULL

    @property
    def runs_indexer(self) -> bool:
        """True when the layer computes its own top-k selection."""
        return self.mode in (CSA2LayerMode.FULL, CSA2LayerMode.REINDEX)


@dataclass(frozen=True)
class CSA2Plan:
    """Roles of every model layer plus the shared-selection parameters."""

    layers: Tuple[CSA2LayerPlan, ...]
    kv_source_layers: Tuple[int, ...]
    index_source_layers: Tuple[int, ...]
    candidate_source_layer: Optional[int]
    candidate_topk_blocks: int
    candidate_block_size: int

    def __getitem__(self, layer_id: int) -> CSA2LayerPlan:
        return self.layers[layer_id]

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def uses_candidate_blocks(self) -> bool:
        """True when two-level (block then position) top-k is enabled."""
        return self.candidate_source_layer is not None


def model_layer_id_from_layer_number(layer_number: int) -> int:
    """Map a 1-based Megatron pattern position to the 0-based DeepSeek model layer id."""
    if layer_number < 1:
        raise ValueError(f"layer_number must be 1-based and positive, got {layer_number}")
    return (layer_number - 1) // SYMBOLS_PER_MODEL_LAYER


def is_attention_position(layer_number: int) -> bool:
    """True when the 1-based pattern position holds the attention symbol of its model layer."""
    return (layer_number - 1) % SYMBOLS_PER_MODEL_LAYER == 0


def _as_int(value, name: str) -> int:
    """Accept only genuine integers (no bools, no floats such as 2.9 or 0.5)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r} ({type(value).__name__})")
    return int(value)


def _as_int_list(values: Sequence, name: str) -> List[int]:
    return [_as_int(v, name) for v in values]


def _check_strictly_increasing(values: Sequence[int], name: str) -> None:
    for a, b in zip(values, values[1:]):
        if b <= a:
            raise ValueError(f"{name} must be strictly increasing, got {list(values)}")


def _latest_source_at_or_before(sources: Sequence[int], layer_id: int) -> Optional[int]:
    chosen = None
    for src in sources:
        if src <= layer_id:
            chosen = src
        else:
            break
    return chosen


def resolve_csa2_plan(
    compress_ratios: Sequence[int],
    kv_source_layers: Optional[Sequence[int]],
    index_source_layers: Optional[Sequence[int]],
    candidate_source_layer: Optional[int] = None,
    candidate_topk_blocks: int = 0,
    candidate_block_size: int = 0,
) -> CSA2Plan:
    """Resolve and validate the CSA2 layer roles.

    Args:
        compress_ratios: one entry per *model* layer. ``0`` is a window-only layer,
            ``r >= 1`` attends to compressed positions produced ``r`` tokens to one.
        kv_source_layers: model layers that run a compressor and publish compressed KV.
        index_source_layers: model layers that run an indexer and publish top-k.
        candidate_source_layer: model layer that publishes candidate blocks, or ``None`` /
            a negative value to disable two-level selection.
        candidate_topk_blocks: number of candidate blocks kept per query.
        candidate_block_size: compressed positions per candidate block.

    Raises:
        ValueError: on any inconsistent layout. Every message names the offending field.
    """
    ratios = _as_int_list(compress_ratios, "compress_ratios")
    num_layers = len(ratios)
    if num_layers == 0:
        raise ValueError("compress_ratios must not be empty")
    if any(r < 0 for r in ratios):
        raise ValueError(f"compress_ratios must be non-negative, got {ratios}")

    kv_sources = tuple(_as_int_list(kv_source_layers or (), "csa2_kv_source_layers"))
    index_sources = tuple(_as_int_list(index_source_layers or (), "csa2_index_source_layers"))
    if candidate_source_layer is not None:
        candidate_source_layer = _as_int(candidate_source_layer, "csa2_candidate_source_layer")
    candidate_topk_blocks = _as_int(candidate_topk_blocks, "csa2_candidate_topk_blocks")
    candidate_block_size = _as_int(candidate_block_size, "csa2_candidate_block_size")
    _check_strictly_increasing(kv_sources, "csa2_kv_source_layers")
    _check_strictly_increasing(index_sources, "csa2_index_source_layers")

    for name, sources in (
        ("csa2_kv_source_layers", kv_sources),
        ("csa2_index_source_layers", index_sources),
    ):
        for src in sources:
            if not 0 <= src < num_layers:
                raise ValueError(f"{name} entry {src} is outside [0, {num_layers})")
            if ratios[src] == 0:
                raise ValueError(
                    f"{name} entry {src} is a window-only layer (compress ratio 0); "
                    "a source layer must have a positive compress ratio"
                )

    index_source_set = set(index_sources)
    for src in kv_sources:
        if src not in index_source_set:
            raise ValueError(
                f"csa2_kv_source_layers entry {src} must also be listed in "
                "csa2_index_source_layers: indexer keys are derived from the compressor output "
                "and only a KV source can publish them"
            )

    if candidate_source_layer is not None and candidate_source_layer < 0:
        candidate_source_layer = None
    if candidate_source_layer is not None:
        if candidate_source_layer not in kv_sources:
            raise ValueError(
                f"csa2_candidate_source_layer {candidate_source_layer} must be a KV source layer"
            )
        if candidate_topk_blocks <= 0 or candidate_block_size <= 0:
            raise ValueError(
                "csa2_candidate_topk_blocks and csa2_candidate_block_size must be positive when "
                f"csa2_candidate_source_layer is set, got {candidate_topk_blocks} / "
                f"{candidate_block_size}"
            )
        if any(src > candidate_source_layer for src in kv_sources):
            raise ValueError(
                "csa2_candidate_source_layer must be the last KV source layer: a later KV source "
                "would change the compressed axis the candidate mask refers to "
                f"(candidate {candidate_source_layer}, kv sources {list(kv_sources)})"
            )
    else:
        if candidate_topk_blocks or candidate_block_size:
            raise ValueError(
                "csa2_candidate_topk_blocks / csa2_candidate_block_size are set but "
                "csa2_candidate_source_layer is not"
            )

    layers: List[CSA2LayerPlan] = []
    for layer_id, ratio in enumerate(ratios):
        if ratio == 0:
            layers.append(
                CSA2LayerPlan(
                    layer_id=layer_id,
                    compress_ratio=0,
                    mode=CSA2LayerMode.WINDOW,
                    kv_source=None,
                    index_source=None,
                    is_candidate_source=False,
                    uses_candidates=False,
                )
            )
            continue

        kv_source = _latest_source_at_or_before(kv_sources, layer_id)
        index_source = _latest_source_at_or_before(index_sources, layer_id)
        if kv_source is None:
            raise ValueError(
                f"model layer {layer_id} has compress ratio {ratio} but no KV source layer at "
                "or before it; the first compressing layer must be in csa2_kv_source_layers"
            )
        if index_source is None:
            raise ValueError(
                f"model layer {layer_id} has compress ratio {ratio} but no index source layer "
                "at or before it"
            )
        if ratios[kv_source] != ratio:
            raise ValueError(
                f"model layer {layer_id} (compress ratio {ratio}) reads compressed KV from layer "
                f"{kv_source} (compress ratio {ratios[kv_source]}); layers sharing a KV source "
                "must use the same compress ratio"
            )
        if index_source < kv_source:
            raise ValueError(
                f"model layer {layer_id} would read top-k from layer {index_source} but compressed "
                f"KV from the later layer {kv_source}; every KV source must also be an index source"
            )

        if layer_id in kv_sources:
            mode = CSA2LayerMode.FULL
        elif layer_id in index_source_set:
            mode = CSA2LayerMode.REINDEX
        else:
            mode = CSA2LayerMode.REUSE

        is_candidate_source = candidate_source_layer == layer_id
        uses_candidates = (
            candidate_source_layer is not None
            and layer_id > candidate_source_layer
            and mode in (CSA2LayerMode.FULL, CSA2LayerMode.REINDEX)
        )
        layers.append(
            CSA2LayerPlan(
                layer_id=layer_id,
                compress_ratio=ratio,
                mode=mode,
                kv_source=kv_source,
                index_source=index_source,
                is_candidate_source=is_candidate_source,
                uses_candidates=uses_candidates,
            )
        )

    return CSA2Plan(
        layers=tuple(layers),
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=(
            int(candidate_topk_blocks) if candidate_source_layer is not None else 0
        ),
        candidate_block_size=int(candidate_block_size) if candidate_source_layer is not None else 0,
    )


def model_layer_ratios_from_pattern_ratios(pattern_ratios: Sequence[int]) -> List[int]:
    """Extract per-model-layer ratios from the per-pattern-position ("full form") list.

    Megatron-Core stores ``csa_compress_ratios`` with one entry per hybrid pattern symbol,
    MoE positions carrying ``0``. For DeepSeek-V4.1 the pattern alternates attention / MoE,
    so the attention entries sit at even positions.
    """
    ratios = list(pattern_ratios)
    if len(ratios) % SYMBOLS_PER_MODEL_LAYER != 0:
        raise ValueError(
            f"csa_compress_ratios length {len(ratios)} is not a multiple of "
            f"{SYMBOLS_PER_MODEL_LAYER}; DeepSeek-V4.1 expects one attention and one MoE "
            "position per model layer"
        )
    moe_entries = ratios[1::SYMBOLS_PER_MODEL_LAYER]
    if any(r != 0 for r in moe_entries):
        raise ValueError(
            "csa_compress_ratios must carry 0 at MoE pattern positions (odd indices), got "
            f"{moe_entries}"
        )
    return ratios[0::SYMBOLS_PER_MODEL_LAYER]


def pattern_ratios_from_model_layer_ratios(model_layer_ratios: Sequence[int]) -> List[int]:
    """Inverse of :func:`model_layer_ratios_from_pattern_ratios`."""
    full: List[int] = []
    for ratio in model_layer_ratios:
        full.append(int(ratio))
        full.extend([0] * (SYMBOLS_PER_MODEL_LAYER - 1))
    return full

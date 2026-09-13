# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-microbatch cross-layer state for DeepSeek-V4.1 (CSA2 sharing and single-pass mHC).

One :class:`DSv41SharedState` instance is created by the V4.1 hybrid stack at the start of
every forward and handed to each layer as an explicit keyword argument. Consumers address
producers by *static* ids (the model layer id of the KV / index / candidate source, or the
previous pattern position for the hyper-connection handoff), so a layer re-executed under
activation recomputation reads exactly the entries it read in the original forward. The
tensors stay attached to the autograd graph of their producer; gradients from every
consumer flow back into the source layer.

Under full activation recomputation the stack threads the published tensors explicitly
through the checkpoint boundaries: :meth:`DSv41SharedState.export` flattens the entries that
later layers still need into a tensor list (checkpoint outputs / inputs) and
:meth:`DSv41SharedState.load` rebuilds a state from it, so replay publishes into a fresh
state and gradients cross the boundary like any other checkpoint output.

Design reference: the ``SharedAttentionRuntime`` of the official DeepSeek-V4.1 inference
code holds one slot per quantity because inference runs layers strictly in order. Training
with recomputation and pipelining needs the per-producer addressing used here. Independent
implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# (kind, id): kind in {"kv", "ik", "topk", "cand", "hpre"}; id is the model layer id of the
# producer, or the 1-based pattern position for "hpre".
StateKey = Tuple[str, int]


@dataclass
class CompressedKVRecord:
    """What a KV-source layer publishes for its consumers."""

    source_layer: int
    compress_ratio: int
    n_compressed: int
    # SBHD: [n_compressed, b, v_head_dim]; THD: [n_compressed, v_head_dim]. RoPE applied.
    kv: torch.Tensor
    # SBHD: [n_compressed, b, index_head_dim]; THD: [n_compressed, index_head_dim]. RoPE applied.
    index_keys: torch.Tensor


class DSv41SharedState:
    """Container for everything DeepSeek-V4.1 layers hand down the stack."""

    def __init__(self) -> None:
        self._compressed: Dict[int, CompressedKVRecord] = {}
        self._topk: Dict[int, torch.Tensor] = {}
        self._candidates: Dict[int, torch.Tensor] = {}
        self._h_pre: Dict[int, torch.Tensor] = {}
        # [s, b, n_engram_layers, n_hash_cols] int64, or None when Engram is disabled.
        self.engram_hash_ids: Optional[torch.Tensor] = None

    # ---- CSA2 -----------------------------------------------------------------------------

    def publish_compressed(self, record: CompressedKVRecord) -> None:
        """Store the compressed KV / indexer keys of a KV-source layer."""
        if record.source_layer in self._compressed:
            raise RuntimeError(
                f"compressed KV of model layer {record.source_layer} published twice in one "
                "forward; every KV source must run exactly once per microbatch"
            )
        self._compressed[record.source_layer] = record

    def get_compressed(self, source_layer: int) -> CompressedKVRecord:
        """Fetch the compressed KV published by ``source_layer``."""
        try:
            return self._compressed[source_layer]
        except KeyError:
            raise RuntimeError(
                f"compressed KV of model layer {source_layer} requested before it was "
                "published; check the pipeline split does not separate a KV source from its "
                "consumers"
            ) from None

    def publish_topk(self, source_layer: int, topk_indices: torch.Tensor) -> None:
        """Store the top-k selection of an index-source layer."""
        if source_layer in self._topk:
            raise RuntimeError(f"top-k of model layer {source_layer} published twice")
        self._topk[source_layer] = topk_indices

    def get_topk(self, source_layer: int) -> torch.Tensor:
        """Fetch the top-k selection published by ``source_layer``."""
        try:
            return self._topk[source_layer]
        except KeyError:
            raise RuntimeError(
                f"top-k of model layer {source_layer} requested before it was published"
            ) from None

    def publish_candidates(self, source_layer: int, block_ids: torch.Tensor) -> None:
        """Store the candidate block ids (``[..., topk_blocks]`` int32, ``-1`` padded) of the
        candidate-source layer."""
        if source_layer in self._candidates:
            raise RuntimeError(f"candidate blocks of model layer {source_layer} published twice")
        self._candidates[source_layer] = block_ids

    def get_candidates(self, source_layer: int) -> torch.Tensor:
        """Fetch the candidate block ids published by ``source_layer``."""
        try:
            return self._candidates[source_layer]
        except KeyError:
            raise RuntimeError(
                f"candidate blocks of model layer {source_layer} requested before publication"
            ) from None

    # ---- single-pass hyper-connections --------------------------------------------------

    def publish_h_pre(self, layer_number: int, h_pre: torch.Tensor) -> None:
        """Store the aggregation weights a layer computed for its successor."""
        if layer_number in self._h_pre:
            raise RuntimeError(f"h_pre of pattern position {layer_number} published twice")
        self._h_pre[layer_number] = h_pre

    def get_h_pre_for(self, layer_number: int) -> Optional[torch.Tensor]:
        """Aggregation weights for ``layer_number`` (published by ``layer_number - 1``).

        Returns ``None`` when no predecessor published anything, i.e. for the first layer of
        the model; the caller then uses the identity mix (first stream only).
        """
        return self._h_pre.get(layer_number - 1)

    def final_h_pre(self, last_layer_number: int) -> torch.Tensor:
        """Aggregation weights for the output head, published by the last layer."""
        try:
            return self._h_pre[last_layer_number]
        except KeyError:
            raise RuntimeError(
                f"final h_pre from pattern position {last_layer_number} was not published"
            ) from None

    # ---- checkpoint transport ---------------------------------------------------------------

    def keys(self) -> List[StateKey]:
        """Every published entry as a sorted key list (deterministic across replay)."""
        keys: List[StateKey] = []
        keys += [("kv", i) for i in sorted(self._compressed)]
        keys += [("ik", i) for i in sorted(self._compressed)]
        keys += [("topk", i) for i in sorted(self._topk)]
        keys += [("cand", i) for i in sorted(self._candidates)]
        keys += [("hpre", i) for i in sorted(self._h_pre)]
        return keys

    def export(self, keys: Sequence[StateKey]) -> List[torch.Tensor]:
        """Tensors of the given keys, in order."""
        out: List[torch.Tensor] = []
        for kind, ident in keys:
            if kind == "kv":
                out.append(self._compressed[ident].kv)
            elif kind == "ik":
                out.append(self._compressed[ident].index_keys)
            elif kind == "topk":
                out.append(self._topk[ident])
            elif kind == "cand":
                out.append(self._candidates[ident])
            elif kind == "hpre":
                out.append(self._h_pre[ident])
            else:
                raise KeyError(kind)
        return out

    def load(
        self, keys: Sequence[StateKey], tensors: Sequence[torch.Tensor], ratios: Dict[int, int]
    ) -> None:
        """Rebuild entries from an exported tensor list. ``ratios`` maps a KV-source model
        layer to its compress ratio (needed to rebuild the compressed record)."""
        pending_kv: Dict[int, torch.Tensor] = {}
        pending_ik: Dict[int, torch.Tensor] = {}
        for (kind, ident), tensor in zip(keys, tensors):
            if kind == "kv":
                pending_kv[ident] = tensor
            elif kind == "ik":
                pending_ik[ident] = tensor
            elif kind == "topk":
                self._topk[ident] = tensor
            elif kind == "cand":
                self._candidates[ident] = tensor
            elif kind == "hpre":
                self._h_pre[ident] = tensor
            else:
                raise KeyError(kind)
        for ident, kv in pending_kv.items():
            self._compressed[ident] = CompressedKVRecord(
                source_layer=ident,
                compress_ratio=ratios[ident],
                n_compressed=kv.shape[0],
                kv=kv,
                index_keys=pending_ik[ident],
            )


class SharedStateSlot:
    """Mutable handle through which a layer wrapper exposes the current state to submodules.

    The wrapper sets ``slot.state`` right before calling its inner layer and clears it
    afterwards, so the attention module never caches a state across microbatches.
    """

    __slots__ = ("state",)

    def __init__(self) -> None:
        self.state: Optional[DSv41SharedState] = None

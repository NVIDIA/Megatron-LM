# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TP/PP/EP/ETP KV-shard layouts and the range-intersection reshard planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from megatron.core.inference.disaggregation.utils import intersect


@dataclass(frozen=True)
class KVShardLayout:
    """A worker's KV-cache ownership within the global model.

    ``num_layers`` / ``num_heads`` are the *global* attention layer count
    and KV-head count (for GQA, the number of KV heads). ``global_rank``
    is the worker's torch rank (used as the transport peer id).
    """

    num_layers: int
    num_heads: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    global_rank: int
    # Expert dimensions. KV-replica dimensions only: they shard the MoE
    # expert weights, never the attention KV cache, so they don't affect
    # head_range/layer_range -- only representative (source) selection.
    ep_size: int = 1
    ep_rank: int = 0
    etp_size: int = 1
    etp_rank: int = 0
    # Optional explicit PP layer window for this stage. When None, an even split
    # of num_layers across pp_size is assumed -- correct for pure-attention
    # models. Models that do NOT split attention layers evenly across PP stages
    # (e.g. hybrid Mamba+attention) must pass an explicit (layer_start,
    # num_local_layers); the even-split default would otherwise map the wrong
    # global layer indices.
    layer_start: Optional[int] = None
    num_local_layers: Optional[int] = None

    def __post_init__(self) -> None:
        # TP must divide heads (the head split is always even).
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads={self.num_heads} not divisible by tp_size={self.tp_size}"
            )
        # layer_start and num_local_layers are an all-or-nothing explicit window:
        # setting only one would silently fall back to the even-split count and
        # defeat the purpose (uneven stage with an even count).
        if (self.layer_start is None) != (self.num_local_layers is None):
            raise ValueError(
                "layer_start and num_local_layers must be set together (or both omitted)"
            )
        # Only the even-split path requires PP to divide layers; an explicit
        # window may be uneven across stages.
        if self.layer_start is None and self.num_layers % self.pp_size != 0:
            raise ValueError(
                f"num_layers={self.num_layers} not divisible by pp_size={self.pp_size}; "
                "pass an explicit (layer_start, num_local_layers) for uneven PP splits"
            )

    def kv_shard_key(self) -> Tuple[int, int]:
        """The attention shard this rank holds: ``(tp_rank, pp_rank)``.
        Ranks sharing a key hold identical KV (EP/ETP replicas of it)."""
        return (self.tp_rank, self.pp_rank)

    def layer_range(self) -> Tuple[int, int]:
        # num_local_layers is guaranteed set whenever layer_start is (see __post_init__).
        if self.layer_start is not None:
            return (self.layer_start, self.layer_start + self.num_local_layers)
        per = self.num_layers // self.pp_size
        return (self.pp_rank * per, (self.pp_rank + 1) * per)

    def head_range(self) -> Tuple[int, int]:
        per = self.num_heads // self.tp_size
        return (self.tp_rank * per, (self.tp_rank + 1) * per)

    def local_num_layers(self) -> int:
        lo, hi = self.layer_range()
        return hi - lo

    def local_num_heads(self) -> int:
        lo, hi = self.head_range()
        return hi - lo


@dataclass(frozen=True)
class KVReshardTransfer:
    """One sub-block exchange between a (src, dst) rank pair.

    Global coords identify the intersection; the local-slice helpers
    convert to each side's buffer offsets. There is at most one transfer
    per (src, dst) pair (each owns a contiguous rectangle, so the
    intersection is a single rectangle).
    """

    src_rank: int
    dst_rank: int
    # The transferred sub-block's GLOBAL bounds as half-open ranges:
    # layers [global_layer_lo, global_layer_hi) x kv-heads [global_head_lo, global_head_hi).
    global_layer_lo: int
    global_layer_hi: int
    global_head_lo: int
    global_head_hi: int

    def src_layer_slice(self, src: KVShardLayout) -> slice:
        off = src.layer_range()[0]
        return slice(self.global_layer_lo - off, self.global_layer_hi - off)

    def src_head_slice(self, src: KVShardLayout) -> slice:
        off = src.head_range()[0]
        return slice(self.global_head_lo - off, self.global_head_hi - off)

    def dst_layer_slice(self, dst: KVShardLayout) -> slice:
        off = dst.layer_range()[0]
        return slice(self.global_layer_lo - off, self.global_layer_hi - off)

    def dst_head_slice(self, dst: KVShardLayout) -> slice:
        off = dst.head_range()[0]
        return slice(self.global_head_lo - off, self.global_head_hi - off)


def plan_kv_reshard(
    srcs: List[KVShardLayout], dsts: List[KVShardLayout]
) -> List[KVReshardTransfer]:
    """Full reshard plan: every sub-block that must move src -> dst.

    Both sides compute the same plan from the same layouts and filter to
    their own rank (``transfers_for_src`` / ``transfers_for_dst``).

    KV is replicated across the EP and ETP dimensions, so each attention
    shard ``(tp_rank, pp_rank)`` may be held by several source ranks. We
    source each shard from exactly one of them -- the smallest
    ``global_rank`` -- which avoids duplicate sends and is independent of
    how EP/ETP map onto ranks.
    """
    if srcs and dsts:
        if srcs[0].num_layers != dsts[0].num_layers or srcs[0].num_heads != dsts[0].num_heads:
            raise ValueError("src and dst describe different global models")

    # One representative source rank per attention shard (dedupe EP/ETP
    # replicas that hold identical KV).
    rep_rank: dict = {}
    for s in srcs:
        key = s.kv_shard_key()
        if key not in rep_rank or s.global_rank < rep_rank[key]:
            rep_rank[key] = s.global_rank
    source_ranks = set(rep_rank.values())

    transfers: List[KVReshardTransfer] = []
    for d in dsts:
        dl, dh = d.layer_range(), d.head_range()
        for s in srcs:
            if s.global_rank not in source_ranks:
                continue
            li = intersect(s.layer_range(), dl)
            if li is None:
                continue
            hi = intersect(s.head_range(), dh)
            if hi is None:
                continue
            transfers.append(
                KVReshardTransfer(
                    src_rank=s.global_rank,
                    dst_rank=d.global_rank,
                    global_layer_lo=li[0],
                    global_layer_hi=li[1],
                    global_head_lo=hi[0],
                    global_head_hi=hi[1],
                )
            )
    return transfers

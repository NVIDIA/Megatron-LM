# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous TP/PP reshard of SSM boundary-snapshot state between
prefill and decode shard layouts (the SSM analog of attention KV resharding).

A snapshot's conv state packs three channel bands, [x | B | C], on one axis:
x is head-sharded (d_inner) and B/C are group-sharded (ngroups * d_state).
The recurrent state is head-sharded. ``plan_ssm_reshard`` emits one transfer per
(src rank, dst rank, global layer, band) whose layer and channel ranges
overlap; both sides compute the same plan from the same layout lists, so the
send and receive orders match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from megatron.core.inference.disaggregation.utils import intersect

# Channel bands of an SSM layer's conv state, in the order the conv state
# concatenates them on its channel axis. The recurrent band is the head axis
# of its own tensor.
_CONV_BANDS = ("x", "B", "C")


@dataclass(frozen=True)
class SSMStateDims:
    """The model's global (unsharded) SSM structural dimensions.

    Carried as one unit so there is a single source and the dims cannot drift
    apart. The producer should read them from the model config rather than
    deriving them from tensor shapes. TP shards nheads/ngroups; the rest are
    unsharded.
    """

    nheads: int
    headdim: int
    d_state: int
    ngroups: int
    d_conv: int


@dataclass(frozen=True)
class SSMShardLayout:
    """One rank's SSM-state ownership: which global layers and TP rank,
    plus the model's structural dims. Per-rank local sizes follow by dividing
    by tp_size."""

    global_rank: int
    tp_size: int
    tp_rank: int
    layer_start: int  # global SSM-layer index of this rank's first layer
    num_layers: int  # SSM layers held locally (this PP stage)
    dims: SSMStateDims

    def __post_init__(self) -> None:
        # Wire reconstruction (SSMShardLayout(**dict)) hands dims as a plain
        # dict; coerce it back to SSMStateDims.
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", SSMStateDims(**self.dims))
        # TP shards heads and groups; both must divide evenly or the local
        # band widths are wrong.
        if self.dims.nheads % self.tp_size != 0:
            raise ValueError(f"nheads={self.dims.nheads} not divisible by tp_size={self.tp_size}")
        if self.dims.ngroups % self.tp_size != 0:
            raise ValueError(f"ngroups={self.dims.ngroups} not divisible by tp_size={self.tp_size}")

    @property
    def d_inner(self) -> int:
        """Global inner dimension (nheads * headdim)."""
        return self.dims.nheads * self.dims.headdim

    @property
    def nheads_local(self) -> int:
        """SSM heads held by this TP rank."""
        return self.dims.nheads // self.tp_size

    @property
    def d_inner_local(self) -> int:
        """Local inner dimension for this TP rank."""
        return self.d_inner // self.tp_size

    @property
    def ngroups_local(self) -> int:
        """B/C groups held by this TP rank."""
        return self.dims.ngroups // self.tp_size

    @property
    def conv_dim_local(self) -> int:
        """Total local conv channel width (x + B + C bands)."""
        return self.d_inner_local + 2 * self.ngroups_local * self.dims.d_state

    def shard_key(self) -> Tuple[int, int]:
        """The SSM shard this rank holds: (tp_rank, layer_start). Ranks
        sharing a key hold identical state (e.g. EP/DP replicas)."""
        return (self.tp_rank, self.layer_start)

    def layer_range(self) -> Tuple[int, int]:
        """Global SSM-layer range [lo, hi) owned by this rank."""
        return (self.layer_start, self.layer_start + self.num_layers)

    def band(self, name: str) -> Tuple[int, int, int]:
        """Return (global_total, local_size, local_offset) for a band.

        local_offset is the band's start on the local conv channel axis; for
        the "recurrent" (head) band it is the start on the local head axis
        (always 0, heads are the whole tensor).
        """
        if name == "x":
            return self.d_inner, self.d_inner_local, 0
        if name == "B":
            g = self.dims.ngroups * self.dims.d_state
            return g, self.ngroups_local * self.dims.d_state, self.d_inner_local
        if name == "C":
            g = self.dims.ngroups * self.dims.d_state
            return (
                g,
                self.ngroups_local * self.dims.d_state,
                self.d_inner_local + self.ngroups_local * self.dims.d_state,
            )
        if name == "recurrent":
            return self.dims.nheads, self.nheads_local, 0
        raise KeyError(name)


@dataclass(frozen=True)
class SSMReshardTransfer:
    """One sub-block move of the snapshot reshard.

    band is "x"/"B"/"C" (conv channel axis) or "recurrent" (head axis).
    src_layer/dst_layer are local layer indices on each side; *_lo/*_hi are
    the local channel or head slice bounds.
    """

    src_rank: int
    dst_rank: int
    band: str
    global_layer: int
    src_layer: int
    dst_layer: int
    src_lo: int
    src_hi: int
    dst_lo: int
    dst_hi: int

    @property
    def is_conv(self) -> bool:
        """True if this transfer targets the conv state; False for recurrent."""
        return self.band in _CONV_BANDS


def plan_ssm_reshard(
    src_layouts: List[SSMShardLayout], dst_layouts: List[SSMShardLayout]
) -> List[SSMReshardTransfer]:
    """Plan the conv/recurrent sub-block moves from the prefill (src) layouts to
    the decode (dst) layouts: one transfer per (src rank, dst rank, global
    layer, band) where both the layer ranges and the channel ranges overlap.

    Ranks sharing (tp_rank, layer_start) hold identical SSM state (e.g.
    EP/DP replicas), so each shard is sourced from exactly one of them, the
    smallest global_rank. Deterministic given the layout lists, so both sides
    enumerate the same transfers in the same order.
    """
    rep_rank: dict = {}
    for s in src_layouts:
        key = s.shard_key()
        if key not in rep_rank or s.global_rank < rep_rank[key]:
            rep_rank[key] = s.global_rank
    source_ranks = set(rep_rank.values())

    out: List[SSMReshardTransfer] = []
    for s in src_layouts:
        if s.global_rank not in source_ranks:
            continue
        s_lr = s.layer_range()
        for d in dst_layouts:
            layer_ov = intersect(s_lr, d.layer_range())
            if layer_ov is None:
                continue
            for band in (*_CONV_BANDS, "recurrent"):
                _, s_size, s_off = s.band(band)
                _, d_size, d_off = d.band(band)
                s_glo = (s.tp_rank * s_size, s.tp_rank * s_size + s_size)
                d_glo = (d.tp_rank * d_size, d.tp_rank * d_size + d_size)
                chan_ov = intersect(s_glo, d_glo)
                if chan_ov is None:
                    continue
                lo, hi = chan_ov
                for g in range(layer_ov[0], layer_ov[1]):
                    out.append(
                        SSMReshardTransfer(
                            src_rank=s.global_rank,
                            dst_rank=d.global_rank,
                            band=band,
                            global_layer=g,
                            src_layer=g - s.layer_start,
                            dst_layer=g - d.layer_start,
                            src_lo=s_off + (lo - s_glo[0]),
                            src_hi=s_off + (hi - s_glo[0]),
                            dst_lo=d_off + (lo - d_glo[0]),
                            dst_hi=d_off + (hi - d_glo[0]),
                        )
                    )
    return out

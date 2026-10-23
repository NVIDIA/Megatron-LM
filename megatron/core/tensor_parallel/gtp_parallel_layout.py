# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rank layouts for GTP weight sharding and replication, and sequence and batch partitioning.

Layouts are computed without initializing distributed process groups. TP is the
innermost axis, followed by GTP_remat weight shards, with PP as the outermost axis.
A contiguous TP x GTP_remat tile can therefore be placed within one NVLink domain by the launcher.
Context parallelism may span several such tiles without enlarging weight gathers.
"""

from dataclasses import dataclass
from math import prod
from typing import Optional


def resolve_tensor_parallel_sequence_shards(
    tensor_model_parallel_size: int,
    tensor_parallel_num_sequence_shards: Optional[int],
    gtp_weight_remat_size: int,
    sequence_parallel: bool,
) -> tuple[int, int]:
    """Return sequence-shard counts including and excluding TP/SP, respectively."""
    sp_size = tensor_model_parallel_size if sequence_parallel else 1
    shards = tensor_parallel_num_sequence_shards
    if shards is None:
        shards = sp_size
    if shards < sp_size or shards % sp_size:
        raise ValueError(
            f"tensor_parallel_num_sequence_shards ({shards}) must be positive and "
            f"divisible by the sequence-parallel size ({sp_size})"
        )
    num_sequence_shards = shards // sp_size
    if num_sequence_shards > 1 and tensor_model_parallel_size > 1 and not sequence_parallel:
        raise ValueError("GTP sequence sharding requires sequence parallelism when TP > 1")
    if gtp_weight_remat_size % num_sequence_shards:
        raise ValueError(
            f"GTP sequence shards ({num_sequence_shards}) must divide gtp_weight_remat_size "
            f"({gtp_weight_remat_size})"
        )
    return shards, num_sequence_shards


def get_batch_parallel_size(args) -> int:
    """Return independent batch partitions: DP x GTP_remat batch shards.

    This replaces DP x GTP_remat when some GTP_remat ranks partition sequences.
    The fallback supports manually built launch args without a resolved batch size.
    """
    size = getattr(args, "batch_parallel_size", None)
    if size is not None:
        return size
    return (
        args.data_parallel_size
        * args.gtp_weight_remat_size
        // getattr(args, "gtp_remat_num_sequence_shards", 1)
    )


@dataclass(frozen=True)
class GTPParallelLayout:
    """Describe two views of the same ranks, keeping actual weight shards fixed.

    ``cp`` is independent of GTP_remat. ``num_sequence_shards`` partitions sequences
    within each weight group, excluding TP/SP. ``weight_size`` excludes TP. ``dp`` is
    the residual independent-data axis; callers should use ``batch_parallel_size`` for
    batch accounting and ``num_weight_replicas`` for dense optimizer sharding.
    """

    world_size: int
    tp: int
    pp: int
    cp: int
    weight_size: int
    num_sequence_shards: int = 1
    rank_offset: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.world_size,
                self.tp,
                self.pp,
                self.cp,
                self.weight_size,
                self.num_sequence_shards,
            )
            < 1
        ):
            raise ValueError("Parallel sizes must be positive")
        if self.weight_size % self.num_sequence_shards:
            raise ValueError("GTP sequence shards must divide the weight-remat size")
        if self.world_size % self.minimum_world_size:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by TP x PP x "
                f"CP x weight-remat size ({self.minimum_world_size})"
            )

    @property
    def minimum_world_size(self) -> int:
        """Smallest world that can hold both layouts."""
        return self.tp * self.pp * self.cp * self.weight_size

    @property
    def dp(self) -> int:
        """Residual independent-data degree, excluding batch shards inside GTP_remat."""
        return self.world_size // self.minimum_world_size

    @property
    def batch_parallel_size(self) -> int:
        """Number of independently assigned microbatches per pipeline."""
        return self.world_size // (self.tp * self.pp * self.cp * self.num_sequence_shards)

    @property
    def num_weight_replicas(self) -> int:
        """Number of copies of each TP-local weight shard."""
        return self.world_size // (self.tp * self.pp * self.weight_size)

    def _groups(self, axes: set[str]) -> list[list[int]]:
        names = ("tp", "gtp_remat_sequence_shards", "extra_data", "cp", "dp", "pp")
        sizes = (
            self.tp,
            self.num_sequence_shards,
            self.weight_size // self.num_sequence_shards,
            self.cp,
            self.dp,
            self.pp,
        )
        strides = [prod(sizes[:i]) for i in range(len(sizes))]
        groups = {}
        for rank in range(self.world_size):
            fixed = tuple(
                (rank // stride) % size
                for name, size, stride in zip(names, sizes, strides)
                if name not in axes
            )
            groups.setdefault(fixed, []).append(rank + self.rank_offset)
        return list(groups.values())

    @property
    def data(self) -> "_RankView":
        """CP x GTP_remat sequence shards split sequences; extra_data x DP assigns samples."""
        return _RankView(
            self,
            {
                "tp": ("tp",),
                "pp": ("pp",),
                "cp": ("cp", "gtp_remat_sequence_shards"),
                "gtp_remat": ("extra_data",),
                "dp": ("dp",),
                "ep": (),
            },
        )

    @property
    def weights(self) -> "_RankView":
        """Weight layout: TP x GTP_remat shards, DP replicates those shards."""
        return _RankView(
            self,
            {
                "tp": ("tp",),
                "pp": ("pp",),
                "gtp_remat": ("gtp_remat_sequence_shards", "extra_data"),
                "dp": ("cp", "dp"),
            },
        )


@dataclass(frozen=True)
class _RankView:
    """Bootstrap adapter for RankGenerator's rank-list interface."""

    layout: GTPParallelLayout
    axes: dict[str, tuple[str, ...]]

    def get_ranks(self, token: str) -> list[list[int]]:
        """Enumerate groups in a deterministic order on every process."""
        axes = {axis for name in token.split("-") for axis in self.axes[name]}
        return self.layout._groups(axes)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compatible weight and token layouts for GTP, without distributed initialization.

Ranks are TP-fastest, then weight-shard-fastest, with PP last. A contiguous
TP x GTP tile can therefore be placed within one NVLink domain by the launcher.
Context parallelism may span several such tiles without enlarging weight gathers.
"""

from dataclasses import dataclass
from math import gcd, prod


def get_sample_parallel_size(args) -> int:
    """Read resolved sample ownership, supporting legacy manually built launch args."""
    size = getattr(args, "sample_parallel_size", None)
    if size is not None:
        return size
    if getattr(args, "gtp_remat_fold_cp", False):
        return args.world_size // (
            args.tensor_model_parallel_size
            * args.pipeline_model_parallel_size
            * args.context_parallel_size
        )
    return args.data_parallel_size * args.gtp_weight_remat_size


@dataclass(frozen=True)
class GTPParallelLayout:
    """Describe two views of the same ranks, keeping actual weight shards fixed.

    ``weight_size`` is the rematerialization degree, excluding TP. ``dp`` is
    the residual independent-data axis; callers should use ``sample_size`` for
    batch accounting and ``replica_size`` for dense optimizer sharding.
    """

    world_size: int
    tp: int
    pp: int
    cp: int
    weight_size: int
    rank_offset: int = 0

    def __post_init__(self) -> None:
        if min(self.world_size, self.tp, self.pp, self.cp, self.weight_size) < 1:
            raise ValueError("Parallel sizes must be positive")
        if self.world_size % self.minimum_world_size:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by TP x PP x "
                f"lcm(CP, GTP) ({self.minimum_world_size})"
            )

    @property
    def overlap(self) -> int:
        """Number of context ranks participating in each weight group."""
        return gcd(self.cp, self.weight_size)

    @property
    def minimum_world_size(self) -> int:
        """Smallest world that can hold both layouts."""
        return self.tp * self.pp * self.cp * (self.weight_size // self.overlap)

    @property
    def dp(self) -> int:
        """Residual independent-data degree, excluding extra data inside GTP."""
        return self.world_size // self.minimum_world_size

    @property
    def sample_size(self) -> int:
        """Number of independently assigned microbatches per pipeline."""
        return self.world_size // (self.tp * self.pp * self.cp)

    @property
    def replica_size(self) -> int:
        """Number of copies of each TP-local weight shard."""
        return self.world_size // (self.tp * self.pp * self.weight_size)

    def _groups(self, axes: set[str]) -> list[list[int]]:
        names = ("tp", "cp_inner", "extra_data", "cp_outer", "dp", "pp")
        sizes = (
            self.tp,
            self.overlap,
            self.weight_size // self.overlap,
            self.cp // self.overlap,
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
        """Token layout: CP partitions sequences; extra_data x DP assigns samples."""
        return _RankView(
            self,
            {
                "tp": ("tp",),
                "pp": ("pp",),
                "cp": ("cp_inner", "cp_outer"),
                "gtp_remat": ("extra_data",),
                "dp": ("dp",),
                "ep": (),
            },
        )

    @property
    def weights(self) -> "_RankView":
        """Weight layout: TP x GTP shards, DP replicates those shards."""
        return _RankView(
            self,
            {
                "tp": ("tp",),
                "pp": ("pp",),
                "gtp_remat": ("cp_inner", "extra_data"),
                "dp": ("cp_outer", "dp"),
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

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Physical row layouts for fused projections optimized by per-head Muon.

Metadata only: the model keeps its fused parameters, GEMMs and checkpoint names.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MuonProjectionLayout:
    """Independent matrices and elementwise AdamW slices in physical row order.

    Args:
        splits: Row counts of consecutive logical matrices.
        adamw: Whether each slice uses AdamW instead of Newton-Schulz.
        tp_local: Layout describes one TP rank, before GTP rematerialization sharding.
        tp_partitioned: Each local matrix is itself row-sharded over TP (SwiGLU).
        tp_reorder_splits: Global projection blocks stored as one fragment of each
            block per TP rank (fused MLA Q-down/KV-down). Gather then reorder these
            blocks before applying the logical matrix splits.
    """

    splits: tuple[int, ...]
    adamw: tuple[bool, ...]
    tp_local: bool = False
    tp_partitioned: bool = False
    tp_reorder_splits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.splits or any(n <= 0 for n in self.splits):
            raise ValueError(f"Projection split sizes must be positive: {self.splits}")
        if len(self.splits) != len(self.adamw):
            raise ValueError("Every projection slice needs an optimizer assignment")
        if self.tp_partitioned and not self.tp_local:
            raise ValueError("TP-partitioned matrices require a TP-local physical layout")

        if self.tp_reorder_splits:
            if self.tp_local or any(self.adamw):
                raise ValueError("TP reordering requires a global, all-Muon layout")
            if any(n <= 0 for n in self.tp_reorder_splits) or sum(self.tp_reorder_splits) != sum(
                self.splits
            ):
                raise ValueError("TP reorder blocks must cover the full projection")

    @classmethod
    def matrices(cls, splits: tuple[int, ...], **kwargs) -> 'MuonProjectionLayout':
        """Build a layout of independent matrices, all using Muon."""
        # MLA permits disabling RoPE; an empty projection has no NS operation.
        splits = tuple(n for n in splits if n != 0)
        return cls(splits, (False,) * len(splits), **kwargs)

    @classmethod
    def from_qkv_layout(cls, layout) -> 'MuonProjectionLayout':
        """Add optimizer assignments to the shared layer-owned QKV boundaries."""
        adamw = layout.per_head_adamw or (False,) * len(layout.per_head_split_shapes)
        return cls(layout.per_head_split_shapes * layout.num_groups, adamw * layout.num_groups)

    @classmethod
    def gdn(cls, names, sections, key_head_dim: int, value_head_dim: int) -> 'MuonProjectionLayout':
        """Build GDN1 or GDN2 from the variant's actual TP-local split table."""
        splits, adamw = [], []
        if len(names) != len(sections) or tuple(names[:3]) != ('query', 'key', 'value'):
            raise ValueError("Unsupported GDN projection order")
        for name, size in zip(names, sections):
            if name in ('query', 'key', 'value'):
                head_dim = value_head_dim if name == 'value' else key_head_dim
                if size % head_dim:
                    raise ValueError(f"GDN {name} does not contain complete heads")
                splits.extend([head_dim] * (size // head_dim))
                adamw.extend([False] * (size // head_dim))
            elif name in ('z', 'beta', 'alpha', 'f', 'b', 'w'):
                # Control projections are elementwise AdamW, never 1-row NS matrices.
                splits.append(size)
                adamw.append(True)
            else:
                raise ValueError(f"Unknown GDN projection section: {name}")
        return cls(tuple(splits), tuple(adamw), tp_local=True)

    def adamw_ranges(self, start: int, rows: int) -> tuple[tuple[int, int], ...]:
        """Intersect and coalesce AdamW rows with a physical local shard (padding excluded)."""
        ranges: list[tuple[int, int]] = []
        offset = 0
        for size, use_adam in zip(self.splits, self.adamw):
            lo, hi = max(offset, start), min(offset + size, start + rows)
            if use_adam and lo < hi:
                lo, hi = lo - start, hi - start
                if ranges and ranges[-1][1] == lo:
                    ranges[-1] = (ranges[-1][0], hi)
                else:
                    ranges.append((lo, hi))
            offset += size
        return tuple(ranges)

    def localize(self, start: int, rows: int) -> tuple['MuonProjectionLayout', bool]:
        """Intersect a layout with a row shard and report complete Muon matrices."""
        splits, adamw, complete = [], [], True
        offset = 0
        for size, use_adam in zip(self.splits, self.adamw):
            width = max(0, min(offset + size, start + rows) - max(offset, start))
            if width:
                splits.append(width)
                adamw.append(use_adam)
                complete &= use_adam or width == size
            offset += size
        if sum(splits) != rows:
            raise ValueError("Projection shard falls outside the logical layout")
        return type(self)(tuple(splits), tuple(adamw), self.tp_local, self.tp_partitioned), complete

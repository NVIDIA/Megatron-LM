# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Memoized exact pack costs and physical-capacity constraints."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PackCostModel:
    """Memoize physical pack costs and enforce capacity and sequence-count limits."""

    config: object  # dcp_scheduler SchedulerConfig (alignments, tokens_per_rank, ...)
    forward: object  # callable(lengths, cp, ranks=) -> ms per layer
    backward: object | None = None
    program_costs: object | None = None  # callable(lengths, cp, ranks=) -> {name: ms}
    placement_period: int | None = (
        None  # domain size whose crossing pattern changes costs (8 on H100 NVL8)
    )
    max_group_seqs: int | None = None
    stats: dict = field(default_factory=lambda: dict(calls=0, hits=0, errors=0, signature_hits=0))
    _memo: dict = field(default_factory=dict, repr=False)
    _min_cp: dict = field(default_factory=dict, repr=False)
    _stored: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ capacity
    @property
    def tokens_per_rank(self):
        """Return the physical token capacity of one rank."""
        return self.config.tokens_per_rank

    @property
    def allowed(self):
        """Return the CP widths available in the runtime geometry."""
        return tuple(self.config.allowed_cp_sizes)

    def stored(self, length, cp):
        """Return the cached physically padded token count for one sequence."""
        key = (length, cp)
        value = self._stored.get(key)
        if value is None:
            value = self._stored[key] = self.config.stored_length(int(length), int(cp))
        return value

    def capacity(self, cp):
        """Return the physical token capacity across a CP group."""
        return cp * self.tokens_per_rank

    def stored_total(self, lengths, cp):
        """Sum physically padded token counts for a proposed pack."""
        return sum(self.stored(n, cp) for n in lengths)

    def fits(self, lengths, cp):
        """Check the pack against sequence-count and physical-token limits."""
        if self.max_group_seqs is not None and len(lengths) > self.max_group_seqs:
            return False
        return self.stored_total(lengths, cp) <= self.capacity(cp)

    def min_cp(self, length):
        """Find the first allowed CP width that can store the sequence."""
        value = self._min_cp.get(length)
        if value is None:
            for cp in self.allowed:
                if self.stored(length, cp) <= self.capacity(cp):
                    value = cp
                    break
            if value is None:
                raise ValueError(f'sequence of {length} tokens does not fit any allowed CP group')
            self._min_cp[length] = value
        return value

    # --------------------------------------------------------------- exact
    def pack(self, lengths, cp, start):
        """Exact (forward_ms, backward_ms, program_costs_or_None); None if the predictor rejects."""
        lengths = tuple(int(n) for n in lengths)
        placement = 0 if self.placement_period is None or cp == 1 else start % self.placement_period
        key = (lengths, cp, placement)
        if key in self._memo:
            self.stats['hits'] += 1
            return self._memo[key]
        self.stats['calls'] += 1
        ranks = list(range(start, start + cp))
        try:
            program = (
                dict(self.program_costs(lengths, cp, ranks=ranks))
                if self.program_costs is not None
                else None
            )
            fwd = float(self.forward(lengths, cp, ranks=ranks))
            bwd = (
                float(self.backward(lengths, cp, ranks=ranks)) if self.backward is not None else 0.0
            )
            value = (fwd, bwd, program)
        except ValueError:
            self.stats['errors'] += 1
            value = None
        self._memo[key] = value
        return value

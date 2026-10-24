# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Memoized exact pack costs and physical-capacity constraints."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PackCostModel:
    config: object  # dcp_scheduler SchedulerConfig (alignments, tokens_per_rank, ...)
    forward: object  # callable(lengths, cp, ranks=) -> ms per layer
    backward: object | None = None
    program_costs: object | None = None  # callable(lengths, cp, ranks=) -> {name: ms}
    placement_period: int | None = (
        None  # domain size whose crossing pattern changes costs (8 on H100 NVL8)
    )
    signature_estimates: bool = True  # reuse forward values across packs with the same signature
    stats: dict = field(default_factory=lambda: dict(calls=0, hits=0, errors=0, signature_hits=0))
    _memo: dict = field(default_factory=dict, repr=False)
    _signature: dict = field(default_factory=dict, repr=False)
    _estimates: dict = field(default_factory=dict, repr=False)
    _min_cp: dict = field(default_factory=dict, repr=False)
    _stored: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ capacity
    @property
    def world_size(self):
        return self.config.world_size

    @property
    def tokens_per_rank(self):
        return self.config.tokens_per_rank

    @property
    def allowed(self):
        return tuple(self.config.allowed_cp_sizes)

    def stored(self, length, cp):
        key = (length, cp)
        value = self._stored.get(key)
        if value is None:
            value = self._stored[key] = self.config.stored_length(int(length), int(cp))
        return value

    def capacity(self, cp):
        return cp * self.tokens_per_rank

    def stored_total(self, lengths, cp):
        return sum(self.stored(n, cp) for n in lengths)

    def fits(self, lengths, cp):
        return self.stored_total(lengths, cp) <= self.capacity(cp)

    def min_cp(self, length):
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

    # --------------------------------------------------------------- keys
    def _placement_class(self, cp, start):
        if self.placement_period is None or cp == 1:
            return 0
        return start % self.placement_period

    def _key(self, lengths, cp, start):
        return (lengths, cp, self._placement_class(cp, start))

    def _sig(self, lengths, cp, start):
        stored = self.stored_total(lengths, cp)
        capacity = self.capacity(cp)
        util = round(sum(lengths) / capacity, 2) if len(lengths) == 1 else None
        return (cp, len(lengths), self._placement_class(cp, start), stored < capacity, util)

    # --------------------------------------------------------------- exact
    def pack(self, lengths, cp, start):
        """Exact (forward_ms, backward_ms, program_costs_or_None); None if the predictor rejects."""
        lengths = tuple(int(n) for n in lengths)
        key = self._key(lengths, cp, start)
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
            if program is None:
                self._signature.setdefault(self._sig(lengths, cp, start), fwd)
        except ValueError:
            self.stats['errors'] += 1
            value = None
        self._memo[key] = value
        return value

    def layer_cost(self, lengths, cp, start):
        """Exact forward + backward attention ms per layer (None if rejected)."""
        value = self.pack(lengths, cp, start)
        if value is None:
            return None
        return value[0] + value[1]

    # ------------------------------------------------------------ estimate
    def estimate(self, lengths, cp, start):
        """Cheap forward+backward ms per layer for search decisions (None if infeasible)."""
        lengths = tuple(int(n) for n in lengths)
        key = self._key(lengths, cp, start)
        if key in self._estimates:
            return self._estimates[key]
        value = self._estimate(lengths, cp, start, key)
        self._estimates[key] = value
        return value

    def _estimate(self, lengths, cp, start, key):
        if not self.fits(lengths, cp):
            return None
        exact = self._memo.get(key)
        if exact is not None:
            return exact[0] + exact[1]
        if key in self._memo:  # exact call previously failed
            return None
        if self.program_costs is not None or not self.signature_estimates:
            return self.layer_cost(lengths, cp, start)
        fwd = self._signature.get(self._sig(lengths, cp, start))
        if fwd is None:
            return self.layer_cost(lengths, cp, start)
        self.stats['signature_hits'] += 1
        ranks = list(range(start, start + cp))
        try:
            bwd = (
                float(self.backward(lengths, cp, ranks=ranks)) if self.backward is not None else 0.0
            )
        except ValueError:
            return None
        return fwd + bwd

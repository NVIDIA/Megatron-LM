# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""What the autotune interception should do, as one object.

Triton picks a kernel config by benchmarking its candidates at first call, so
the winner depends on wall-clock timings. Two consequences follow, and only one
of them is about determinism:

- Two identical runs can select different tile shapes, which changes the
  floating-point reduction order and therefore the result.
- Every cold process pays for the benchmark, and repeated benchmarks of the same
  workload disagree with each other, so measurements are noisy.

Pinning the choice to a pure function of the candidate list fixes both. This
module holds the intent; :mod:`megatron.core.tuning.interception` carries it out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import torch

_deterministic_override = None


def use_deterministic_mode() -> bool:
    """Whether deterministic behaviour is requested for kernel selection."""
    if _deterministic_override is not None:
        return _deterministic_override
    return torch.are_deterministic_algorithms_enabled()


def set_deterministic_mode(value):
    """Override :func:`use_deterministic_mode` for the current process."""
    global _deterministic_override
    _deterministic_override = value


@dataclass(frozen=True)
class AutotunePolicy:
    """Immutable configuration inputs for choosing Triton kernel launch parameters.

    This object contains settings only. Installation resolves defaults into a
    separate policy value; loaded tables, selected configs, and recorded winners
    belong to the runtime adapter in :mod:`megatron.core.tuning.interception`.

    Attributes:
        mode: ``None`` derives the mode from recording and determinism settings.
            ``auto`` leaves Triton alone. ``pinned`` replaces the candidate
            list with one entry chosen without measuring anything. ``record``
            lets Triton benchmark as usual and captures the winners, so a table
            can be built; a recording run is deliberately not reproducible.
        modules: Module prefixes to act on. Kernels outside them keep their
            tuned performance.
        table_path: Directories searched for tuned tables before the packaged
            defaults.
        record_path: Where a ``record`` run writes its per-rank captures.
        on_miss: What to do when no table entry matches. ``min_cost`` is still
            deterministic, just possibly slower; ``error`` refuses to guess.
        block_sizes: Explicit ``BLOCK_*`` kernel arguments to match when the
            table misses. Accepts a mapping or pairs; stored as immutable pairs.
        verify_every: Cross-rank agreement check cadence, in steps. 0 disables.
            Honoured by the training loop through
            :func:`megatron.core.tuning.maybe_verify_choices`.
        enumerate_autotuners: Report every multi-config autotuner reached,
            without changing what is chosen.
        chaos: Make each rank pick a different config on purpose. A positive
            control for divergence detectors; never for real runs.
    """

    mode: Literal["auto", "pinned", "record"] | None = None
    modules: tuple[str, ...] = ("mamba_ssm", "transformer_engine", "megatron.core.ssm.ops")
    table_path: tuple[str, ...] = ()
    record_path: str | None = None
    on_miss: Literal["min_cost", "error"] = "min_cost"
    block_sizes: tuple[tuple[str, int], ...] = ()
    verify_every: int = 0
    verify_strict: bool = False
    enumerate_autotuners: bool = False
    chaos: bool = False

    def __post_init__(self):
        if self.mode not in (None, "auto", "pinned", "record"):
            raise ValueError(f"Unknown autotune mode: {self.mode!r}")
        if self.on_miss not in ("min_cost", "error"):
            raise ValueError(f"Unknown autotune miss policy: {self.on_miss!r}")
        if self.verify_every < 0:
            raise ValueError("Autotune verification cadence must be nonnegative")
        if self.mode == "record" and not self.record_path:
            raise ValueError("Record mode requires record_path")
        if isinstance(self.modules, str) or isinstance(self.table_path, str):
            raise TypeError("modules and table_path must be sequences, not strings")
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "table_path", tuple(str(path) for path in self.table_path))
        if self.record_path is not None:
            object.__setattr__(self, "record_path", str(self.record_path))
        pairs = (
            self.block_sizes.items() if isinstance(self.block_sizes, Mapping) else self.block_sizes
        )
        pairs = tuple(pairs)
        for name, value in pairs:
            if not isinstance(name, str) or not name.startswith("BLOCK"):
                raise ValueError(f"Invalid block-size argument: {name!r}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"Block size {name!r} must be a positive integer")
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Block-size arguments must be unique")
        object.__setattr__(
            self, "block_sizes", tuple(sorted((name, value) for name, value in pairs))
        )

    def resolve(self, *, deterministic: bool = False) -> "AutotunePolicy":
        """Resolve an omitted mode, leaving explicit configuration unchanged.

        A recording path implies ``record``. Otherwise, model/PyTorch
        deterministic mode implies ``pinned``; ordinary execution uses ``auto``.
        """
        if self.mode is not None:
            return self
        if self.record_path:
            mode = "record"
        elif deterministic or use_deterministic_mode():
            mode = "pinned"
        else:
            mode = "auto"
        return replace(self, mode=mode)

    @property
    def intercepts(self) -> bool:
        """Whether this policy needs the autotuner patched at all.

        Enumeration and the agreement check both read what Triton chose, which
        only the interception records, so either one needs the patch even in
        ``auto`` mode — where it observes the timed choice without changing it.
        """
        return self.mode not in (None, "auto") or self.enumerate_autotuners or self.verify_every > 0

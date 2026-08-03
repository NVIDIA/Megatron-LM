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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

_deterministic_override = None


def use_deterministic_mode() -> bool:
    """Whether deterministic behaviour is requested for kernel selection."""
    if _deterministic_override is not None:
        return _deterministic_override
    env = os.environ.get('MAMBA_DETERMINISTIC')
    if env:
        return env[0] == '1'
    return torch.are_deterministic_algorithms_enabled()


def set_deterministic_mode(value):
    """Override :func:`use_deterministic_mode` for the current process."""
    global _deterministic_override
    _deterministic_override = value


def _env(*names: str, default: str = "") -> str:
    """First set value among ``names``, so old variable names keep working."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class AutotunePolicy:
    """How to choose Triton kernel configs.

    Attributes:
        mode: ``auto`` leaves Triton alone. ``pinned`` replaces the candidate
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
        verify_every: Cross-rank agreement check cadence, in steps. 0 disables.
        enumerate_autotuners: Report every multi-config autotuner reached,
            without changing what is chosen.
        chaos: Make each rank pick a different config on purpose. A positive
            control for divergence detectors; never for real runs.
    """

    mode: Literal["auto", "pinned", "record"] = "auto"
    modules: tuple[str, ...] = ("mamba_ssm", "transformer_engine")
    table_path: tuple[Path, ...] = ()
    record_path: str | None = None
    on_miss: Literal["min_cost", "error"] = "min_cost"
    verify_every: int = 0
    verify_strict: bool = False
    enumerate_autotuners: bool = False
    chaos: bool = False
    _explicit_mode: bool = field(default=False, repr=False)

    @classmethod
    def from_env(cls) -> "AutotunePolicy":
        """Build a policy from the environment.

        Deterministic mode implies ``pinned``; an explicit ``MCORE_AUTOTUNE_MODE``
        wins over that, so a determinism run can still be put into ``record``.
        """
        record_path = _env("MCORE_AUTOTUNE_RECORD", "MCORE_DET_TUNE_RECORD") or None
        explicit = _env("MCORE_AUTOTUNE_MODE")
        if explicit:
            mode = explicit
        elif record_path:
            mode = "record"
        elif use_deterministic_mode():
            mode = "pinned"
        else:
            mode = "auto"

        modules = tuple(
            prefix.strip()
            for prefix in _env(
                "MCORE_AUTOTUNE_MODULES",
                "DET_AUTOTUNE_PIN_MODULES",
                default="mamba_ssm,transformer_engine",
            ).split(",")
            if prefix.strip()
        )
        table_path = tuple(
            Path(p) for p in _env("MCORE_AUTOTUNE_TABLE_PATH").split(os.pathsep) if p
        )
        return cls(
            mode=mode,
            modules=modules,
            table_path=table_path,
            record_path=record_path,
            on_miss=_env("MCORE_AUTOTUNE_ON_MISS", default="min_cost"),
            verify_every=int(_env("MCORE_AUTOTUNE_VERIFY", "DET_AUTOTUNE_VERIFY", default="0")),
            verify_strict=_env("MCORE_AUTOTUNE_VERIFY_STRICT", "DET_AUTOTUNE_VERIFY_STRICT") == "1",
            enumerate_autotuners=_env("MCORE_AUTOTUNE_ENUMERATE", "DET_AUTOTUNE_ENUMERATE") == "1",
            chaos=_env("MCORE_AUTOTUNE_CHAOS", "DET_AUTOTUNE_CHAOS") == "1",
            _explicit_mode=bool(explicit),
        )

    @property
    def intercepts(self) -> bool:
        """Whether this policy needs the autotuner patched at all."""
        return self.mode != "auto" or self.enumerate_autotuners

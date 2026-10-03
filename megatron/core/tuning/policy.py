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
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Literal

import torch

_deterministic_override = None

DEFAULT_MODULES = ("mamba_ssm", "transformer_engine", "megatron.core")

# Kernels whose outputs do not depend on the launch configuration, so a timed choice
# cannot change results and pinning would only cost throughput. Only pure data movement
# qualifies by default: even elementwise arithmetic can round differently between
# configs, because the layout decides whether values are computed packed or promoted
# and whether multiplies and adds are fused. A GPU test forces every candidate of each
# entry and requires bit-identical outputs.
DEFAULT_CONFIG_INVARIANT = tuple(
    f"{module}.{kernel}"
    for module in (
        "transformer_engine.common.triton.permutation",
        "transformer_engine.pytorch.triton.permutation",
    )
    for kernel in ("_permute_kernel", "_unpermute_kernel", "_sort_chunks_by_map_kernel")
)


def use_deterministic_mode() -> bool:
    """Whether deterministic behaviour is requested for kernel selection."""
    if _deterministic_override is not None:
        return _deterministic_override
    return torch.are_deterministic_algorithms_enabled()


def set_deterministic_mode(value):
    """Override :func:`use_deterministic_mode` for the current process."""
    global _deterministic_override
    _deterministic_override = value


def _string_tuple(name: str, value) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"AutotunePolicy.{name} must be a sequence of strings, got {value!r}")
    items = tuple(os.fspath(item) if isinstance(item, os.PathLike) else item for item in value)
    for item in items:
        if not isinstance(item, str) or not item:
            raise TypeError(f"AutotunePolicy.{name} must contain non-empty strings, got {item!r}")
    return items


@dataclass(frozen=True)
class AutotunePolicy:
    """Immutable configuration inputs for choosing Triton kernel launch parameters.

    This object contains settings only. Installation resolves defaults into a
    separate policy value; loaded tables, selected configs, and recorded winners
    belong to the runtime adapter in :mod:`megatron.core.tuning.interception`.
    See ``megatron/core/tuning/README.md`` for modes, scope, tables and
    configuration.
    """

    mode: Literal["auto", "pinned", "record"] | None = None
    """``None`` derives the mode from recording and determinism settings. ``auto``
    leaves Triton alone. ``pinned`` replaces the candidate list with one entry chosen
    without measuring anything. ``record`` lets Triton benchmark as usual and captures
    the winners, so a table can be built; a recording run is deliberately not
    reproducible."""

    modules: tuple[str, ...] = DEFAULT_MODULES
    """Module prefixes to act on, matched at package boundaries. Kernels outside
    them keep Triton's timed choice."""

    config_invariant: tuple[str, ...] = DEFAULT_CONFIG_INVARIANT
    """Qualified kernel names (``module.function``) whose outputs do not depend on
    the launch configuration. They keep Triton's timed choice even in ``pinned`` mode,
    since pinning them costs throughput without changing any result."""

    table_path: tuple[str, ...] = ()
    """Directories searched for tuned tables before the packaged defaults. ``~`` is
    expanded."""

    record_path: str | None = None
    """File prefix for a ``record`` run's per-rank captures, written as
    ``<record_path>.rank<N>.json``. ``~`` is expanded."""

    on_miss: Literal["min_cost", "error"] = "min_cost"
    """What to do when no table entry matches. ``min_cost`` is still deterministic,
    just possibly slower; ``error`` refuses to guess."""

    block_sizes: tuple[tuple[str, int], ...] = ()
    """Explicit ``BLOCK_*`` kernel arguments to match when the table misses. Accepts
    a mapping or pairs; stored as sorted immutable pairs."""

    verify_every: int = 0
    """Cross-rank agreement check cadence, in training steps. 0 disables. Honoured
    by the training loop through :func:`megatron.core.tuning.maybe_verify_choices`."""

    verify_strict: bool = False
    """Raise instead of logging a warning when the agreement check finds ranks
    running different configurations for the same kernel and shape."""

    enumerate_autotuners: bool = False
    """Log every multi-config autotuner reached, with whether it is pinned, without
    changing what is chosen."""

    chaos: bool = False
    """Make each rank pick a different config on purpose. A positive control for
    divergence detectors; never for real runs."""

    def __post_init__(self):
        if self.mode not in (None, "auto", "pinned", "record"):
            raise ValueError(f"Unknown autotune mode: {self.mode!r}")
        if self.on_miss not in ("min_cost", "error"):
            raise ValueError(f"Unknown autotune miss policy: {self.on_miss!r}")
        for name in ("verify_strict", "enumerate_autotuners", "chaos"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"AutotunePolicy.{name} must be a bool, got {value!r}")
        if type(self.verify_every) is not int:
            raise TypeError(
                f"AutotunePolicy.verify_every must be an int, got {self.verify_every!r}"
            )
        if self.verify_every < 0:
            raise ValueError("Autotune verification cadence must be nonnegative")
        for name in ("modules", "config_invariant", "table_path"):
            object.__setattr__(self, name, _string_tuple(name, getattr(self, name)))
        object.__setattr__(
            self, "table_path", tuple(os.path.expanduser(path) for path in self.table_path)
        )
        if self.record_path is not None:
            record_path = os.fspath(self.record_path)
            if not isinstance(record_path, str) or not record_path:
                raise TypeError(
                    f"AutotunePolicy.record_path must be a path prefix, got {self.record_path!r}"
                )
            if record_path.endswith(("/", os.sep)):
                raise ValueError(
                    f"AutotunePolicy.record_path is a file prefix, not a directory: "
                    f"{record_path!r}; captures are written as <prefix>.rank<N>.json"
                )
            object.__setattr__(self, "record_path", os.path.expanduser(record_path))
        if self.mode == "record" and not self.record_path:
            raise ValueError("Record mode requires record_path")
        if self.block_sizes is None:
            raise TypeError("AutotunePolicy.block_sizes must be a mapping or pairs, got None")
        pairs = (
            self.block_sizes.items() if isinstance(self.block_sizes, Mapping) else self.block_sizes
        )
        pairs = tuple(tuple(pair) for pair in pairs)
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Invalid block-size pair: {pair!r}")
            name, value = pair
            if not isinstance(name, str) or not name.startswith("BLOCK") or not name.isidentifier():
                raise ValueError(f"Invalid block-size argument: {name!r}")
            if type(value) is not int or value <= 0:
                raise ValueError(f"Block size {name!r} must be a positive integer")
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Block-size arguments must be unique")
        object.__setattr__(
            self, "block_sizes", tuple(sorted((name, value) for name, value in pairs))
        )

    @classmethod
    def from_mapping(cls, values: Mapping) -> "AutotunePolicy":
        """Build a policy from a mapping, such as a parsed YAML section.

        Keys set to ``None`` fall back to their defaults, as an omitted key would.
        Unknown keys raise, so a misspelled option cannot be silently ignored.
        """
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise TypeError(f"Unknown AutotunePolicy option(s): {', '.join(unknown)}")
        return cls(**{key: value for key, value in values.items() if value is not None})

    def resolve(self, *, deterministic: bool = False) -> "AutotunePolicy":
        """Resolve an omitted mode, leaving explicit configuration unchanged.

        A recording path implies ``record``. Otherwise, ``deterministic`` or
        PyTorch's deterministic flag implies ``pinned``; ordinary execution uses
        ``auto``.
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


def coerce_policy(value) -> AutotunePolicy | None:
    """Return ``value`` as an :class:`AutotunePolicy`, converting a mapping.

    ``None`` stays ``None``. Anything else is rejected before it can reach the
    process-wide adapter.
    """
    if value is None or isinstance(value, AutotunePolicy):
        return value
    if isinstance(value, Mapping):
        return AutotunePolicy.from_mapping(value)
    raise TypeError(
        f"An autotune policy must be an AutotunePolicy or a mapping, got {type(value).__name__}"
    )

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lightweight kernel declarations and construction-time validation, not dispatch.

Importing declarations never imports their optional dependencies. These checks
describe a local implementation, not the surrounding model or distributed run.
Input-dependent restrictions remain in the family's existing call contract.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from importlib import import_module, metadata
from operator import attrgetter
from typing import Callable, Collection

from packaging.requirements import Requirement


class Determinism(str, Enum):
    """Bit-exact repeatability within the explicitly documented execution scope."""

    DETERMINISTIC = "deterministic"
    NONDETERMINISTIC = "nondeterministic"
    UNKNOWN = "unknown"


class DeterminismPolicy(str, Enum):
    """IGNORE preserves ordinary execution; WARN allows unknowns with a warning.

    WARN and ERROR both reject known nondeterminism. ERROR also rejects UNKNOWN.
    Existing model-specific deterministic-mode guards remain authoritative.
    """

    IGNORE = "ignore"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Dependency:
    """An import requirement, optionally versioned or conditional on a feature.

    ``requirement`` is a distribution name with an optional PEP 440 specifier;
    ``module`` is its actual Python import path. ``symbols`` are required exports,
    optionally dotted for lazy namespaces such as ``cudnn.DSA``.
    ``feature`` limits the check to a caller-selected feature such as qk_l2norm.
    Source installs without distribution metadata work for unversioned imports;
    version constraints require installed metadata rather than guessing a version.
    """

    requirement: str
    module: str
    symbols: tuple[str, ...] = ()
    feature: str | None = None

    def __post_init__(self) -> None:
        requirement = Requirement(self.requirement)
        if requirement.extras or requirement.marker or requirement.url:
            raise ValueError("Kernel dependencies accept a name and version specifier only.")
        if not self.module or self.feature == "":
            raise ValueError("Dependency module and feature names must not be empty.")
        if not isinstance(self.symbols, tuple) or not all(
            isinstance(symbol, str) and symbol for symbol in self.symbols
        ):
            raise ValueError("Dependency exports must be a tuple of nonempty names.")

    def validate(self, kernel_name: str) -> None:
        """Check the actual import and exports, then any declared version bound."""
        requirement = Requirement(self.requirement)
        description = f"Kernel {kernel_name} requires {self.requirement} ({self.module})"
        try:
            module = import_module(self.module)
            for symbol in self.symbols:
                try:
                    value = attrgetter(symbol)(module)
                except AttributeError as exc:
                    raise ImportError(f"missing export {self.module}.{symbol}") from exc
                if value is None:
                    raise ImportError(f"missing export {self.module}.{symbol}")
        except (ImportError, OSError) as exc:
            raise ImportError(f"{description}: {exc}") from exc
        if requirement.specifier:
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError as exc:
                raise ImportError(f"{description}: cannot verify the installed version.") from exc
            if not requirement.specifier.contains(installed, prereleases=True):
                raise ImportError(f"{description}; found version {installed}.")


@dataclass(frozen=True)
class DeterminismResult:
    """A scoped assessment with a reason, not a cross-platform guarantee."""

    status: Determinism
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, Determinism) or not self.reason.strip():
            raise ValueError("Determinism requires an explicit status and a nonempty reason.")


@dataclass(frozen=True)
class KernelMetadata:
    """Unified declaration for a selectable or phase-specific kernel entry point.

    ``contract`` points to the family's documented call surface. ``determinism``
    states the conservative default; an optional checker assesses the current
    environment at construction. Neither is inferred from the implementation's
    language or backend name. No callable is wrapped or registered by this type.
    """

    name: str
    requires: tuple[Dependency, ...]
    determinism: DeterminismResult
    contract: str
    determinism_check: Callable[[], DeterminismResult] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.contract.strip():
            raise ValueError("Kernel metadata requires a name and a family contract.")
        if not isinstance(self.determinism, DeterminismResult):
            raise ValueError("Kernel metadata requires an explicit determinism assessment.")
        if not isinstance(self.requires, tuple) or not all(
            isinstance(dependency, Dependency) for dependency in self.requires
        ):
            raise ValueError("Kernel dependencies must be a tuple of Dependency declarations.")


def validate_kernel(
    kernel: KernelMetadata,
    *,
    determinism: DeterminismPolicy = DeterminismPolicy.IGNORE,
    features: Collection[str] = (),
) -> None:
    """Validate only the selected implementation without changing its callable.

    Call once during construction. ``features`` enables conditional dependencies;
    callers must include the features they intend to use. Shapes, runtime dtypes,
    packed inputs and device-specific execution checks remain in the kernel.
    Strict ERROR policy is opt-in, independent of existing configuration flags.
    """
    if not isinstance(determinism, DeterminismPolicy):
        raise ValueError("Expected an explicit DeterminismPolicy.")
    for dependency in kernel.requires:
        if dependency.feature is None or dependency.feature in features:
            dependency.validate(kernel.name)
    validate_determinism(kernel, determinism)


def validate_determinism(kernel: KernelMetadata, policy: DeterminismPolicy) -> None:
    """Apply a caller's determinism policy without repeating dependency imports."""
    if not isinstance(policy, DeterminismPolicy):
        raise ValueError("Expected an explicit DeterminismPolicy.")
    if policy is DeterminismPolicy.IGNORE:
        return
    result = kernel.determinism_check() if kernel.determinism_check else kernel.determinism
    if not isinstance(result, DeterminismResult):
        raise TypeError(f"Kernel {kernel.name} returned an invalid determinism assessment.")
    if result.status is Determinism.DETERMINISTIC:
        return
    message = f"Kernel {kernel.name} determinism is {result.status.value}: {result.reason}"
    if result.status is Determinism.NONDETERMINISTIC or policy is DeterminismPolicy.ERROR:
        raise RuntimeError(message)
    warnings.warn(message, UserWarning, stacklevel=2)


def validate_kernels(
    kernels: Collection[KernelMetadata],
    *,
    determinism: DeterminismPolicy = DeterminismPolicy.IGNORE,
    features: Collection[str] = (),
) -> None:
    """Validate an owner's selected entry points once during initialization.

    Pass the actual selection, not a family's inventory of available backends.
    No global registry or cached availability result is created.
    """
    for kernel in kernels:
        validate_kernel(kernel, determinism=determinism, features=features)

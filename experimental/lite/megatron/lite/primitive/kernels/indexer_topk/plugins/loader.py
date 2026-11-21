# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit-path loaders of the external indexer top-k plugins.

:func:`load_litetopk_plugin` loads a LiteTopK plugin (ABI v1) from the directory named in its
:class:`LiteTopKPluginConfig`:

1. check that the adapter, its CUDA sources, DeepGEMM and the prebuilt extension exist, listing
   every missing item at once;
2. compute the source id of the CUDA sources and the SHA-256 of the adapter and the prebuilt
   extension, and compare them with the pins before any plugin code runs;
3. render the settings into the process environment (:mod:`.env`);
4. import the adapter under a private module name, validate its ABI and ``plugin_info()``, check
   the configuration it snapshotted, and only then load its CUDA extension.

A failed load leaves nothing behind: the module is removed from ``sys.modules`` and the keys
the load added to the environment are removed again. A successful load is cached for the
process lifetime by source id, adapter hash and settings; one source id can only be loaded with
one settings profile, because its extension reads the environment of the process.

:func:`load_exact_topk` imports the exact-tie top-k package the same way, as a private package.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from megatron.lite.primitive.kernels.indexer_topk.config import (
    EXACT_TOPK_FILES,
    ExactTopKConfig,
    IndexerTopKPluginError,
    LiteTopKPluginConfig,
    LiteTopKPluginSettings,
)
from megatron.lite.primitive.kernels.indexer_topk.plugins import env
from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import (
    LITETOPK_KERNEL_FILES,
    LiteTopKModuleV1,
    PluginInfo,
    parse_plugin_info,
    validate_litetopk_module,
)

__all__ = [
    "LoadedExactTopK",
    "LoadedLiteTopKPlugin",
    "compute_litetopk_source_id",
    "load_exact_topk",
    "load_litetopk_plugin",
    "loaded_litetopk_plugins",
]

logger = logging.getLogger(__name__)

_ADAPTER_FILE = "litetopk.py"
_KERNEL_DIR = "litetopk_kernels"
_RUNTIME_PACKAGES = ("deep_gemm",)
_EXACT_TOPK_MODULE = "indexer_top_k_decode_varlen"
_MAX_EXACT_TOPK_KEYS = 1 << 20
_MAX_EXACT_TOPK = 2048

_LOCK = threading.RLock()
_LOADED: dict[tuple[str, str, LiteTopKPluginSettings], LoadedLiteTopKPlugin] = {}
_EXACT_LOADED: dict[str, LoadedExactTopK] = {}


@dataclass(frozen=True, eq=False)
class LoadedLiteTopKPlugin:
    """A validated LiteTopK plugin whose CUDA extension is loaded.

    Attributes:
        module: The imported adapter module.
        info: Its parsed ``plugin_info()``.
        source: The resolved plugin directory.
        adapter_sha256: The SHA-256 of the adapter module file.
        settings: The import-time settings the plugin was loaded with.
        rendered_env: The environment keys rendered for the plugin.
        diagnostic_env: The diagnostic keys passed through from the process.
        launch_time_env: The launch-time keys of the plugin and their values at load (None when
            unset); :meth:`check_launch_time_env` compares against them.
        prebuilt_extension: The resolved prebuilt extension, or None for a JIT build.
        prebuilt_extension_sha256: The SHA-256 of the prebuilt extension.
        build_dir: The resolved JIT build directory, or None for the plugin's default.
        deepgemm_include_dir: The resolved JIT header directory, or None for the default.
        pinned: The pins the configuration provided, among ``expected_source_id``,
            ``expected_adapter_sha256`` and ``prebuilt_extension_sha256``.
    """

    module: LiteTopKModuleV1
    info: PluginInfo
    source: str
    adapter_sha256: str
    settings: LiteTopKPluginSettings
    rendered_env: dict[str, str]
    diagnostic_env: dict[str, str]
    launch_time_env: dict[str, str | None]
    prebuilt_extension: str | None
    prebuilt_extension_sha256: str | None
    build_dir: str | None
    deepgemm_include_dir: str | None
    pinned: frozenset[str]

    @property
    def source_id(self) -> str:
        """The 12-hex id of the plugin's CUDA sources."""
        return self.info.source_id

    @property
    def owner(self) -> str:
        """How error messages name this plugin: its source id and adapter hash prefix."""
        return f"{self.info.source_id} (adapter {self.adapter_sha256[:8]})"

    def check_launch_time_env(self) -> None:
        """Check that the keys the CUDA extension reads on every launch are unchanged.

        Selectors call this before every selection; it only compares a few environment values.

        Raises:
            IndexerTopKRuntimeError: If a launch-time key changed after the load.
        """
        env.check_launch_time_env(self.launch_time_env, owner=self.owner)

    def provenance(self) -> dict[str, Any]:
        """Return a JSON-ready record of what was loaded, for result files."""
        return {
            "source": self.source,
            "module": self.module.__name__,
            "source_id": self.source_id,
            "adapter_sha256": self.adapter_sha256,
            "prebuilt_extension": self.prebuilt_extension,
            "prebuilt_extension_sha256": self.prebuilt_extension_sha256,
            "build_dir": self.build_dir,
            "deepgemm_include_dir": self.deepgemm_include_dir,
            "pinned": sorted(self.pinned),
            "rendered_env": dict(self.rendered_env),
            "diagnostic_env": dict(self.diagnostic_env),
            "launch_time_env": dict(self.launch_time_env),
            "plugin_info": self.info.as_dict(),
        }

    def __deepcopy__(self, memo: dict) -> LoadedLiteTopKPlugin:
        """Return self: a loaded plugin is process state, shared by every copy of its users."""
        return self


@dataclass(frozen=True, eq=False)
class LoadedExactTopK:
    """The imported exact-tie top-k package.

    Calling it selects, for every row, the ``top_k`` highest scores among the row's first
    ``lengths[row]`` columns, ordering equal scores by ascending column id.

    Attributes:
        source: The resolved package directory.
        package: The private package name it is imported under.
        file_sha256: The SHA-256 of every package file.
        wrapper: The package's ``cute_dsl_topk_wrapper``.
    """

    source: str
    package: str
    file_sha256: dict[str, str]
    wrapper: Callable[..., Any]

    def __deepcopy__(self, memo: dict) -> LoadedExactTopK:
        """Return self: an imported package is process state, shared by every copy of its users."""
        return self

    def __call__(
        self, scores: Tensor, lengths: Tensor, *, top_k: int, return_values: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Select the ``top_k`` best columns of every row without sorting them.

        No device-to-host synchronization: the arguments are checked from their metadata only.
        The package allocates its outputs and scratch on the current stream of the scores'
        device.

        Args:
            scores: CUDA float32 ``[rows, keys]`` with unit column stride and ``keys <= 2**20``;
                rows may be padded (row stride >= keys).
            lengths: CUDA int32 ``[rows]``, the valid columns of each row, within ``[0, keys]``.
            top_k: The number of columns per row, at most 2048.
            return_values: Also return the selected scores.

        Returns:
            ``(indices, values)``: int32 ``[rows, top_k]`` column ids, and the float32 scores or
            None.

        Raises:
            ValueError: If an argument or a returned tensor has unexpected metadata.
        """
        if (
            not isinstance(scores, Tensor)
            or not isinstance(lengths, Tensor)
            or scores.ndim != 2
            or scores.dtype != torch.float32
            or scores.shape[0] <= 0
            or not 0 < scores.shape[1] <= _MAX_EXACT_TOPK_KEYS
            or not scores.is_cuda
            or scores.stride(1) != 1
            or scores.stride(0) < scores.shape[1]
            or lengths.shape != (scores.shape[0],)
            or lengths.dtype != torch.int32
            or lengths.device != scores.device
            or not lengths.is_contiguous()
            or type(top_k) is not int
            or not 0 < top_k <= _MAX_EXACT_TOPK
            or type(return_values) is not bool
        ):
            raise ValueError(
                "exact top-k expects CUDA float32 scores [rows, keys <= 2**20] with unit column "
                "stride, int32 row lengths on the same device and 0 < top_k <= 2048"
            )
        with torch.cuda.device(scores.device):
            stream = torch.cuda.current_stream(scores.device)
            scores.record_stream(stream)
            lengths.record_stream(stream)
            indices, values = self.wrapper(scores, lengths, top_k, 1, return_val=return_values)
            # Record the outputs before checking them, so that a rejected result cannot free
            # storage that the package still writes asynchronously.
            for output in (indices, values):
                if isinstance(output, Tensor) and output.is_cuda:
                    output.record_stream(stream)
            if (
                not isinstance(indices, Tensor)
                or indices.shape != (scores.shape[0], top_k)
                or indices.dtype != torch.int32
                or indices.device != scores.device
                or (not return_values and values is not None)
                or (
                    return_values
                    and (
                        not isinstance(values, Tensor)
                        or values.shape != indices.shape
                        or values.dtype != torch.float32
                        or values.device != scores.device
                    )
                )
            ):
                raise ValueError("exact top-k package returned outputs with unexpected metadata")
        return indices, values


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_digest(directory: Path, names: tuple[str, ...]) -> str:
    """SHA-256 over every file name followed by the file's bytes, in the given order."""
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        with (directory / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def compute_litetopk_source_id(source: str | Path) -> str:
    """Return the source id of a LiteTopK plugin directory.

    The id is the plugin's own algorithm: the first 12 hex digits of the SHA-256 over the file
    name and then the bytes of ``dsa_litetopk.cu``, ``sm100_dsa_litetopk.cuh`` and
    ``dense_topk_litetopk.cuh`` in ``litetopk_kernels/``, in that order. The adapter module is
    not covered. Nothing is imported or executed.

    Args:
        source: The plugin directory.

    Returns:
        The 12-hex source id.

    Raises:
        IndexerTopKPluginError: If a CUDA source is missing.
    """
    kernel_dir = Path(source) / _KERNEL_DIR
    missing = [name for name in LITETOPK_KERNEL_FILES if not (kernel_dir / name).is_file()]
    if missing:
        raise IndexerTopKPluginError(
            f"LiteTopK plugin at {source} lacks {', '.join(f'{_KERNEL_DIR}/{m}' for m in missing)}"
        )
    return _files_digest(kernel_dir, LITETOPK_KERNEL_FILES)[:12]


def _missing_packages() -> list[str]:
    """Return the runtime packages of the plugins that cannot be found (none is imported)."""
    missing = []
    for name in _RUNTIME_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(name)
    return missing


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _pin_mismatches(origin: Path, checks: tuple[tuple[str, str | None, str | None], ...]) -> None:
    problems = [
        f"LiteTopK plugin at {origin}: {what} is {actual}, expected {expected}"
        for what, actual, expected in checks
        if expected is not None and actual != expected
    ]
    if problems:
        raise IndexerTopKPluginError("; ".join(problems))


def _wrap_plugin_failure(origin: Path, step: str, exc: Exception) -> IndexerTopKPluginError:
    if isinstance(exc, IndexerTopKPluginError):
        return exc
    return IndexerTopKPluginError(
        f"LiteTopK plugin at {origin}: {step} failed: {type(exc).__name__}: {exc}"
    )


def load_litetopk_plugin(
    config: LiteTopKPluginConfig, settings: LiteTopKPluginSettings | None = None
) -> LoadedLiteTopKPlugin:
    """Load, validate and cache a LiteTopK plugin; idempotent for the same request.

    Args:
        config: Where the plugin is and what it must hash to.
        settings: The import-time settings; None renders only the fixed environment keys.

    Returns:
        The loaded plugin; the same object for every later call with the same source, adapter
        and settings.

    Raises:
        IndexerTopKPluginError: If a file or dependency is missing, a pin does not match, the
            environment conflicts, the module violates ABI v1 or reports a configuration other
            than the rendered one, the source id is already loaded with other settings or
            another extension, or the extension fails to load.
    """
    if not isinstance(config, LiteTopKPluginConfig):
        raise TypeError(f"expected a LiteTopKPluginConfig, got {type(config).__name__}")
    if settings is None:
        settings = LiteTopKPluginSettings()
    elif not isinstance(settings, LiteTopKPluginSettings):
        raise TypeError(f"expected LiteTopKPluginSettings or None, got {type(settings).__name__}")
    with _LOCK:
        source = _resolve(config.source)
        adapter = source / _ADAPTER_FILE
        kernel_dir = source / _KERNEL_DIR
        prebuilt = (
            None if config.prebuilt_extension is None else _resolve(config.prebuilt_extension)
        )
        issues = []
        if not source.is_dir():
            issues.append("the plugin directory does not exist")
        elif not adapter.is_file():
            issues.append(f"{_ADAPTER_FILE} is missing")
        if source.is_dir():
            issues.extend(
                f"{_KERNEL_DIR}/{name} is missing"
                for name in LITETOPK_KERNEL_FILES
                if not (kernel_dir / name).is_file()
            )
        issues.extend(f"Python package {name} is not installed" for name in _missing_packages())
        if prebuilt is not None and not prebuilt.is_file():
            issues.append(f"prebuilt extension {prebuilt} does not exist")
        if issues:
            raise IndexerTopKPluginError(
                f"LiteTopK plugin at {source} cannot be loaded: {'; '.join(issues)}"
            )

        source_id = compute_litetopk_source_id(source)
        adapter_sha256 = _sha256_file(adapter)
        prebuilt_sha256 = None if prebuilt is None else _sha256_file(prebuilt)
        _pin_mismatches(
            source,
            (
                ("source id", source_id, config.expected_source_id),
                ("adapter sha256", adapter_sha256, config.expected_adapter_sha256),
                ("prebuilt extension sha256", prebuilt_sha256, config.prebuilt_extension_sha256),
            ),
        )
        request = tuple(
            None if path is None else str(_resolve(path))
            for path in (config.prebuilt_extension, config.build_dir, config.deepgemm_include_dir)
        )
        loaded = _LOADED.get((source_id, adapter_sha256, settings))
        if loaded is not None:
            loaded_request = (
                loaded.prebuilt_extension,
                loaded.build_dir,
                loaded.deepgemm_include_dir,
            )
            if loaded_request != request:
                raise IndexerTopKPluginError(
                    f"LiteTopK source {source_id} is already loaded in this process with "
                    f"(prebuilt_extension, build_dir, deepgemm_include_dir) = {loaded_request}; "
                    f"the configuration requests {request}"
                )
            return loaded
        if any(
            other.source_id == source_id and other.settings != settings
            for other in _LOADED.values()
        ):
            raise IndexerTopKPluginError(
                f"LiteTopK source {source_id} is already loaded in this process with different "
                "settings; ABI v1 snapshots settings at import. Use one settings profile per "
                "process."
            )
        pins = ["expected_source_id", "expected_adapter_sha256"]
        if prebuilt is not None:
            pins.append("prebuilt_extension_sha256")
        pinned = frozenset(name for name in pins if getattr(config, name) is not None)
        loaded = _import_litetopk_plugin(
            source, source_id, adapter_sha256, settings, request, prebuilt_sha256, pinned
        )
        _LOADED[(source_id, adapter_sha256, settings)] = loaded
        unpinned = [name for name in pins if name not in pinned]
        if unpinned:
            logger.warning(
                "LiteTopK plugin at %s was loaded without %s; source id %s, adapter sha256 %s, "
                "prebuilt extension sha256 %s",
                source,
                ", ".join(unpinned),
                source_id,
                adapter_sha256,
                prebuilt_sha256,
            )
        return loaded


def _import_litetopk_plugin(
    source: Path,
    source_id: str,
    adapter_sha256: str,
    settings: LiteTopKPluginSettings,
    request: tuple[str | None, str | None, str | None],
    prebuilt_sha256: str | None,
    pinned: frozenset[str],
) -> LoadedLiteTopKPlugin:
    prebuilt_extension, build_dir, deepgemm_include_dir = request
    owner = f"{source_id} (adapter {adapter_sha256[:8]})"
    module_name = f"megatron_lite_litetopk_{source_id}_{adapter_sha256[:8]}"
    if module_name in sys.modules:
        raise IndexerTopKPluginError(
            f"LiteTopK plugin at {source}: module name {module_name} is already taken in "
            "sys.modules"
        )
    rendered = env.render_plugin_env(settings)
    claim = env.claim_plugin_env(owner, rendered)
    try:
        spec = importlib.util.spec_from_file_location(module_name, source / _ADAPTER_FILE)
        if spec is None or spec.loader is None:
            raise IndexerTopKPluginError(
                f"LiteTopK plugin at {source}: cannot create a module spec for {_ADAPTER_FILE}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise _wrap_plugin_failure(source, f"importing {_ADAPTER_FILE}", exc) from exc
        validate_litetopk_module(module, origin=str(source))
        try:
            raw_info = module.plugin_info()
        except Exception as exc:
            raise _wrap_plugin_failure(source, "plugin_info()", exc) from exc
        info = parse_plugin_info(raw_info, origin=str(source))
        if info.source_id != source_id:
            raise IndexerTopKPluginError(
                f"LiteTopK plugin at {source} reports source id {info.source_id}, but its CUDA "
                f"sources hash to {source_id}"
            )
        claim.check_launch_time_keys(info.launch_time_env_keys)
        claim.check_effective_config(info.effective_config)
        for name, requested in (
            ("tie_policy", settings.tie_policy),
            ("score_policy", settings.score_policy),
        ):
            reported = getattr(info, name)
            if requested is not None and reported != requested:
                raise IndexerTopKPluginError(
                    f"LiteTopK plugin at {source} reports {name} {reported!r}; the settings "
                    f"request {requested!r}"
                )
        try:
            module.load_extension(
                prebuilt_path=prebuilt_extension,
                prebuilt_sha256=prebuilt_sha256,
                build_dir=build_dir,
                deepgemm_include_dir=deepgemm_include_dir,
            )
        except Exception as exc:
            raise _wrap_plugin_failure(source, "load_extension()", exc) from exc
    except BaseException:
        sys.modules.pop(module_name, None)
        claim.rollback()
        raise
    launch_time_env = claim.commit(info.launch_time_env_keys)
    return LoadedLiteTopKPlugin(
        module=module,
        info=info,
        source=str(source),
        adapter_sha256=adapter_sha256,
        settings=settings,
        rendered_env=rendered,
        diagnostic_env=claim.diagnostics,
        launch_time_env=launch_time_env,
        prebuilt_extension=prebuilt_extension,
        prebuilt_extension_sha256=prebuilt_sha256,
        build_dir=build_dir,
        deepgemm_include_dir=deepgemm_include_dir,
        pinned=pinned,
    )


def loaded_litetopk_plugins() -> tuple[LoadedLiteTopKPlugin, ...]:
    """Return the LiteTopK plugins loaded in this process, in load order."""
    with _LOCK:
        return tuple(_LOADED.values())


def load_exact_topk(config: ExactTopKConfig) -> LoadedExactTopK:
    """Import the exact-tie top-k package as a private package; idempotent for the same files.

    The package is registered as ``megatron_lite_exact_topk_<digest>`` with the source directory
    as its path, so its relative imports work without changing ``sys.path`` or any existing
    module. Identical files map to the same package.

    Args:
        config: Where the package is and what its files must hash to.

    Returns:
        The imported package's selector.

    Raises:
        IndexerTopKPluginError: If a file is missing, a pin does not match, or the import fails
            (for example when its cuDNN frontend or CUTLASS DSL dependencies are missing).
    """
    if not isinstance(config, ExactTopKConfig):
        raise TypeError(f"expected an ExactTopKConfig, got {type(config).__name__}")
    with _LOCK:
        source = _resolve(config.source)
        missing = [name for name in EXACT_TOPK_FILES if not (source / name).is_file()]
        if missing:
            raise IndexerTopKPluginError(
                f"exact top-k package at {source} cannot be loaded: {', '.join(missing)} missing"
            )
        file_sha256 = {name: _sha256_file(source / name) for name in EXACT_TOPK_FILES}
        problems = [
            f"exact top-k package at {source}: sha256 of {name} is {file_sha256[name]}, "
            f"expected {expected}"
            for name, expected in sorted((config.expected_sha256 or {}).items())
            if file_sha256[name] != expected
        ]
        if problems:
            raise IndexerTopKPluginError("; ".join(problems))
        digest = _files_digest(source, EXACT_TOPK_FILES)
        loaded = _EXACT_LOADED.get(digest)
        if loaded is not None:
            return loaded
        package_name = f"megatron_lite_exact_topk_{digest[:8]}"
        if package_name in sys.modules:
            raise IndexerTopKPluginError(
                f"exact top-k package at {source}: module name {package_name} is already taken "
                "in sys.modules"
            )
        spec = importlib.machinery.ModuleSpec(package_name, None, is_package=True)
        spec.submodule_search_locations = [str(source)]
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        try:
            module = importlib.import_module(f"{package_name}.{_EXACT_TOPK_MODULE}")
            wrapper = getattr(module, "cute_dsl_topk_wrapper", None)
            if not callable(wrapper):
                raise IndexerTopKPluginError(
                    f"exact top-k package at {source}: {_EXACT_TOPK_MODULE} has no callable "
                    "cute_dsl_topk_wrapper"
                )
        except BaseException as exc:
            for name in [
                name
                for name in sys.modules
                if name == package_name or name.startswith(f"{package_name}.")
            ]:
                sys.modules.pop(name, None)
            if isinstance(exc, Exception) and not isinstance(exc, IndexerTopKPluginError):
                raise IndexerTopKPluginError(
                    f"exact top-k package at {source} failed to import: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            raise
        loaded = LoadedExactTopK(
            source=str(source), package=package_name, file_sha256=file_sha256, wrapper=wrapper
        )
        _EXACT_LOADED[digest] = loaded
        return loaded

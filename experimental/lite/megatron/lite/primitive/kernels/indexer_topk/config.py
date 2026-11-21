# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and error types of the indexer top-k selectors.

The LiteTopK kernels and the exact-tie top-k used by the reference selector are external
dependencies: Megatron Lite does not bundle them. They are loaded from the explicit paths given
here, never from environment variables, and every path can be pinned by content hash.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ExactTopKConfig",
    "IndexerTopKConfigError",
    "IndexerTopKPluginError",
    "IndexerTopKRuntimeError",
    "LiteTopKPluginConfig",
    "LiteTopKPluginSettings",
]

# The modules of the external exact-tie top-k package (a modified cuDNN CuTe DSL radix top-k).
EXACT_TOPK_FILES = (
    "block_scan.py",
    "indexer_top_k_varlen_util.py",
    "indexer_top_k_decode_varlen.py",
)

_SOURCE_ID = re.compile(r"[0-9a-f]{12}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LAYOUT = re.compile(r"[a-z0-9]+")
_TIE_POLICIES = ("storage", "logical-id", "logical-id-desc")
_SCORE_POLICIES = ("folded", "native-fp32")
_ROW_TILES = (1, 2, 4, 8)


class IndexerTopKConfigError(ValueError):
    """An indexer top-k configuration value, or a combination of values, is invalid."""


class IndexerTopKPluginError(RuntimeError):
    """An external indexer top-k plugin is missing, does not match its pins, or fails to load."""


class IndexerTopKRuntimeError(RuntimeError):
    """An indexer top-k selection cannot run in the current process state."""


def _check_path(owner: str, name: str, value: object, *, required: bool) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not value:
        raise IndexerTopKConfigError(
            f"{owner}.{name} must be a non-empty path string, got {value!r}"
        )


def _check_hex(owner: str, name: str, value: object, pattern: re.Pattern[str], what: str) -> None:
    if value is not None and (not isinstance(value, str) or pattern.fullmatch(value) is None):
        raise IndexerTopKConfigError(
            f"{owner}.{name} must be {what} in lowercase hexadecimal, got {value!r}"
        )


@dataclass(frozen=True)
class LiteTopKPluginConfig:
    """An external LiteTopK plugin (ABI v1), loaded from an explicit path.

    The plugin is a directory holding the adapter module ``litetopk.py`` and its CUDA sources
    ``litetopk_kernels/{dsa_litetopk.cu, sm100_dsa_litetopk.cuh, dense_topk_litetopk.cuh}``. Its
    CUDA extension is either a prebuilt shared library or a JIT build of those sources. Every
    ``expected_*`` pin is optional; unpinned values are computed, logged once and recorded in
    the load provenance.

    Attributes:
        source: The plugin directory.
        expected_source_id: The 12-hex source id of the three CUDA files (the plugin's own
            source id algorithm, which the loader reproduces before importing anything).
        expected_adapter_sha256: The SHA-256 of ``litetopk.py``, which the source id does not
            cover.
        prebuilt_extension: The prebuilt CUDA extension to load instead of a JIT build.
        prebuilt_extension_sha256: The SHA-256 of ``prebuilt_extension``.
        build_dir: The JIT build directory; None keeps the plugin's default.
        deepgemm_include_dir: The DeepGEMM checkout or include directory holding the headers of
            a JIT build; None keeps the plugin's default.
    """

    source: str
    expected_source_id: str | None = None
    expected_adapter_sha256: str | None = None
    prebuilt_extension: str | None = None
    prebuilt_extension_sha256: str | None = None
    build_dir: str | None = None
    deepgemm_include_dir: str | None = None

    def __post_init__(self) -> None:
        owner = "indexer_topk.litetopk"
        _check_path(owner, "source", self.source, required=True)
        _check_hex(
            owner, "expected_source_id", self.expected_source_id, _SOURCE_ID, "12 characters"
        )
        _check_hex(
            owner, "expected_adapter_sha256", self.expected_adapter_sha256, _SHA256, "a SHA-256"
        )
        for name in ("prebuilt_extension", "build_dir", "deepgemm_include_dir"):
            _check_path(owner, name, getattr(self, name), required=False)
        _check_hex(
            owner, "prebuilt_extension_sha256", self.prebuilt_extension_sha256, _SHA256, "a SHA-256"
        )
        if self.prebuilt_extension is None and self.prebuilt_extension_sha256 is not None:
            raise IndexerTopKConfigError(
                f"{owner}.prebuilt_extension_sha256 pins a prebuilt extension; "
                "set prebuilt_extension as well"
            )
        if self.prebuilt_extension is not None and (
            self.build_dir is not None or self.deepgemm_include_dir is not None
        ):
            raise IndexerTopKConfigError(
                f"{owner}.build_dir and deepgemm_include_dir configure a JIT build and cannot "
                "be combined with prebuilt_extension"
            )


@dataclass(frozen=True)
class ExactTopKConfig:
    """The external exact-tie top-k package, loaded from an explicit path.

    The package is a modified cuDNN CuTe DSL radix top-k that orders equal scores by ascending
    key id. It is imported as a private package so that its relative imports work without
    touching ``sys.path``.

    Attributes:
        source: The directory holding ``block_scan.py``, ``indexer_top_k_varlen_util.py`` and
            ``indexer_top_k_decode_varlen.py``.
        expected_sha256: Optional per-file SHA-256 pins, keyed by those file names.
    """

    source: str
    expected_sha256: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        owner = "indexer_topk.exact_topk"
        _check_path(owner, "source", self.source, required=True)
        if self.expected_sha256 is None:
            return
        if not isinstance(self.expected_sha256, Mapping):
            raise IndexerTopKConfigError(
                f"{owner}.expected_sha256 must map file names to SHA-256 values, "
                f"got {type(self.expected_sha256).__name__}"
            )
        for name, digest in self.expected_sha256.items():
            if name not in EXACT_TOPK_FILES:
                raise IndexerTopKConfigError(
                    f"{owner}.expected_sha256 pins unknown file {name!r}; "
                    f"expected one of {', '.join(EXACT_TOPK_FILES)}"
                )
            _check_hex(owner, f"expected_sha256[{name!r}]", digest, _SHA256, "a SHA-256")


def _check_choice(name: str, value: object, choices: tuple[object, ...]) -> None:
    if value is not None and value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise IndexerTopKConfigError(
            f"LiteTopKPluginSettings.{name} must be one of {allowed} or None, got {value!r}"
        )


def _check_int(name: str, value: object, minimum: int) -> None:
    if value is not None and (type(value) is not int or value < minimum):
        raise IndexerTopKConfigError(
            f"LiteTopKPluginSettings.{name} must be an integer >= {minimum} or None, got {value!r}"
        )


def _check_bool(name: str, value: object) -> None:
    if value is not None and type(value) is not bool:
        raise IndexerTopKConfigError(
            f"LiteTopKPluginSettings.{name} must be a bool or None, got {value!r}"
        )


@dataclass(frozen=True)
class LiteTopKPluginSettings:
    """Import-time settings of a LiteTopK plugin (an expert knob; never part of ImplConfig).

    ABI v1 plugins read their configuration from the process environment once, when they are
    imported, and their CUDA extensions read a few keys again on every launch. The loader
    renders these settings into those environment keys before the import and keeps them for the
    process lifetime, so one process holds at most one settings profile per plugin source.
    ``None`` leaves a key unrendered, so the plugin's built-in default applies.

    Attributes:
        tie_policy: How the selector orders keys whose selection codes are equal.
        score_policy: ``native-fp32`` scores with the float32 epilogue of the reference score
            kernel; ``folded`` uses the plugin's folded epilogue.
        paged_pool_pages_per_row: Candidate pool pages per query row of the paged route.
        fp8_row_tiles: Internal query-row tiles of large FP8 calls (1, 2, 4 or 8).
        fp8_paged_admit_max_query_len: When positive, the FP8 paged route also admits query
            tiles of at most this many rows (a multiple of four); 0 admits only the qualified
            tile lengths.
        tiered_seed_12k: Use the tiered HOT seed calibration.
        coldstart_identity: Start a sequence without a carried seed from the identity HOT seed.
        raw32_staging: Staging layout of the scan of a raw-FP32-key plugin, a launch-time
            setting (for example ``u40x18k3``, that plugin's compiled default, or ``u40x14``;
            the plugin validates the name). Only a plugin that lists the key among its
            launch-time keys accepts it. None leaves it unset.
    """

    tie_policy: Literal["storage", "logical-id", "logical-id-desc"] | None = None
    score_policy: Literal["folded", "native-fp32"] | None = None
    paged_pool_pages_per_row: int | None = None
    fp8_row_tiles: int | None = None
    fp8_paged_admit_max_query_len: int | None = None
    tiered_seed_12k: bool | None = None
    coldstart_identity: bool | None = None
    raw32_staging: str | None = None

    def __post_init__(self) -> None:
        _check_choice("tie_policy", self.tie_policy, _TIE_POLICIES)
        _check_choice("score_policy", self.score_policy, _SCORE_POLICIES)
        _check_int("paged_pool_pages_per_row", self.paged_pool_pages_per_row, 1)
        _check_int("fp8_row_tiles", self.fp8_row_tiles, 1)
        _check_choice("fp8_row_tiles", self.fp8_row_tiles, _ROW_TILES)
        _check_int("fp8_paged_admit_max_query_len", self.fp8_paged_admit_max_query_len, 0)
        _check_bool("tiered_seed_12k", self.tiered_seed_12k)
        _check_bool("coldstart_identity", self.coldstart_identity)
        if self.raw32_staging is not None and (
            not isinstance(self.raw32_staging, str) or _LAYOUT.fullmatch(self.raw32_staging) is None
        ):
            raise IndexerTopKConfigError(
                "LiteTopKPluginSettings.raw32_staging must be a layout name of lowercase "
                f"letters and digits or None, got {self.raw32_staging!r}"
            )

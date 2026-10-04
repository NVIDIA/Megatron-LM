# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and error types of the indexer top-k selectors.

The LiteTopK kernels and the exact-tie top-k used by the reference selector are external
dependencies: Megatron Lite does not bundle them. They are loaded from the explicit paths given
here, never from environment variables, and every path can be pinned by content hash.

Expert tuning of the selection plan goes through one policy seam,
:func:`resolve_indexer_topk_tuning`: it derives every value from the plugin route, the operand
format, the indexer geometry and the device, and applies the overrides of an
:class:`IndexerTopKTuning`. Model configurations never carry tuning values.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from megatron.lite.primitive.kernels.indexer_topk.layout import IndexerGeometry
    from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability

__all__ = [
    "ExactTopKConfig",
    "IndexerTopKConfigError",
    "IndexerTopKFormat",
    "IndexerTopKPluginError",
    "IndexerTopKPrecision",
    "IndexerTopKRuntimeError",
    "IndexerTopKTuning",
    "LiteTopKPluginConfig",
    "LiteTopKPluginSettings",
    "ResolvedIndexerTopKTuning",
    "resolve_indexer_topk_tuning",
]

IndexerTopKFormat = Literal["fp8"]
IndexerTopKPrecision = Literal["exact", "fast"]

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
_FORMATS = ("fp8",)
_PRECISIONS = ("exact", "fast")
_SEED_BOOTSTRAPS = ("reference", "identity")
_INDEX_ORDERS = ("ascending", "selector")
_STATUS_CHECKS = ("sync_recompute", "sync_recompute_tail", "device_assert")

# Byte budget of the float32 scores of one reference scoring call, per operand format.
REFERENCE_BUDGET_BYTES = {"fp8": 2 << 30}
# Rows per top-k kernel call. The cuDNN frontend radix top-k (and the exact-tie package derived
# from it) is verified only up to this row count per call; a fused call with more rows silently
# corrupts the rows past it once the process has made any earlier call.
TOPK_ROWS_PER_CALL_LIMIT = 32768
# Scores of the FP8 score kernels are computed for 128 // heads query rows per SM, and a LiteTopK
# FP8 tile covers this many such waves over all SMs.
_FP8_TILE_WAVES = 3
# FP8 LiteTopK starts this many positions before the route's qualified minimum key count.
_FP8_STARTUP_MARGIN = 8192
# Candidate pool pages per query row of the FP8 paged route. With cold per-layer timing on
# GLM-5.2 indexer inputs (six in-model layers at 256K, real-weight layer 0 at 512K and 1M,
# synthetic inputs up to 1M) the pool size does not change the speed of a tile, and the worst
# tile measured used 10.4% of a 13-page pool: 9.6x headroom, and 1.19 GiB per device less than
# 32 pages with 8-byte candidate records. An exhausted pool is a FAILED row status that the
# bindings recompute, not a kernel fault.
_FP8_POOL_PAGES_PER_ROW = 13
# The tiered 12K HOT seed is the plugin setting that decides the speed of the FP8 paged route:
# without it the tiles of the synthetic inputs collect 3-4x the candidates, and a call on the
# strict-gap inputs was 1.5x (256K) and 2.8x (512K) slower than the reference selector alone.
_FP8_TIERED_SEED = True


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


def _check_choice(owner: str, name: str, value: object, choices: tuple[object, ...]) -> None:
    if value is not None and value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise IndexerTopKConfigError(
            f"{owner}.{name} must be one of {allowed} or None, got {value!r}"
        )


def _check_int(owner: str, name: str, value: object, minimum: int) -> None:
    if value is not None and (type(value) is not int or value < minimum):
        raise IndexerTopKConfigError(
            f"{owner}.{name} must be an integer >= {minimum} or None, got {value!r}"
        )


def _check_bool(owner: str, name: str, value: object) -> None:
    if value is not None and type(value) is not bool:
        raise IndexerTopKConfigError(f"{owner}.{name} must be a bool or None, got {value!r}")


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
        owner = "LiteTopKPluginSettings"
        _check_choice(owner, "tie_policy", self.tie_policy, _TIE_POLICIES)
        _check_choice(owner, "score_policy", self.score_policy, _SCORE_POLICIES)
        _check_int(owner, "paged_pool_pages_per_row", self.paged_pool_pages_per_row, 1)
        _check_int(owner, "fp8_row_tiles", self.fp8_row_tiles, 1)
        _check_choice(owner, "fp8_row_tiles", self.fp8_row_tiles, _ROW_TILES)
        _check_int(owner, "fp8_paged_admit_max_query_len", self.fp8_paged_admit_max_query_len, 0)
        _check_bool(owner, "tiered_seed_12k", self.tiered_seed_12k)
        _check_bool(owner, "coldstart_identity", self.coldstart_identity)
        if self.raw32_staging is not None and (
            not isinstance(self.raw32_staging, str) or _LAYOUT.fullmatch(self.raw32_staging) is None
        ):
            raise IndexerTopKConfigError(
                f"{owner}.raw32_staging must be a layout name of lowercase letters and digits "
                f"or None, got {self.raw32_staging!r}"
            )


@dataclass(frozen=True)
class IndexerTopKTuning:
    """Expert overrides of the indexer top-k selection plan (harnesses and tests only).

    Models never set these values. Every field defaults to None, which keeps the value
    :func:`resolve_indexer_topk_tuning` derives for the route, operand format, indexer geometry
    and device (see :class:`ResolvedIndexerTopKTuning` for the meaning and default of each
    field). ``plugin_settings`` overrides only its own non-None fields.
    """

    required: bool | None = None
    tile_rows: int | None = None
    startup_position: int | None = None
    group_tiles: int | None = None
    seed_bootstrap: Literal["reference", "identity"] | None = None
    min_litetopk_pairs: int | None = None
    reference_budget_bytes: int | None = None
    reference_rows_per_call: int | None = None
    index_order: Literal["ascending", "selector"] | None = None
    status_check: Literal["sync_recompute", "sync_recompute_tail", "device_assert"] | None = None
    plugin_settings: LiteTopKPluginSettings | None = None

    def __post_init__(self) -> None:
        _check_tuning_fields(self, "IndexerTopKTuning")


def _check_tuning_fields(tuning: object, owner: str) -> None:
    _check_bool(owner, "required", tuning.required)
    _check_int(owner, "tile_rows", tuning.tile_rows, 4)
    if tuning.tile_rows is not None and tuning.tile_rows % 4:
        raise IndexerTopKConfigError(
            f"{owner}.tile_rows must be a multiple of 4, got {tuning.tile_rows}"
        )
    _check_int(owner, "startup_position", tuning.startup_position, 0)
    _check_int(owner, "group_tiles", tuning.group_tiles, 1)
    _check_choice(owner, "seed_bootstrap", tuning.seed_bootstrap, _SEED_BOOTSTRAPS)
    _check_int(owner, "min_litetopk_pairs", tuning.min_litetopk_pairs, 0)
    _check_int(owner, "reference_budget_bytes", tuning.reference_budget_bytes, 1)
    _check_int(owner, "reference_rows_per_call", tuning.reference_rows_per_call, 1)
    _check_choice(owner, "index_order", tuning.index_order, _INDEX_ORDERS)
    _check_choice(owner, "status_check", tuning.status_check, _STATUS_CHECKS)
    if tuning.plugin_settings is not None and not isinstance(
        tuning.plugin_settings, LiteTopKPluginSettings
    ):
        raise IndexerTopKConfigError(
            f"{owner}.plugin_settings must be LiteTopKPluginSettings or None, got "
            f"{type(tuning.plugin_settings).__name__}"
        )


@dataclass(frozen=True)
class ResolvedIndexerTopKTuning:
    """The selection plan settings of one layer, resolved by :func:`resolve_indexer_topk_tuning`.

    Attributes:
        required: Raise an ``IndexerTopKRuntimeError`` instead of falling back to the reference
            selector when rows planned for LiteTopK are declined or fail (for benchmarks).
            Default False.
        tile_rows: Query rows of a LiteTopK tile, a multiple of 4. Default: three waves of the
            score kernel (``3 * num_sms * (128 // num_heads)``, rounded down to a multiple of 4;
            1776 for 32 heads on 148 SMs).
        startup_position: First causal position a LiteTopK tile may start at (a tile also
            needs rows that see the route's HOT prefix). Default: 8192 before the route's
            qualified minimum key count (188416 for a 196608-key minimum). 0 lets only the HOT
            prefix limit it.
        group_tiles: Consecutive tiles of equal length that share one plan. Default 8.
        seed_bootstrap: How the first tile group of a segment gets its HOT seed: ``reference``
            votes with the reference selections of the rows that precede it (or of the
            segment's first tile, then computed by the reference selector, when too few rows
            precede it); ``identity`` starts from the plugin's cold-start seed. Default
            ``reference``.
        min_litetopk_pairs: The per-segment crossover: a segment uses LiteTopK only when its
            LiteTopK tiles cover at least this many query-key pairs (the visible keys summed
            over their rows, not counting a reference bootstrap tile). Default 0 (every eligible
            segment).
        reference_budget_bytes: Byte budget of the float32 scores of one reference scoring
            call; the top-k kernel needs scratch of about twice that on top. Default 2 GiB.
        reference_rows_per_call: Rows per reference scoring call, bounded by the budget; None
            plans whole SM waves of the score kernel within the budget.
        index_order: ``ascending`` sorts every output row (ids ascending, -1 last);
            ``selector`` keeps the selectors' slot order. Default ``ascending``.
        status_check: How the per-row status of the plugin is handled. ``sync_recompute`` reads
            it once per call and recomputes with the reference selector the rows that report
            a candidate overflow and the tiles that report a failure; ``sync_recompute_tail``
            also recomputes every LiteTopK row after a failed tile in its segment;
            ``device_assert`` asserts on the device that every status is 0 and reads nothing.
            Default ``sync_recompute``.
        plugin_settings: Import-time settings of the LiteTopK plugin. Default: the tiered 12K
            HOT seed, 13 pool pages per row, 2 row tiles, the identity cold-start seed,
            paged tiles of up to ``tile_rows`` rows admitted and, for precision ``exact``, the
            policies exact selection needs (``logical-id`` ties on ``native-fp32`` scores).
    """

    required: bool
    tile_rows: int
    startup_position: int
    group_tiles: int
    seed_bootstrap: Literal["reference", "identity"]
    min_litetopk_pairs: int
    reference_budget_bytes: int
    reference_rows_per_call: int | None
    index_order: Literal["ascending", "selector"]
    status_check: Literal["sync_recompute", "sync_recompute_tail", "device_assert"]
    plugin_settings: LiteTopKPluginSettings

    def __post_init__(self) -> None:
        owner = "ResolvedIndexerTopKTuning"
        optional = ("reference_rows_per_call",)
        unset = [
            field.name
            for field in dataclasses.fields(self)
            if field.name not in optional and getattr(self, field.name) is None
        ]
        if unset:
            raise IndexerTopKConfigError(f"{owner} needs concrete values for {unset}")
        _check_tuning_fields(self, owner)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy, for logs and provenance records."""
        return dataclasses.asdict(self)


def resolve_indexer_topk_tuning(
    tuning: IndexerTopKTuning | None,
    *,
    fmt: IndexerTopKFormat,
    route: RouteCapability | None,
    geometry: IndexerGeometry,
    precision: IndexerTopKPrecision,
    num_sms: int,
) -> ResolvedIndexerTopKTuning:
    """Resolve the selection plan settings of one layer: the single tuning policy seam.

    Derives every value (see :class:`ResolvedIndexerTopKTuning`) and applies the non-None
    overrides of ``tuning``.

    Args:
        tuning: Expert overrides, or None for the derived values.
        fmt: Operand format of the layer: ``fp8``.
        route: The plugin route that serves the layer, or None when only the reference
            selector runs (its route-dependent values are then unused).
        geometry: The indexer geometry of the layer.
        precision: ``exact`` or ``fast``.
        num_sms: Streaming multiprocessors of the device.

    Returns:
        The resolved settings.

    Raises:
        IndexerTopKConfigError: If an argument is invalid, the route serves another format, or
            the overrides conflict with the precision or the geometry.
    """
    if tuning is None:
        tuning = IndexerTopKTuning()
    elif not isinstance(tuning, IndexerTopKTuning):
        raise TypeError(f"expected IndexerTopKTuning or None, got {type(tuning).__name__}")
    if fmt not in _FORMATS:
        raise IndexerTopKConfigError(f"fmt must be one of {_FORMATS}, got {fmt!r}")
    if precision not in _PRECISIONS:
        raise IndexerTopKConfigError(
            f"indexer_topk.precision must be one of {_PRECISIONS}, got {precision!r}"
        )
    if route is not None and route.fmt != fmt:
        raise IndexerTopKConfigError(
            f"LiteTopK route {route.name} serves {route.fmt} operands; this layer uses {fmt}"
        )
    if type(num_sms) is not int or num_sms < 1:
        raise IndexerTopKConfigError(f"num_sms must be a positive integer, got {num_sms!r}")

    waves = _FP8_TILE_WAVES * num_sms * max(1, 128 // geometry.num_heads)
    tile_rows = _override(tuning.tile_rows, max(4, waves // 4 * 4))
    startup = 0 if route is None else max(0, route.min_keys - _FP8_STARTUP_MARGIN)
    exact = precision == "exact"
    settings = LiteTopKPluginSettings(
        tie_policy="logical-id" if exact else None,
        score_policy="native-fp32" if exact else None,
        paged_pool_pages_per_row=_FP8_POOL_PAGES_PER_ROW,
        fp8_row_tiles=2,
        fp8_paged_admit_max_query_len=tile_rows,
        tiered_seed_12k=_FP8_TIERED_SEED,
        coldstart_identity=True,
    )
    group_tiles, seed_bootstrap = 8, "reference"
    if tuning.plugin_settings is not None:
        settings = dataclasses.replace(
            settings,
            **{
                field.name: getattr(tuning.plugin_settings, field.name)
                for field in dataclasses.fields(LiteTopKPluginSettings)
                if getattr(tuning.plugin_settings, field.name) is not None
            },
        )
    resolved = ResolvedIndexerTopKTuning(
        required=_override(tuning.required, False),
        tile_rows=tile_rows,
        startup_position=_override(tuning.startup_position, startup),
        group_tiles=_override(tuning.group_tiles, group_tiles),
        seed_bootstrap=_override(tuning.seed_bootstrap, seed_bootstrap),
        min_litetopk_pairs=_override(tuning.min_litetopk_pairs, 0),
        reference_budget_bytes=_override(
            tuning.reference_budget_bytes, REFERENCE_BUDGET_BYTES[fmt]
        ),
        reference_rows_per_call=tuning.reference_rows_per_call,
        index_order=_override(tuning.index_order, "ascending"),
        status_check=_override(tuning.status_check, "sync_recompute"),
        plugin_settings=settings,
    )
    if precision == "exact" and (
        (settings.tie_policy, settings.score_policy) != ("logical-id", "native-fp32")
    ):
        raise IndexerTopKConfigError(
            "precision='exact' needs plugin_settings tie_policy='logical-id' and "
            "score_policy='native-fp32' (exact selection ranks the reference score kernel's "
            f"float32 scores, equal scores by ascending key id); got {settings.tie_policy!r} "
            f"and {settings.score_policy!r}"
        )
    return resolved


def _override(value: Any, default: Any) -> Any:
    return default if value is None else value

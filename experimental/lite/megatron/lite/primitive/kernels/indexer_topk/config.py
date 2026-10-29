# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration and error types of the indexer top-k selectors.

:class:`IndexerTopKConfig` is what a model configuration carries: which selector runs, how
precise it must be, and where its external dependencies are. The LiteTopK kernels and the
exact-tie top-k used by the reference selector are not bundled with Megatron Lite. They are
loaded from the explicit paths given here, never from environment variables, and every path
can be pinned by content hash.

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
    from megatron.lite.primitive.kernels.indexer_topk.heads import IndexerHeads
    from megatron.lite.primitive.kernels.indexer_topk.layout import IndexerGeometry
    from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability

__all__ = [
    "ExactTopKConfig",
    "IndexerTopKBackend",
    "IndexerTopKConfig",
    "IndexerTopKConfigError",
    "IndexerTopKFormat",
    "IndexerTopKPluginError",
    "IndexerTopKPrecision",
    "IndexerTopKRuntimeError",
    "IndexerTopKTuning",
    "LiteTopKPluginConfig",
    "LiteTopKPluginSettings",
    "ResolvedIndexerTopKTuning",
    "normalize_indexer_topk_config",
    "resolve_indexer_topk_tuning",
]

IndexerTopKBackend = Literal["default", "reference", "litetopk"]
IndexerTopKFormat = Literal["fp8", "mxfp4"]
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
_BACKENDS = ("default", "reference", "litetopk")
_FORMATS = ("fp8", "mxfp4")
_PRECISIONS = ("exact", "fast")
_SEED_BOOTSTRAPS = ("reference", "identity")
_INDEX_ORDERS = ("ascending", "selector")
_STATUS_CHECKS = ("sync_recompute", "sync_recompute_tail", "device_assert")

# Byte budget of the float32 scores of one reference scoring call, per operand format.
REFERENCE_BUDGET_BYTES = {"fp8": 2 << 30, "mxfp4": 1 << 30}
# Rows per top-k kernel call. The cuDNN frontend radix top-k (and the exact-tie package derived
# from it) is verified only up to this row count per call; a fused call with more rows silently
# corrupts the rows past it once the process has made any earlier call.
TOPK_ROWS_PER_CALL_LIMIT = 32768
# Scores of the FP8 score kernels are computed for 128 // heads query rows per SM, and a LiteTopK
# FP8 tile covers this many such waves over all SMs, and at least _FP8_TILE_ROWS_PER_SM rows per
# SM: the 1776-row tiles of 148 SMs, which the 32-head kernels were measured fastest with. With
# 64 kernel heads (two rows per SM and wave) cold per-call timing at 512K to 1M keys found 1776-row
# tiles 0.1% to 2.1% faster than three-wave (888-row) ones on every input.
_FP8_TILE_WAVES = 3
_FP8_TILE_ROWS_PER_SM = 12
# FP8 LiteTopK starts this many positions before the route's qualified minimum key count, the
# start measured on GLM-5.2 indexer inputs (32 heads).
_FP8_STARTUP_MARGIN = 8192
# Start positions of FP8 LiteTopK tiles measured for other kernel head counts, which replace the
# rule above; None: the default plan gives LiteTopK no row. 64 heads (BLOCK_Q 2, the
# glm-litetopk-raw32h64-abi1 route), cold per-call timing against the reference backend on
# synthetic (structured, strict-gap) and real-weight (GLM-5.2 layer 0, heads duplicated) inputs:
# with tiles from position 188416 the calls of the real-weight input were 1.4% slower at 512K
# keys, with tiles from 524288 or 655360 those of the strict-gap input 1.7% to 2.8% slower at
# 768K; only at 1M keys was LiteTopK faster on every input (by 1.1% to 5.9%), largely because
# the reference backend's scoring calls are smallest there.
_FP8_MEASURED_STARTUP: dict[int, int | None] = {64: None}
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

# MXFP4 LiteTopK tiles start at the causal position whose row sees this many (compressed) keys.
# Cold per-call timing of DeepSeek-V4 indexer inputs on B200 (in-model layers 2 and 22 and
# synthetic inputs at 256K tokens, synthetic inputs at 512K): a slab tile, with its plan and query
# operands, costs as much as the reference selector on the same rows at 37K-53K visible keys,
# more below and less above. Of the start positions measured, 45056 keys gave the fastest calls:
# the reference selector alone took 1.009-1.024x their time at 256K (0.968-0.991x when the tiles
# start at the HOT prefix, 12288 keys, as in the previous integration) and 1.08-1.10x at 512K.
_MXFP4_STARTUP_KEYS = 45056

# The candidate slab of the MXFP4 slab route holds, per query row of a tile, one record of
# SLAB_RECORD_BYTES bytes (a 16-bit score code and a 32-bit key id) per candidate; the plugin
# accepts slabs of at least SLAB_MIN_CANDIDATES and SLAB_CANDIDATES_PER_TOPK * topk candidates.
SLAB_RECORD_BYTES = 6
SLAB_MIN_CANDIDATES = 16384
SLAB_CANDIDATES_PER_TOPK = 32
# Byte budget of the slab of one MXFP4 tile: 65536 candidates per row for 4096-row tiles, about
# the 60000-candidate slab (1.37 GiB) the previous integration ran DeepSeek-V4 with. Rows with
# more candidates report a capacity status and are recomputed by the reference selector.
CANDIDATE_BUDGET_BYTES = 3 << 29


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


@dataclass(frozen=True)
class IndexerTopKConfig:
    """How a model selects its indexer top-k (the ``indexer_topk`` field of an ``ImplConfig``).

    Attributes:
        backend: ``default`` installs nothing: every module keeps its upstream selector.
            ``reference`` selects every row with the matched-precision reference selector
            (operands quantized to the model's indexer format, scored by DeepGEMM, memory
            bounded for long contexts). ``litetopk`` selects with an external LiteTopK plugin
            where its plan allows and with the reference selector elsewhere.
        precision: ``exact``: every selected set is the exact top-k of the matched-precision
            scores (score descending, lower key id first on equal scores); it needs
            ``exact_topk`` and, for LiteTopK, a plugin route that advertises exact selection.
            ``fast``: LiteTopK rows may differ from that set among nearly tied keys, and
            without ``exact_topk`` equal scores are ordered arbitrarily.
        litetopk: The LiteTopK plugin; required by backend ``litetopk``.
        exact_topk: The exact-tie top-k package of the reference selector; required by
            precision ``exact`` unless the backend is ``default``.
        head_padding: Let LiteTopK serve a layer whose indexer head count its route has no
            kernels for by appending zero heads up to the smallest larger head count of the
            route (for a head count that is a multiple of four). Zero heads change no score
            value (see :mod:`.heads`); the scoring work grows by the padded head count over the
            layer's. Without it such a layer fails when it is bound. Backends other than
            ``litetopk`` ignore it: the reference selector pads only where its score kernel has
            no kernel for the head count, with or without it.
    """

    backend: IndexerTopKBackend = "default"
    precision: IndexerTopKPrecision = "exact"
    litetopk: LiteTopKPluginConfig | None = None
    exact_topk: ExactTopKConfig | None = None
    head_padding: bool = False

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise IndexerTopKConfigError(
                "indexer_topk.backend must be one of 'default', 'reference', 'litetopk'; "
                f"got {self.backend!r}"
            )
        if self.precision not in _PRECISIONS:
            raise IndexerTopKConfigError(
                f"indexer_topk.precision must be one of 'exact', 'fast'; got {self.precision!r}"
            )
        if type(self.head_padding) is not bool:
            raise IndexerTopKConfigError(
                f"indexer_topk.head_padding must be a bool, got {self.head_padding!r}"
            )
        for name, expected in (("litetopk", LiteTopKPluginConfig), ("exact_topk", ExactTopKConfig)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise IndexerTopKConfigError(
                    f"indexer_topk.{name} must be {expected.__name__} or None, got "
                    f"{type(value).__name__}"
                )
        if self.backend == "litetopk" and self.litetopk is None:
            raise IndexerTopKConfigError(
                "indexer_topk.litetopk.source is required: LiteTopK kernels are not bundled "
                "with Megatron Lite (see experimental/lite/docs/indexer_topk.md#dependencies)"
            )
        if self.backend != "default" and self.precision == "exact" and self.exact_topk is None:
            raise IndexerTopKConfigError(
                "indexer_topk.exact_topk.source is required for precision='exact': the "
                "exact-tie top-k is not bundled with Megatron Lite (set it, or use "
                "precision='fast')"
            )


def _from_mapping(owner: str, cls: type, value: object) -> Any:
    """Build the config dataclass ``cls`` from a mapping, rejecting unknown keys."""
    if value is None or isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise IndexerTopKConfigError(
            f"{owner} must be {cls.__name__}, a mapping or None, got {type(value).__name__}"
        )
    known = [field.name for field in dataclasses.fields(cls)]
    unknown = sorted(set(value) - set(known), key=str)
    if unknown:
        raise IndexerTopKConfigError(
            f"{owner} has unknown keys {unknown}; valid keys are {', '.join(known)}"
        )
    try:
        return cls(**value)
    except TypeError as exc:
        raise IndexerTopKConfigError(f"{owner} is incomplete: {exc}") from None


def normalize_indexer_topk_config(
    value: IndexerTopKConfig | Mapping[str, Any] | None,
) -> IndexerTopKConfig | None:
    """Return an indexer top-k configuration as a validated :class:`IndexerTopKConfig`.

    Args:
        value: A configuration, its mapping form (for example from a YAML file; ``litetopk``
            and ``exact_topk`` may be mappings too), or None.

    Returns:
        The configuration, or None when ``value`` is None.

    Raises:
        IndexerTopKConfigError: If a key is unknown or a value or combination is invalid.
    """
    if value is None or isinstance(value, IndexerTopKConfig):
        return value
    if not isinstance(value, Mapping):
        raise IndexerTopKConfigError(
            "indexer_topk must be IndexerTopKConfig, a mapping or None, got "
            f"{type(value).__name__}"
        )
    fields = dict(value)
    for name, cls in (("litetopk", LiteTopKPluginConfig), ("exact_topk", ExactTopKConfig)):
        if name in fields:
            fields[name] = _from_mapping(f"indexer_topk.{name}", cls, fields[name])
    return _from_mapping("indexer_topk", IndexerTopKConfig, fields)


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
        merge_cap: Candidate capacity of the legacy contiguous slab route.
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
    merge_cap: int | None = None
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
        _check_int(owner, "merge_cap", self.merge_cap, 1)
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
    candidate_capacity: int | None = None
    candidate_budget_bytes: int | None = None
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
    _check_int(owner, "candidate_capacity", tuning.candidate_capacity, 1)
    _check_int(owner, "candidate_budget_bytes", tuning.candidate_budget_bytes, 1)
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
        tile_rows: Query rows of a LiteTopK tile, a multiple of 4. Default: FP8 three waves of
            the score kernel for the heads the LiteTopK kernels run with
            (``3 * num_sms * (128 // kernel_heads)``) and at least 12 rows per SM, rounded down
            to a multiple of 4: 1776 rows for 32 and for 64 heads on 148 SMs; MXFP4 4096.
        startup_position: First causal position a LiteTopK tile may start at (a tile also
            needs rows that see the route's HOT prefix), or None when the plan gives LiteTopK
            no row. Default: FP8 8192 before the route's qualified minimum key count (188416 for
            a 196608-key minimum), the start measured with 32 kernel heads, and None with 64
            kernel heads, where no start made LiteTopK faster than the reference backend at
            every measured length (512K, 768K and 1M keys); MXFP4 the position whose row sees
            45056 keys (180224 with four query tokens per key), where a slab tile starts to be
            faster than the reference selector on the same rows. None as well when LiteTopK runs
            zero-padded heads and the reference selector would score the layer with fewer
            heads on its own: no crossover was measured for that case. 0 lets only the HOT
            prefix limit it, the previous integration's MXFP4 plan.
        group_tiles: Consecutive tiles of equal length that share one plan. Default: FP8 8,
            MXFP4 1.
        seed_bootstrap: How the first tile group of a segment gets its HOT seed: ``reference``
            votes with the reference selections of the rows that precede it (or of the
            segment's first tile, then computed by the reference selector, when too few rows
            precede it); ``identity`` starts from the plugin's cold-start seed. Default: FP8
            ``reference``, MXFP4 ``identity``.
        min_litetopk_pairs: The per-segment crossover: a segment uses LiteTopK only when its
            LiteTopK tiles cover at least this many query-key pairs (the visible keys summed
            over their rows, not counting a reference bootstrap tile). Default 0 (every eligible
            segment).
        candidate_capacity: Candidates per query row of the slab of an MXFP4 slab tile, as an
            explicit bound: the slab of a segment's tiles holds as many candidates as the
            segment's last tile sees keys (a tile row has at most one candidate per visible
            key), at least ``max(16384, 32 * topk)``, and at most this value. None bounds it by
            ``candidate_budget_bytes`` instead.
        candidate_budget_bytes: Byte budget of the candidate slab of one MXFP4 slab tile
            (``tile_rows * candidates * 6`` bytes) when ``candidate_capacity`` is None. Default
            1.5 GiB: 65536 candidates per row for 4096-row tiles. A row with more candidates
            than its slab holds reports a capacity status and is recomputed by the reference
            selector. Must hold at least ``max(16384, 32 * topk)`` candidates per row; None for
            FP8 (the paged route's pool is the ``paged_pool_pages_per_row`` plugin setting).
        reference_budget_bytes: Byte budget of the float32 scores of one reference scoring
            call; the top-k kernel needs scratch of about twice that on top. Default: FP8
            2 GiB, MXFP4 1 GiB.
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
        plugin_settings: Import-time settings of the LiteTopK plugin. Default FP8: the tiered
            12K HOT seed, 13 pool pages per row, 2 row tiles, the identity cold-start seed,
            paged tiles of up to ``tile_rows`` rows admitted (the default tiles of the layer's
            own head count: the settings are rendered before the plugin, and with it the
            route's head counts, is loaded, and the tiles of zero-padded heads are shorter)
            and, for precision ``exact``, the policies exact selection needs (``logical-id``
            ties on ``native-fp32`` scores).
            Default MXFP4: the environment the previous integration ran its slab route with
            (32 pool pages per row, 2 row tiles, the identity cold-start seed, no admission
            beyond the qualified tile lengths). Plugin settings are process-wide (the plugins
            read them from the environment), and the two defaults differ in
            ``fp8_paged_admit_max_query_len`` (and in the pool and seed settings), so an FP8
            model (GLM-5) and an MXFP4 model (DeepSeek-V4) with default settings cannot load
            their plugins in one process: run them in separate jobs.
    """

    required: bool
    tile_rows: int
    startup_position: int | None
    group_tiles: int
    seed_bootstrap: Literal["reference", "identity"]
    min_litetopk_pairs: int
    candidate_capacity: int | None
    candidate_budget_bytes: int | None
    reference_budget_bytes: int
    reference_rows_per_call: int | None
    index_order: Literal["ascending", "selector"]
    status_check: Literal["sync_recompute", "sync_recompute_tail", "device_assert"]
    plugin_settings: LiteTopKPluginSettings

    def __post_init__(self) -> None:
        owner = "ResolvedIndexerTopKTuning"
        optional = (
            "startup_position",
            "candidate_capacity",
            "candidate_budget_bytes",
            "reference_rows_per_call",
        )
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
    heads: IndexerHeads | None = None,
) -> ResolvedIndexerTopKTuning:
    """Resolve the selection plan settings of one layer: the single tuning policy seam.

    Derives every value (see :class:`ResolvedIndexerTopKTuning`) and applies the non-None
    overrides of ``tuning``.

    Args:
        tuning: Expert overrides, or None for the derived values.
        fmt: Operand format of the layer: ``fp8`` or ``mxfp4``.
        route: The plugin route that serves the layer, or None when only the reference
            selector runs (its route-dependent values are then unused).
        geometry: The indexer geometry of the layer.
        precision: ``exact`` or ``fast``.
        num_sms: Streaming multiprocessors of the device.
        heads: The negotiated head counts of the layer
            (:func:`~.heads.negotiate_indexer_heads`); None when every selector scores the
            layer's own head count.

    Returns:
        The resolved settings.

    Raises:
        IndexerTopKConfigError: If an argument is invalid, the route serves another format, the
            overrides conflict with the precision or the geometry, or LiteTopK is required but
            the plan gives it no row.
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
    if heads is not None and heads.num_heads != geometry.num_heads:
        raise IndexerTopKConfigError(
            f"the negotiated heads are for {heads.num_heads} indexer heads; this layer has "
            f"{geometry.num_heads}"
        )
    kernel_heads = geometry.num_heads
    if heads is not None and heads.litetopk_heads is not None:
        kernel_heads = heads.litetopk_heads
    startup, why = _default_startup(fmt, route, geometry, heads, kernel_heads)

    if fmt == "fp8":
        tile_rows = _override(tuning.tile_rows, _fp8_tile_rows(num_sms, kernel_heads))
        exact = precision == "exact"
        settings = LiteTopKPluginSettings(
            tie_policy="logical-id" if exact else None,
            score_policy="native-fp32" if exact else None,
            paged_pool_pages_per_row=_FP8_POOL_PAGES_PER_ROW,
            fp8_row_tiles=2,
            fp8_paged_admit_max_query_len=_override(
                tuning.tile_rows, _fp8_tile_rows(num_sms, geometry.num_heads)
            ),
            tiered_seed_12k=_FP8_TIERED_SEED,
            coldstart_identity=True,
        )
        group_tiles, seed_bootstrap = 8, "reference"
        candidate_budget = tuning.candidate_budget_bytes
    else:
        tile_rows = _override(tuning.tile_rows, 4096)
        settings = LiteTopKPluginSettings(
            paged_pool_pages_per_row=32,
            fp8_row_tiles=2,
            fp8_paged_admit_max_query_len=0,
            coldstart_identity=True,
        )
        group_tiles, seed_bootstrap = 1, "identity"
        candidate_budget = _override(tuning.candidate_budget_bytes, CANDIDATE_BUDGET_BYTES)
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
        candidate_capacity=tuning.candidate_capacity,
        candidate_budget_bytes=candidate_budget,
        reference_budget_bytes=_override(
            tuning.reference_budget_bytes, REFERENCE_BUDGET_BYTES[fmt]
        ),
        reference_rows_per_call=tuning.reference_rows_per_call,
        index_order=_override(tuning.index_order, "ascending"),
        status_check=_override(tuning.status_check, "sync_recompute"),
        plugin_settings=settings,
    )
    if (
        precision == "exact"
        and fmt == "fp8"
        and (settings.tie_policy, settings.score_policy) != ("logical-id", "native-fp32")
    ):
        raise IndexerTopKConfigError(
            "precision='exact' needs plugin_settings tie_policy='logical-id' and "
            "score_policy='native-fp32' (exact selection ranks the reference score kernel's "
            f"float32 scores, equal scores by ascending key id); got {settings.tie_policy!r} "
            f"and {settings.score_policy!r}"
        )
    if resolved.candidate_capacity is not None and resolved.candidate_capacity < geometry.topk:
        raise IndexerTopKConfigError(
            f"tuning.candidate_capacity={resolved.candidate_capacity} is below top-k "
            f"{geometry.topk}"
        )
    if fmt == "mxfp4" and resolved.candidate_capacity is None:
        smallest = max(SLAB_MIN_CANDIDATES, SLAB_CANDIDATES_PER_TOPK * geometry.topk)
        if candidate_budget // (tile_rows * SLAB_RECORD_BYTES) < smallest:
            raise IndexerTopKConfigError(
                f"tuning.candidate_budget_bytes={candidate_budget} cannot hold the smallest slab "
                f"of a {tile_rows}-row tile ({smallest} candidates of {SLAB_RECORD_BYTES} bytes "
                "per row)"
            )
    if resolved.required and route is not None and resolved.startup_position is None:
        raise IndexerTopKConfigError(
            f"IndexerTopKTuning.required needs LiteTopK rows, but the default plan gives "
            f"LiteTopK none for {fmt} operands with {kernel_heads} kernel heads ({why}); set "
            "IndexerTopKTuning.startup_position to select with LiteTopK anyway"
        )
    return resolved


def _fp8_tile_rows(num_sms: int, heads: int) -> int:
    rows = num_sms * max(_FP8_TILE_ROWS_PER_SM, _FP8_TILE_WAVES * max(1, 128 // heads))
    return max(4, rows // 4 * 4)


def _default_startup(
    fmt: IndexerTopKFormat,
    route: RouteCapability | None,
    geometry: IndexerGeometry,
    heads: IndexerHeads | None,
    kernel_heads: int,
) -> tuple[int | None, str]:
    """The default first position of LiteTopK tiles, and why it is None when it is."""
    if fmt == "mxfp4":
        startup = _MXFP4_STARTUP_KEYS * geometry.key_ratio
    elif route is None:
        return 0, ""
    elif kernel_heads in _FP8_MEASURED_STARTUP:
        startup = _FP8_MEASURED_STARTUP[kernel_heads]
        if startup is None:
            return None, "no start was faster than the reference selector at every measured length"
    else:
        startup = max(0, route.min_keys - _FP8_STARTUP_MARGIN)
    # The start positions above were measured against a reference selector that scores as many
    # heads as LiteTopK. One that scores fewer, because LiteTopK runs zero-padded heads and the
    # reference score kernel supports the layer's own head count, is faster than measured.
    if route is not None and heads is not None:
        baseline = heads.num_heads if heads.baseline_heads is None else heads.baseline_heads
        if baseline < kernel_heads:
            return None, (
                f"LiteTopK runs {kernel_heads} zero-padded heads, the reference selector "
                f"{baseline}"
            )
    return startup, ""


def _override(value: Any, default: Any) -> Any:
    return default if value is None else value

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The external plugin ABI v1: the module surface Megatron Lite calls and what plugins report.

A LiteTopK plugin module exposes ``LITETOPK_ABI_VERSION == 1`` and the functions of
:class:`LiteTopKModuleV1`; keyword names are part of the ABI. ``plugin_info()`` describes the
source, its routes and the configuration the module snapshotted at import, and is parsed into a
:class:`PluginInfo`. An exact-tie top-k package exposes the function of
:class:`ExactTopKModule`.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from megatron.lite.primitive.kernels.indexer_topk.config import IndexerTopKPluginError

__all__ = [
    "LITETOPK_KERNEL_FILES",
    "PLUGIN_ABI_VERSION",
    "ExactTopKModule",
    "LiteTopKModuleV1",
    "PluginInfo",
    "RouteCapability",
    "parse_plugin_info",
    "validate_litetopk_module",
]

PLUGIN_ABI_VERSION = 1

# The CUDA sources of a LiteTopK plugin, in the order its source id hashes them.
LITETOPK_KERNEL_FILES = ("dsa_litetopk.cu", "sm100_dsa_litetopk.cuh", "dense_topk_litetopk.cuh")

_SOURCE_ID = re.compile(r"[0-9a-f]{12}")
_ROUTE_FORMATS = {"fp8_paged": "fp8"}
_ROUTE_KEYS = frozenset(
    (
        "name",
        "fmt",
        "heads",
        "head_dims",
        "topk",
        "max_topk",
        "qualified_query_lengths",
        "admitted_max_query_len",
        "min_keys",
        "max_keys",
        "hot_prefix",
        "exact",
        "tie_policies",
        "score_policies",
    )
)
_INFO_KEYS = frozenset(
    (
        "abi",
        "source_id",
        "routes",
        "effective_config",
        "launch_time_env_keys",
        "tie_policy",
        "score_policy",
    )
)

# How Megatron Lite calls each ABI function: the number of positional arguments and the
# keyword arguments it may pass. validate_litetopk_module binds these against the signatures.
_ABI_CALLS: dict[str, tuple[int, tuple[str, ...]]] = {
    "plugin_info": (0, ()),
    "load_extension": (
        0,
        ("prebuilt_path", "prebuilt_sha256", "build_dir", "deepgemm_include_dir", "cuda_arch"),
    ),
    "production_min_s": (1, ()),
    "carry_vote_rows": (0, ()),
    "begin_call": (3, ()),
    "prepare_permuted_gather": (
        4,
        ("sequence_length", "query_length", "num_reqs", "common_end", "window_start", "hot_key"),
    ),
    "try_large_exact_once_chunk": (
        8,
        (
            "permuted_plan",
            "num_reqs",
            "ke_min_hint",
            "cap",
            "hot_key",
            "ks_common_hint",
            "carry_extent_hint",
            "carry_recent_rows_hint",
            "q_sf",
            "carry_io",
            "exact",
            "status_out",
        ),
    ),
    "stash_carry": (3, ("min_index", "recent_rows_hint")),
    "drop_carry": (2, ()),
    "release": (1, ("release_scratch",)),
}


@dataclass(frozen=True)
class RouteCapability:
    """One selection route of a LiteTopK plugin, as its ``plugin_info()`` declares it.

    Attributes:
        name: ``fp8_paged`` (FP8 operands, paged candidate pool).
        fmt: The operand format of the route: ``fp8``.
        heads: The indexer head counts the route's kernels are built for.
        head_dims: The supported indexer head dimensions.
        topk: The supported top-k sizes; None means any top-k up to ``max_topk``.
        max_topk: The largest supported top-k.
        qualified_query_lengths: The qualified query tile lengths.
        admitted_max_query_len: With a positive value the route also admits every query tile
            length up to it that is a multiple of four; 0 admits only the qualified lengths.
        min_keys: The shortest key sequence the route is qualified for.
        max_keys: The longest key sequence the route supports.
        hot_prefix: The keys scored for the HOT seed of every query tile.
        exact: Whether the route selects exactly: each selected set equals the exact top-k of
            the reference score kernel on the same operands (score descending, lower key id
            first on equal scores).
        tie_policies: The tie policies the route supports.
        score_policies: The score policies the route supports.
    """

    name: Literal["fp8_paged"]
    fmt: Literal["fp8"]
    heads: frozenset[int]
    head_dims: frozenset[int]
    topk: frozenset[int] | None
    max_topk: int
    qualified_query_lengths: frozenset[int]
    admitted_max_query_len: int
    min_keys: int
    max_keys: int
    hot_prefix: int
    exact: bool
    tie_policies: frozenset[str]
    score_policies: frozenset[str]

    def supports_topk(self, topk: int) -> bool:
        """Return whether the route selects ``topk`` keys per query row."""
        if self.topk is not None:
            return topk in self.topk
        return 0 < topk <= self.max_topk

    def admits_query_length(self, query_length: int) -> bool:
        """Return whether the route accepts a query tile of ``query_length`` rows."""
        return query_length in self.qualified_query_lengths or (
            0 < query_length <= self.admitted_max_query_len and query_length % 4 == 0
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the capability in the ``plugin_info()`` layout, with sorted lists."""
        return {
            "name": self.name,
            "fmt": self.fmt,
            "heads": sorted(self.heads),
            "head_dims": sorted(self.head_dims),
            "topk": None if self.topk is None else sorted(self.topk),
            "max_topk": self.max_topk,
            "qualified_query_lengths": sorted(self.qualified_query_lengths),
            "admitted_max_query_len": self.admitted_max_query_len,
            "min_keys": self.min_keys,
            "max_keys": self.max_keys,
            "hot_prefix": self.hot_prefix,
            "exact": self.exact,
            "tie_policies": sorted(self.tie_policies),
            "score_policies": sorted(self.score_policies),
        }


@dataclass(frozen=True, eq=False)
class PluginInfo:
    """The parsed ``plugin_info()`` of a LiteTopK plugin.

    Attributes:
        abi: The plugin ABI version (always :data:`PLUGIN_ABI_VERSION`).
        source_id: The 12-hex id of the plugin's CUDA sources.
        routes: The selection routes, one per route name.
        effective_config: Every environment key the module read at import, mapped to the value
            it snapshotted (None when the key was unset).
        launch_time_env_keys: The keys the CUDA extension reads on every launch.
        tie_policy: The tie policy the module was configured with.
        score_policy: The score policy the module was configured with.
    """

    abi: int
    source_id: str
    routes: tuple[RouteCapability, ...]
    effective_config: dict[str, str | None]
    launch_time_env_keys: frozenset[str]
    tie_policy: str
    score_policy: str

    def route(self, name: str) -> RouteCapability | None:
        """Return the route called ``name``, or None when the plugin has no such route."""
        return next((route for route in self.routes if route.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy in the ``plugin_info()`` layout, for provenance records."""
        return {
            "abi": self.abi,
            "source_id": self.source_id,
            "routes": [route.as_dict() for route in self.routes],
            "effective_config": dict(sorted(self.effective_config.items())),
            "launch_time_env_keys": sorted(self.launch_time_env_keys),
            "tie_policy": self.tie_policy,
            "score_policy": self.score_policy,
        }


class LiteTopKModuleV1(Protocol):
    """The module surface of a LiteTopK plugin (ABI v1)."""

    LITETOPK_ABI_VERSION: int

    def plugin_info(self) -> Mapping[str, Any]:
        """Describe the source, its routes and the import-time configuration."""
        ...

    def load_extension(
        self,
        *,
        prebuilt_path: str | None,
        prebuilt_sha256: str | None,
        build_dir: str | None,
        deepgemm_include_dir: str | None,
        cuda_arch: str = "10.0a",
    ) -> None:
        """Load the prebuilt CUDA extension or JIT-build it; raise on any failure."""
        ...

    def production_min_s(self, use_fp4: bool) -> int:
        """Return the qualified minimum key count of the plugin's FP8 routes (``use_fp4=False``).

        Megatron Lite reads ``min_keys`` from ``plugin_info()``.
        """
        ...

    def carry_vote_rows(self) -> int:
        """Return the number of recent query rows whose selections vote for a HOT carry."""
        ...

    def begin_call(self, device: torch.device, hot_key: Hashable, sequence_length: int) -> None:
        """Start a selector call: retire rolled-back state and drop the carry of ``hot_key``."""
        ...

    def prepare_permuted_gather(
        self,
        kv_cache: Tensor,
        dst_k: Tensor,
        dst_scale: Tensor,
        block_table: Tensor,
        *,
        sequence_length: int,
        query_length: int,
        num_reqs: int,
        common_end: int,
        window_start: int,
        hot_key: Hashable,
    ) -> object | None:
        """Gather the keys HOT-prefix first for a group of tiles; None declines the group."""
        ...

    def try_large_exact_once_chunk(
        self,
        q: Tensor,
        k: Tensor,
        k_scale: Tensor,
        weights: Tensor,
        ks: Tensor,
        ke: Tensor,
        out_idx: Tensor,
        topk: int,
        *,
        permuted_plan: object,
        num_reqs: int,
        ke_min_hint: int,
        cap: int | None = None,
        hot_key: Hashable | None = None,
        ks_common_hint: int = 0,
        carry_extent_hint: int | None = None,
        carry_recent_rows_hint: int | None = None,
        q_sf: Tensor | None = None,
        carry_io: bool = True,
        exact: bool = False,
        status_out: Tensor | None = None,
    ) -> bool:
        """Select one query tile; False is a host-side decline that wrote nothing."""
        ...

    def stash_carry(
        self,
        hot_key: Hashable,
        idx: Tensor,
        S: int,
        min_index: int = 0,
        *,
        recent_rows_hint: int | None = None,
    ) -> None:
        """Publish a HOT carry for ``hot_key`` from selected key ids."""
        ...

    def drop_carry(self, device: torch.device, hot_key: Hashable) -> None:
        """Remove the carry stored under ``hot_key`` on ``device``, if any."""
        ...

    def release(self, device: torch.device, *, release_scratch: bool = True) -> None:
        """Free the plugin's scratch memory on ``device``; never loads the extension."""
        ...


class ExactTopKModule(Protocol):
    """The module surface of the exact-tie top-k package."""

    def cute_dsl_topk_wrapper(
        self, input_values: Tensor, seq_lens: Tensor, top_k: int, next_n: int, return_val: bool
    ) -> tuple[Tensor, Tensor | None]:
        """Select the ``top_k`` best scores of every row (lower key id first on equal scores)."""
        ...


def validate_litetopk_module(module: object, *, origin: str) -> None:
    """Check that an imported module implements the LiteTopK plugin ABI v1.

    Args:
        module: The imported plugin module.
        origin: The plugin location, used in error messages.

    Raises:
        IndexerTopKPluginError: If the ABI version differs, an ABI function is missing or not
            callable, or a function signature cannot take Megatron Lite's calls.
    """
    version = getattr(module, "LITETOPK_ABI_VERSION", None)
    if type(version) is not int or version != PLUGIN_ABI_VERSION:
        raise IndexerTopKPluginError(
            f"LiteTopK plugin at {origin} exposes ABI {version!r}; Megatron Lite needs "
            f"LITETOPK_ABI_VERSION == {PLUGIN_ABI_VERSION} (see docs/indexer_topk.md#plugin-abi)"
        )
    missing = [name for name in _ABI_CALLS if not callable(getattr(module, name, None))]
    if missing:
        raise IndexerTopKPluginError(
            f"LiteTopK plugin at {origin} lacks the ABI v1 functions {', '.join(missing)}"
        )
    for name, (positional, keywords) in _ABI_CALLS.items():
        function = getattr(module, name)
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            continue  # No introspectable signature (for example a builtin); nothing to check.
        try:
            signature.bind(*([None] * positional), **dict.fromkeys(keywords))
        except TypeError as exc:
            call = ", ".join(["arg"] * positional + [f"{keyword}=" for keyword in keywords])
            raise IndexerTopKPluginError(
                f"LiteTopK plugin at {origin}: {name}{signature} cannot take the ABI v1 call "
                f"{name}({call}): {exc}"
            ) from None


def _fail(origin: str, where: str, problem: str) -> IndexerTopKPluginError:
    return IndexerTopKPluginError(f"LiteTopK plugin at {origin}: plugin_info(){where} {problem}")


def _int_set(origin: str, where: str, value: object, *, allow_empty: bool) -> frozenset[int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or not all(type(item) is int and item > 0 for item in value)
        or (not value and not allow_empty)
    ):
        emptiness = "a" if allow_empty else "a non-empty"
        raise _fail(origin, where, f"must be {emptiness} list of positive integers, got {value!r}")
    return frozenset(value)


def _str_set(origin: str, where: str, value: object) -> frozenset[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise _fail(origin, where, f"must be a non-empty list of strings, got {value!r}")
    return frozenset(value)


def _int(origin: str, where: str, value: object, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(origin, where, f"must be an integer >= {minimum}, got {value!r}")
    return value


def _exact_keys(origin: str, where: str, raw: object, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise _fail(origin, where, f"must be a mapping, got {type(raw).__name__}")
    missing = sorted(keys - set(raw))
    unknown = sorted(set(raw) - keys)
    if missing or unknown:
        raise _fail(origin, where, f"has missing keys {missing} and unknown keys {unknown}")
    return raw


def _parse_route(origin: str, index: int, raw: object) -> RouteCapability:
    where = f"['routes'][{index}]"
    route = _exact_keys(origin, where, raw, _ROUTE_KEYS)
    name = route["name"]
    if name not in _ROUTE_FORMATS:
        raise _fail(origin, f"{where}['name']", f"must be one of {sorted(_ROUTE_FORMATS)}")
    if route["fmt"] != _ROUTE_FORMATS[name]:
        raise _fail(origin, f"{where}['fmt']", f"must be {_ROUTE_FORMATS[name]!r} for {name}")
    max_topk = _int(origin, f"{where}['max_topk']", route["max_topk"], 1)
    topk = None
    if route["topk"] is not None:
        topk = _int_set(origin, f"{where}['topk']", route["topk"], allow_empty=False)
        if max(topk) > max_topk:
            raise _fail(origin, f"{where}['topk']", f"exceeds max_topk {max_topk}")
    min_keys = _int(origin, f"{where}['min_keys']", route["min_keys"], 1)
    if not isinstance(route["exact"], bool):
        raise _fail(origin, f"{where}['exact']", f"must be a bool, got {route['exact']!r}")
    return RouteCapability(
        name=name,
        fmt=route["fmt"],
        heads=_int_set(origin, f"{where}['heads']", route["heads"], allow_empty=False),
        head_dims=_int_set(origin, f"{where}['head_dims']", route["head_dims"], allow_empty=False),
        topk=topk,
        max_topk=max_topk,
        qualified_query_lengths=_int_set(
            origin,
            f"{where}['qualified_query_lengths']",
            route["qualified_query_lengths"],
            allow_empty=True,
        ),
        admitted_max_query_len=_int(
            origin, f"{where}['admitted_max_query_len']", route["admitted_max_query_len"], 0
        ),
        min_keys=min_keys,
        max_keys=_int(origin, f"{where}['max_keys']", route["max_keys"], min_keys),
        hot_prefix=_int(origin, f"{where}['hot_prefix']", route["hot_prefix"], 1),
        exact=route["exact"],
        tie_policies=_str_set(origin, f"{where}['tie_policies']", route["tie_policies"]),
        score_policies=_str_set(origin, f"{where}['score_policies']", route["score_policies"]),
    )


def parse_plugin_info(raw: object, *, origin: str) -> PluginInfo:
    """Validate a ``plugin_info()`` result against the ABI v1 schema and parse it.

    Args:
        raw: The value ``plugin_info()`` returned.
        origin: The plugin location, used in error messages.

    Returns:
        The parsed plugin description.

    Raises:
        IndexerTopKPluginError: If a key is missing or unknown, a value has the wrong type, the
            ABI version differs, or the plugin declares no route or a route twice.
    """
    info = _exact_keys(origin, "", raw, _INFO_KEYS)
    if type(info["abi"]) is not int or info["abi"] != PLUGIN_ABI_VERSION:
        raise _fail(origin, "['abi']", f"must be {PLUGIN_ABI_VERSION}, got {info['abi']!r}")
    source_id = info["source_id"]
    if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
        raise _fail(origin, "['source_id']", f"must be a 12-hex source id, got {source_id!r}")
    raw_routes = info["routes"]
    if not isinstance(raw_routes, Sequence) or isinstance(raw_routes, str) or not raw_routes:
        raise _fail(origin, "['routes']", f"must be a non-empty list, got {raw_routes!r}")
    routes = tuple(_parse_route(origin, index, route) for index, route in enumerate(raw_routes))
    names = [route.name for route in routes]
    if len(set(names)) != len(names):
        raise _fail(origin, "['routes']", f"declares a route twice: {names}")
    effective = info["effective_config"]
    if not isinstance(effective, Mapping) or not all(
        isinstance(key, str) and (value is None or isinstance(value, str))
        for key, value in effective.items()
    ):
        raise _fail(origin, "['effective_config']", "must map key names to strings or None")
    launch_keys = info["launch_time_env_keys"]
    if (
        not isinstance(launch_keys, Sequence)
        or isinstance(launch_keys, str)
        or not all(isinstance(key, str) and key for key in launch_keys)
    ):
        raise _fail(origin, "['launch_time_env_keys']", "must be a list of key names")
    for key in ("tie_policy", "score_policy"):
        if not isinstance(info[key], str) or not info[key]:
            raise _fail(origin, f"[{key!r}]", f"must be a policy name, got {info[key]!r}")
    return PluginInfo(
        abi=info["abi"],
        source_id=source_id,
        routes=routes,
        effective_config=dict(effective),
        launch_time_env_keys=frozenset(launch_keys),
        tie_policy=info["tie_policy"],
        score_policy=info["score_policy"],
    )

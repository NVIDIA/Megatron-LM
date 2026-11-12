# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

#!/usr/bin/env python3
"""Predict H100 or GB200 BF16 causal GQA forward-attention time.

The public function accepts packed logical sequence lengths, CP size, and GQA
geometry. TE P2P uses hardware-specific measured artifacts through CP64.
Compute-only and Ulysses modes intentionally omit communication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from .cp_gqa_attention_predictor import (
    CPGQAAttentionPredictor,
    DEFAULT_PHYSICAL_TOKENS_PER_RANK,
    SUPPORTED_CP_IMPLEMENTATIONS,
    SUPPORTED_CP_SIZES,
    SUPPORTED_PHYSICAL_TOKENS_PER_RANK,
    SUPPORTED_ULYSSES_KV_STRATEGIES,
    normalize_physical_tokens_per_rank,
    predict_compute_only_attention_with_details,
    predict_ulysses_compute_only_attention_with_details,
)
from .cp1_gqa_attention_predictor import FrozenCP1GQAAttentionPredictor
from .gb200_cp1_attention_predictor import GB200CP1AttentionPredictor
from .singleton_sol_attention_predictor import FrozenSingletonSOLPredictor

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "gqa_attention_model_registry.json"
SUPPORTED_HARDWARE = ("h100", "gb200")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if int(registry.get("schema_version", 0)) != 1:
        raise ValueError("unsupported GQA attention registry schema")
    if set(registry.get("hardware", {})) != set(SUPPORTED_HARDWARE):
        raise ValueError("registry hardware set is incomplete")
    return registry


def _hardware_contract(hardware: str) -> dict[str, Any]:
    if hardware not in SUPPORTED_HARDWARE:
        raise ValueError(f"hardware must be one of {SUPPORTED_HARDWARE}")
    return load_registry()["hardware"][hardware]


def _geometry(query_heads: int, kv_heads: int, head_dim: int) -> tuple[int, int, int]:
    return int(query_heads), int(kv_heads), int(head_dim)


def _cp1_profiles(hardware: str) -> dict[tuple[int, int, int], str]:
    return {
        _geometry(
            entry["query_heads"],
            entry["kv_heads"],
            entry["head_dim"],
        ): str(entry["profile_name"])
        for entry in _hardware_contract(hardware)["cp1_geometries"]
    }


def _te_entries(hardware: str) -> dict[tuple[int, int, int], dict[str, Any]]:
    result = {}
    for profile_name, entry in _hardware_contract(hardware)["te_p2p"].items():
        geometry = _geometry(entry["query_heads"], entry["kv_heads"], entry["head_dim"])
        result[geometry] = {"profile_name": profile_name, **entry}
    return result


def _te_storage_branches(entry: dict[str, Any]) -> dict[int, dict[str, str]]:
    """Normalize legacy 4K entries and storage-aware registry entries."""
    raw_branches = entry.get("storage_branches")
    if raw_branches is None:
        return {
            DEFAULT_PHYSICAL_TOKENS_PER_RANK: {
                "artifact": str(entry["artifact"]),
                "sha256": str(entry["sha256"]),
            }
        }
    branches = {
        normalize_physical_tokens_per_rank(int(storage)): {
            "artifact": str(branch["artifact"]),
            "sha256": str(branch["sha256"]),
        }
        for storage, branch in raw_branches.items()
    }
    if not branches:
        raise ValueError("TE-P2P registry entry has no storage branches")
    return branches


def _resolve_te_p2p_storage(
    hardware: str,
    geometry: tuple[int, int, int],
    cp_size: int,
    requested: int | None,
) -> int:
    if requested is not None:
        return normalize_physical_tokens_per_rank(requested)
    if cp_size == 1:
        return DEFAULT_PHYSICAL_TOKENS_PER_RANK

    entry = _te_entries(hardware).get(geometry)
    if entry is None:
        # Preserve the legacy exact-singleton and unsupported-geometry paths.
        return DEFAULT_PHYSICAL_TOKENS_PER_RANK
    branches = tuple(sorted(_te_storage_branches(entry)))
    if len(branches) == 1:
        return branches[0]
    raise ValueError(
        "physical_tokens_per_rank is required for multi-CP TE-P2P prediction "
        f"when multiple calibrated branches are available: {branches}"
    )


def supported_attention_geometries(
    hardware: str = "h100", *, cp_size: int = 1
) -> tuple[tuple[int, int, int], ...]:
    """Return measured geometries for one hardware and CP size."""
    if int(cp_size) == 1:
        geometries = _cp1_profiles(hardware)
    else:
        geometries = _te_entries(hardware)
    return tuple(sorted(geometries))


def supported_physical_tokens_per_rank(
    hardware: str,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[int, ...]:
    """Return calibrated multi-CP physical buffer sizes for one geometry."""
    entry = _te_entries(hardware).get(_geometry(query_heads, kv_heads, head_dim))
    if entry is None:
        return ()
    return tuple(sorted(_te_storage_branches(entry)))


def supported_singleton_sol_geometries(
    hardware: str = "h100", *, cp_size: int = 1
) -> tuple[tuple[int, int, int], ...]:
    """Return geometries with an exact measured singleton SOL lookup."""
    _, predictor = load_singleton_sol_predictor(hardware)
    return predictor.supported_geometries(hardware, cp_size)


def _verified_path(relative_path: str, expected_sha256: str) -> Path:
    path = ROOT / relative_path
    observed = sha256(path)
    if observed != expected_sha256:
        raise ValueError(
            f"model hash mismatch for {path}: expected {expected_sha256}, "
            f"observed {observed}"
        )
    return path


@lru_cache(maxsize=None)
def load_te_p2p_predictor(
    hardware: str,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_tokens_per_rank: int = DEFAULT_PHYSICAL_TOKENS_PER_RANK,
) -> tuple[str, str, CPGQAAttentionPredictor]:
    geometry = _geometry(query_heads, kv_heads, head_dim)
    entry = _te_entries(hardware).get(geometry)
    if entry is None:
        contract = _hardware_contract(hardware)
        unavailable = contract.get("te_p2p_unavailable", {})
        if int(head_dim) == 256 and "head_dim_256" in unavailable:
            raise ValueError(
                f"{hardware} TE-P2P does not support this geometry: "
                f"{unavailable['head_dim_256']}"
            )
        supported = ", ".join(
            f"{hq}/{hkv}/{dimension}"
            for hq, hkv, dimension in supported_attention_geometries(
                hardware, cp_size=2
            )
        )
        raise ValueError(
            f"no measured {hardware} TE-P2P branch for "
            f"{query_heads}/{kv_heads}/{head_dim}; supported: {supported}"
        )
    storage = normalize_physical_tokens_per_rank(physical_tokens_per_rank)
    branches = _te_storage_branches(entry)
    branch = branches.get(storage)
    if branch is None:
        available = tuple(sorted(branches))
        raise ValueError(
            f"no measured {hardware} TE-P2P {storage}-token-per-rank branch for "
            f"{query_heads}/{kv_heads}/{head_dim}; calibrated storage branches: "
            f"{available}"
        )
    path = _verified_path(branch["artifact"], branch["sha256"])
    predictor = CPGQAAttentionPredictor.from_path(path)
    if (predictor.query_heads, predictor.kv_heads, predictor.head_dim) != geometry:
        raise ValueError(f"registry geometry differs from model artifact {path}")
    if predictor.physical_tokens_per_rank != storage:
        raise ValueError(
            f"registry selects {storage} physical tokens per rank but {path} "
            f"contains {predictor.physical_tokens_per_rank}"
        )
    return str(entry["profile_name"]), str(branch["sha256"]), predictor


@lru_cache(maxsize=1)
def load_gb200_cp1_predictor() -> GB200CP1AttentionPredictor:
    contract = _hardware_contract("gb200")["cp1"]
    standard = _verified_path(
        contract["standard_artifact"], contract["standard_sha256"]
    )
    backend = _verified_path(
        contract["backend_state_artifact"], contract["backend_state_sha256"]
    )
    return GB200CP1AttentionPredictor.from_paths(standard, backend)


@lru_cache(maxsize=None)
def load_fixed_storage_cp1_predictor(
    hardware: str,
) -> tuple[str, FrozenCP1GQAAttentionPredictor] | None:
    """Load the optional calibrated 4K/8K/16K CP1 artifact."""
    contract = _hardware_contract(hardware).get("cp1_fixed_storage")
    if contract is None:
        return None
    path = _verified_path(contract["artifact"], contract["sha256"])
    predictor = FrozenCP1GQAAttentionPredictor.from_path(path)
    expected_gpu = str(_hardware_contract(hardware)["gpu"])
    if not predictor.gpu.startswith(expected_gpu):
        raise ValueError(
            f"registry hardware {hardware} differs from CP1 artifact GPU "
            f"{predictor.gpu!r}"
        )
    return str(contract["sha256"]), predictor


@lru_cache(maxsize=None)
def load_singleton_sol_predictor(
    hardware: str,
) -> tuple[str, FrozenSingletonSOLPredictor]:
    contract = _hardware_contract(hardware)["singleton_sol"]
    path = _verified_path(contract["artifact"], contract["sha256"])
    return str(contract["sha256"]), FrozenSingletonSOLPredictor.from_path(path)


def _cp1_model(hardware: str) -> Any | None:
    fixed_storage = load_fixed_storage_cp1_predictor(hardware)
    if fixed_storage is not None:
        return fixed_storage[1]
    return load_gb200_cp1_predictor() if hardware == "gb200" else None


def _profile_name(hardware: str, query_heads: int, kv_heads: int, head_dim: int) -> str:
    return _cp1_profiles(hardware).get(
        _geometry(query_heads, kv_heads, head_dim), "variable_head_fallback"
    )


def _singleton_sol_details(
    sequence_lengths: Sequence[int],
    hardware: str,
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    physical_tokens_per_rank: int,
) -> dict[str, Any] | None:
    if physical_tokens_per_rank != DEFAULT_PHYSICAL_TOKENS_PER_RANK:
        return None
    model_sha256, predictor = load_singleton_sol_predictor(hardware)
    point = predictor.lookup(
        sequence_lengths,
        hardware,
        cp_size,
        query_heads,
        kv_heads,
        head_dim,
    )
    if point is None:
        return None
    return {
        "time_ms": point.time_ms,
        "cp_size": point.cp_size,
        "cp_implementation": "te_p2p",
        "communication_included": point.cp_size > 1,
        "profile_name": point.profile_name,
        "hardware": point.hardware,
        "gpu": _hardware_contract(hardware)["gpu"],
        "geometry": {
            "query_heads": point.query_heads,
            "kv_heads": point.kv_heads,
            "head_dim": point.head_dim,
        },
        "physical_tokens_per_rank": physical_tokens_per_rank,
        "sequence_count": 1,
        "model_kind": "exact_singleton_sol_lookup",
        "model_sha256": model_sha256,
        "measurement_run": point.measurement_run,
        "measurement_provenance": point.measurement_provenance,
        "prediction_envelope": {
            "policy": "exact_only",
            "sequence_lengths": [point.sequence_length],
            "cp_size": point.cp_size,
        },
    }


def _gb200_cp1_details(
    sequence_lengths: Sequence[int],
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    geometry = _geometry(query_heads, kv_heads, head_dim)
    profiles = _cp1_profiles("gb200")
    if geometry not in profiles:
        supported = ", ".join(
            f"{hq}/{hkv}/{dimension}"
            for hq, hkv, dimension in supported_attention_geometries("gb200")
        )
        raise ValueError(
            f"no measured gb200 CP1 branch for {query_heads}/{kv_heads}/{head_dim}; "
            f"supported: {supported}"
        )
    lengths = [int(value) for value in sequence_lengths]
    time_ms = load_gb200_cp1_predictor().predict_time_ms(
        lengths, query_heads, kv_heads, head_dim
    )
    total = sum(lengths)
    storage = 4096 if total <= 4096 else 8192
    cp1_contract = _hardware_contract("gb200")["cp1"]
    head_dim_256 = int(head_dim) == 256
    return {
        "time_ms": time_ms,
        "cp_size": 1,
        "cp_implementation": "te_p2p",
        "communication_included": False,
        "profile_name": profiles[geometry],
        "hardware": "gb200",
        "gpu": _hardware_contract("gb200")["gpu"],
        "geometry": {
            "query_heads": int(query_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
        },
        "storage_tokens": storage,
        "backend_state": "exact_storage" if total == storage else "tail_padded",
        "model_sha256": (
            cp1_contract["backend_state_sha256"]
            if head_dim_256
            else cp1_contract["standard_sha256"]
        ),
    }


def _fixed_storage_cp1_details(
    sequence_lengths: Sequence[int],
    hardware: str,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    storage: int,
    model_sha256: str,
    predictor: FrozenCP1GQAAttentionPredictor,
    max_cp_size: int | None,
) -> dict[str, Any]:
    geometry = _geometry(query_heads, kv_heads, head_dim)
    if not predictor.has_geometry(geometry):
        supported = ", ".join(
            f"{hq}/{hkv}/{dimension}"
            for hq, hkv, dimension in sorted(predictor.models_by_geometry)
        )
        raise ValueError(
            f"no measured {hardware} CP1 branch for "
            f"{query_heads}/{kv_heads}/{head_dim}; supported: {supported}"
        )
    time_ms = predictor.predict_time_ms(
        sequence_lengths,
        query_heads,
        kv_heads,
        head_dim,
        max_cp_size=max_cp_size,
        storage_tokens=storage,
    )
    layout = predictor.physical_layout(
        sequence_lengths,
        max_cp_size=max_cp_size,
        storage_tokens=storage,
    )
    model_storage = predictor.model_storage_tokens(
        sequence_lengths,
        query_heads,
        kv_heads,
        head_dim,
        max_cp_size=max_cp_size,
    )
    return {
        "time_ms": time_ms,
        "cp_size": 1,
        "cp_implementation": "te_p2p",
        "communication_included": False,
        "profile_name": predictor.models_by_geometry[geometry].name,
        "hardware": hardware,
        "gpu": _hardware_contract(hardware)["gpu"],
        "geometry": {
            "query_heads": int(query_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
        },
        "physical_tokens_per_rank": storage,
        "max_cp_size": layout.max_cp_size,
        "sequence_alignment": layout.sequence_alignment,
        "storage_tokens": layout.storage_tokens,
        "model_storage_tokens": model_storage,
        "per_sequence_padded_lengths": list(layout.per_sequence_padded_lengths),
        "final_padded_lengths": list(layout.final_padded_lengths),
        "per_sequence_padding_tokens": layout.per_sequence_padding_tokens,
        "tail_padding_tokens": layout.tail_padding_tokens,
        "real_sequence_count": len(layout.logical_lengths),
        "tail_padding_is_sequence": False,
        "pad_between_seqs": bool(
            layout.per_sequence_padding_tokens or layout.tail_padding_tokens
        ),
        "model_kind": "fixed_storage_cp1_fit",
        "tail_padding_time_invariant": True,
        "model_sha256": model_sha256,
    }


def predict_attention_with_details(
    sequence_lengths: Sequence[int],
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    hardware: str = "h100",
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
    cp_implementation: str = "te_p2p",
    ulysses_kv_strategy: str = "replicate",
    physical_tokens_per_rank: int | None = None,
) -> dict[str, Any]:
    """Predict forward core-attention wall time in milliseconds."""
    contract = _hardware_contract(hardware)
    if cp_implementation not in SUPPORTED_CP_IMPLEMENTATIONS:
        raise ValueError(
            f"cp_implementation must be one of {SUPPORTED_CP_IMPLEMENTATIONS}"
        )
    cp_size = int(cp_size)
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if max_cp_size is not None:
        max_cp_size = int(max_cp_size)
        if max_cp_size not in SUPPORTED_CP_SIZES or max_cp_size < cp_size:
            raise ValueError(
                f"max_cp_size must be in {SUPPORTED_CP_SIZES} and >= cp_size"
            )
    if sequence_alignment is not None:
        sequence_alignment = int(sequence_alignment)
        if cp_implementation != "te_p2p":
            raise ValueError(
                "sequence_alignment is only supported by the te_p2p predictor"
            )
        if cp_size == 1:
            raise ValueError(
                "explicit sequence_alignment is currently supported only for CP > 1"
            )
    storage = _resolve_te_p2p_storage(
        hardware,
        _geometry(query_heads, kv_heads, head_dim),
        cp_size,
        physical_tokens_per_rank,
    )
    fixed_storage_cp1 = (
        load_fixed_storage_cp1_predictor(hardware) if cp_size == 1 else None
    )
    if fixed_storage_cp1 is not None and physical_tokens_per_rank is None:
        storage = fixed_storage_cp1[1].physical_layout(
            sequence_lengths, max_cp_size=max_cp_size
        ).storage_tokens

    if cp_implementation == "te_p2p":
        singleton_sol = _singleton_sol_details(
            sequence_lengths,
            hardware,
            cp_size,
            query_heads,
            kv_heads,
            head_dim,
            storage,
        )
        if singleton_sol is not None:
            return singleton_sol
        if fixed_storage_cp1 is not None:
            model_sha256, predictor = fixed_storage_cp1
            return _fixed_storage_cp1_details(
                sequence_lengths,
                hardware,
                query_heads,
                kv_heads,
                head_dim,
                storage,
                model_sha256,
                predictor,
                max_cp_size,
            )
        if (
            hardware == "gb200"
            and cp_size == 1
            and storage == DEFAULT_PHYSICAL_TOKENS_PER_RANK
        ):
            return _gb200_cp1_details(sequence_lengths, query_heads, kv_heads, head_dim)
        profile_name, model_sha256, predictor = load_te_p2p_predictor(
            hardware,
            query_heads,
            kv_heads,
            head_dim,
            storage,
        )
        result = predictor.predict_with_details(
            sequence_lengths,
            cp_size,
            max_cp_size=max_cp_size,
            sequence_alignment=sequence_alignment,
            physical_tokens_per_rank=storage,
        )
        return {
            "profile_name": profile_name,
            "hardware": hardware,
            "gpu": contract["gpu"],
            "geometry": {
                "query_heads": int(query_heads),
                "kv_heads": int(kv_heads),
                "head_dim": int(head_dim),
            },
            "model_sha256": model_sha256,
            "cp_implementation": "te_p2p",
            "communication_included": cp_size > 1,
            **result,
        }

    cp1_model = _cp1_model(hardware)
    if cp_implementation == "compute_only":
        result = predict_compute_only_attention_with_details(
            sequence_lengths,
            cp_size,
            query_heads,
            kv_heads,
            head_dim,
            max_cp_size=max_cp_size,
            physical_tokens_per_rank=physical_tokens_per_rank,
            cp1_model_artifact=cp1_model,
        )
    else:
        result = predict_ulysses_compute_only_attention_with_details(
            sequence_lengths,
            cp_size,
            query_heads,
            kv_heads,
            head_dim,
            max_cp_size=max_cp_size,
            ulysses_kv_strategy=ulysses_kv_strategy,
            physical_tokens_per_rank=physical_tokens_per_rank,
            cp1_model_artifact=cp1_model,
        )
    return {
        "profile_name": _profile_name(hardware, query_heads, kv_heads, head_dim),
        "hardware": hardware,
        "gpu": contract["gpu"],
        "geometry": {
            "query_heads": int(query_heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
        },
        "model_sha256": None,
        **result,
    }


def predict_attention_time_ms(
    sequence_lengths: Sequence[int],
    cp_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    hardware: str = "h100",
    max_cp_size: int | None = None,
    sequence_alignment: int | None = None,
    cp_implementation: str = "te_p2p",
    ulysses_kv_strategy: str = "replicate",
    physical_tokens_per_rank: int | None = None,
) -> float:
    """Return predicted forward core-attention wall time in milliseconds."""
    return float(
        predict_attention_with_details(
            sequence_lengths,
            cp_size,
            query_heads,
            kv_heads,
            head_dim,
            hardware=hardware,
            max_cp_size=max_cp_size,
            sequence_alignment=sequence_alignment,
            cp_implementation=cp_implementation,
            ulysses_kv_strategy=ulysses_kv_strategy,
            physical_tokens_per_rank=physical_tokens_per_rank,
        )["time_ms"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=SUPPORTED_HARDWARE, default="h100")
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--cp-size", type=int, required=True)
    parser.add_argument(
        "--max-cp-size",
        type=int,
        choices=SUPPORTED_CP_SIZES,
        help=(
            "Scheduler maximum CP size. Without --sequence-alignment, physical "
            "per-sequence alignment is 2 * max_cp_size."
        ),
    )
    parser.add_argument(
        "--sequence-alignment",
        type=int,
        help=(
            "Explicit alignment for non-divisor CP sets; must be divisible "
            "by 2 * CP."
        ),
    )
    parser.add_argument("--query-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument(
        "--physical-tokens-per-rank",
        type=int,
        choices=SUPPORTED_PHYSICAL_TOKENS_PER_RANK,
    )
    parser.add_argument(
        "--cp-implementation",
        choices=SUPPORTED_CP_IMPLEMENTATIONS,
        default="te_p2p",
    )
    parser.add_argument(
        "--ulysses-kv-strategy",
        choices=SUPPORTED_ULYSSES_KV_STRATEGIES,
        default="replicate",
    )
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    result = predict_attention_with_details(
        [int(value) for value in args.lengths.split(",") if value],
        args.cp_size,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
        hardware=args.hardware,
        max_cp_size=args.max_cp_size,
        sequence_alignment=args.sequence_alignment,
        cp_implementation=args.cp_implementation,
        ulysses_kv_strategy=args.ulysses_kv_strategy,
        physical_tokens_per_rank=args.physical_tokens_per_rank,
    )
    if args.details:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['time_ms']:.9f}")


if __name__ == "__main__":
    main()

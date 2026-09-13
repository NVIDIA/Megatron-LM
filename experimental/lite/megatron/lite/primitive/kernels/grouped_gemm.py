# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optional grouped GEMM kernel accessors."""

from __future__ import annotations

try:
    import grouped_gemm  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional fused kernel or missing shared libraries.
    grouped_gemm = None  # type: ignore[assignment]


def grouped_gemm_is_available() -> bool:
    return grouped_gemm is not None


def assert_grouped_gemm_is_available() -> None:
    if grouped_gemm is None:
        raise AssertionError(
            "Grouped GEMM is not available. Please install the grouped_gemm package."
        )


ops = grouped_gemm.ops if grouped_gemm_is_available() else None


__all__ = ["assert_grouped_gemm_is_available", "grouped_gemm_is_available", "ops"]

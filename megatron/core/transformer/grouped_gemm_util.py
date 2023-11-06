# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

def grouped_gemm_is_available():
    return grouped_gemm is not None

def assert_grouped_gemm_is_available():
    assert grouped_gemm_is_available(), (
        "Grouped GEMM not available. Please run "
        "`pip install git+https://github.com/tgale96/grouped_gemm@main`.")

ops = grouped_gemm.ops if grouped_gemm_is_available() else None
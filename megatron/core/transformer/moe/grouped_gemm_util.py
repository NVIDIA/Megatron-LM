# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from importlib.metadata import version

from pkg_resources import packaging

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


def grouped_gemm_is_available():
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    assert grouped_gemm_is_available(), (
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.2`."
    )

    _gg_version = packaging.version.Version(version("grouped_gemm"))
    assert _gg_version >= packaging.version.Version("1.1.2"), (
        "Grouped GEMM should be v1.1.2 or newer. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.2`."
    )


ops = grouped_gemm.ops if grouped_gemm_is_available() else None

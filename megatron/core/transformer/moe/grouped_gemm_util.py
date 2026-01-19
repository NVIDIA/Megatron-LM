# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


def grouped_gemm_is_available():
    """Check if grouped_gemm is available."""
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    """Assert that grouped_gemm is available."""
    error_msg = (
        "The legacy Grouped GEMM is not available. To use MoE with the legacy grouped GEMM, "
        "you need to install nv-grouped-gemm.\n\n"
        "Installation options:\n"
        "1. Install from PyPI (requires CUDA toolkit and CUTLASS headers):\n"
        "   pip install 'megatron-core[moe]'\n"
        "   or\n"
        "   pip install nv-grouped-gemm\n\n"
        "2. Build from source:\n"
        "   pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4\n\n"
        "Note: Building from source requires:\n"
        "- CUDA toolkit (nvcc)\n"
        "- CUTLASS headers (can be installed via 'apt-get install libcutlass-dev' on Ubuntu)\n"
        "- Compatible GPU with compute capability >= 8.0\n\n"
        "We recommend installing TE (>= 1.9) for more comprehensive grouped gemm support. "
        "If you don't have TE nor nv-grouped-gemm, please remove the '--moe-grouped-gemm' argument."
    )
    assert grouped_gemm_is_available(), error_msg


ops = grouped_gemm.ops if grouped_gemm_is_available() else None

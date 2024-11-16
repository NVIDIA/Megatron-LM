# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch


try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None


def grouped_gemm_is_available():
    """Check if grouped_gemm is available."""
    return grouped_gemm is not None


def assert_grouped_gemm_is_available():
    """Assert that grouped_gemm is available."""
    assert grouped_gemm_is_available(), (
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`."
    )

class TorchGroupedGemm:

    @staticmethod
    def gmm(A:torch.Tensor, B:torch.Tensor, num_groups:int, trans_b:bool=False) -> torch.Tensor:

        if trans_b:
            B =torch.transpose(B, 0, 1)
        batch_size, m, k = A.shape
        _, k, n = B.shape

        A = A.reshape(num_groups, batch_size // num_groups, m, k)
        B = B.reshape(num_groups, batch_size // num_groups, k, n)

        C = torch.bmm(A, B)
        C = C.reshape(batch_size, m, n)
        return C

ops = grouped_gemm.ops if grouped_gemm_is_available() else TorchGroupedGemm

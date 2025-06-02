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
    def gmm(a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor, trans_b:bool=False):

        assert torch.all(batch_sizes != 0), "Input batch_sizes should not be all zeros!"
        batch_sizes = torch.split(batch_sizes, 1, dim=0)

        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            B = torch.transpose(b[i, :, :], 0, 1) if trans_b else b[i, :, :]
            A = a[start:start + size, :]
            C = torch.mm(A, B)
            out.append(C)
            start += size
        
        result = torch.cat(out).to(device=a.device)
        return result

ops = grouped_gemm.ops if grouped_gemm_is_available() else TorchGroupedGemm

import logging
import math
import os
import sys

import torch
import transformer_engine.pytorch.cpp_extensions.gemm  # type: ignore
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor  # type: ignore
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor  # type: ignore
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import (  # type: ignore
    MXFP8TensorStorage,
)
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (  # type: ignore
    NVFP4TensorStorage,
)

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_GLOBAL_IS_PATCHED_ = None

orig_general_gemm = transformer_engine.pytorch.cpp_extensions.gemm.general_gemm
orig_general_grouped_gemm = transformer_engine.pytorch.cpp_extensions.gemm.general_grouped_gemm

CKSUM_TOLERANCE = 0.01
EPSILON = 1e-5

DTYPES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def get_rank_info():
    """Return rank and device info for logging."""
    rank = os.environ.get("RANK", 0)
    node: str = os.uname()[1]
    device: int = torch.cuda.current_device()
    return f"{rank}|{node}|{device}"


def get_dtype(t):
    """Return dtype info about a quantized tensor."""
    if isinstance(t, (MXFP8Tensor, MXFP8TensorStorage)):
        return "mxfp8"
    elif isinstance(t, (NVFP4Tensor, NVFP4TensorStorage)):
        return "nvfp4"
    elif isinstance(t, torch.Tensor):
        return DTYPES.get(t.dtype, "torch-unknown")
    else:
        return "unknown"


def new_general_gemm(A, B, workspace, out=None, accumulate=False, bias=None, **kwargs):
    """
    A wrapper for general_gemm that calls the original method twice
    and compares results for consistency.
    """
    cksum = [0, 0]
    # Run twice for comparison.
    for run_idx in range(2):
        is_first_iter = run_idx == 0
        need_output_copy = out is not None and (bias is out or accumulate)
        if need_output_copy and is_first_iter:
            out_arg = out.clone().detach()
            if bias == out:
                bias_arg = out_arg
            else:
                bias_arg = bias
        else:
            out_arg = out
            bias_arg = bias

        res = orig_general_gemm(
            A, B, workspace, out=out_arg, accumulate=accumulate, bias=bias_arg, **kwargs
        )
        out_res = res[0]
        cksum[run_idx] = out_res.sum(dtype=torch.float32).item()
        if is_first_iter:
            rerun_res = out_res
            continue

    # Check the consistency of the first and second run and return.
    same_cksum = math.fabs(cksum[1] - cksum[0]) / (math.fabs(cksum[1]) + EPSILON) <= CKSUM_TOLERANCE
    if not same_cksum:
        logger.log(
            logging.ERROR,
            f"RANK {get_rank_info()}: DIFFERENT CHECKSUM ON 2ND RUN for dtype "
            f"{get_dtype(A)}. {cksum[1]} != {cksum[0]}, "
            f"shapes A:{list(A.size())},B:{list(B.size())},out:{list(out_res.shape)}",
        )
    elif not torch.equal(rerun_res, out_res):
        logger.log(
            logging.ERROR,
            f"RANK {get_rank_info()}: DIFFERENT BITWISE on 2ND RUN for dtype "
            f"{get_dtype(A)}. "
            f"shapes A:{list(A.size())},B:{list(B.size())},out:{list(out_res.shape)}",
        )
    del rerun_res
    return res


def new_general_grouped_gemm(
    A, B, out, out_dtype, workspaces, grad=False, accumulate=False, bias=None, **kwargs
):
    """
    A wrapper for general_grouped_gemm that calls the original method twice
    and compares results for consistency.
    """
    saved_output = None
    out_first_iter = None
    bias_first_iter = None
    # Run twice for comparison
    for run_idx in range(2):
        is_first_iter = run_idx == 0
        if is_first_iter:
            out_first_iter = (
                [t.clone().detach() for t in out]
                if out is not None and (accumulate or out is bias)
                else out
            )
            bias_first_iter = out_first_iter if bias is not None and out is bias else bias
        res = orig_general_grouped_gemm(
            A,
            B,
            out_first_iter if is_first_iter else out,
            out_dtype,
            workspaces,
            grad=grad if is_second_iter else False,
            accumulate=accumulate,
            bias=bias_first_iter if is_first_iter else bias,
            **kwargs,
        )
        output = res[0]
        if is_first_iter:
            saved_output = [t.clone().detach() for t in output]
            continue

    # Check comparison and return
    for tidx in range(len(output)):
        if not torch.equal(output[tidx], saved_output[tidx]):
            logging.log(
                logging.ERROR,
                f"RANK {get_rank_info()}: DIFFERENT GROUPED_GEMM RESULT ON 2ND RUN FOR OP "
                f"GroupedGEMM group {tidx} = {get_dtype(A[tidx])}({list(A[tidx].size())}) "
                f"x {get_dtype(B[tidx])}({list(B[tidx].size())}) -> "
                f"{get_dtype(output[tidx])}({list(output[tidx].size())})",
            )
    return res


def patch_te_gemms() -> None:
    """
    Replaces general_gemm and general_grouped_gemm methods with wrappers to repeat
    calculations and check consistency.
    """
    global _GLOBAL_IS_PATCHED_
    if _GLOBAL_IS_PATCHED_ is not None and _GLOBAL_IS_PATCHED_:
        return
    transformer_engine.pytorch.cpp_extensions.gemm.general_gemm = new_general_gemm
    transformer_engine.pytorch.cpp_extensions.gemm.general_grouped_gemm = new_general_grouped_gemm

    for module in sys.modules:
        if 'general_gemm' in dir(sys.modules[module]):
            if sys.modules[module].general_gemm is orig_general_gemm:
                log_single_rank(logger, logging.INFO, f"PATCHING general_gemm IN MODULE: {module}")
                sys.modules[module].general_gemm = new_general_gemm  # type: ignore[attr-defined]
            else:
                log_single_rank(
                    logger, logging.WARN, f"SKIP PATCHING general_gemm IN MODULE: {module}"
                )
        if 'general_grouped_gemm' in dir(sys.modules[module]):
            if sys.modules[module].general_grouped_gemm is orig_general_grouped_gemm:
                log_single_rank(
                    logger, logging.INFO, f"PATCHING general_grouped_gemm IN MODULE: {module}"
                )
                sys.modules[module].general_grouped_gemm = (  # type: ignore[attr-defined]
                    new_general_grouped_gemm
                )
            else:
                log_single_rank(
                    logger, logging.WARN, f"SKIP PATCHING general_gemm IN MODULE: {module}"
                )
    _GLOBAL_IS_PATCHED_ = True

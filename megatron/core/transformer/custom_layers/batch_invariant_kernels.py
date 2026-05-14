# Copyright 2025 Thinking Machines Lab
# The following code has been adapted
# from the following repo: https://github.com/thinking-machines-lab/batch_invariant_ops


import contextlib
import importlib
import importlib.util
import logging
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

try:
    import deep_gemm

    HAVE_DEEPGEMM_BF16 = all(
        hasattr(deep_gemm, name)
        for name in (
            "m_grouped_bf16_gemm_nt_contiguous",
            "k_grouped_bf16_gemm_tn_contiguous",
            "bf16_gemm_nt",
        )
    )
except ImportError:
    deep_gemm = None
    HAVE_DEEPGEMM_BF16 = False

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
    "grouped_gemm_batch_invariant",
    "BatchInvariantGroupedGemmFn",
    "HAVE_DEEPGEMM_BF16",
    "deterministic_index_add",
]


_LOGGER = logging.getLogger(__name__)


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Build launch metadata for Triton matmul kernels used in BIK matmul."""
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Persistent matmul Triton kernel backing `matmul_persistent`."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def get_compute_units():
    """
    Returns the number of streaming multiprocessors (SMs) or equivalent compute units
    for the available accelerator. Assigns the value to NUM_SMS.
    """
    NUM_SMS = None
    device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")

    # Use match/case for device-specific logic (Python 3.10+)
    match device_type:
        case "cuda":
            device_properties = torch.cuda.get_device_properties(0)
            NUM_SMS = device_properties.multi_processor_count
        case "xpu":
            device_properties = torch.xpu.get_device_properties(0)
            NUM_SMS = device_properties.max_compute_units
        case _:
            _LOGGER.warning("No CUDA or XPU device available. Using CPU.")
            # For CPU, you might want to use the number of CPU cores
            NUM_SMS = torch.get_num_threads()

    return NUM_SMS


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    """Persistent matmul kernel used by batch-invariant GEMM."""
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert (
        bias is None or bias.dim() == 1
    ), "Currently assuming bias is 1D, let Horace know if you run into this"

    NUM_SMS = get_compute_units()
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        blocks_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
        blocks_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
        return (min(NUM_SMS, blocks_m * blocks_n),)

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c


@triton.jit
def _log_softmax_kernel(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 or last dim supported)
    >> Stashed changes
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")
    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d, output, input_2d.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def deterministic_index_add(
    out: torch.Tensor,
    idx: torch.Tensor,
    src: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deterministic, CUDA-graph-compatible equivalent of `out.index_add_(0, idx, src)`.

    Math: ``out[i] += sum over j where idx[j]==i of src[j]``.

    Unlike `torch.Tensor.index_add_`, the result is bitwise-stable across
    runs *without* requiring `torch.use_deterministic_algorithms(True)`, and
    captures cleanly into a `torch.cuda.CUDAGraph` (no `.item()`, no
    data-dependent shapes).

    Algorithm: sort by `idx` (duplicates become contiguous segments) →
    inclusive fp32 cumsum on the sorted src → `searchsorted` gives a fixed
    `(M+1,)` boundary tensor → segment sums via gather + subtract.

    Args:
        out: (M, H) destination, updated in place.
        idx: (N,) integer indices in [0, M). Need not be sorted; we sort
            internally with stable=True.
        src: (N, H) contributions in the same dtype as `out` (any precision —
            cumsum accumulates in fp32 then casts back).
        valid_mask: optional (N,) bool. Rows with mask=False contribute zero;
            their idx is clamped to a valid range so the corresponding update
            is a no-op. Use this instead of slicing `src[:n_used]` to keep
            shapes static under CG capture.
    Returns:
        `out` (same tensor, updated).

    Memory: an fp32 (N+1, H) cumsum buffer; ~4*N*H bytes peak.
    """
    if valid_mask is not None:
        src = src * valid_mask.unsqueeze(-1).to(src.dtype)
        idx = idx.clamp(min=0, max=out.shape[0] - 1)

    sorted_idx, perm = idx.sort(stable=True)
    sorted_src = src.index_select(0, perm)
    H = sorted_src.shape[-1]
    csum = sorted_src.to(torch.float32).cumsum(dim=0)
    zero = torch.zeros(1, H, device=csum.device, dtype=torch.float32)
    csum_with_zero = torch.cat([zero, csum], dim=0)

    M = out.shape[0]
    queries = torch.arange(M + 1, device=idx.device, dtype=sorted_idx.dtype)
    boundaries = torch.searchsorted(sorted_idx, queries)
    seg_sum = csum_with_zero[boundaries[1:]] - csum_with_zero[boundaries[:-1]]
    out.add_(seg_sum.to(out.dtype))
    return out


# Kernel backend for mm / addmm. Selected at `enable_batch_invariant_mode` time
# from `TransformerConfig.batch_invariant_kernel_backend`.
#   "deepgemm" (default): DeepGEMM `bf16_gemm_nt` — bitwise-identical to
#       `torch.mm`. Requires bf16 CUDA inputs on Hopper/Blackwell.
#   "triton": BIK Triton `matmul_persistent` — works on any CUDA device with
#       bf16/fp16/fp32. Has small rounding drift vs `torch.mm`.
_BIK_BACKENDS = ("deepgemm", "triton")
_BIK_BACKEND: str = "deepgemm"


def _mm_deepgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a @ b` via DeepGEMM `bf16_gemm_nt`.

    `aten::mm` is (M, K) @ (K, N). DeepGEMM NT layout is (M, K) @ (N, K).T,
    so we transpose B before passing. Bitwise-identical to `torch.mm` on
    Hopper/Blackwell, deterministic across runs, batch-invariant.
    """
    if a.dtype != torch.bfloat16:
        raise RuntimeError(
            f"batch_invariant_kernel_backend='deepgemm' requires bf16 inputs "
            f"(got {a.dtype}); use backend='triton' for fp16/fp32."
        )
    M = a.shape[0]
    N = b.shape[1]
    d = torch.empty(M, N, device=a.device, dtype=a.dtype)
    deep_gemm.bf16_gemm_nt(a, b.transpose(0, 1).contiguous(), d)
    return d


def mm_batch_invariant(a, b):
    """Batch-invariant replacement for `aten::mm`."""
    if _BIK_BACKEND == "deepgemm":
        return _mm_deepgemm(a, b)
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    """Batch-invariant replacement for `aten::addmm`."""
    if _BIK_BACKEND == "deepgemm":
        out = _mm_deepgemm(a, b)
        if bias is not None:
            out = out + bias
        return out
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    """Batch-invariant replacement for `aten::mean.dim` over one or more dimensions."""
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        assert input.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }, "only float types supported for now"
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    """Return the (block_m, block_n) tiling used for batch-invariant attention."""
    return AttentionBlockSize(block_m=16, block_n=16)


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_ORIG_FWD = None
_MEG_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_FUNC_ORIGS: Dict[str, Any] = {}
_TE_GEMM_FUNC_ORIGS: Dict[str, Any] = {}
_TE_GROUPED_GEMM_FUNC_ORIGS: Dict[str, Any] = {}


def _import_module_if_available(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return importlib.import_module(name)


def _te_patch_for_batch_invariant():
    """Patch Transformer Engine modules to use batch-invariant GEMM and RMSNorm.

    This monkey-patches TE's GEMM and RMSNorm entry points to dispatch to the
    batch-invariant implementations when batch-invariant mode is enabled.
    Safe no-op if TE is unavailable.
    """
    global _TE_GENERAL_GEMM_ORIG, _TE_RMSNORM_ORIG_FWD, _MEG_TE_GENERAL_GEMM_ORIG
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.cpp_extensions as te_cpp

    # Patch general_gemm once
    if _TE_GENERAL_GEMM_ORIG is None and hasattr(te_cpp, "general_gemm"):
        _TE_GENERAL_GEMM_ORIG = te_cpp.general_gemm
        te_cpp.general_gemm = _te_general_gemm_patched

    # Also patch the symbol imported inside TE's module.linear
    # (from ..cpp_extensions import general_gemm)
    import transformer_engine.pytorch.module.linear as te_linear_mod

    if hasattr(te_linear_mod, "general_gemm"):
        if "module.linear.general_gemm" not in _TE_GEMM_FUNC_ORIGS:
            _TE_GEMM_FUNC_ORIGS["module.linear.general_gemm"] = te_linear_mod.general_gemm
            te_linear_mod.general_gemm = _te_general_gemm_patched

    # Also patch the symbol imported inside TE's module.layernorm_linear
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod

    if hasattr(te_layernorm_linear_mod, "general_gemm"):
        if "module.layernorm_linear.general_gemm" not in _TE_GEMM_FUNC_ORIGS:
            _TE_GEMM_FUNC_ORIGS["module.layernorm_linear.general_gemm"] = (
                te_layernorm_linear_mod.general_gemm
            )
            te_layernorm_linear_mod.general_gemm = _te_general_gemm_patched

    # Also patch the symbol imported into Megatron's TE wrapper module
    import megatron.core.extensions.transformer_engine as meg_te

    if _MEG_TE_GENERAL_GEMM_ORIG is None and hasattr(meg_te, "general_gemm"):
        _MEG_TE_GENERAL_GEMM_ORIG = meg_te.general_gemm
        meg_te.general_gemm = _te_general_gemm_patched

    # Patch RMSNorm.forward once (class may be on te or te.pytorch)
    rms_cls = getattr(te, "RMSNorm", None)
    if rms_cls is None:
        rms_cls = getattr(te, "pytorch", None)
        rms_cls = getattr(rms_cls, "RMSNorm", None)
    if rms_cls is not None and _TE_RMSNORM_ORIG_FWD is None and hasattr(rms_cls, "forward"):
        _TE_RMSNORM_ORIG_FWD = rms_cls.forward
        rms_cls.forward = _te_rmsnorm_forward_patched

    # Patch TE module-level RMSNorm functions used by fused LayerNormLinear
    import transformer_engine.pytorch.module.layernorm as te_layernorm_mod

    def _make_rmsnorm_patched(orig_func):
        # Module-level helpers (e.g. transformer_engine.pytorch.module.layernorm.rmsnorm)
        # do not go through the RMSNorm class, so we also wrap those functions here.
        def _patched(*args, **kwargs):
            # If batch-invariant mode is off, use original
            if not is_batch_invariant_mode_enabled():
                return orig_func(*args, **kwargs)

            # Extract x, weight, eps from args/kwargs per TE signatures
            x = args[0] if len(args) > 0 else kwargs.get("x")
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            eps = (args[2] if len(args) > 2 else None) if "eps" not in kwargs else kwargs.get("eps")
            if eps is None:
                eps = 1e-5
            if x is None or weight is None:
                return orig_func(*args, **kwargs)

            y = rmsnorm_batch_invariant(x, weight, float(eps))
            # Match TE behavior: cast output to parameter dtype
            return y.to(weight.dtype)

        return _patched

    for name in ("rmsnorm", "rmsnorm_forward", "rmsnorm_fwd"):
        if hasattr(te_layernorm_mod, name) and name not in _TE_RMSNORM_FUNC_ORIGS:
            orig = getattr(te_layernorm_mod, name)
            _TE_RMSNORM_FUNC_ORIGS[name] = orig
            setattr(te_layernorm_mod, name, _make_rmsnorm_patched(orig))

    # Patch TE.general_grouped_gemm at every known import site so that
    # TEGroupedMLP (forward + dgrad + wgrad) goes through DeepGEMM in bf16.
    _te_patch_general_grouped_gemm()


def _te_patch_general_grouped_gemm() -> None:
    """Replace TE.general_grouped_gemm with a batch-invariant dispatcher.

    Patches the symbol at three import sites — the consumer module
    (transformer_engine.pytorch.module.grouped_linear), the package-level
    re-export (transformer_engine.pytorch.cpp_extensions), and the source
    module (transformer_engine.pytorch.cpp_extensions.gemm). Stores originals
    in _TE_GROUPED_GEMM_FUNC_ORIGS so the unpatch can restore them.
    """
    te_grouped_linear_mod = _import_module_if_available(
        "transformer_engine.pytorch.module.grouped_linear"
    )
    if te_grouped_linear_mod is not None and hasattr(te_grouped_linear_mod, "general_grouped_gemm"):
        key = "module.grouped_linear.general_grouped_gemm"
        if key not in _TE_GROUPED_GEMM_FUNC_ORIGS:
            _TE_GROUPED_GEMM_FUNC_ORIGS[key] = te_grouped_linear_mod.general_grouped_gemm
            te_grouped_linear_mod.general_grouped_gemm = _te_general_grouped_gemm_patched

    te_cpp = _import_module_if_available("transformer_engine.pytorch.cpp_extensions")
    if te_cpp is not None and hasattr(te_cpp, "general_grouped_gemm"):
        key = "cpp_extensions.general_grouped_gemm"
        if key not in _TE_GROUPED_GEMM_FUNC_ORIGS:
            _TE_GROUPED_GEMM_FUNC_ORIGS[key] = te_cpp.general_grouped_gemm
            te_cpp.general_grouped_gemm = _te_general_grouped_gemm_patched

    te_cpp_gemm = _import_module_if_available("transformer_engine.pytorch.cpp_extensions.gemm")
    if te_cpp_gemm is not None and hasattr(te_cpp_gemm, "general_grouped_gemm"):
        key = "cpp_extensions.gemm.general_grouped_gemm"
        if key not in _TE_GROUPED_GEMM_FUNC_ORIGS:
            _TE_GROUPED_GEMM_FUNC_ORIGS[key] = te_cpp_gemm.general_grouped_gemm
            te_cpp_gemm.general_grouped_gemm = _te_general_grouped_gemm_patched


def _te_unpatch_general_grouped_gemm() -> None:
    """Restore the originals captured by _te_patch_general_grouped_gemm."""
    module_paths = {
        "module.grouped_linear.general_grouped_gemm": (
            "transformer_engine.pytorch.module.grouped_linear",
            "general_grouped_gemm",
        ),
        "cpp_extensions.general_grouped_gemm": (
            "transformer_engine.pytorch.cpp_extensions",
            "general_grouped_gemm",
        ),
        "cpp_extensions.gemm.general_grouped_gemm": (
            "transformer_engine.pytorch.cpp_extensions.gemm",
            "general_grouped_gemm",
        ),
    }
    for key, (mod_name, attr) in module_paths.items():
        if key not in _TE_GROUPED_GEMM_FUNC_ORIGS:
            continue
        mod = _import_module_if_available(mod_name)
        if mod is not None and hasattr(mod, attr):
            setattr(mod, attr, _TE_GROUPED_GEMM_FUNC_ORIGS[key])
        _TE_GROUPED_GEMM_FUNC_ORIGS.pop(key, None)


def _is_bf16_grouped_path(A, B, quantization_params, gelu: bool) -> bool:
    """Decide if TE's general_grouped_gemm call can be served by DeepGEMM bf16."""
    if gelu:
        return False
    if not HAVE_DEEPGEMM_BF16:
        return False
    if not (isinstance(A, list) and isinstance(B, list)):
        return False
    if len(A) != len(B) or len(A) == 0:
        return False
    if quantization_params is not None and any(q is not None for q in quantization_params):
        return False
    for t in (*A, *B):
        if not isinstance(t, torch.Tensor):
            return False
        if t.dtype != torch.bfloat16:
            return False
    return True


def _te_general_grouped_gemm_patched(
    A,
    B,
    out,
    quantization_params,
    out_dtype,
    layout: str = "TN",
    m_splits=None,
    gelu: bool = False,
    grad: bool = False,
    accumulate: bool = False,
    bias=None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype=None,
    single_output: bool = False,
):
    """Batch-invariant replacement for TE general_grouped_gemm.

    Dispatches by (layout, single_output, grad) to forward / dgrad / wgrad
    implementations backed by DeepGEMM. Falls back to TE's original for any
    case we cannot guarantee batch-invariant: quantized inputs, gelu fusion,
    non-bf16 dtypes, or unsupported (layout, mode) combinations.
    """
    if not _is_bf16_grouped_path(A, B, quantization_params, gelu):
        orig = _TE_GROUPED_GEMM_FUNC_ORIGS.get("module.grouped_linear.general_grouped_gemm")
        if orig is None:
            orig = _TE_GROUPED_GEMM_FUNC_ORIGS.get("cpp_extensions.general_grouped_gemm")
        if orig is None:
            orig = _TE_GROUPED_GEMM_FUNC_ORIGS.get("cpp_extensions.gemm.general_grouped_gemm")
        if orig is None:
            raise RuntimeError(
                "Batch-invariant grouped GEMM patch was invoked but no original "
                "TE general_grouped_gemm was captured; patching order issue."
            )
        return orig(
            A,
            B,
            out,
            quantization_params,
            out_dtype,
            layout=layout,
            m_splits=m_splits,
            gelu=gelu,
            grad=grad,
            accumulate=accumulate,
            bias=bias,
            use_bias=use_bias,
            use_split_accumulator=use_split_accumulator,
            D_dtype=D_dtype,
            single_output=single_output,
        )

    # Dispatch by TE's call convention.
    # In TE _GroupedLinear:
    #   forward  -> layout="TN", single_output=True,  grad=False  (A=weights, B=inputmats)
    #   dgrad    -> layout="NN", single_output=True,  grad=True   (A=weights, B=grad_y)
    #   wgrad    -> layout="NT", single_output=False, grad=True   (A=inputmats, B=grad_y)
    if single_output and layout == "TN" and not grad:
        return _bik_te_grouped_forward(A, B, out, m_splits, bias, use_bias, accumulate)
    if single_output and layout == "NN" and grad:
        return _bik_te_grouped_dgrad(A, B, out, m_splits, accumulate)
    if (not single_output) and layout == "NT" and grad:
        return _bik_te_grouped_wgrad(A, B, out, m_splits, use_bias, accumulate)
    # Unknown TE call shape — defer to the original.
    orig = _TE_GROUPED_GEMM_FUNC_ORIGS.get("module.grouped_linear.general_grouped_gemm")
    return orig(
        A,
        B,
        out,
        quantization_params,
        out_dtype,
        layout=layout,
        m_splits=m_splits,
        gelu=gelu,
        grad=grad,
        accumulate=accumulate,
        bias=bias,
        use_bias=use_bias,
        use_split_accumulator=use_split_accumulator,
        D_dtype=D_dtype,
        single_output=single_output,
    )


def _stack_weights_for_deepgemm(weights: List[torch.Tensor]) -> torch.Tensor:
    """Stack a per-expert weight list into a contiguous [E, N, K] buffer.

    For TEGroupedMLP with the optional contiguous-weight optimization (see
    experts.py), all weights are already views into a single [E, N, K] tensor;
    we detect that case and return the underlying tensor without copying.
    """
    if not weights:
        return torch.empty(0)
    first = weights[0]
    base = first._base if first._base is not None else first
    expected_shape = (len(weights), *first.shape)
    if (
        base.dim() == 3
        and base.shape == expected_shape
        and all(
            (w._base is base) and w.data_ptr() == base[i].data_ptr() for i, w in enumerate(weights)
        )
        and base.is_contiguous()
    ):
        return base
    return torch.stack([w.contiguous() for w in weights], dim=0)


def _bik_te_grouped_forward(A, B, out, m_splits, bias, use_bias, accumulate):
    """TE forward: Y = X @ W^T per expert, then optional bias.

    A = weights:   List[Tensor[N, K]]
    B = inputmats: List[Tensor[m_i, K]]
    out:           [single Tensor[M_total, N]]  (single_output=True)
    """
    assert not accumulate, "Forward never accumulates"
    assert len(out) == 1, "single_output=True forward expects a single out tensor"
    out_buf = out[0]
    w_stack = _stack_weights_for_deepgemm(A)
    x_cat = torch.cat([b.contiguous() for b in B], dim=0)
    m_total = x_cat.shape[0]
    m_indices = _m_splits_to_m_indices(m_splits, x_cat.device, m_total)

    y = _bf16_grouped_gemm_contiguous(x_cat, w_stack, m_indices)
    if use_bias and bias is not None:
        offset = 0
        for i, m in enumerate(m_splits):
            if m == 0:
                continue
            b_i = bias[i] if i < len(bias) else None
            if b_i is not None and b_i.numel() > 0:
                y[offset : offset + m] = y[offset : offset + m] + b_i.to(y.dtype)
            offset += m

    if y.dtype != out_buf.dtype:
        y = y.to(out_buf.dtype)
    out_buf.copy_(y)
    # TE's contract: (out_list, bias_or_grad_bias, gelu_input)
    return out, bias if use_bias else [None] * len(A), None


def _bik_te_grouped_dgrad(A, B, out, m_splits, accumulate):
    """TE dgrad: dX = dY @ W per expert.

    A = weights:    List[Tensor[N, K]]
    B = grad_y_per_expert: List[Tensor[m_i, N]]
    out:            [single Tensor[M_total, K]] (single_output=True)
    """
    assert not accumulate, "Dgrad never accumulates"
    assert len(out) == 1
    out_buf = out[0]
    w_stack = _stack_weights_for_deepgemm(A)
    dy_cat = torch.cat([b.contiguous() for b in B], dim=0)
    m_total = dy_cat.shape[0]
    m_indices = _m_splits_to_m_indices(m_splits, dy_cat.device, m_total)
    # NT call interprets B as [E, out_dim, in_dim]; for dgrad we need W as [E, K, N]
    w_kn = w_stack.transpose(1, 2).contiguous()
    dx = _bf16_grouped_gemm_contiguous(dy_cat, w_kn, m_indices)
    if dx.dtype != out_buf.dtype:
        dx = dx.to(out_buf.dtype)
    out_buf.copy_(dx)
    return out, [None] * len(A), None


def _bik_te_grouped_wgrad(A, B, out, m_splits, use_bias, accumulate):
    """TE wgrad: dW[g] = dY[g]^T @ X[g], plus optional dbias[g] = sum(dY[g], dim=0).

    A = inputmats: List[Tensor[m_i, K]]
    B = grad_y:    List[Tensor[m_i, N]]
    out:           List[Tensor[N, K]] per expert (single_output=False)
    """
    E = len(m_splits)
    x_cat = torch.cat([a.contiguous() for a in A], dim=0)
    dy_cat = torch.cat([b.contiguous() for b in B], dim=0)
    m_total = x_cat.shape[0]
    k_indices = _m_splits_to_m_indices(m_splits, x_cat.device, m_total)

    dw_stack = _bf16_grouped_gemm_wgrad_contiguous(dy_cat, x_cat, k_indices, E)

    grad_bias = [None] * E
    if use_bias:
        offset = 0
        for i, m in enumerate(m_splits):
            if m > 0:
                grad_bias[i] = dy_cat[offset : offset + m].sum(dim=0)
                offset += m

    for i in range(E):
        target = out[i]
        contrib = dw_stack[i]
        if contrib.dtype != target.dtype:
            contrib = contrib.to(target.dtype)
        if accumulate:
            target.add_(contrib)
        else:
            target.copy_(contrib)
    return out, grad_bias, None


def _te_unpatch_for_batch_invariant():
    """Restore original Transformer Engine functions if they were patched."""
    global _TE_GENERAL_GEMM_ORIG, _TE_RMSNORM_ORIG_FWD, _MEG_TE_GENERAL_GEMM_ORIG
    te_cpp = _import_module_if_available("transformer_engine.pytorch.cpp_extensions")
    te = _import_module_if_available("transformer_engine.pytorch")
    if te_cpp is None or te is None:
        _TE_GENERAL_GEMM_ORIG = None
        _TE_RMSNORM_ORIG_FWD = None
        _MEG_TE_GENERAL_GEMM_ORIG = None
        return

    if _TE_GENERAL_GEMM_ORIG is not None and hasattr(te_cpp, "general_gemm"):
        te_cpp.general_gemm = _TE_GENERAL_GEMM_ORIG
        _TE_GENERAL_GEMM_ORIG = None

    rms_cls = getattr(te, "RMSNorm", None)
    if rms_cls is None:
        rms_cls = getattr(te, "pytorch", None)
        rms_cls = getattr(rms_cls, "RMSNorm", None)
    if rms_cls is not None and _TE_RMSNORM_ORIG_FWD is not None:
        rms_cls.forward = _TE_RMSNORM_ORIG_FWD
        _TE_RMSNORM_ORIG_FWD = None

    meg_te = _import_module_if_available("megatron.core.extensions.transformer_engine")
    if (
        meg_te is not None
        and _MEG_TE_GENERAL_GEMM_ORIG is not None
        and hasattr(meg_te, "general_gemm")
    ):
        meg_te.general_gemm = _MEG_TE_GENERAL_GEMM_ORIG
        _MEG_TE_GENERAL_GEMM_ORIG = None
    elif meg_te is None:
        _MEG_TE_GENERAL_GEMM_ORIG = None

    # Restore TE module-level RMSNorm functions
    te_layernorm_mod = _import_module_if_available("transformer_engine.pytorch.module.layernorm")
    if te_layernorm_mod is not None:
        for name, orig in list(_TE_RMSNORM_FUNC_ORIGS.items()):
            if hasattr(te_layernorm_mod, name):
                setattr(te_layernorm_mod, name, orig)
            _TE_RMSNORM_FUNC_ORIGS.pop(name, None)
    else:
        _TE_RMSNORM_FUNC_ORIGS.clear()

    # Restore TE module.linear imported symbol for general_gemm if patched
    te_linear_mod = _import_module_if_available("transformer_engine.pytorch.module.linear")
    key = "module.linear.general_gemm"
    if (
        te_linear_mod is not None
        and key in _TE_GEMM_FUNC_ORIGS
        and hasattr(te_linear_mod, "general_gemm")
    ):
        te_linear_mod.general_gemm = _TE_GEMM_FUNC_ORIGS[key]
        _TE_GEMM_FUNC_ORIGS.pop(key, None)
    else:
        _TE_GEMM_FUNC_ORIGS.pop(key, None)

    # Restore TE module.layernorm_linear imported symbol for general_gemm if patched
    te_layernorm_linear_mod = _import_module_if_available(
        "transformer_engine.pytorch.module.layernorm_linear"
    )
    key = "module.layernorm_linear.general_gemm"
    if (
        te_layernorm_linear_mod is not None
        and key in _TE_GEMM_FUNC_ORIGS
        and hasattr(te_layernorm_linear_mod, "general_gemm")
    ):
        te_layernorm_linear_mod.general_gemm = _TE_GEMM_FUNC_ORIGS[key]
        _TE_GEMM_FUNC_ORIGS.pop(key, None)
    else:
        _TE_GEMM_FUNC_ORIGS.pop(key, None)

    # Restore TE general_grouped_gemm at every patched import site.
    _te_unpatch_general_grouped_gemm()


def _extract_te_gemm_args(args: tuple, kwargs: Dict[str, Any]):
    """Utility to parse TE general_gemm flexible signature.

    Returns (A, B, out_dtype, layout, out, bias, grad).
    """
    A = args[0] if len(args) > 0 else kwargs.get("A")
    B = args[1] if len(args) > 1 else kwargs.get("B")
    out_dtype = kwargs.get("out_dtype")
    layout = kwargs.get("layout", "TN")
    out = kwargs.get("out")
    bias = kwargs.get("bias")
    grad = kwargs.get("grad", False)
    return A, B, out_dtype, layout, out, bias, grad


def _is_supported_dtype_for_bik(t: torch.dtype) -> bool:
    return t in {torch.float16, torch.bfloat16, torch.float32}


class BatchInvariantTEGemmFn(torch.autograd.Function):
    """Autograd function implementing batch-invariant TE GEMM."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_dtype: Optional[torch.dtype],
        layout: str,
    ):
        """Forward pass computing batch-invariant TE GEMM.

        Respects TE's flexible `layout` semantics, flattens leading dimensions of
        the input as needed, applies optional bias, and casts to `out_dtype`.
        """
        assert isinstance(layout, str) and len(layout) == 2, f"Unsupported layout: {layout}"
        transa = layout[0].upper() == "T"
        transb = layout[1].upper() == "T"

        opA = A.transpose(0, 1).contiguous() if transa else A.contiguous()  # [K, O] or [I, O]
        opB = B.transpose(0, 1).contiguous() if transb else B.contiguous()  # [..., K]

        # Flatten opA to 2D if needed (weight tensors should be 2D, but validate)
        if opA.dim() > 2:
            opA = opA.reshape(-1, opA.shape[-1])
        elif opA.dim() < 2:
            raise ValueError(f"opA has insufficient dimensions: {opA.shape}")
        assert opA.dim() == 2, f"opA must be 2D for matmul_persistent, got shape {opA.shape}"

        # Flatten all leading dims of opB except the last feature dim to match TE behavior
        if opB.dim() >= 2:
            leading_shape = opB.shape[:-1]
            K = opB.shape[-1]
            opB_2d = opB.reshape(-1, K)
        else:
            leading_shape = ()
            opB_2d = opB

        # Perform GEMM: (N_total, K) @ (K, O) -> (N_total, O)
        base_2d = matmul_persistent(opB_2d, opA, bias=None)

        # Reshape back to original leading dims with output features at the end
        out = base_2d.reshape(*leading_shape, base_2d.shape[-1])

        # Add bias after reshaping to match output structure
        if bias is not None:
            out = out + bias

        if out_dtype is not None:
            out = out.to(out_dtype)

        # Save for backward
        ctx.transa = transa
        ctx.transb = transb
        ctx.leading_shape = leading_shape
        ctx.bias_present = bias is not None
        ctx.save_for_backward(A, B)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for batch-invariant TE GEMM.

        Computes gradients w.r.t. A, B, and optional bias while mirroring the
        reshaping/layout logic used in the forward pass.
        """
        A, B = ctx.saved_tensors
        transa = ctx.transa
        transb = ctx.transb
        leading_shape = ctx.leading_shape

        # Reconstruct opA/opB for gradients
        opA = A.transpose(0, 1).contiguous() if transa else A  # [K, O]
        opB = B.transpose(0, 1).contiguous() if transb else B  # [..., K]

        # Flatten grad_output to 2D to mirror forward flatten
        if grad_output.dim() >= 2 and isinstance(leading_shape, tuple) and len(leading_shape) > 0:
            N_total = 1
            for s in leading_shape:
                N_total *= s
            grad_out_2d = grad_output.reshape(N_total, grad_output.shape[-1])
        else:
            grad_out_2d = grad_output

        # Y = B_flat @ A -> dB_flat = dY @ A^T ; dA = B_flat^T @ dY
        d_opB_2d = grad_out_2d.matmul(opA.transpose(0, 1).contiguous())
        d_opA = opB.reshape(-1, opB.shape[-1]).transpose(0, 1).contiguous().matmul(grad_out_2d)

        # Reshape d_opB back to original opB shape
        d_opB = (
            d_opB_2d.reshape(*leading_shape, d_opB_2d.shape[-1])
            if grad_output.dim() >= 2
            else d_opB_2d
        )

        # Map back to dA, dB based on trans flags
        if transa:
            dA = d_opA.transpose(0, 1).contiguous()
        else:
            dA = d_opA

        if transb:
            dB = d_opB.transpose(0, 1).contiguous()
        else:
            dB = d_opB

        # Bias grad along last dimension of Y, if bias was added in forward
        if ctx.bias_present:
            dbias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
        else:
            dbias = None

        return dA, dB, dbias, None, None


def _te_general_gemm_patched(*args, **kwargs) -> List[torch.Tensor]:
    """
    Batch-invariant replacement for TE general_gemm.
    Returns a list of tensors to match TE's API: (gemm_out, bias_grad, gelu_input, extra_output)
    """
    global _TE_GENERAL_GEMM_ORIG
    # If original not captured, do nothing
    if _TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError("TE general_gemm original not captured; patching order issue")

    A, B, out_dtype, layout, out, bias, grad = _extract_te_gemm_args(args, kwargs)
    extra_output = kwargs.get("extra_output", None)
    ub = kwargs.get("ub", None)
    ub_type = kwargs.get("ub_type", None)
    bulk_overlap = kwargs.get("bulk_overlap", False)

    # Guardrails: validate inputs
    if A is None or B is None:
        raise ValueError("Batch-invariant GEMM requires A and B tensors.")
    if (not A.is_cuda) or (not B.is_cuda):
        raise RuntimeError("Batch-invariant GEMM requires CUDA tensors.")
    if not _is_supported_dtype_for_bik(A.dtype) or not _is_supported_dtype_for_bik(B.dtype):
        raise RuntimeError(f"Unsupported dtype for batch-invariant GEMM: {A.dtype}, {B.dtype}")

    # Disallow GEMM-comm overlap in batch-invariant mode
    if extra_output is not None or ub is not None or ub_type is not None or bulk_overlap:
        raise RuntimeError(
            "Batch-invariant GEMM does not support Userbuffers/overlap "
            "(extra_output/ub/ub_type/bulk_overlap)."
        )

    # Compute via autograd-aware function matching TE's layout semantics
    result = BatchInvariantTEGemmFn.apply(A, B, bias if not grad else None, out_dtype, layout)

    bias_grad = None
    if grad and bias is not None:
        # Flatten B to 2D and sum over batch/sequence dimension (first dim)
        B_flat = B.reshape(-1, B.shape[-1]) if B.dim() > 2 else B
        bias_grad = B_flat.sum(dim=0)  # Sum over batch/sequence, keeping output dim

    if out is not None:
        out.copy_(result)
        # TE expects (gemm_out, bias_grad, gelu_input, extra_output)
        return (out, bias_grad, None, extra_output)
    return (result, bias_grad, None, extra_output)


class BatchInvariantRMSNormFn(torch.autograd.Function):
    """Autograd function implementing batch-invariant RMSNorm."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered_gamma: bool):
        """Forward pass for batch-invariant RMSNorm.

        Normalizes `x` using an RMSNorm-style statistic computed via `mean_dim`,
        applies affine `weight`, and stores intermediate rsigma for backward.
        """
        if not x.is_cuda:
            raise RuntimeError("Batch-invariant RMSNorm requires CUDA tensors.")
        if not _is_supported_dtype_for_bik(x.dtype):
            raise RuntimeError(f"Unsupported dtype for batch-invariant RMSNorm: {x.dtype}")
        weight_eff = weight + 1.0 if zero_centered_gamma else weight

        # We do everything in rmsnorm_batch_invariant manually here so that we can
        # save rsigma in full precision for backward to match the TE behavior.
        x_dtype = x.dtype
        x_fp32 = x.float()
        w_fp32 = weight.to(device=x.device, dtype=torch.float32)
        ms = mean_dim(x_fp32 * x_fp32, dim=-1, keepdim=True)
        rsigma = torch.rsqrt(ms + eps)
        out_fp32 = (x_fp32 * rsigma) * w_fp32
        out = out_fp32.to(x_dtype)

        # Save for backward
        ctx.eps = eps
        ctx.zero_centered_gamma = zero_centered_gamma
        ctx.rsigma = rsigma

        ctx.save_for_backward(x, weight, rsigma)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for batch-invariant RMSNorm.

        Computes gradients w.r.t. input and weight while matching TE's fp32
        accumulation and reduction behavior for numerical stability.
        """
        x, weight, rsigma = ctx.saved_tensors
        w_eff = (weight + 1.0) if ctx.zero_centered_gamma else weight

        go_fp32 = grad_output.float()
        x_fp32 = x.float()
        w_fp32 = w_eff.to(device=x.device, dtype=torch.float32)
        r = rsigma
        r3 = r * r * r
        D = x.shape[-1]

        red_dims = tuple(range(0, go_fp32.ndim - 1))
        g_w = (go_fp32 * x_fp32 * r).sum(dim=red_dims).to(weight.dtype)

        s = (go_fp32 * x_fp32 * w_fp32).sum(dim=-1, keepdim=True)
        dx = go_fp32 * (w_fp32 * r) - (w_fp32 * r3) * (s * x_fp32) / D
        dx = dx.to(x.dtype)

        return dx, g_w, None, None


def rmsnorm_batch_invariant(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Batch-invariant RMSNorm wrapper that delegates to autograd-aware implementation.

    This provides a simple functional interface while using the optimized BatchInvariantRMSNormFn
    which has better numerics (fp32 precision in forward/backward).
    """
    # Delegate to the autograd function with zero_centered_gamma=False (standard RMSNorm)
    return BatchInvariantRMSNormFn.apply(x, weight, eps, False)


# ---------------------------------------------------------------------------
# Batch-invariant grouped GEMM (DeepGEMM-backed). Used by MoE so that training
# (TEGroupedMLP via patched TE.general_grouped_gemm) and inference
# (InferenceGroupedMLP via patched _bf16_grouped_mm) produce bitwise-identical
# outputs for the same inputs. This is what gives RL rollout==train log-prob
# parity for MoE models.
# ---------------------------------------------------------------------------


def _require_deepgemm_bf16(op: str) -> None:
    """Raise a clear error if DeepGEMM bf16 grouped bindings are unavailable."""
    if not HAVE_DEEPGEMM_BF16:
        raise RuntimeError(
            f"Batch-invariant grouped GEMM ({op}) requires DeepGEMM with bf16 bindings. "
            "Install via `uv pip install -e .[batch_invariant]` (pins a DeepGEMM commit "
            "that exposes m_grouped_bf16_gemm_nt_contiguous), or disable "
            "transformer_config.batch_invariant_mode for MoE models."
        )


def _offs_to_m_indices(offs: torch.Tensor, m_total: int) -> torch.Tensor:
    """Convert inclusive cumulative per-expert offsets to per-row expert ids.

    offs: int32 [num_experts] inclusive offsets — offs[i] is the (exclusive) end
          row of expert i in the contiguous M dimension. Equivalently, offs[i] is
          the start of expert i+1.
    Returns: int32 [m_total] m_indices[r] = expert id for row r. Rows past offs[-1]
             (post-padding tail when m_total > offs[-1]) get -1; DeepGEMM skips
             those rows.
    """
    rows = torch.arange(m_total, device=offs.device, dtype=torch.int32)
    # For row r, expert id = bisect_right(offs, r). torch.searchsorted is deterministic.
    m_indices = torch.searchsorted(offs, rows, right=True).to(torch.int32)
    n_used = offs[-1].to(torch.int32)
    m_indices = torch.where(rows < n_used, m_indices, torch.full_like(m_indices, -1))
    return m_indices


def _m_splits_to_m_indices(m_splits: List[int], device: torch.device, m_total: int) -> torch.Tensor:
    """Convert TE per-expert token counts (List[int]) to int32 [m_total] m_indices.

    No padding rows in TE training path — sum(m_splits) == m_total exactly.
    """
    assert sum(m_splits) == m_total, f"m_splits sum ({sum(m_splits)}) != m_total ({m_total})"
    parts = [
        torch.full((n,), i, device=device, dtype=torch.int32)
        for i, n in enumerate(m_splits)
        if n > 0
    ]
    if not parts:
        return torch.empty(0, device=device, dtype=torch.int32)
    return torch.cat(parts, dim=0)


# DeepGEMM's contiguous M-grouped and K-grouped bf16 GEMMs require each
# per-expert block on the grouped axis to be a multiple of this alignment
# (typically 128 on SM90/SM100). We pad inputs to satisfy this, then strip the
# padding from the output. Padding rows are zeros (correct identity for the
# reduction sum) and tagged with m_indices=-1 for the M-grouped case so the
# kernel can skip them in store.
_DEEPGEMM_M_ALIGNMENT: Optional[int] = None


def _deepgemm_m_alignment() -> int:
    """Lazily fetch DeepGEMM's required per-expert block alignment."""
    global _DEEPGEMM_M_ALIGNMENT
    if _DEEPGEMM_M_ALIGNMENT is None:
        _DEEPGEMM_M_ALIGNMENT = int(deep_gemm.get_m_alignment_for_contiguous_layout())
    return _DEEPGEMM_M_ALIGNMENT


def _expert_counts_from_m_indices(m_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Per-expert token counts from a sorted m_indices tensor. Ignores -1 rows."""
    valid = m_indices[m_indices >= 0]
    return torch.bincount(valid.long(), minlength=num_experts).to(torch.int64)


def _pad_for_m_grouped(a: torch.Tensor, m_indices: torch.Tensor, num_experts: int) -> tuple:
    """Pad an M-grouped contiguous input to satisfy DeepGEMM's per-expert M alignment.

    Returns (a_padded, m_indices_padded, padded_counts_cpu, true_counts_cpu).
    The padded layout groups expert i's true rows contiguously at the start of
    its 128-aligned block; remaining rows in the block are zero with m_indices=-1.
    """
    A = _deepgemm_m_alignment()
    counts = _expert_counts_from_m_indices(m_indices, num_experts)
    counts_cpu = counts.tolist()
    padded_counts_cpu = [((c + A - 1) // A) * A for c in counts_cpu]
    M_pad = sum(padded_counts_cpu)
    if M_pad == 0:
        return (
            torch.empty(0, a.shape[1], device=a.device, dtype=a.dtype),
            torch.empty(0, device=a.device, dtype=torch.int32),
            padded_counts_cpu,
            counts_cpu,
        )

    a_padded = torch.zeros(M_pad, a.shape[1], device=a.device, dtype=a.dtype)
    m_indices_padded = torch.full((M_pad,), -1, device=a.device, dtype=torch.int32)
    src = 0
    dst = 0
    for i, (c, cp) in enumerate(zip(counts_cpu, padded_counts_cpu)):
        if c > 0:
            a_padded[dst : dst + c] = a[src : src + c]
            m_indices_padded[dst : dst + c] = i
        src += c
        dst += cp
    return a_padded, m_indices_padded, padded_counts_cpu, counts_cpu


def _bf16_grouped_gemm_contiguous(
    a: torch.Tensor, b: torch.Tensor, m_indices: torch.Tensor
) -> torch.Tensor:
    """bf16 M-grouped GEMM via DeepGEMM. Deterministic / batch-invariant.

    a:         [M_total, K] bf16, contiguous, expert-grouped (rows of expert i
                                              are contiguous; m_indices is sorted).
    b:         [E, N, K]    bf16, contiguous (per-expert weights, NT layout —
                                              DeepGEMM transposes B internally).
    m_indices: [M_total]    int32, expert id per row (-1 to skip).
    Returns:   [M_total, N] bf16 with rows in the same order as `a`.

    Handles DeepGEMM's per-expert M alignment requirement by padding/unpadding
    internally; the caller does not need pre-padded inputs.
    """
    _require_deepgemm_bf16("m_grouped_bf16_gemm_nt_contiguous")
    assert (
        a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    ), f"bf16 grouped GEMM requires bf16; got a.dtype={a.dtype}, b.dtype={b.dtype}"
    assert a.is_contiguous() and b.is_contiguous(), "a, b must be contiguous"
    assert (
        m_indices.dtype == torch.int32 and m_indices.is_contiguous()
    ), "m_indices must be int32 contiguous"
    M_total, K = a.shape
    E, N, K_b = b.shape
    assert K == K_b, f"K mismatch between a ({K}) and b ({K_b})"
    assert (
        m_indices.shape[0] == M_total
    ), f"m_indices length {m_indices.shape[0]} != M_total {M_total}"

    a_padded, m_indices_padded, padded_counts_cpu, counts_cpu = _pad_for_m_grouped(a, m_indices, E)
    M_pad = a_padded.shape[0]
    if M_pad == 0:
        return torch.zeros(M_total, N, device=a.device, dtype=torch.bfloat16)

    d_padded = torch.empty(M_pad, N, device=a.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_padded, b, d_padded, m_indices_padded)

    # Strip padding: copy each expert's true rows back to a [M_total, N] tensor.
    d = torch.empty(M_total, N, device=a.device, dtype=torch.bfloat16)
    # Rows with m_indices == -1 (post-padding tail in the input) get zero output.
    if (m_indices < 0).any():
        d.zero_()
    src = 0
    dst = 0
    for c, cp in zip(counts_cpu, padded_counts_cpu):
        if c > 0:
            d[src : src + c] = d_padded[dst : dst + c]
        src += c
        dst += cp
    return d


def _bf16_grouped_gemm_wgrad_contiguous(
    grad_y: torch.Tensor, x: torch.Tensor, k_indices: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """K-grouped TN GEMM producing per-expert weight gradients via DeepGEMM.

    grad_y:    [M_total, N] bf16, contiguous, expert-grouped (rows of expert i
                                              are contiguous; k_indices is sorted).
    x:         [M_total, K] bf16, contiguous, expert-grouped (same row ordering).
    k_indices: [M_total]    int32, expert id per row (the K-axis grouping label).
    Returns:   [E, N, K]    bf16 stacked per-expert wgrad.

    DeepGEMM's k_grouped_bf16 kernel computes in fp32 and requires fp32 d/c
    accumulators; we cast the result back to bf16. Per-expert K alignment is
    handled by padding internally.
    """
    _require_deepgemm_bf16("k_grouped_bf16_gemm_tn_contiguous")
    assert grad_y.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    assert grad_y.is_contiguous() and x.is_contiguous()
    assert k_indices.dtype == torch.int32 and k_indices.is_contiguous()
    M_total, N = grad_y.shape
    M_total_b, K = x.shape
    assert M_total == M_total_b

    A = _deepgemm_m_alignment()
    counts = _expert_counts_from_m_indices(k_indices, num_experts).tolist()
    padded_counts = [((c + A - 1) // A) * A for c in counts]
    M_pad = sum(padded_counts)
    if M_pad == 0:
        return torch.zeros(num_experts, N, K, device=grad_y.device, dtype=torch.bfloat16)

    grad_y_pad = torch.zeros(M_pad, N, device=grad_y.device, dtype=torch.bfloat16)
    x_pad = torch.zeros(M_pad, K, device=x.device, dtype=torch.bfloat16)
    src = 0
    dst = 0
    for c, cp in zip(counts, padded_counts):
        if c > 0:
            grad_y_pad[dst : dst + c] = grad_y[src : src + c]
            x_pad[dst : dst + c] = x[src : src + c]
        src += c
        dst += cp

    ks_tensor = torch.tensor(padded_counts, dtype=torch.int32, device=grad_y.device)
    d_fp32 = torch.zeros(num_experts, N, K, device=grad_y.device, dtype=torch.float32)
    c_zero = torch.zeros(num_experts, N, K, device=grad_y.device, dtype=torch.float32)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        grad_y_pad, x_pad, d_fp32, padded_counts, ks_tensor, c_zero
    )
    return d_fp32.to(torch.bfloat16)


def grouped_gemm_batch_invariant(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    m_indices: Optional[torch.Tensor] = None,
    offs: Optional[torch.Tensor] = None,
    m_total: Optional[int] = None,
) -> torch.Tensor:
    """Public functional API for batch-invariant bf16 grouped GEMM.

    Either pass m_indices directly, or pass (offs, m_total) and we'll build the
    indices via _offs_to_m_indices.
    """
    if m_indices is None:
        assert (
            offs is not None and m_total is not None
        ), "grouped_gemm_batch_invariant: either m_indices or (offs, m_total) required"
        m_indices = _offs_to_m_indices(offs, m_total)
    return _bf16_grouped_gemm_contiguous(a.contiguous(), b.contiguous(), m_indices)


class BatchInvariantGroupedGemmFn(torch.autograd.Function):
    """Autograd-aware batch-invariant bf16 grouped GEMM.

    Used by the TE training-path patch to make TEGroupedMLP differentiable while
    still producing bitwise-identical activations to the inference path.

    Conventions:
        x:         [M_total, K]   bf16 inputs, contiguous, expert-grouped.
        w_stack:   [E, N, K]      bf16 weights, NT layout (DeepGEMM transposes B).
        m_indices: [M_total]      int32 expert id per row.
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, w_stack: torch.Tensor, m_indices: torch.Tensor, num_experts: int
    ):
        """Y[r] = X[r] @ W[m_indices[r]]^T   for each row r."""
        assert x.dtype == torch.bfloat16 and w_stack.dtype == torch.bfloat16
        x = x.contiguous()
        w_stack = w_stack.contiguous()
        m_indices = m_indices.contiguous()
        y = _bf16_grouped_gemm_contiguous(x, w_stack, m_indices)
        ctx.save_for_backward(x, w_stack, m_indices)
        ctx.num_experts = num_experts
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        """dX = grad_y @ W[g] (M-grouped NT) ; dW[g] = grad_y[g]^T @ X[g] (K-grouped TN)."""
        x, w_stack, m_indices = ctx.saved_tensors
        E = ctx.num_experts
        grad_y = grad_y.contiguous()

        # dgrad: pass w_stack transposed to [E, K, N] so DeepGEMM's NT call computes
        # grad_y @ W (with W treated as [in=K, out=N] per expert).
        w_kn = w_stack.transpose(1, 2).contiguous()
        dx = _bf16_grouped_gemm_contiguous(grad_y, w_kn, m_indices)

        dw_stack = _bf16_grouped_gemm_wgrad_contiguous(grad_y, x, m_indices, E)
        return dx, dw_stack, None, None


def _te_rmsnorm_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    """Patched TE RMSNorm.forward that routes to batch-invariant
    implementation with autograd support.
    """
    weight = getattr(self, "weight", None)
    if weight is None:
        raise RuntimeError("Batch-invariant RMSNorm requires affine weight.")
    eps = getattr(self, "eps", 1e-5)
    zero_centered_gamma = getattr(self, "zero_centered_gamma", False)
    return BatchInvariantRMSNormFn.apply(x, weight, eps, zero_centered_gamma)


def is_batch_invariant_mode_enabled():
    """Return True if global batch-invariant mode is currently enabled."""
    return _batch_invariant_MODE


def enable_batch_invariant_mode(backend: str = "deepgemm"):
    """Enable global batch-invariant mode and patch Aten/TE kernels.

    Args:
        backend: which kernel to dispatch `aten::mm`/`aten::addmm` through.
            "deepgemm" (default) routes bf16 CUDA inputs through DeepGEMM
            `bf16_gemm_nt`. "triton" routes through the BIK Triton
            `matmul_persistent` kernel (works for bf16/fp16/fp32 and on
            any CUDA device). Grouped GEMM always uses DeepGEMM regardless.
    """
    global _batch_invariant_MODE, _batch_invariant_LIB, _BIK_BACKEND
    if _batch_invariant_MODE:
        return
    if backend not in _BIK_BACKENDS:
        raise ValueError(
            f"Unknown batch_invariant_kernel_backend={backend!r}; "
            f"expected one of {_BIK_BACKENDS}."
        )
    if backend == "deepgemm" and not HAVE_DEEPGEMM_BF16:
        raise RuntimeError(
            "batch_invariant_kernel_backend='deepgemm' requires DeepGEMM with "
            "bf16 bindings. Install DeepGEMM or use backend='triton'."
        )
    _BIK_BACKEND = backend
    dispatch_key = getattr(torch.accelerator.current_accelerator(), "type", "cpu").upper()
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, dispatch_key)
    # Also patch Transformer Engine kernels when available
    _te_patch_for_batch_invariant()


def disable_batch_invariant_mode():
    """Disable global batch-invariant mode and restore original kernels."""
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
    # Restore Transformer Engine kernels if previously patched
    _te_unpatch_for_batch_invariant()


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    """Context manager to toggle global batch-invariant mode.

    When `enabled` is True, batch-invariant kernels are enabled for the duration of
    the context; when False, they are disabled for the duration. This implementation
    is re-entrant and correctly restores the previous state even under nesting.
    """
    global _batch_invariant_MODE, _batch_invariant_LIB
    # Save the previous on/off state so we can correctly restore it, even under
    # nested usage or when toggling from True->False inside an outer True scope.
    prev_enabled = _batch_invariant_MODE

    # Apply the requested state only if it differs from the current one.
    if enabled and not prev_enabled:
        enable_batch_invariant_mode()
    elif not enabled and prev_enabled:
        disable_batch_invariant_mode()

    try:
        yield
    finally:
        # Restore the previous state. If we turned BIK on at entry, turn it off here.
        # If we turned it off at entry (inside an outer True scope), turn it back on.
        if enabled and not prev_enabled:
            disable_batch_invariant_mode()
        elif not enabled and prev_enabled:
            enable_batch_invariant_mode()

import contextlib
import importlib
import importlib.util
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]
) -> Dict[str, Any]:
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
            print("No CUDA or XPU device available. Using CPU.")
            # For CPU, you might want to use the number of CPU cores
            NUM_SMS = torch.get_num_threads()

    return NUM_SMS


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
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
        return (
            min(
                NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])
            ),
        )

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


def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
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
    return AttentionBlockSize(block_m=16, block_n=16)


# Everything above is from the blog https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
# and its github repo https://github.com/thinking-machines-lab/batch_invariant_ops


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_ORIG_FWD = None
_MEG_TE_GENERAL_GEMM_ORIG = None
_TE_RMSNORM_FUNC_ORIGS: Dict[str, Any] = {}
_TE_GEMM_FUNC_ORIGS: Dict[str, Any] = {}


def _import_module_if_available(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return importlib.import_module(name)


def _te_patch_for_batch_invariant():
    """
    Monkey-patch Transformer Engine to route GEMM and RMSNorm to batch-invariant kernels.
    Safe no-op if TE is unavailable.
    """
    global _TE_GENERAL_GEMM_ORIG, _TE_RMSNORM_ORIG_FWD, _MEG_TE_GENERAL_GEMM_ORIG
    import transformer_engine.pytorch as te  # type: ignore
    import transformer_engine.pytorch.cpp_extensions as te_cpp  # type: ignore

    # Patch general_gemm once
    if _TE_GENERAL_GEMM_ORIG is None and hasattr(te_cpp, "general_gemm"):
        _TE_GENERAL_GEMM_ORIG = te_cpp.general_gemm
        te_cpp.general_gemm = _te_general_gemm_patched  # type: ignore[attr-defined]

    # Also patch the symbol imported inside TE's module.linear (from ..cpp_extensions import general_gemm)
    import transformer_engine.pytorch.module.linear as te_linear_mod  # type: ignore

    if hasattr(te_linear_mod, "general_gemm"):
        if "module.linear.general_gemm" not in _TE_GEMM_FUNC_ORIGS:
            _TE_GEMM_FUNC_ORIGS["module.linear.general_gemm"] = te_linear_mod.general_gemm
            te_linear_mod.general_gemm = _te_general_gemm_patched  # type: ignore[attr-defined]

    # Also patch the symbol imported inside TE's module.layernorm_linear
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod  # type: ignore

    if hasattr(te_layernorm_linear_mod, "general_gemm"):
        if "module.layernorm_linear.general_gemm" not in _TE_GEMM_FUNC_ORIGS:
            _TE_GEMM_FUNC_ORIGS["module.layernorm_linear.general_gemm"] = (
                te_layernorm_linear_mod.general_gemm
            )
            te_layernorm_linear_mod.general_gemm = _te_general_gemm_patched  # type: ignore[attr-defined]

        # Also patch the symbol imported into Megatron's TE wrapper module
    import megatron.core.extensions.transformer_engine as meg_te  # type: ignore

    if _MEG_TE_GENERAL_GEMM_ORIG is None and hasattr(meg_te, "general_gemm"):
        _MEG_TE_GENERAL_GEMM_ORIG = meg_te.general_gemm
        meg_te.general_gemm = _te_general_gemm_patched  # type: ignore[attr-defined]

    # Patch RMSNorm.forward once (class may be on te or te.pytorch)
    rms_cls = getattr(te, "RMSNorm", None)
    if rms_cls is None:
        rms_cls = getattr(te, "pytorch", None)
        rms_cls = getattr(rms_cls, "RMSNorm", None)
    if rms_cls is not None and _TE_RMSNORM_ORIG_FWD is None and hasattr(rms_cls, "forward"):
        _TE_RMSNORM_ORIG_FWD = rms_cls.forward
        rms_cls.forward = _te_rmsnorm_forward_patched  # type: ignore[attr-defined]

    # Patch TE module-level RMSNorm functions used by fused LayerNormLinear
    import transformer_engine.pytorch.module.layernorm as te_layernorm_mod  # type: ignore

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
        te_cpp.general_gemm = _TE_GENERAL_GEMM_ORIG  # type: ignore[assignment]
        _TE_GENERAL_GEMM_ORIG = None

    rms_cls = getattr(te, "RMSNorm", None)
    if rms_cls is None:
        rms_cls = getattr(te, "pytorch", None)
        rms_cls = getattr(rms_cls, "RMSNorm", None)
    if rms_cls is not None and _TE_RMSNORM_ORIG_FWD is not None:
        rms_cls.forward = _TE_RMSNORM_ORIG_FWD  # type: ignore[assignment]
        _TE_RMSNORM_ORIG_FWD = None

    meg_te = _import_module_if_available("megatron.core.extensions.transformer_engine")
    if (
        meg_te is not None
        and _MEG_TE_GENERAL_GEMM_ORIG is not None
        and hasattr(meg_te, "general_gemm")
    ):
        meg_te.general_gemm = _MEG_TE_GENERAL_GEMM_ORIG  # type: ignore[assignment]
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
        te_linear_mod.general_gemm = _TE_GEMM_FUNC_ORIGS[key]  # type: ignore[assignment]
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
        te_layernorm_linear_mod.general_gemm = _TE_GEMM_FUNC_ORIGS[key]  # type: ignore[assignment]
        _TE_GEMM_FUNC_ORIGS.pop(key, None)
    else:
        _TE_GEMM_FUNC_ORIGS.pop(key, None)


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
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        bias: Optional[torch.Tensor],
        out_dtype: Optional[torch.dtype],
        layout: str,
    ):
        assert isinstance(layout, str) and len(layout) == 2, f"Unsupported layout: {layout}"
        transa = layout[0].upper() == "T"
        transb = layout[1].upper() == "T"

        opA = A.transpose(0, 1).contiguous() if transa else A.contiguous()  # [K, O] or [I, O]
        opB = B.transpose(0, 1).contiguous() if transb else B.contiguous()  # [..., K]

        # Flatten opA to 2D if needed (weight tensors should be 2D, but validate)
        if opA.dim() > 2:
            # If A has extra dims, flatten all but last to match matmul_persistent expectations
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
            # For transposed original B, map gradient accordingly; B was transposed before forward
            # Here d_opB matches the shape of (B after possible transpose); reverse it
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
            "Batch-invariant GEMM does not support Userbuffers/overlap (extra_output/ub/ub_type/bulk_overlap)."
        )

    # Compute via autograd-aware function matching TE's layout semantics
    result = BatchInvariantTEGemmFn.apply(A, B, bias if not grad else None, out_dtype, layout)

    # Compute bias gradient if needed (for wgrad GEMM)
    # TODO: Note (Peter): I dont get this at all. This seems very wrong.
    # In wgrad: general_gemm(x, dy, layout="NT", grad=True, bias=bias)
    # - Computes: dw = dy^T @ x (this is 'result')
    # - Should also compute: db = sum(dy) where dy is B (grad_output)
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
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered_gamma: bool):
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
        x, weight, rsigma = ctx.saved_tensors
        w_eff = (weight + 1.0) if ctx.zero_centered_gamma else weight

        go_fp32 = grad_output.float()
        x_fp32 = x.float()
        w_fp32 = w_eff.to(device=x.device, dtype=torch.float32)
        r = rsigma
        r3 = r * r * r
        D = x.shape[-1]

        # dγ = Σ (g ⊙ x ⊙ r)   over all leading dims
        red_dims = tuple(range(0, go_fp32.ndim - 1))
        g_w = (go_fp32 * x_fp32 * r).sum(dim=red_dims).to(weight.dtype)

        # dx = g ⊙ (w r) − (w r^3) * x * Σ(g ⊙ x ⊙ w) / D
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


def _te_rmsnorm_forward_patched(self, x: torch.Tensor) -> torch.Tensor:
    """Patched TE RMSNorm.forward that routes to batch-invariant implementation with autograd support."""
    weight = getattr(self, "weight", None)
    if weight is None:
        raise RuntimeError("Batch-invariant RMSNorm requires affine weight.")
    # TODO(peter): Should I even allow defaults here like this?
    eps = getattr(self, "eps", 1e-5)
    zero_centered_gamma = getattr(self, "zero_centered_gamma", False)
    return BatchInvariantRMSNormFn.apply(x, weight, eps, zero_centered_gamma)


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
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
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None
    # Restore Transformer Engine kernels if previously patched
    _te_unpatch_for_batch_invariant()


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    global _batch_invariant_MODE, _batch_invariant_LIB
    old_data = (_batch_invariant_MODE, _batch_invariant_LIB)
    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    yield
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE, _batch_invariant_LIB = old_data

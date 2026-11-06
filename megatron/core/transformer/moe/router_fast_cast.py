# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact, vectorized BF16/FP16 -> FP32 cast for the router GEMM.

``Tensor.to(torch.float32)`` of a BF16 tensor runs PyTorch's unrolled, non-vectorized casting
copy: about 104 us for the [8192, 5120] router input on GB300 (2.4 TB/s). Widening a 16-bit float
to FP32 is exact, so a copy kernel with 16-byte loads returns bit-identical values (it converts
with the same ``cvt`` instruction) at close to HBM bandwidth.

CUDA-graph safety: every launch uses 16-byte aligned pointers and a whole number of blocks (a
ragged tail is copied by ``Tensor.copy_``) and passes no integer arguments, so each (device, source
dtype, block, warps, variant) has exactly one Triton specialization and a static shape never
recompiles. Inside CUDA-graph capture a specialization is launched only if it already ran outside
capture (the training warm-up iterations do that); otherwise the call uses ``Tensor.to``, which is
exact as well, so Triton never compiles or loads a module during capture.
"""

import logging

import torch

from megatron.core.transformer.moe import moe_env_flags

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # Triton ships with the PyTorch container; keep CPU-only imports working.
    HAVE_TRITON = False

logger = logging.getLogger(__name__)

_SOURCE_DTYPES = (torch.bfloat16, torch.float16)
_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)
_DEFAULT_LAUNCH_CONFIG = (4096, 8, False)

# Specializations launched outside CUDA-graph capture, and those that failed to compile or launch.
_WARM_KEYS = set()
_FAILED_KEYS = set()
_WARNED_KEYS = set()
# Calls inside capture that used Tensor.to because their specialization was not warm yet.
capture_fallbacks = 0


if HAVE_TRITON:

    @triton.jit
    def _upcast_fp32_kernel(src_ptr, dst_ptr, BLOCK: tl.constexpr, SPLIT_STORE: tl.constexpr):
        """Widen BLOCK consecutive 16-bit floats to FP32. Callers launch whole blocks only."""
        base = tl.program_id(0).to(tl.int64) * BLOCK
        if SPLIT_STORE:
            # A row holds 8 inputs (one 16-byte load); its 8 outputs leave as two 16-byte stores
            # of the same registers, so the load and store layouts need no conversion.
            rows = tl.arange(0, BLOCK // 8)[:, None] * 8
            values = tl.load(src_ptr + base + rows + tl.arange(0, 8)[None, :]).to(tl.float32)
            low, high = tl.split(tl.permute(tl.reshape(values, (BLOCK // 8, 2, 4)), (0, 2, 1)))
            dst = dst_ptr + base + rows + tl.arange(0, 4)[None, :]
            tl.store(dst, low)
            tl.store(dst + 4, high)
        else:
            offsets = base + tl.arange(0, BLOCK)
            tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets).to(tl.float32))

else:
    _upcast_fp32_kernel = None


def _valid_launch_config(block: int, num_warps: int) -> bool:
    """A block is a power of two in [256, 65536]; warps is a power of two up to 32."""
    return 256 <= block <= 65536 and block & (block - 1) == 0 and num_warps in (1, 2, 4, 8, 16, 32)


def _launch_config_from_flags() -> tuple:
    """Return the validated launch config selected by the MCORE_ROUTER_FAST_CAST_* switches."""
    config = (
        moe_env_flags.ROUTER_FAST_CAST_BLOCK,
        moe_env_flags.ROUTER_FAST_CAST_WARPS,
        moe_env_flags.ROUTER_FAST_CAST_SPLIT,
    )
    if _valid_launch_config(config[0], config[1]):
        return config
    logger.warning(
        "Ignoring invalid router cast launch config block=%d warps=%d; using block=%d warps=%d",
        config[0],
        config[1],
        _DEFAULT_LAUNCH_CONFIG[0],
        _DEFAULT_LAUNCH_CONFIG[1],
    )
    return (_DEFAULT_LAUNCH_CONFIG[0], _DEFAULT_LAUNCH_CONFIG[1], config[2])


# (elements per program, warps per program, split-store variant), fixed at import.
LAUNCH_CONFIG = _launch_config_from_flags()


def _warn_once(key: tuple, message: str) -> None:
    """Log ``message`` once per specialization key."""
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        logger.warning("Router fast cast %s: %s", key, message)


def _kernel_eligible(tensor: torch.Tensor, block: int) -> bool:
    """Whether the kernel's single specialization can widen ``tensor`` (see module doc)."""
    return (
        HAVE_TRITON
        and tensor.dtype in _SOURCE_DTYPES
        and type(tensor) in _PLAIN_TENSOR_TYPES
        and tensor.is_cuda
        and tensor.is_contiguous()
        and tensor.data_ptr() % 16 == 0
        and tensor.numel() >= block
        and not (tensor.requires_grad and torch.is_grad_enabled())
        and tensor.device.index == torch.cuda.current_device()
        and not torch.compiler.is_compiling()
    )


def cast_fp32_with_config(
    tensor: torch.Tensor, block: int, num_warps: int, split_store: bool
) -> torch.Tensor:
    """``tensor.to(torch.float32)`` with an explicit kernel launch shape (for tests and tuning).

    ``fast_cast`` is the entry point. Ineligible inputs (see ``_kernel_eligible``), a failed
    compilation and a not-yet-warm specialization inside CUDA-graph capture all use ``Tensor.to``.
    """
    global capture_fallbacks
    if not _kernel_eligible(tensor, block) or not _valid_launch_config(block, num_warps):
        return tensor.to(torch.float32)
    key = (tensor.device.index, tensor.dtype, block, num_warps, bool(split_store))
    if key in _FAILED_KEYS:
        return tensor.to(torch.float32)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _WARM_KEYS:
        capture_fallbacks += 1
        _warn_once(key, "first used inside CUDA-graph capture; this graph uses Tensor.to")
        return tensor.to(torch.float32)
    output = torch.empty(tensor.shape, dtype=torch.float32, device=tensor.device)
    if output.data_ptr() % 16 != 0:
        # Another pointer alignment would be another Triton specialization; keep the single one.
        return output.copy_(tensor)
    source = tensor.view(-1)
    destination = output.view(-1)
    num_blocks = source.numel() // block
    try:
        _upcast_fp32_kernel[(num_blocks,)](
            source, destination, BLOCK=block, SPLIT_STORE=bool(split_store), num_warps=num_warps
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        # Compilation or launch failure: Tensor.to gives the same values.
        if capturing:
            raise
        _FAILED_KEYS.add(key)
        _warn_once(key, f"kernel unavailable ({error!r}); using Tensor.to")
        return tensor.to(torch.float32)
    if not capturing:
        _WARM_KEYS.add(key)
    converted = num_blocks * block
    if converted < source.numel():
        destination[converted:].copy_(source[converted:])
    return output


def fast_cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bit-identical drop-in replacement for ``tensor.to(dtype)``.

    Widening a contiguous, 16-byte aligned CUDA BF16/FP16 tensor of at least one block to FP32 on
    the current device, outside autograd recording and torch.compile, uses the vectorized kernel;
    every other case calls ``tensor.to(dtype)``.
    """
    if dtype != torch.float32 or tensor.dtype == dtype:
        return tensor.to(dtype)
    block, num_warps, split_store = LAUNCH_CONFIG
    return cast_fp32_with_config(tensor, block, num_warps, split_store)

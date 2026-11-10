# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CUDA VMM (Virtual Memory Management) allocator for NCCL symmetric-memory pools.

Implements NCCL's requirements on user-allocated communication buffers
(https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)
minimally: VMM allocations at the recommended granularity, exportable handle
types (POSIX FD, plus FABRIC when supported — dropped on retry if cuMemCreate
rejects it), and GPUDirect-RDMA-capable physical memory when supported.

``ncclMemAlloc`` uses the same VMM driver calls; the difference is that this
allocator maps memory only on the allocation's device (peers access it through
NCCL windows), while ``ncclMemAlloc`` additionally maps every allocation on all
P2P-visible peer GPUs — and those persistent peer mappings slow CPU-side kernel
launching for the whole training step.

Window registration is delegated to ``nccl_allocator.register_mem_pool``
(which calls ``ncclCommWindowRegister``), which accepts this memory and runs its
symmetric kernels on it. Building the extension requires nvcc and libcuda at
runtime; ``init``/``create_vmm_mem_pool`` raise if it cannot build, and callers
decide whether to fall back to ``ncclMemAlloc``-backed pools.
"""

import logging
import os

import torch

# This import is needed for the cpp extension to work.
# pylint: disable=unused-import
from torch.utils import cpp_extension

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core.nccl_allocator import get_func_args
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_allocator = None
_build_error = None


def _build_vmm_allocator():
    global _allocator, _build_error
    # If the allocator is already built, return; if the build already failed, do not
    # retry the compilation on every call.
    if _allocator is not None:
        return
    if _build_error is not None:
        raise RuntimeError(
            "[MCORE][VMM_SYMM_ALLOCATOR] The VMM allocator extension failed to build "
            "(requires nvcc and libcuda at runtime)."
        ) from _build_error

    # load_inline writes main.cpp before acquiring PyTorch's build lock, so ranks
    # can overwrite the source while another rank compiles it. Use a packaged
    # source file that is never rewritten at runtime; load serializes the build.
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(os.path.dirname(__file__), "csrc", "vmm_symm_allocator.cpp")
    build_dir = os.path.join(module_dir, "build", "vmm_symm_allocator")
    os.makedirs(build_dir, exist_ok=True)
    try:
        vmm_allocator = torch.utils.cpp_extension.load(
            name="vmm_symm_allocator",
            sources=[source_path],
            with_cuda=True,
            extra_ldflags=["-lcuda"],
            verbose=True,
            is_python_module=True,
            build_directory=build_dir,
        )
    except Exception as e:
        _build_error = e
        raise RuntimeError(
            "[MCORE][VMM_SYMM_ALLOCATOR] Failed to build the VMM allocator extension "
            "(requires nvcc and libcuda at runtime)."
        ) from e

    _allocator = vmm_allocator.get_vmm_allocator()


def create_vmm_mem_pool() -> torch.cuda.MemPool:
    """
    Create a symmetric memory pool using the VMM allocator. Callers enforce the
    torch >= 2.9 floor that symmetric pools need.
    """
    _build_vmm_allocator()
    assert _allocator is not None, "VMM allocator is not initialized"
    if 'symmetric' in get_func_args(torch.cuda.MemPool):
        # PyTorch >= 2.9.0a0 and before PyTorch PR #161238 takes the symmetric knob at
        # MemPool construction; since #161238 it lives in the registration function.
        return torch.cuda.MemPool(_allocator, symmetric=True)
    if 'symm_mem' in get_func_args(torch.cuda.MemPool):
        # Argument name divergence between nvidia pytorch and the official pytorch.
        return torch.cuda.MemPool(_allocator, symm_mem=True)
    # The symmetric knob is in the registration function.
    return torch.cuda.MemPool(_allocator)


def init() -> None:
    """
    Initialize the VMM allocator, including the NCCL environment its pools are
    registered under (same settings as nccl_allocator.init()).
    """
    # Enables NCCL NVLS algorithm
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    # Disables the use of the tensor register allocator hook
    os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "0"
    _build_vmm_allocator()
    log_single_rank(
        logger, logging.INFO, "[MCORE][VMM_SYMM_ALLOCATOR] Initialized the VMM Allocator"
    )


def register_mem_pool(pool: torch.cuda.MemPool, group) -> None:
    """
    Window-register a pool's segments on ``group`` (always symmetric).
    Delegating to nccl_allocator is safe because its (de)registration walks the
    pool's segments and never touches the allocator that produced them.
    """
    nccl_allocator.register_mem_pool(pool, group, symmetric=True)


def deregister_mem_pool(pool: torch.cuda.MemPool, group) -> None:
    """
    Deregister a pool's windows from ``group``. Delegates to nccl_allocator.
    """
    nccl_allocator.deregister_mem_pool(pool, group)

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GTP symmetric memory: NCCL window registration for GTP communication buffers.

This module keeps one ``ncclMemAlloc``-backed ``torch.cuda.MemPool`` per GTP process
group. Once ``register_gtp_symm_pool(group)`` registers a pool on its group, PyTorch's
ProcessGroupNCCL hook window-registers every allocation made inside
``gtp_symm_pool_ctx(group)``, which lets NCCL run its symmetric / NVLS kernels on
those buffers.

Two parts:
  - Pool lifecycle: create, register, query, and tear down the per-group pools,
    plus the allocation context ``gtp_symm_pool_ctx``.
  - ``symmetric_wgrad_pool`` (a ``RegisteredLIFOPool``): recycled, window-registered
    send buffers for eager wgrad reduce-scatter. Local CUDA graphs allocate their
    plan-owned persistent send arenas through ``gtp_symm_pool_ctx`` instead.
"""

from __future__ import annotations

import logging
import math
import typing
from collections import defaultdict
from contextlib import AbstractContextManager

import torch
import torch.distributed as dist

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core.utils import is_torch_min_version, log_single_rank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registries (process-global, keyed by group.group_name)
# ---------------------------------------------------------------------------

# One MemPool per GTP/EGTP group, created once.
_pools: typing.Dict[str, torch.cuda.MemPool] = {}

# Groups whose pool registration is live. Maps name -> group (not a set) because
# teardown needs the group object to deregister.
_registered: typing.Dict[str, typing.Any] = {}

# ---------------------------------------------------------------------------
# Pool lifecycle: create -> warm -> register -> allocate-into -> query -> deregister
# ---------------------------------------------------------------------------


def _get_gtp_symm_pool(group: dist.ProcessGroup) -> torch.cuda.MemPool:
    """Return the per-group ``ncclMemAlloc``-backed MemPool, creating it once."""
    name = group.group_name
    pool = _pools.get(name)
    if pool is None:
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)
        _pools[name] = pool
    return pool


def register_gtp_symm_pool(group: dist.ProcessGroup | None) -> torch.cuda.MemPool | None:
    """Create (if needed) and register the group's pool. Safe to call more than once;
    does nothing for ``None`` or single-rank groups.

    Issues a collective, so call it during model construction — before the first
    forward or any CUDA-graph capture. New segments register automatically afterwards.
    """
    if group is None or group.size() <= 1:
        return None
    if not is_torch_min_version("2.9.0a0"):
        raise RuntimeError(
            "[GTP] --gtp-remat-nccl-ub/--gtp-expert-remat-nccl-ub require PyTorch >= 2.9: older "
            "versions cannot create a symmetric memory pool (create_nccl_mem_pool silently "
            "falls back to a non-symmetric one, so the reduce-scatter would not be symmetric)."
        )
    pool = _get_gtp_symm_pool(group)
    if group.group_name in _registered:
        return pool
    # NCCL creates communicators lazily on the first collective; run one tiny all-reduce
    # so the registration below sees an initialized communicator.
    warmup = torch.zeros(1, device=torch.cuda.current_device())
    dist.all_reduce(warmup, group=group)
    nccl_allocator.register_mem_pool(pool, group, symmetric=True)
    _registered[group.group_name] = group
    log_single_rank(
        logger,
        logging.INFO,
        f"[MCORE][GTP] Registered GTP cache pool on group {group.group_name} "
        f"(size={group.size()})",
    )
    return pool


def gtp_symm_pool_ctx(group: dist.ProcessGroup) -> AbstractContextManager[None]:
    """Context manager: allocations inside it come from ``group``'s pool. Collective-free
    (capture-safe); register the pool first or allocations are not window-registered."""
    return torch.cuda.use_mem_pool(_get_gtp_symm_pool(group))


def is_gtp_symm_pool_registered(group: dist.ProcessGroup | None) -> bool:
    """True once ``register_gtp_symm_pool`` has registered this group's pool; also False for
    ``None`` and single-rank groups, which are never registered."""
    return group is not None and group.size() > 1 and group.group_name in _registered


# ---------------------------------------------------------------------------
# RS send-buffer LIFO: recycled window-registered scratch for wgrad reduce-scatters
# ---------------------------------------------------------------------------


class RegisteredLIFOPool:
    """A recycling cache of window-registered buffers, one free list per group.

    The wgrad reduce-scatter can only use symmetric collectives if its send buffer is
    window-registered, so the wgrad is written into a buffer from this cache. ``alloc``
    pops a free buffer (or allocates a new one through ``gtp_symm_pool_ctx``); ``free``
    returns it once the reduce-scatter has finished reading it. Buffers are shared by
    all weights of the same size, so memory stays at the peak number of in-flight
    reduce-scatters instead of one buffer per weight.

    CUDA graphs: the eager warmup iterations run the same reduce-scatter overlap as
    the captured steps and are expected to pre-populate the free lists, so that during
    capture ``alloc`` only ever pops. Allocating a new buffer during capture would be
    illegal, so ``alloc`` raises if that expectation did not hold.

    Buffers are stored 1-D, so one free list serves every shape with the same element
    count. ``alloc`` returns a view tagged with ``_gtp_symm_group``; ``free`` ignores
    untagged tensors, which lets callers pass mixed buffer lists to both this pool and
    the plain scratch pool and have each take only its own.

    Why LIFO: ordering cannot affect correctness (buffers enter the free list only
    after their reduce-scatter has been waited on), but LIFO keeps the same buffer
    reused for the same operation at steady state even if a key ever over-allocates,
    whereas FIFO would rotate the assignment every iteration -- LIFO keeps memory
    behavior deterministic and repeatable across iterations.
    """

    def __init__(self) -> None:
        # (numel, dtype, group_name) -> list of free 1-D buffers.
        self._free: dict[tuple, list] = defaultdict(list)
        self._warned_unregistered: set = set()

    def alloc(
        self,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Return a buffer of ``shape`` from ``group``'s free list, allocating one if empty.

        Raises RuntimeError if a fresh allocation would be needed during CUDA-graph capture.
        """
        numel = int(math.prod(shape))
        bucket = self._free[(numel, dtype, group.group_name)]
        if bucket:
            flat = bucket.pop()
        else:
            if torch.cuda.is_current_stream_capturing():
                mine = sum(len(v) for k, v in self._free.items() if k[2] == group.group_name)
                others = sum(len(v) for k, v in self._free.items() if k[2] != group.group_name)
                hint = (
                    "this group's free lists are empty while other groups have "
                    f"{others} free buffer(s) -- likely stranded send buffers from "
                    "backwards that never reduce-scattered"
                    if mine == 0 and others > 0
                    else "likely the warmup depth or RS concurrency changed between "
                    "warmup and capture -- run more warmup iters"
                )
                raise RuntimeError(
                    "[GTP] RegisteredLIFOPool exhausted during CUDA-graph capture "
                    f"(group={group.group_name}, numel={numel}, dtype={dtype}): {hint}."
                )
            # Allocate from the group's registered pool when it has one; else plain memory.
            if is_gtp_symm_pool_registered(group):
                with gtp_symm_pool_ctx(group):
                    flat = torch.empty(numel, dtype=dtype, device=device)
            else:
                # Unreachable in production (callers gate on registration); the buffer
                # would be sent with regular kernels, so make a misconfiguration visible.
                if group.group_name not in self._warned_unregistered:
                    self._warned_unregistered.add(group.group_name)
                    log_single_rank(
                        logger,
                        logging.WARNING,
                        f"[GTP] RegisteredLIFOPool.alloc on unregistered group "
                        f"{group.group_name}: buffer will not be window-registered.",
                    )
                flat = torch.empty(numel, dtype=dtype, device=device)
        out = flat.view(shape)
        out._gtp_symm_group = group  # marks the buffer as pool-owned; free() keys on this
        return out

    def free(self, buf: torch.Tensor) -> None:
        """Return ``buf`` to its group's free list; no-op for untagged (foreign) buffers."""
        group = getattr(buf, "_gtp_symm_group", None)
        if group is None:
            return
        self._free[(buf.numel(), buf.dtype, group.group_name)].append(buf.view(-1))

    def clear(self) -> None:
        """Drop every cached buffer. Called at teardown, before the pools they alias go away."""
        self._free.clear()


# The process-wide send-buffer cache, used by generalized_tensor_parallelism. Lives here
# so deregister_and_clear_gtp_symm_pools can drop its buffers at teardown.
symmetric_wgrad_pool = RegisteredLIFOPool()


def deregister_and_clear_gtp_symm_pools() -> None:
    """Tear down what this module owns: deregister the pools' windows, then drop the
    recycled send buffers. Allocations owned by others (e.g. CUDA-graph persistent arenas)
    are not freed here. Call on all ranks before teardown; no-op if never registered."""
    # Wait for all GPU work to finish first: a kernel or collective still reading
    # pool memory would fault once the windows go away.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    # Deregister while the recycled send buffers are still alive. Their memory keeps
    # the pool non-empty, so deregister_mem_pool (which skips empty pools) always runs.
    for name in sorted(_registered):
        nccl_allocator.deregister_mem_pool(_pools[name], _registered[name])
    # Only now drop the buffers and the pools; the windows are gone, so the memory
    # is safe to release.
    symmetric_wgrad_pool.clear()
    _registered.clear()
    _pools.clear()

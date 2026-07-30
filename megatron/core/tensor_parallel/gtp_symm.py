# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GTP symmetric-memory pools: the shared registration primitive.

One ``ncclMemAlloc``-backed ``torch.cuda.MemPool`` per GTP process group, registered once
via ``backend.register_mem_pool(pool, symm=True)``. PyTorch's ProcessGroupNCCL segment hook
then auto-registers every later allocation made under ``gtp_mem_pool_ctx(group)``. With both
ends of a collective in such a pool on the same comm, NCCL selects its symmetric/NVLS kernels.

This module owns only the pool + registration concern. Consumers ride on ``gtp_mem_pool_ctx``:
``GTPWeightCache`` for AG output buffers, ``RegisteredLifoPool`` for RS send buffers, and
``register_ddp_buffers_on_gtp_groups`` for the DDP param buffer (the AG input).

Which params participate is decided at wrap time from ``--gtp-nccl-ub`` (dense) /
``--egtp-nccl-ub`` (expert) -- independent of ``--use-nccl-ub``, which covers the DP group --
and stamped by ``_gtp_attach_attrs`` as ``gtp_smr`` (use this GTP pool) and
``param_needs_nccl_mem`` (back the DDP param buffer with ncclMemAlloc). Sites read both via
getattr; there is no module-level env gate here.

Launcher must export ``NCCL_NVLS_ENABLE=1`` and
``TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0`` before ``init_process_group``.
"""

import logging
import math
from collections import defaultdict
from contextlib import AbstractContextManager

import torch
import torch.distributed as dist

import megatron.core.nccl_allocator as nccl_allocator
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

# group.group_name -> per-group MemPool (one pool per group, registered once).
_pools: "dict[str, torch.cuda.MemPool]" = {}
# group.group_name -> group for pools whose registration is live. A dict (not a set) so the
# shutdown deregister can reach each group; membership still gates re-registration.
_registered: "dict[str, object]" = {}
# group.group_name for groups whose NCCL comm has been warmed (lazy comms are created on the
# first collective). Shared by all GTP register paths so each group is warmed exactly once.
_warmed_groups: "set[str]" = set()


def get_gtp_pool(group: dist.ProcessGroup) -> torch.cuda.MemPool:
    """Return the per-group ``ncclMemAlloc``-backed MemPool, creating it once."""
    name = group.group_name
    pool = _pools.get(name)
    if pool is None:
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)
        _pools[name] = pool
    return pool


def _warmup_group_comm(group: dist.ProcessGroup) -> None:
    """Force lazy NCCL comm creation for ``group`` once, so a subsequent register_mem_pool
    sees an initialized communicator. Idempotent across all GTP register paths."""
    if group.group_name in _warmed_groups:
        return
    warmup = torch.zeros(1, device=torch.cuda.current_device())
    dist.all_reduce(warmup, group=group)
    _warmed_groups.add(group.group_name)


def register_gtp_pool(group: dist.ProcessGroup) -> torch.cuda.MemPool:
    """Create (if needed) and register the per-group pool on ``group``. Idempotent.

    Call once at model-construction time (before any CUDA-graph capture or
    forward), because it issues a collective (comm warmup). After this, the
    segment hook auto-registers future segments, so ``gtp_mem_pool_ctx`` itself
    is capture-safe and collective-free.
    """
    pool = get_gtp_pool(group)
    if group.group_name in _registered:
        return pool
    _warmup_group_comm(group)
    nccl_allocator.register_mem_pool(pool, group, symmetric=True)
    _registered[group.group_name] = group
    log_single_rank(
        logger,
        logging.INFO,
        f"[MCORE][GTP] Registered GTP cache pool on group {group.group_name} "
        f"(size={group.size()})",
    )
    return pool


def gtp_mem_pool_ctx(group: dist.ProcessGroup) -> AbstractContextManager[None]:
    """Context manager: allocations inside land in ``group``'s registered pool.

    Pure ``use_mem_pool`` -- no collective -- so it is safe inside CUDA-graph
    capture. The pool must already be registered via ``register_gtp_pool`` for
    new segments to be window-registered by the segment hook.
    """
    return torch.cuda.use_mem_pool(get_gtp_pool(group))


def is_gtp_pool_registered(group: dist.ProcessGroup | None) -> bool:
    """True once ``register_gtp_pool`` has registered this group's symmetric pool.

    Full "use the symm pool for this group?" predicate: also rejects None and trivial
    (size-1) groups, which are never registered.
    """
    return group is not None and group.size() > 1 and group.group_name in _registered


class RegisteredLifoPool:
    """Group-aware LIFO cache of reduce-scatter send buffers in the per-group symmetric pool.

    Holds the full-shape wgrad the bwd GEMM writes and then scatters. Fresh allocations go
    through ``gtp_mem_pool_ctx`` so they are window-registered; freed buffers are recycled
    (keyed by numel + dtype + group), keeping memory flat rather than one buffer per weight.
    Storage is 1-D so one key serves any shape with that numel, and ``alloc`` returns a view
    tagged with ``_gtp_symm_group`` for recycling.

    The free-list grows to peak RS concurrency during the eager warmup iterations, so under
    CUDA-graph capture ``alloc`` only ever pops. A fresh allocation during capture would
    re-register the pool mid-graph, so it raises instead.
    """

    def __init__(self) -> None:
        # (numel, dtype, group_name) -> list of free 1-D buffers.
        self._free: "dict[tuple, list]" = defaultdict(list)

    def alloc(
        self,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Return a buffer of ``shape`` from ``group``'s free-list, allocating one if empty.

        Raises RuntimeError if a fresh allocation would be needed during CUDA-graph capture.
        """
        numel = int(math.prod(shape))
        bucket = self._free[(numel, dtype, group.group_name)]
        if bucket:
            flat = bucket.pop()
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "[GTP] RegisteredLifoPool exhausted during CUDA-graph capture "
                    f"(group={group.group_name}, numel={numel}, dtype={dtype}). The "
                    "eager warmup did not pre-populate enough RS send buffers for "
                    "the reduce-scatter overlap depth -- run more warmup iters, or "
                    "the RS concurrency changed between warmup and capture."
                )
            # Symm pool only when registered (gtp_smr implies it); else a plain buffer (non-symm).
            if is_gtp_pool_registered(group):
                with gtp_mem_pool_ctx(group):
                    flat = torch.empty(numel, dtype=dtype, device=device)
            else:
                flat = torch.empty(numel, dtype=dtype, device=device)
        out = flat.view(shape)
        out._gtp_symm_group = group  # tag so callers can recycle via tag dispatch
        return out

    def free(self, buf: torch.Tensor) -> None:
        """Return ``buf`` to the free-list of the group it was allocated from.

        No-op for a buffer this pool did not allocate (no ``_gtp_symm_group`` tag).
        """
        group = getattr(buf, "_gtp_symm_group", None)
        if group is None:
            return
        self._free[(buf.numel(), buf.dtype, group.group_name)].append(buf.reshape(-1))

    def clear(self) -> None:
        """Drop every cached buffer. Called at teardown, before the pools they alias go away."""
        self._free.clear()


# Imported by generalized_tensor_parallelism. Lives here so deregister_gtp_pools can drop its
# buffers at teardown, alongside the pools they alias.
_gtp_wgrad_pool = RegisteredLifoPool()


def _ddp_buffers(ddp_module: torch.nn.Module) -> list:
    """All param/grad buffers of a DDP-wrapped module (dense + expert-parallel)."""
    return list(getattr(ddp_module, "buffers", [])) + list(
        getattr(ddp_module, "expert_parallel_buffers", [])
    )


def _buffer_symm_groups(buf) -> list[dist.ProcessGroup]:
    """GTP comm groups this buffer's pool must be (de)registered on: params with
    ``param_needs_nccl_mem`` set and group size > 1.

    Shared by register and deregister so the two stay in step. Sorted by name because the
    collective (de)register order must match across ranks or the comms cross-deadlock.
    """
    if getattr(buf, "nccl_mem_pool", None) is None:
        return []
    # Needs a param_data all-gather input: without distopt the pool backs only grad_data, which
    # the GTP AG never reads, so registering it would cost a window for nothing.
    if getattr(buf, "param_data", None) is None:
        return []
    groups = {}
    for param in buf.params:
        if not getattr(param, "param_needs_nccl_mem", False):
            continue
        group = getattr(param, "group", None)
        if group is not None and group.size() > 1:
            groups.setdefault(group.group_name, group)
    return [group for _, group in sorted(groups.items())]


def register_ddp_buffers_on_gtp_groups(ddp_module: torch.nn.Module) -> None:
    """Register each DDP buffer's NCCL pool on the GTP group(s) its params opted into
    via ``param_needs_nccl_mem``, so the DDP param buffer (the AG input) is in the
    symmetric window. The GTP-owned cache/RS pool (AG/RS output) is registered separately
    by ``configure_gtp_remat_from_recipe`` before construction. Call once per model chunk
    *after* DDP construction (buffers/pools must exist).

    Always symmetric: ``--disable-symmetric-registration`` scopes to the DP-group
    registration, not the GTP groups, which are opted into by --gtp-nccl-ub/--egtp-nccl-ub.
    """
    for buf in _ddp_buffers(ddp_module):
        for group in _buffer_symm_groups(buf):
            # buf.nccl_mem_pool is non-None here (checked in _buffer_symm_groups).
            _warmup_group_comm(group)
            nccl_allocator.register_mem_pool(buf.nccl_mem_pool, group, symmetric=True)
            log_single_rank(
                logger,
                logging.INFO,
                f"[MCORE][GTP] Registered DDP param/grad pool on GTP group "
                f"{group.group_name} (size={group.size()})",
            )


def deregister_ddp_buffers_from_gtp_groups(ddp_module: torch.nn.Module) -> None:
    """Mirror of ``register_ddp_buffers_on_gtp_groups``: deregister each buffer's pool from
    the same GTP group set. Call at graceful exit *before* the ProcessGroupNCCL destructor
    -- window-registered handles left on a comm make its ncclCommDeregister abort
    ("Could not find handle"). The DP group is the core buffer's concern, handled
    separately by the training loop.
    """
    for buf in _ddp_buffers(ddp_module):
        for group in _buffer_symm_groups(buf):
            # buf.nccl_mem_pool is non-None here (checked in _buffer_symm_groups).
            nccl_allocator.deregister_mem_pool(buf.nccl_mem_pool, group)
            log_single_rank(
                logger,
                logging.INFO,
                f"[MCORE][GTP] Deregistered DDP param/grad pool from GTP group "
                f"{group.group_name} (size={group.size()})",
            )


def deregister_gtp_pools() -> None:
    """Deregister all GTP-owned symmetric pools.

    Must be called (collectively, on all ranks) before process-group teardown when GTP
    symmetric memory was used -- leftover windows abort the ProcessGroupNCCL destructor.
    Training shutdown does this; test fixtures enabling GTP NCCL-UB must do likewise.
    No-op when nothing was registered. Also drops the recycled RS send buffers, which alias
    the pools being torn down here.
    """
    for name in sorted(_registered):
        nccl_allocator.deregister_mem_pool(_pools[name], _registered[name])
    _registered.clear()
    _pools.clear()
    _gtp_wgrad_pool.clear()
    _warmed_groups.clear()

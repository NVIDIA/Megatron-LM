# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
This module provides a singleton instance of AsyncCallsQueue which manages
the async checkpoint save calls.
"""
import inspect
import logging
import time
from abc import ABC
from typing import TYPE_CHECKING, Any

from megatron.core.dist_checkpointing.strategies.nvrx import (
    make_nvrx_async_request,
)
from megatron.training import get_args
from megatron.training.utils import print_rank_0


def _tag_current_span_call_idx(call_idx):
    """Stamp the async request's call_idx onto whatever OTel span is currently
    active -- the shared join key that ties an async checkpoint's three
    separate-in-time spans together (dispatch's megatron.checkpoint.save subtree,
    the worker's nvrx.checkpoint.* with nvrx.call_idx, and finalize's
    megatron.checkpoint.save.finalize), since they can't share a parent span. No-op
    when telemetry is off (get_current_span returns a non-recording span) or
    nemo-lens is absent. call_idx may be an int (dispatch) or a list (finalize
    may complete more than one call)."""
    if call_idx is None or call_idx == []:
        return
    try:
        from opentelemetry import trace as _ot
        from nemo.lens.helpers import safe_set_span_attributes as _set

        _set(_ot.get_current_span(), {'nvrx.call_idx': call_idx})
    except Exception:  # noqa: BLE001 -- telemetry must never break checkpointing
        pass

if TYPE_CHECKING:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest
else:
    AsyncRequest = Any

try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import _results_queue
except (ImportError, ModuleNotFoundError):
    _results_queue = None


logger = logging.getLogger(__name__)

# Singleton manager of async calls
_async_calls_queue = None


def _get_async_calls_queue():
    """Get or lazily initialize the async calls queue."""
    global _async_calls_queue

    if _async_calls_queue is None:
        from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue

        args = get_args()
        _async_calls_queue = AsyncCallsQueue(
            persistent=getattr(args, "use_persistent_ckpt_worker", False)
        )

    return _async_calls_queue


def build_otel_worker_bootstrap(args):
    """Build the plain-dict otel config + resource attributes to hand to the
    persistent checkpoint worker process.

    The worker is spawned via multiprocessing 'spawn' and inherits nothing from
    this process -- no TracerProvider, no open file -- so it has to bootstrap
    its own independent telemetry from data passed explicitly through the
    ctx.Process args (same channel cpu_priority/io_priority already use). Must
    stay a plain dict of primitives: it crosses a pickling boundary, and a
    class instance there would tie correctness to an import path staying
    resolvable in a separate process.
    """
    import os

    from megatron.training.global_vars import build_telemetry_resource_attrs

    service_name = getattr(args, 'otel_service_name', None)
    if not service_name and not os.environ.get('OTEL_SERVICE_NAME', '').strip():
        service_name = 'megatron-lm'

    # Span groups cross to the worker two ways, for backward compatibility with
    # a baked (not-yet-rebuilt) NVRx worker:
    #
    #   span_groups (str)         -- BASE-resolvable spec only. The worker's
    #       setup_telemetry() resolves this with the base SpanGroup class (not
    #       MegatronSpanGroup), so Megatron-only names ('trace_region', etc.)
    #       would raise "Unknown span group". We resolve here with
    #       MegatronSpanGroup and intersect with the base group set, so an OLD
    #       worker only ever sees names it can resolve (keeps its 'checkpoint'
    #       spans; misses Megatron-only save-side detail). Never crashes.
    #
    #   resolved_span_groups (list[str]) -- the FULL set (incl. Megatron-only
    #       groups), already resolved here. A NEW worker calls
    #       set_enabled_span_groups() with this AFTER setup_telemetry, overriding
    #       the base-safe set -- giving it trace_region etc. (save-side
    #       checkpoint internals). An OLD worker ignores this key.
    worker_span_groups = getattr(args, 'otel_span_groups', None)
    resolved_span_groups = None
    if worker_span_groups:
        try:
            from nemo.lens.groups import SpanGroup as _BaseSpanGroup
            from megatron.core.telemetry.span_groups import MegatronSpanGroup

            resolved = MegatronSpanGroup.resolve(worker_span_groups)
            resolved_span_groups = sorted(resolved)
            base_safe = resolved & _BaseSpanGroup.ALL_GROUPS
            worker_span_groups = ','.join(sorted(base_safe)) if base_safe else 'default'
        except Exception:
            # nemo-lens absent or resolve failed -- fall back to a base preset
            # rather than risk handing the worker something it can't resolve.
            worker_span_groups = 'per_step'

    enabled = bool(getattr(args, 'otel_enabled', False))
    return {
        'enabled': enabled,
        'service_name': service_name,
        'span_groups': worker_span_groups,
        'resolved_span_groups': resolved_span_groups,
        'rank': args.rank,
        'world_size': args.world_size,
        # Same reasoning as _set_telemetry(): building these calls NVML
        # (_detect_gpu_identity), and the worker discards them when 'enabled' is
        # False, so don't build them on the disabled path. The key is always
        # present, so the worker-side contract is unchanged.
        'resource_attrs': build_telemetry_resource_attrs(args) if enabled else {},
    }


def init_persistent_async_worker(rank: int, mp_mode: str = 'spawn'):
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import get_write_results_queue

    global _async_calls_queue
    args = get_args()
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    time_start = time.time()
    if rank == 0:
        print(f"init_persistent_async_worker: {rank}, Starting Async Caller", flush=True)
    _async_calls_queue = AsyncCallsQueue(
        persistent=True,
        **(
            {"cpu_shm_mode": args.async_ckpt_use_cpu_shm}
            if "cpu_shm_mode" in inspect.signature(AsyncCallsQueue.__init__).parameters
            else {}
        ),
    )
    # initialize the persistent caller with QoS priorities from args
    warmup_kwargs = {}
    if "cpu_shm_mode" in inspect.signature(AsyncCallsQueue.warmup_persistent_caller).parameters:
        warmup_kwargs["cpu_shm_mode"] = args.async_ckpt_use_cpu_shm
    elif args.async_ckpt_use_cpu_shm:
        raise AssertionError(
            "Installed nvidia-resiliency-ext does not support cpu_shm_mode. "
            "Update nvidia-resiliency-ext to use --async-ckpt-use-cpu-shm."
        )
    # Older nvidia-resiliency-ext installs won't have this parameter yet --
    # degrade to no worker-side telemetry rather than hard-failing checkpointing.
    if "otel_bootstrap" in inspect.signature(AsyncCallsQueue.warmup_persistent_caller).parameters:
        warmup_kwargs["otel_bootstrap"] = build_otel_worker_bootstrap(args)
    AsyncCallsQueue.warmup_persistent_caller(
        rank,
        cpu_priority=args.async_ckpt_cpu_priority,
        io_priority=args.async_ckpt_io_priority,
        **warmup_kwargs,
    )
    # initialize ckpt write results queue
    if "mp_mode" not in inspect.signature(get_write_results_queue).parameters:
        raise AssertionError(
            "Installed nvidia-resiliency-ext does not support "
            "get_write_results_queue(mp_mode=...). Update nvidia-resiliency-ext."
        )
    get_write_results_queue(mp_mode="fork")
    if rank == 0:
        print(f"init_persistent_async_worker: rank {rank}, Async Caller Started in {time.time() - time_start} seconds", flush=True)


def schedule_async_save(async_request: AsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    call_idx = _get_async_calls_queue().schedule_async_request(async_request)
    _tag_current_span_call_idx(call_idx)


def maybe_finalize_async_save(blocking: bool = False, terminate=False):
    """Finalizes active async save calls and cleans up deletion processes.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    args = get_args()
    if not args.async_save:
        return

    if blocking and not is_empty_async_queue():
        print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

    async_calls_queue = _async_calls_queue
    if async_calls_queue is not None:
        finalized = async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)
        # Tag the active finalize span (megatron.checkpoint.save.finalize) with the
        # call_idx(es) that actually completed this call, so it groups with the
        # matching dispatch + worker spans.
        _tag_current_span_call_idx(finalized)

    # Clean up finished deletion processes to prevent zombies
    # Import here to avoid circular dependency
    from .checkpointing import finalize_deletion_processes
    finalize_deletion_processes(blocking=blocking or terminate)

    if terminate and async_calls_queue is not None:
        async_calls_queue.close()


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Returns:
        bool: True if there is any ongoing async call.
    """
    return _async_calls_queue is None or _async_calls_queue.get_num_unfinalized_calls() == 0


def reset_persistent_async_worker():
    from nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader import ( # pylint: disable=line-too-long
        CachedMetadataFileSystemReader,
    )

    global _async_calls_queue, _results_queue

    if _async_calls_queue is not None:
        _async_calls_queue.close(abort=True)
        del _async_calls_queue
    if _results_queue is not None:
        _results_queue._manager.shutdown()
        del _results_queue
    _results_queue = None
    _async_calls_queue = None
    CachedMetadataFileSystemReader.clear_metadata_cache()


def get_save_and_finalize_callbacks(writer, save_state_dict_ret) -> AsyncRequest:
    """Creates an async save request for fsdp_dtensor & torch_dcp with a finalize function."""
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest
    from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import save_state_dict_async_finalize # pylint: disable=line-too-long

    save_fn, preload_fn, save_args = writer.get_save_function_and_args()

    def finalize_fn():
        """Finalizes async checkpointing and synchronizes processes."""
        save_state_dict_async_finalize(*save_state_dict_ret)

    return make_nvrx_async_request(
        AsyncRequest,
        save_fn,
        save_args,
        [finalize_fn],
        async_fn_kwargs={},
        preload_fn=preload_fn,
    )

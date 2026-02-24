# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Storage writer for PyT Distributed format allowing asynchronous save."""

import dataclasses
import inspect
import logging
import os
import pickle
import queue
from functools import partial
from heapq import heappop, heappush
from itertools import chain
from operator import itemgetter
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import multiprocessing as mp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.api import WRAPPED_EXCEPTION, _wrap_exception
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _StoragePrefix, _write_item
from torch.distributed.checkpoint.metadata import Metadata

try:
    from torch.distributed.checkpoint.filesystem import _StorageWriterTransforms
except ImportError:
    _StorageWriterTransforms = Any

from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

from .async_utils import PersistentAsyncCaller, _disable_gc

logger = logging.getLogger(__name__)

WriteBucket = Tuple[str, str, Tuple[list, list]]  # represents writes to a single file
# Compact structure: plan items + resolved data, avoiding expanded bucket structure
ResolvedPlanData = Tuple[
    str, str, List[WriteItem], List[Any], int, _StoragePrefix
]  # checkpoint_dir, separation_hint, items, resolved_data, thread_count, storage_plan


class ConsistentDataIdentifier:
    """Identifier for consistent data structure stored in worker cache.

    This allows passing a lightweight identifier instead of pickling
    the entire data structure (which includes IPC handles) across process boundaries.
    """

    def __init__(self, key: int):
        self.key = key


def _compute_data_structure_key_from_plan(items: List[WriteItem]) -> int:
    """Compute a hash key based on plan items only (no data resolution needed).

    This creates a deterministic key from plan metadata that's available without
    resolving the actual tensor data.

    Args:
        items: List of WriteItem from the plan

    Returns:
        Integer hash key representing the data structure
    """
    structure_info = []

    for item in items:
        # Include item metadata that defines the structure
        item_info = (
            item.index.fqn,  # Fully qualified name
            item.type,  # WriteItemType (BYTE_IO or TENSOR)
        )

        # Include metadata from plan (available without resolving data)
        if item.tensor_data is not None:
            # Use tensor metadata from the plan
            data_info = (
                tuple(item.tensor_data.chunk.sizes),  # Tensor chunk shape
                str(item.tensor_data.properties.dtype),  # Data type
            )
        else:
            # For non-tensor data (BYTE_IO), use placeholder
            data_info = (("BYTE_IO",), "BYTE_IO")
        structure_info.append((item_info, data_info))

    # Create a hash from the structure info
    return hash(tuple(structure_info))


try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

_results_queue = None


def get_write_results_queue():
    global _results_queue
    if _results_queue is None:
        ctx = mp.get_context("spawn")
        _results_queue = ctx.Manager().Queue()
    return _results_queue


class FileSystemWriterAsync(FileSystemWriter):
    """
    Async-enabled implementation of FileSystemWriter using file I/O.

    This class does not spawn the async process itself but relies on an external async mechanism.

    **Flow:**

    1. Call `write_data`
    2. Externally start an async process with `get_save_function_and_args` and its arguments.
    3. The async function `writer_proxy_func` calls `write_preloaded_data` across multiple
        processes.
    4. Once saving is finalized on all ranks, call `super().finish` with the results stored
        in `self.writer_result`.

    **Note:** Step (3) can also be executed synchronously.

    Currently, it is assumed that a separate writer is created for each ckpt save
    (intermediate state is stored as writer attributes).
    """

    # Class-level cache to track identifiers that have been sent to worker across instances
    _cached_identifiers: set = set()

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *args,
        separation_hint: Optional[str] = None,
        use_msc: bool = False,
        sequential: bool = False,
        use_cached_data_structure: bool = False,
        **kwargs,
    ):
        self.checkpoint_dir = path
        self.use_msc = use_msc

        super().__init__(path, *args, **kwargs)
        if not self.single_file_per_rank:
            raise NotImplementedError(
                "single_file_per_rank flag not supported for FileSystemWriterAsync"
            )

        self.can_run_decentralized_global_plan: bool = True

        # Intermediate state between preparation and finalization
        self.resolved_plan_data: Optional[ResolvedPlanData] = None
        self.has_data_to_write: bool = False
        self.results_queue: Optional[mp.Queue] = None
        self.separation_hint = separation_hint
        self.sequential = sequential
        self.use_cached_data_structure = use_cached_data_structure
        self.consistent_data_structure = None
        self.consistent_data_identifier: Optional[ConsistentDataIdentifier] = None

    def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
        """
        First stage of async saving. Resolve data and store in compact format.

        Args:
            plan (SavePlan): save plan generated by the PyT Distributed compatible planner
            planner (SavePlanner): save planner used to resolve the bytes and tensor data

        Returns: None, but stores the resolved plan data in `self.resolved_plan_data`
        """
        start = time()
        logger.debug(f"thread_count: {self.thread_count}, time: {start}")
        if self.separation_hint:
            assert (
                self.thread_count > 1
            ), "thread_count must be at least 2 if separation_hint is provided"

        def _clone_if_needed(ten: torch.Tensor):
            """Clone if we detect incontiguous storage for CPU tensors.

            Makes sure we perform a `clone` only if we detect incontiguous storage,
            so that we don't blow up host memory unnecessarily.

            For GPU tensors, returns as-is since they'll be moved to CPU later.
            """
            ten = ten.detach()
            if ten.device.type != "cpu":
                # GPU tensors will be moved to CPU in preload_tensors
                return ten
            # For CPU tensors, clone if they are views to ensure contiguous storage
            is_view = ten.untyped_storage().size() != ten.numel() * ten.itemsize
            return ten.clone() if is_view else ten

        def resolve_data(items):
            resolved = []
            for item in items:
                data = planner.resolve_data(item)
                # Apply cloning logic during resolution
                if isinstance(data, torch.Tensor):
                    data = _clone_if_needed(data)
                resolved.append(data)
            return resolved

        # Separate items by type: only GPU tensors can be cached via IPC
        # CPU tensors and ByteIO must be resolved fresh (cannot use IPC)
        tensor_items = [item for item in plan.items if item.type != WriteItemType.BYTE_IO]
        byte_io_items = [item for item in plan.items if item.type == WriteItemType.BYTE_IO]

        # Helper to separate resolved tensors by device
        def separate_by_device(items, resolved_data):
            """Separate tensor items and data into GPU and CPU categories."""
            gpu_items, gpu_data = [], []
            cpu_items, cpu_data = [], []

            for item, data in zip(items, resolved_data):
                if isinstance(data, torch.Tensor) and data.device.type == "cpu":
                    cpu_items.append(item)
                    cpu_data.append(data)
                else:
                    gpu_items.append(item)
                    gpu_data.append(data)

            return (gpu_items, gpu_data), (cpu_items, cpu_data)

        # Handle GPU tensor caching (only GPU tensors can benefit from IPC)
        # CPU tensors will be separated and treated like ByteIO (always resolved fresh)
        if self.use_cached_data_structure and tensor_items:
            key = _compute_data_structure_key_from_plan(tensor_items)
            self.consistent_data_identifier = ConsistentDataIdentifier(key)
            cache_exists = key in FileSystemWriterAsync._cached_identifiers

            # Always resolve tensors to separate CPU tensors (which can't be cached)
            resolved_tensors = resolve_data(tensor_items)
            (gpu_items, gpu_data), (cpu_items, cpu_data) = separate_by_device(
                tensor_items, resolved_tensors
            )

            if cache_exists:
                # Reuse cached GPU tensors from worker
                self.cached_tensor_data = None  # Signal to reuse cached data
                logger.debug(
                    f"Reusing cached GPU tensors (key={key}), "
                    f"resolved {len(cpu_items)} CPU tensors fresh"
                )
            else:
                # First time caching - send GPU tensor data to worker
                self.cached_tensor_data = (gpu_items, gpu_data) if gpu_items else None
                FileSystemWriterAsync._cached_identifiers.add(key)
                logger.debug(
                    f"Caching {len(gpu_items)} GPU tensors (key={key}), "
                    f"{len(cpu_items)} CPU tensors passed fresh"
                )

            # CPU tensors are always passed fresh (never cached)
            self.cpu_tensor_data = (cpu_items, cpu_data) if cpu_items else None
        else:
            # No caching - resolve and separate all tensors
            self.consistent_data_identifier = None

            if tensor_items:
                resolved_tensors = resolve_data(tensor_items)
                (gpu_items, gpu_data), (cpu_items, cpu_data) = separate_by_device(
                    tensor_items, resolved_tensors
                )
                self.cached_tensor_data = (gpu_items, gpu_data) if gpu_items else None
                self.cpu_tensor_data = (cpu_items, cpu_data) if cpu_items else None
            else:
                self.cached_tensor_data = None
                self.cpu_tensor_data = None

        # Always resolve ByteIO fresh (cannot use IPC)
        self.byte_io_data = (byte_io_items, resolve_data(byte_io_items)) if byte_io_items else None
        self.storage_plan = plan.storage_data

        # Setup results queue if there's data to write
        self.has_data_to_write = len(plan.items) > 0
        self.results_queue = get_write_results_queue() if self.has_data_to_write else None
        end = time()
        logger.debug(f"prepare_write_data, time: {end - start}")

    def get_save_function_and_args(self) -> Tuple[Optional[Callable], Optional[Callable], List]:
        """
        Get function that saves the data to storage along with its arguments.
        Allows the external caller to apply the save function synchronously or asynchronously.

        Returns: None (if there is nothing to write on this rank) or a tuple of:
            1) the function that saves the data.
            2) the function that stages the GPU tensors to a destination for async checkpointing.
               This function should be self-contained.
            3) arguments to that function in 1).
        """
        if not self.has_data_to_write:
            return None, None, []
        transform_list = [self.transforms] if hasattr(self, "transforms") else []

        # Format: (identifier, (separation_hint, cached_tensor_data,
        # cpu_tensor_data, byte_io_data, thread_count, storage_plan))
        # identifier is None when caching is disabled
        # cpu_tensor_data is always passed fresh (like ByteIO), never cached
        cpu_tensor_data = getattr(self, 'cpu_tensor_data', None)
        data_to_pass = (
            self.consistent_data_identifier,
            (
                self.separation_hint,
                self.cached_tensor_data,
                cpu_tensor_data,
                self.byte_io_data,
                self.thread_count,
                self.storage_plan,
            ),
        )

        return (
            partial(
                self.write_preloaded_data_multiproc, transform_list, self.use_msc, self.sequential
            ),
            partial(self.preload_tensors, (str(self.checkpoint_dir), data_to_pass), True),
            [torch.distributed.get_rank(), None, self.results_queue],
        )

    @staticmethod
    def preload_tensors(resolved_plan_data: Tuple, non_blocking=True) -> List[WriteBucket]:
        """
        Creates write_buckets and preloads tensors to host memory.

        Args:
            resolved_plan_data (Tuple): Tuple containing
                (checkpoint_dir, (identifier, data_structure)) where:
                - identifier: ConsistentDataIdentifier (caching) or None
                - data_structure: (separation_hint, cached_tensor_data,
                  cpu_tensor_data, byte_io_data, thread_count, storage_plan)
            non_blocking (bool, optional): Enable pinned D2H memcpy.

        Returns:
            List[WriteBucket]: List of write buckets with tensors moved to CPU
        """
        start = time()
        logger = logging.getLogger(__name__)
        # Unpack the first two elements
        checkpoint_dir, data_or_identifier = resolved_plan_data

        # Helper to combine GPU tensor, CPU tensor, and ByteIO data
        def combine_data(gpu_tensor_data, cpu_tensor_data, byte_io_data):
            items, resolved = [], []
            for data in [gpu_tensor_data, cpu_tensor_data, byte_io_data]:
                if data:
                    items.extend(data[0])
                    resolved.extend(data[1])
            return items, resolved

        # Parse data structure: (identifier, (separation_hint, cached_tensor_data,
        # cpu_tensor_data, byte_io_data, thread_count, storage_plan))
        # identifier is None when disabled, ConsistentDataIdentifier when enabled
        identifier, data_structure = data_or_identifier
        (
            separation_hint,
            cached_tensor_data,
            cpu_tensor_data,
            byte_io_data,
            thread_count,
            storage_plan,
        ) = data_structure

        if isinstance(identifier, ConsistentDataIdentifier):
            # Caching enabled: get or cache GPU tensor data
            # CPU tensors are NOT cached (treated like ByteIO)
            key = identifier.key
            if cached_tensor_data is not None:
                PersistentAsyncCaller._worker_data_cache[key] = cached_tensor_data
                logger.debug(f"Worker cached GPU tensors (key={key})")
            elif key in PersistentAsyncCaller._worker_data_cache:
                cached_tensor_data = PersistentAsyncCaller._worker_data_cache[key]
                logger.debug(f"Worker retrieved cached GPU tensors (key={key})")
            else:
                raise RuntimeError(f"Worker cache miss for key {key}. Worker may have restarted.")
        # else: identifier is None, no caching needed

        items, resolved_data = combine_data(cached_tensor_data, cpu_tensor_data, byte_io_data)

        logger.debug(f"preload_tensors: thread_count: {thread_count}, time: {start}")

        # Create buckets from items
        bins = thread_count // 2 if separation_hint is not None else thread_count
        item_buckets = _split_by_size_and_type(bins, items)
        logger.debug(f"preload_tensors: bucket_prep, time: {time() - start}")

        # Create a mapping from items to resolved data
        item_to_data = {id(item): data for item, data in zip(items, resolved_data)}

        file_count = 0

        def gen_file(prefix=""):
            nonlocal file_count
            file_name = f"{prefix}{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        # Prepare bytes / tensor data in each bucket, which will be assigned to each writer process
        # Note: cloning already done in prepare_write_data
        write_buckets = []
        for group_name, group_buckets in _split_by_separation_hint(
            item_buckets, separation_hint
        ).items():
            for bucket in group_buckets:
                bytes_data = []
                tensor_data = []
                for item in bucket:
                    data = item_to_data[id(item)]
                    if item.type == WriteItemType.BYTE_IO:
                        bytes_data.append((item, data))
                    else:
                        # Tensor data (GPU or CPU) - already cloned if needed
                        tensor_data.append((item, data))

                if len(bytes_data) > 0 or len(tensor_data) > 0:
                    file_name = gen_file(prefix=group_name)
                    write_buckets.append(
                        (  # type: ignore[arg-type]
                            os.path.join(checkpoint_dir, file_name),
                            file_name,
                            (bytes_data, tensor_data),
                        )
                    )

        # Now move GPU tensors to CPU (CPU tensors are already on CPU)
        result: List[WriteBucket] = []
        for bucket in write_buckets:  # type: ignore[assignment]
            bucket_path, bucket_key, bucket_data = bucket  # type: ignore[misc]
            bytes_data, tensor_data = bucket_data
            tensor_list = []
            for item, tensor in tensor_data:
                # we believe these tensors are detached from the model trainers
                # Move to CPU if needed (no-op if already on CPU)
                tensor_list.append((item, tensor.to("cpu", non_blocking=non_blocking)))
                # This is required for `PersistentAsyncCaller` to remove reference
                # del tensor
            result.append((bucket_path, bucket_key, (bytes_data, tensor_list)))  # type: ignore[arg-type]

        if non_blocking:
            torch.cuda.synchronize()

        end = time()
        logger.debug(f"preload_tensors: D2H and bucket creation, time: {end - start}")
        return result

    @staticmethod
    @_disable_gc()
    def write_preloaded_data_multiproc(
        transform_list: List[_StorageWriterTransforms],
        use_msc: bool,
        sequential: bool,
        rank: int,
        write_buckets: List[WriteBucket],
        global_results_queue: mp.Queue,
    ) -> None:
        """
        Performs saving data to storage with multiple processes.

        Starts predefined number of processes and uses 2 queues to make sure the results
        are complete:
        - local_results_queue - to send the actual results
        - count_queue - small queue to mark worker as completed

        Using just one queue disallowed proper exception handling.

        This method is meant to be run in a forked subprocess.
        Triggering GC during execution leads to CUDA errors
        (cleaning up tensors owned by the parent process).
        To prevent this, we disable the GC explicitly for this function with _disable_gc.

        Args:
            write_buckets (List[WriteBucket]): write plan
            global_results_queue (mp.Queue): mp.Queue to collect Dict[List[WriteResults]]
                (or an Exception) from parallel write processes to the main training process
        Returns: None
        """
        logger = logging.getLogger(__name__)
        w_start = time()
        write_results_or_exc: Union[dict, Exception] = dict()
        ctx = mp.get_context("fork")
        local_results_queue = ctx.Queue()
        count_queue = ctx.JoinableQueue()
        p_list = []

        def check_local_output(local_results_or_exc, local_proc_idx):
            if isinstance(local_results_or_exc, Exception):
                err_msg = (
                    f"Local process {local_proc_idx} encountered"
                    f" an error: {local_results_or_exc}"
                )
                logger.error(err_msg)
            assert isinstance(local_results_or_exc, list), type(local_results_or_exc)

        for i, write_bucket in enumerate(write_buckets):
            try:
                kwargs = {
                    "local_proc_idx": i,
                    "write_bucket": write_bucket,
                    "results_queue": local_results_queue,
                    "count_queue": count_queue,
                    "use_fsync": True,
                }

                if use_msc:
                    import inspect

                    # Remove the inspect after the test_async_save.py is fixed.
                    signature = inspect.signature(FileSystemWriterAsync.write_preloaded_data)
                    if len(signature.parameters) > 6:
                        kwargs['use_msc'] = use_msc
                # Parallel Writers are required
                if i < len(write_buckets) - 1 and not sequential:
                    count_queue.put(i)
                    p_list.append(
                        ctx.Process(
                            target=partial(
                                FileSystemWriterAsync.write_preloaded_data, transform_list
                            ),
                            kwargs=kwargs,
                        )
                    )
                else:
                    kwargs['count_queue'] = None
                    kwargs['results_queue'] = None
                    logger.debug('FileSystemWriterAsync: master worker started')
                    local_output = FileSystemWriterAsync.write_preloaded_data(
                        transform_list, **kwargs
                    )
                    if local_output is not None:
                        logger.debug(
                            'FileSystemWriterAsync: master worker results successfully collected'
                        )
                        check_local_output(local_output[1], local_output[0])
                        write_results_or_exc[local_output[0]] = local_output[1]

            except Exception as e:
                err_msg = f"An error is caught while a proc {i} is created, error: {e}"
                logger.error(err_msg)
                write_results_or_exc = RuntimeError(err_msg)

        if not isinstance(write_results_or_exc, Exception) and len(p_list) > 0 and not sequential:
            for p in p_list:
                p.start()

            logger.debug("FileSystemWriterAsync: collecting worker results...")

            # To make sure all nodes are completed
            count_queue.join()
            # At this point, all workers completed, so the queue should have exactly
            # `len(write_buckets)` items
            for proc_idx in range(0, len(write_buckets) - 1):
                try:
                    local_proc_idx, local_results_or_exc = local_results_queue.get()
                except queue.Empty:
                    write_results_or_exc = RuntimeError(
                        "Unexpected empty `local_results_queue`"
                        f" (got only {proc_idx}/{len(write_buckets)} items)"
                    )
                    break
                else:
                    check_local_output(local_results_or_exc, local_proc_idx)
                    write_results_or_exc[local_proc_idx] = local_results_or_exc
                    p_list[local_proc_idx].join()
            logger.debug('FileSystemWriterAsync: collected worker results successfully')

        global_results_queue.put(write_results_or_exc)

        w_end = time()
        logger.debug(f"{w_end}, rank: {rank}, write(sync,parallel): {w_end - w_start}")

    @staticmethod
    @_disable_gc()
    def write_preloaded_data(
        transform_list: List[_StorageWriterTransforms],
        local_proc_idx: int,
        write_bucket: WriteBucket,
        results_queue: mp.SimpleQueue,
        count_queue: mp.JoinableQueue,
        use_fsync: bool,
        **kwargs,
    ) -> Union[Tuple[int, Exception], None]:
        """
        Performs actual data saving to storage.

        Args:
            local_proc_idx (int): index of a local process that performs writing
            write_bucket (WriteBucket): data to write to storage
            results_queue (mp.Queue): queue to return the write results
                to the proxy checkpoint process.
            count_queue (mp.JoinableQueue): queue to marks worker task as completed
            use_fsync (bool): if True, calls os.fsync at the end of saving

        Returns: None, the write result are put into the `queue`
        """
        logger = logging.getLogger(__name__)
        logger.debug(f"{local_proc_idx} started")
        mem_before = _process_memory()
        use_msc = kwargs.get("use_msc", False)

        local_results = []
        try:
            file_name, storage_key, (bytes_data, tensor_data) = write_bucket
            extra_kwargs = {}
            if "serialization_format" in inspect.signature(_write_item).parameters:
                from torch.distributed.checkpoint.filesystem import SerializationFormat

                extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
            if use_msc:
                import multistorageclient as msc

                open_file = msc.open
            else:
                open_file = open
            with open_file(file_name, "wb") as stream:
                for write_item, data in bytes_data:
                    local_results.append(
                        _write_item(
                            *transform_list, stream, data, write_item, storage_key, **extra_kwargs
                        )
                    )

                for write_item, tensor in tensor_data:
                    assert tensor.is_cpu
                    local_results.append(
                        _write_item(
                            *transform_list, stream, tensor, write_item, storage_key, **extra_kwargs
                        )
                    )

                if use_fsync:
                    if use_msc:
                        stream.fsync()
                    else:
                        os.fsync(stream.fileno())
            local_output = (local_proc_idx, local_results)
        except Exception as e:
            logger.debug(f"{local_proc_idx} failed")
            local_output = (local_proc_idx, e)  # type: ignore[assignment]
        if results_queue is not None:
            results_queue.put(local_output)
        if count_queue is not None:
            # Signal this process is done.
            count_queue.get()
            count_queue.task_done()

        mem_after = _process_memory()
        logger.debug(
            f"{local_proc_idx} consumed: {mem_after - mem_before},"
            f" before: {mem_before}, after: {mem_after}"
        )
        return local_output

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[List[WriteResult]]:
        """Write all items from ``plan``."""
        raise NotImplementedError("write_data not implemented for FileSystemWriterAsync")

    def retrieve_write_results(self) -> Union[List[WriteResult], WRAPPED_EXCEPTION]:
        """
        Turn the latest dict including write results from `self.results_queue`
            into a single results lists. Includes error check.

        Returns (Union(List[WriteResult], WRAPPED_EXCEPTION): the list of write results
            from all local processes performing the save, or a WRAPPED_EXCEPTION if
            an exception was raised during the writing process.
        """
        # Note: consistent_data_structure can be None when caching is enabled and
        # we're reusing a cached identifier, so we don't assert on it

        if self.results_queue is None:
            write_results_or_exc = {}
        else:
            try:
                write_results_or_exc = self.results_queue.get_nowait()
            except queue.Empty:
                return _wrap_exception(RuntimeError("results_queue should not be empty"))

        if isinstance(write_results_or_exc, Exception):
            try:
                raise RuntimeError(
                    f"Worker failure: {write_results_or_exc}"
                ) from write_results_or_exc
            except Exception as e:
                return _wrap_exception(e)
        write_results: dict = write_results_or_exc
        # The number of buckets equals thread_count
        expected_bucket_count = self.thread_count
        if len(write_results) != expected_bucket_count:
            return _wrap_exception(
                RuntimeError(
                    f"Incomplete worker results (expected {expected_bucket_count},"
                    f" got {len(write_results)}. This probably indicates a worker failure."
                )
            )
        return list(chain.from_iterable(write_results.values()))

    def prepare_decentralized_global_plan(self, local_plan: SavePlan) -> SavePlan:
        """Instead of assigning indices by plan order, uses PyT rank (same outcome).

        Args:
            local_plan (SavePlan): local plan to turn to a global plan
                (without interactions with other ranks)

        Returns:
            SavePlan - locally transformed plan equivalent to the plan that would be
                created by the coordinator
        """
        return dataclasses.replace(
            local_plan, storage_data=_StoragePrefix(f"__{torch.distributed.get_rank()}_")
        )

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        """
        Finish the checkpointing process.

        Args:
            metadata (Metadata): metadata to save
            results (List[List[WriteResult]]): results to save
        """
        if self.use_msc:
            import multistorageclient as msc

            storage_md = dict()
            for wr_list in results:
                storage_md.update({wr.index: wr.storage_data for wr in wr_list})

            metadata.storage_data = storage_md
            metadata.storage_meta = self.storage_meta()

            path = os.path.join(self.checkpoint_dir, ".metadata")

            with msc.open(path, "wb") as metadata_file:
                pickle.dump(metadata, metadata_file)
        else:
            super().finish(metadata, results)

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        """
        Prepare the local plan for the checkpointing process.
        """
        if self.use_msc:
            import multistorageclient as msc

            msc.os.makedirs(str(self.checkpoint_dir), exist_ok=True)
        else:
            super().prepare_local_plan(plan)

        return plan

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return str(self.checkpoint_dir)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """
        Validate the checkpoint_id that will be used to save the checkpoint.

        This method is available in PyTorch 2.3 and above.
        """
        checkpoint_id_str = str(checkpoint_id)
        if checkpoint_id_str.startswith("msc://"):
            return True

        if hasattr(FileSystemWriter, "validate_checkpoint_id"):
            return FileSystemWriter.validate_checkpoint_id(checkpoint_id)

        return False


def _split_by_size_and_type(bins: int, items: List[WriteItem]) -> List[List[WriteItem]]:
    """
    Splits write items according to item size into close to uniform bins.

    Same as torch.distributed.checkpoint.filesystem._split_by_size_and_type,
    but with a fixed _item_size function.

    Args:
        bins (int): numbers of bins to split to
        items (List[WriteItem]): list of write items

    Returns (List[List[WriteItem]]): write items split to bins
    """
    if bins == 1:
        return [items]

    bytes_items: List[WriteItem] = []
    tensor_items: List[WriteItem] = []
    for wi in items:
        container = bytes_items if wi.type == WriteItemType.BYTE_IO else tensor_items
        container.append(wi)

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    # Assign bytes with a simple round-robin
    for i, item in enumerate(bytes_items):
        buckets[i % bins].append(item)

    # Sort tensor items by size in decreasing order once and store the size with item
    sized_tensors = [(item, _item_size(item)) for item in tensor_items]
    sized_tensors.sort(key=itemgetter(1), reverse=True)

    # Use a min heap for bin assignment
    # Store (total_size_of_bin, bin_index) tuples
    heap: List[Tuple[int, int]] = [(0, i) for i in range(bins)]

    # Assign tensors using heap
    for item, size in sized_tensors:
        total_bin_size, bin_idx = heappop(heap)
        buckets[bin_idx].append(item)
        heappush(heap, (total_bin_size + size, bin_idx))

    return buckets


def _split_by_separation_hint(
    buckets: List[List[WriteItem]], separation_hint: Optional[str] = None
) -> Dict[str, List[List[WriteItem]]]:
    """
    Splits buckets into those whose keys begin with the separation_hint and those whose keys do not

    Args:
        buckets (List[List[WriteItem]]): buckets to split
        separation_hint (Optional[str]): optional prefix to split on

    Returns (Dict[str, List[List[WriteItem]]]): a dictionary
        mapping the prefix to the relevant buckets
    """
    bins = len(buckets)
    buckets_with_separation_hint: Dict[str, List[List[WriteItem]]] = {}
    if separation_hint is not None:
        buckets_default: List[List[WriteItem]] = [[] for _ in range(bins)]
        buckets_hint: List[List[WriteItem]] = [[] for _ in range(bins)]
        for i in range(bins):
            for item in buckets[i]:
                if item.index.fqn.startswith(separation_hint):
                    buckets_hint[i].append(item)
                else:
                    buckets_default[i].append(item)
        buckets_with_separation_hint[""] = buckets_default
        buckets_with_separation_hint[separation_hint] = buckets_hint
    else:
        buckets_with_separation_hint[""] = buckets
    return buckets_with_separation_hint


def _item_size(item: WriteItem) -> int:
    """
    Calculates size (in bytes) of a single write item.

    Same as torch.distributed.checkpoint.filesystem._item_size,
    but fixes computing chunk size (with item.tensor_data.chunk.sizes)

    Args:
        item (WriteItem): write item to compute the size of

    Returns (int): size of an item in bytes
    """
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.chunk.sizes:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _process_memory() -> int:
    """
    Get memory used by current process.

    Returns (int): memory used by current process
    """
    if not HAVE_PSUTIL:
        raise RuntimeError("psutil is not installed, please install it with `pip install psutil`")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
